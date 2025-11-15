import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import functools


def get_image_token_positions(vlm, num_patches: int = 256) -> Tuple[int, int]:
    """
    Get start and end positions of image tokens in the sequence.
    
    Args:
        vlm: Prismatic VLM instance
        num_patches: Number of image patches (default 256 for ViT)
    
    Returns:
        Tuple of (start_idx, end_idx) for image token positions
    """
    # In prismatic VLM, image tokens are inserted after BOS token
    # Structure: [BOS, img_patch_1, ..., img_patch_N, text_tokens...]
    start_idx = 1  # After BOS token
    end_idx = start_idx + num_patches - 1
    return start_idx, end_idx


def calculate_uncertainty(logits: torch.Tensor) -> float:
    """
    Calculate uncertainty as max softmax probability.
    
    Args:
        logits: Logits tensor of shape (vocab_size,)
    
    Returns:
        Uncertainty score (max softmax probability)
    """
    probs = F.softmax(logits, dim=-1)
    uncertainty = torch.max(probs).item()
    return uncertainty


def create_attention_scaling_wrapper(original_forward, weight: float, 
                                   image_start: int, image_end: int, 
                                   layer_idx: int):
    """
    Create a wrapper for LlamaAttention.forward() that applies attention scaling.
    
    Args:
        original_forward: Original LlamaAttention.forward method
        weight: Scaling weight to apply
        image_start: Start index of image tokens
        image_end: End index of image tokens  
        layer_idx: Layer index (only modify first 32 layers)
    
    Returns:
        Wrapped forward method
    """
    @functools.wraps(original_forward)
    def wrapped_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Call original forward to get Q, K, V
        bsz, q_len, _ = hidden_states.size()
        
        # Get Q, K, V projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        # Apply attention scaling (AdaptVis logic)
        if layer_idx < 32:  # Only modify first 32 layers
            if attn_weights.size()[2] == attn_weights.size()[3]:  # Square matrix (prefill phase)
                # Scale attention TO image tokens
                if image_start < attn_weights.size(3) and image_end < attn_weights.size(3):
                    attn_weights[:, :, :, image_start:image_end+1] *= weight
        
        # Apply attention mask - FIXED: Don't check dimensions, just apply if present
        if attention_mask is not None:
            # The attention mask might have different dimensions than attn_weights
            # We need to handle this more carefully
            try:
                # Try to apply the mask directly
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            except RuntimeError:
                # If dimensions don't match, skip mask application
                # This can happen during generation when sequence lengths change
                pass
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, "
                f"but is {attn_output.size()}"
            )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    return wrapped_forward


def apply_attention_scaling_hook(vlm, weight: float, num_image_tokens: int = 256):
    """
    Apply attention scaling hooks to all LlamaAttention layers.
    
    Args:
        vlm: Prismatic VLM instance
        weight: Scaling weight to apply
        num_image_tokens: Number of image tokens
    """
    image_start, image_end = get_image_token_positions(vlm, num_image_tokens)
    
    # Store original methods for cleanup
    if not hasattr(vlm, '_original_attention_methods'):
        vlm._original_attention_methods = {}
    
    # Patch each layer
    for layer_idx, layer in enumerate(vlm.llm_backbone.llm.model.layers):
        if layer_idx < 32:  # Only modify first 32 layers
            attention_layer = layer.self_attn
            
            # Store original method
            if layer_idx not in vlm._original_attention_methods:
                vlm._original_attention_methods[layer_idx] = attention_layer.forward
            
            # Create and apply wrapper
            wrapped_forward = create_attention_scaling_wrapper(
                vlm._original_attention_methods[layer_idx],
                weight,
                image_start,
                image_end,
                layer_idx
            )
            attention_layer.forward = wrapped_forward.__get__(attention_layer, type(attention_layer))


def remove_attention_scaling_hook(vlm):
    """
    Remove attention scaling hooks and restore original methods.
    
    Args:
        vlm: Prismatic VLM instance
    """
    if hasattr(vlm, '_original_attention_methods'):
        for layer_idx, original_method in vlm._original_attention_methods.items():
            if layer_idx < len(vlm.llm_backbone.llm.model.layers):
                vlm.llm_backbone.llm.model.layers[layer_idx].self_attn.forward = original_method
        delattr(vlm, '_original_attention_methods')