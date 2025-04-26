import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Union, Optional
from torchtyping import TensorType
from transformers import GPTJForCausalLM, GPT2TokenizerFast
from transformers.file_utils import ModelOutput
from PIL import Image
import requests
from io import BytesIO

# Import required components from existing code
from image_prefix import ImagePrefix
from config import MultimodalConfig
from sampling import generate
from utils import build_labels
from transforms import get_transforms


class ImageInput:
    """Class to handle image inputs (either from URL or file)"""
    def __init__(self, source):
        self.source = source
        self.image = None
        
    def get_image(self):
        if self.image is not None:
            return self.image
            
        if isinstance(self.source, str) and self.source.startswith(('http://', 'https://')):
            response = requests.get(self.source)
            self.image = Image.open(BytesIO(response.content)).convert('RGB')
        elif isinstance(self.source, str):
            self.image = Image.open(self.source).convert('RGB')
        elif isinstance(self.source, Image.Image):
            self.image = self.source
        else:
            raise ValueError(f"Unsupported image source type: {type(self.source)}")
            
        return self.image
        
    def get_transformed_image(self, transform_fn=None):
        image = self.get_image()
        if transform_fn is not None:
            return transform_fn(image).unsqueeze(0)  # Add batch dimension
        return image


def get_tokenizer(name, sequence_length=2048):
    """Gets tokenizer for LM"""
    if name == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token_id = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.model_max_length = sequence_length
        # setup lm settings
        tokenizer.add_special_tokens(
            {"cls_token": "<|image|>"}
        )  # add special image token to tokenizer
    else:
        raise ValueError(f"Tokenizer {name} not recognized")
    return tokenizer


class LimberGPTJLightning(pl.LightningModule):
    """PyTorch Lightning implementation of the LIMBER model"""
    
    def __init__(self, config_path, limber_proj_path='auto'):
        super().__init__()
        self.save_hyperparameters()
        self.config_path = config_path
        self.limber_proj_path = limber_proj_path
        
        # Load language model
        self.model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B", 
            revision="float16", 
            torch_dtype=torch.bfloat16
        )
        
        # Setup multimodal components
        self.setup_multimodal(config_path)
        
        # Load projection weights
        self.load_projection_weights(limber_proj_path)
        
    def setup_multimodal(self, config):
        """Sets up multimodal components for the model"""
        if isinstance(config, (str, os.PathLike)):
            config = MultimodalConfig.from_yml(config)
        else:
            assert isinstance(config, MultimodalConfig)
            
        self.multimodal_config = config
        self.seq_len = self.model.config.max_position_embeddings
        self.tokenizer = get_tokenizer('gpt2', sequence_length=self.seq_len)

        self.image_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.word_embedding = self.model.transformer.wte

        if config.freeze_lm:
            for name, param in self.model.named_parameters():
                if config.adapter_config and "adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Create image prefix component
        self.image_prefix = ImagePrefix(
            config=config,
            out_dim=self.model.config.hidden_size,
        )

        self.image_prefix_seq_len = self.image_prefix.out_seq_len
        if 'identity' in config.encoder_name or 'nfresnet' in config.encoder_name:
            inp_rez = None
        else:
            inp_rez = self.image_prefix.enc.input_resolution
            
        self.transforms = get_transforms(
            config.image_size,
            config.encoder_name,
            input_resolution=inp_rez,
        )

        if config.freeze_img_encoder:
            for param in self.image_prefix.enc.parameters():
                param.requires_grad = False
                
    def load_projection_weights(self, limber_proj_path='auto'):
        """Loads the projection weights for the image prefix"""
        if limber_proj_path == 'auto':
            config_path = self.config_path
            if config_path.endswith("beit_ft_linear.yml"):
                limber_proj_path = 'limber_weights/beit_ft_linear/proj.ckpt'
            elif config_path.endswith("beit_linear.yml"):
                limber_proj_path = 'limber_weights/beit_linear/proj.ckpt'
            elif config_path.endswith("nfrn50_4096_linear.yml"):
                limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
            elif config_path.endswith('nfrn50_4096_random_linear.yml'):
                limber_proj_path = 'limber_weights/nfrn50_4096_linear/proj.ckpt'
            elif config_path.endswith('clip_linear.yml'):
                limber_proj_path = 'limber_weights/clip_linear/proj.ckpt'
        
        print(f"Loading projection weights from: {limber_proj_path}")
        proj_ckpt = torch.load(limber_proj_path)
        self.image_prefix.proj.load_state_dict(proj_ckpt)
                
    def preprocess_inputs(self, input_list: List[Union[str, ImageInput]], embed=True) -> List[torch.Tensor]:
        """
        Preprocesses a list of strings and instances of ImageInput
        Converts them into a list of tensors and optionally embeds them
        """
        for i in range(len(input_list)):
            inp = input_list[i]
            if isinstance(inp, str):
                input_list[i] = self.tokenizer.encode(inp, return_tensors="pt")
            elif isinstance(inp, ImageInput):
                input_list[i] = inp.get_transformed_image(transform_fn=self.transforms)
            else:
                raise ValueError(f'Invalid input type: {type(inp)}')

        if embed:
            return self.embed(input_list)
        else: 
            return input_list

    def embed(self, inputs: List[torch.Tensor]) -> TensorType["b", "s", "d"]:
        """
        Embeds a list of tensors in the correct format to input into the LM (b, s, d).
        For each tensor, if it's 2d assume it's text and use word embedding,
        if it's 4d, assume it's an image, and use image_prefix to embed.
        """
        emb_list = []
        for x in inputs:
            if x.ndim == 2:
                x = x.to(self.device)
                emb_list.append(self.word_embedding(x))
            elif x.ndim == 4:
                x = x.to(self.device).half()
                image_embeddings = self.image_prefix(x)
                emb_list.append(image_embeddings)
            elif x.ndim == 3:
                x = x.unsqueeze(0).to(self.device).half()
                image_embeddings = self.image_prefix(x)
                emb_list.append(image_embeddings)
            else:
                raise ValueError(f"Expected 2d or 4d tensor, got {x.ndim}d")
        return torch.cat(emb_list, dim=1)

    @torch.no_grad()
    def generate(self, embeddings: TensorType["b", "s", "d"], 
                max_steps: int = 100,
                temperature: float = 0.7,
                top_k: int = 0,
                top_p: float = 0.9,
                decode: bool = True):
        """
        Generates captions for a batch of embeddings.
        """
        return generate(
            self.model,
            embeddings=embeddings,
            max_steps=max_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            decode=decode,
        )
        
    def forward(self, 
               images: Optional[TensorType["b", "c", "h", "w"]] = None,
               captions: Optional[TensorType["b", "seq"]] = None,
               output_hidden_states: bool = True,
               input_embeddings: Optional[TensorType["b", "s", "d"]] = None,
               attention_mask=None,
               return_dict=None) -> ModelOutput:
        """
        Forward pass for the model
        """
        assert captions is not None, "Must provide captions in training"
        assert any([i is not None for i in [images, input_embeddings]]) and not all(
            [i is not None for i in [images, input_embeddings]]
        ), "Pass in either images, or input embeddings, not both."

        if input_embeddings is None:
            input_embeddings = self.image_prefix(images)

        labels = build_labels(
            input_embeddings, captions, self.eos_token, self.device
        )  # build labels from input_embeddings

        word_embeddings = self.word_embedding(captions)
        input_embeddings = torch.cat(
            (
                input_embeddings,
                word_embeddings,
            ),
            dim=1,
        )
        
        return_dict = return_dict if return_dict is not None else self.model.config.use_return_dict
        lm_outputs = self.model(
            inputs_embeds=input_embeddings,
            labels=labels,
            output_hidden_states=output_hidden_states,
            attention_mask=attention_mask,
            use_cache=False
        )
        return lm_outputs
    
    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        images, captions = batch
        outputs = self(images=images, captions=captions)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        images, captions = batch
        outputs = self(images=images, captions=captions)
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for training"""
        # In inference mode, we don't need an optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.multimodal_config.lr
        )
        return optimizer