import os
import re
import sys
import json
import argparse
import torch
import pandas as pd
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from draccus import decode
from collections import defaultdict
from typing import Dict, List, Tuple
from torch.utils.data import Subset

from prismatic import load, available_model_ids
from prismatic.conf import ModelConfig, DatasetConfig, DatasetRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from eval_utils import get_dataset_and_collator
from adaptvis_utils import (
    apply_attention_scaling_hook, 
    remove_attention_scaling_hook,
    calculate_uncertainty,
    get_image_token_positions
)

IGNORE_INDEX = -100


def identify_relation(question: str) -> str:
    """Identify the spatial relation in a question."""
    spatial_relations = ["right", "left", "behind", "in front of"]
    
    for relation in spatial_relations:
        pattern = r'\b' + re.escape(relation) + r'\b'
        if re.search(pattern, question.lower()):
            return relation
    
    return "other"


def calculate_accuracy(results: List[Dict]) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Calculate accuracy overall and per relation type."""
    relation_results = defaultdict(list)
    
    for result in results:
        relation = result.get("relation", "other")
        is_correct = result["correct"]
        relation_results[relation].append(is_correct)

    all_correct = sum(result["correct"] for result in results)
    all_illegal = sum(result["illegal"] for result in results)
    overall_accuracy = all_correct / len(results) if results else 0
    total_illegal_ratio = all_illegal / len(results) if results else 0
    
    relation_accuracy = {}
    relation_counts = {}
    
    for relation, outcomes in relation_results.items():
        correct = sum(outcomes)
        total = len(outcomes)
        relation_accuracy[relation] = correct / total if total > 0 else 0
        relation_counts[relation] = total
    
    stats_data = {
        'Relation': list(relation_accuracy.keys()),
        'Count': [relation_counts[rel] for rel in relation_accuracy.keys()],
        'Correct': [int(relation_accuracy[rel] * relation_counts[rel]) for rel in relation_accuracy.keys()],
        'Accuracy': [f"{relation_accuracy[rel] * 100:.1f}%" for rel in relation_accuracy.keys()]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    accuracy_dict = {
        "overall": overall_accuracy,
        "total_illegal_ratio": total_illegal_ratio,
        **relation_accuracy
    }
    
    return accuracy_dict, stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on CLEVR spatial relations with AdaptVis")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./runs/dino+siglip-llama-42",
        help="Path to pretrained model or model ID"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/clevr_val_qa_preprocessed.json",
        help="Path to the preprocessed CLEVR question dataset"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default="./data/CLEVR_v1.0/images",
        help="Directory containing CLEVR images"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./evaluation_results.csv",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--max_samples", 
        type=int, 
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    # AdaptVis-specific arguments
    parser.add_argument(
        "--method",
        type=str,
        default="base",
        choices=["base", "scaling_vis", "adapt_vis"],
        help="Evaluation method: base, scaling_vis, or adapt_vis"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.2,
        help="Weight for ScalingVis method"
    )
    parser.add_argument(
        "--weight1",
        type=float,
        default=0.5,
        help="Weight1 for AdaptVis (low uncertainty)"
    )
    parser.add_argument(
        "--weight2",
        type=float,
        default=1.2,
        help="Weight2 for AdaptVis (high uncertainty)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Uncertainty threshold for AdaptVis"
    )
    parser.add_argument(
        "--num_image_tokens",
        type=int,
        default=256,
        help="Number of image tokens (patches)"
    )
    
    args = parser.parse_args()

    if args.model_path in available_model_ids(): # Load a pretrained VLM by ID to auto-download from the HF Hub)
        hf_token = os.environ.get(".hf_token", None)
        model_id = args.model_path
        vlm = load(model_id, hf_token=hf_token)
        # Evaluate on CLEVR
        for dataset_variant in DatasetRegistry:
            if dataset_variant.dataset_id == "clevr-mini":
                dataset_cfg = dataset_variant.value()
        if args.dataset_path:
            dataset_cfg.finetune_stage_components = (
                Path(args.dataset_path),  # By default, this is the preprocessed CLEVR dataset validation split
                Path("data/CLEVR_v1.0/images")
            )
        vision_backbone = vlm.vision_backbone
        llm_backbone = vlm.llm_backbone
        image_transform = vision_backbone.get_image_transform()
        tokenizer = llm_backbone.tokenizer
        inference_mode = True # Pretrained models are always in inference mode
    else:
        if os.path.isdir(args.model_path):
            config_path = os.path.join(args.model_path, "config.json")
        else:
            model_path = os.path.dirname(os.path.dirname(args.model_path))
            config_path = os.path.join(model_path, "config.json")
        
        if "align" in args.model_path:
            print("Evaluation in the ALIGN stage (no fine-tuned LLM weights)")
            inference_mode = False
        elif "finetune" in args.model_path:
            print("Evaluation in the FINETUNE stage (with fine-tuned LLM weights)")
            inference_mode = True
        else:
            print("Warning: Could not determine stage from model_path. Assuming FINETUNE stage.")
            inference_mode = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.environ.get(".hf_token", None)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            cfg = decode(ModelConfig, config["model"])

        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.vision_backbone_id,
            image_resize_strategy=cfg.image_resize_strategy,
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            cfg.llm_backbone_id, 
            llm_max_length=cfg.llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode # >>>>>>>> Set this to True only during finetuning inference because we don't need base weights
        )
        vlm = get_vlm(
            cfg.model_id,
            cfg.arch_specifier,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=cfg.enable_mixed_precision_training,
        )
        
        if args.model_path.endswith(".pt"): # eval on checkpoints
            checkpoint_path = args.model_path
            parent = os.path.dirname(os.path.normpath(checkpoint_path)) 
            model_dir = os.path.basename(os.path.dirname(parent)) # e.g., model name
            base = os.path.basename(checkpoint_path) # e.g., "step-1000.pt"
            name = base.split(".")[0] # e.g., "step-1000"
            os.makedirs(f"./results/{model_dir}", exist_ok=True)
            output_path = f"./results/{model_dir}/evaluation_{name}.csv"
        else: # eval on the full model
            checkpoint_path = os.path.join(args.model_path, "checkpoints", "latest-checkpoint.pt")

        print(f"Loading checkpoint from {checkpoint_path}")    
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("Available keys in checkpoint['model']:", list(checkpoint["model"].keys()))
        print("Loading projector weights from model.projector")

        vlm.projector.load_state_dict(checkpoint["model"]["projector"])
        dataset_cfg = decode(DatasetConfig, config["dataset"])
        dataset_cfg.dataset_root_dir = Path("/share/data/speech/txu/vlm_semantics")
        if inference_mode: # Finetune stage
            print("Loading fine-tuned LLM weights from model.llm_backbone") # Load fine-tuned LLM weights
            vlm.llm_backbone.load_state_dict(checkpoint["model"]["llm_backbone"], strict=False)
            dataset_cfg.finetune_stage_components = [args.dataset_path, "data/CLEVR_v1.0/images"]
        else: # Align stage
            print("Warning: No LLM backbone weights found in checkpoint")
            dataset_cfg.align_stage_components = [args.dataset_path, "data/CLEVR_v1.0/images"]
    
    vlm.to(torch.cuda.current_device())
    vlm.projector.to(torch.cuda.current_device())
    vlm.requires_grad_(False)
    vlm.eval()
    print("Model loaded successfully.")
    
    print(f"Loading dataset: {args.dataset_path}")
    val_dataset, collator = get_dataset_and_collator(
        stage="finetune" if inference_mode else "align",
        dataset_cfg=dataset_cfg,
        image_transform=image_transform,
        tokenizer=tokenizer,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side
    )
    # Sample a subset of the dataset if max_samples is specified
    # Sample the first `max_samples` of samples, not randomized
    if args.max_samples is not None:
        val_dataset = Subset(val_dataset, range(args.max_samples))
    print(f"Loaded {len(val_dataset)} samples from dataset.")

    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        collate_fn=collator,
        shuffle=False,
        num_workers=1
    )
    print(f"Evaluating questions with method: {args.method}")
    
    results = []
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        images = batch["image"]
        prompts = batch['input_text']
        pure_prompts = batch['pure_question']
        prompts_text = []
        for prompt in prompts:
            prompt_text = prompt
            print(f"[Debug] Full prompt:\n{prompt_text}\n") 
            prompts_text.append(prompt_text)
        prompts = prompts_text
        answers = batch['answer']
        candidates = ["Yes", "No"] # ======= added for candidate scoring

        # process one by one
        print(prompts[0])
        for i in range(len(prompts)):
            if args.method == "base":
                # Standard evaluation without attention scaling
                output, _ = vlm.candidate_scoring(
                    images[i],
                    prompts[i],
                    candidates,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0
                )
                
            elif args.method == "scaling_vis":
                # Apply fixed weight scaling
                apply_attention_scaling_hook(vlm, args.weight, args.num_image_tokens)
                output, _ = vlm.candidate_scoring(
                    images[i],
                    prompts[i],
                    candidates,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0
                )
                remove_attention_scaling_hook(vlm)
                
            elif args.method == "adapt_vis":
                # Two-pass evaluation with uncertainty-based weight selection
                # First pass: Generate with weight=1.0 to measure uncertainty
                apply_attention_scaling_hook(vlm, 1.0, args.num_image_tokens)
                output_base = vlm.generate(
                    images[i],
                    prompts[i],
                    max_new_tokens=1,
                    temperature=0.0,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                remove_attention_scaling_hook(vlm)
                
                # Calculate uncertainty
                uncertainty = calculate_uncertainty(output_base.scores[0][0])
                print(f"Uncertainty: {uncertainty:.3f}, Threshold: {args.threshold}")
                
                # Select weight based on uncertainty
                weight = args.weight1 if uncertainty < args.threshold else args.weight2
                print(f"Selected weight: {weight}")
                
                # Second pass: Generate with selected weight
                apply_attention_scaling_hook(vlm, weight, args.num_image_tokens)
                output, _ = vlm.candidate_scoring(
                    images[i],
                    prompts[i],
                    candidates,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0
                )
                remove_attention_scaling_hook(vlm)
            
            predicted = output.strip().lower()
            predicted = re.sub(r'[^\w\s]', '', predicted).strip()
            print(f"Ground truth answer: {answers[i]}, Model's answer: {predicted}\n")
            results.append({
                "question": prompts[i],
                "answer": answers[i],
                "predicted_answer": predicted,
                "relation": identify_relation(pure_prompts[i]),
                "correct": answers[i].strip().lower() == predicted.lower().strip(),
                "illegal": predicted not in ["yes", "no"]
            })

    accuracy_dict, stats_df = calculate_accuracy(results)
    
    print("\nEvaluation Results:")
    print(f"Method: {args.method}")
    if args.method == "scaling_vis":
        print(f"Weight: {args.weight}")
    elif args.method == "adapt_vis":
        print(f"Weight1: {args.weight1}, Weight2: {args.weight2}, Threshold: {args.threshold}")
    print(f"Overall Accuracy: {accuracy_dict['overall'] * 100:.1f}%")
    print(f"Total Illegal Ratio: {accuracy_dict['total_illegal_ratio'] * 100:.1f}%")
    print("\nAccuracy by Relation Type:")
    print(stats_df.to_string(index=False))