import os
import re
import json
import copy
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from typing import Type
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizerBase

from draccus import decode
from prismatic.conf import ModelConfig, DatasetConfig
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
# from prismatic.preprocessing import get_dataset_and_collator
from prismatic.models.backbones.vision import ImageTransform

IGNORE_INDEX = -100

def get_dataset_and_collator(
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
):
    dataset_cls = EvalDataset
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForEval(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    annotation_json, image_dir = dataset_cfg.align_stage_components
    dataset = dataset_cls(
        dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
    )
    return dataset, collator

class EvalDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "eval"
        self.prompt_template = "{caption}" + self.tokenizer.eos_token
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        During the "eval" phase, we return plain text prompts, and the model is expected to generate
        the answer based on the image and the prompt.
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"
        caption = self.prompt_template.format(caption=("Question: " + conversation[0]["value"] + "Answer: ").strip())
        answer = self.prompt_template.format(caption=(conversation[-1]["value"]).strip())
        answer = answer.replace("</s>", "").strip()
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)
        labels[0] = IGNORE_INDEX
        image = Image.open(self.image_dir / image_path).convert("RGB")
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))
        return dict(pixel_values=pixel_values, input_text=caption, answer=answer, input_ids=input_ids, labels=labels, image=image)

    def __len__(self) -> int:
        return len(self.examples)

@dataclass
class PaddedCollatorForEval:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_text, answer, input_ids, labels, image = tuple([instance[key] for instance in instances] for key in ("input_text", "answer", "input_ids", "labels", "image"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]

        attention_mask = input_ids.ne(self.pad_token_id)
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )
        if len(multimodal_indices) == 0:
            pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_text=input_text,
            answer=answer,
            input_ids=input_ids,
            image=image,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


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
    overall_accuracy = all_correct / len(results) if results else 0
    
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
        'Accuracy': [f"{relation_accuracy[rel] * 100:.2f}%" for rel in relation_accuracy.keys()]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    accuracy_dict = {
        "overall": overall_accuracy,
        **relation_accuracy
    }
    
    return accuracy_dict, stats_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLM on CLEVR spatial relations")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./runs/train-clevr-align-42",
        help="Path to pretrained model or model ID"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="data/simple_clevr_val_preprocessed.json",
        help="Path to the filtered CLEVR question dataset"
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
    args = parser.parse_args()

    config_path = os.path.join(args.model_path, "config.json") if os.path.isdir(args.model_path) else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        hf_token=config["hf_token"]
    )

    vlm = get_vlm(
        cfg.model_id,
        cfg.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.enable_mixed_precision_training,
    )
    
    checkpoint_path = os.path.join(args.model_path, "checkpoints", "latest-checkpoint.pt")
    print(f"Loading checkpoint from {checkpoint_path}")    
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    print("Loading projector weights from model.projector")
    vlm.projector.load_state_dict(checkpoint["model"]["projector"])
    
    vlm.to(torch.cuda.current_device())
    vlm.projector.to(torch.cuda.current_device())
    vlm.requires_grad_(False)
    vlm.eval()
    print("Model loaded successfully.")
    
    print(f"Loading dataset: {args.dataset_path}")
    dataset_cfg = decode(DatasetConfig, config["dataset"])
    dataset_cfg.align_stage_components = [args.dataset_path, "data/CLEVR_v1.0/images"]
    val_dataset, collator = get_dataset_and_collator(
        dataset_cfg=dataset_cfg,
        image_transform=image_transform,
        tokenizer=tokenizer,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side
    )
    # Sample a subset of the dataset if max_samples is specified
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
    print(f"Evaluating questions...")
    
    results = []
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        images = batch["image"]
        prompts = batch['input_text']
        input_ids = batch['input_ids']
        labels = batch['labels']
        answers = batch['answer']
        # process one by one
        for i in range(len(prompts)):
            output = vlm.generate(
                images[i],
                prompts[i],
                max_new_tokens=20,
                temperature=None
            )
            predicted = output.strip().lower()
            # print(f"\nQuestion {i+1} input IDs:{input_ids[i]}")
            # print(f"\nQuestion {i+1} labels:{labels[i]}")
            print(f"\nQuestion {i+1} answer:{answers[i]}")
            print(f"\nModel's answer:{predicted}")
            predicted = re.sub(r'[^\w\s]', '', predicted).strip()
            results.append({
                "question": prompts[i],
                "answer": answers[i],
                "predicted_answer": predicted,
                "relation": identify_relation(prompts[i]),
                "correct": answers[i].strip().lower() in predicted.lower().strip()
            })

    accuracy_dict, stats_df = calculate_accuracy(results)
    
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {accuracy_dict['overall'] * 100:.2f}%")
    print("\nAccuracy by Relation Type:")
    print(stats_df.to_string(index=False))
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to {args.output_path}")
    stats_df.to_csv(args.output_path.replace('.csv', '_summary.csv'), index=False)
    print(f"\nSummary statistics saved to {args.output_path.replace('.csv', '_summary.csv')}")