import os
import re
import sys
import json
import copy
import argparse
import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from draccus import decode
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Subset
from transformers import PreTrainedTokenizerBase

from prismatic import load, available_model_ids
from prismatic.conf import ModelConfig, DatasetConfig, DatasetRegistry
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.vision import ImageTransform


IGNORE_INDEX = -100

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
        self.prompts = {"in front of": "In this 3D scene, `in front of` means closer to the camera/viewer (smaller depth value). ",
                        "behind": "In this 3D scene, `behind` means farther away from the camera/viewer (larger depth value). "}
        with open(self.chat_json, "r") as f:
            self.examples = json.load(f)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        During the "eval" phase, we return plain text prompts, and the model is expected to generate
        the answer based on the image and the prompt.
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"
        # question = "Question: " + conversation[0]["value"] + "Answer:"
        q = conversation[0]["value"]
        if True: 
            if identify_relation(q) in ["behind", "in front of"]:
                # special prompts for front and behind
                question = self.prompts[identify_relation(q)] + "Question: " + q + "Answer:"
            else:
                question = "Question: " + q + "Answer:"
        else:
            question = "Question: " + q + "Answer:"
        answer = conversation[-1]["value"]
        input_ids = self.tokenizer(question, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)
        labels[0] = IGNORE_INDEX
        image = Image.open(self.image_dir / image_path).convert("RGB")
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, 
                    input_text=question, 
                    pure_question=q, 
                    answer=answer, 
                    input_ids=input_ids, 
                    labels=labels, 
                    image=image)

    # def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    #     """
    #     Match FinetuneDataset format exactly for consistent evaluation.
    #     """
    #     image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
    #     assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"
        
    #     # Use the same prompt builder as training
    #     from prismatic.models.backbones.llm.prompting.llama2_chat_prompter import LLaMa2ChatPromptBuilder
    
    #     # Create Prompt Builder --> add each message sequentially (same as FinetuneDataset)
    #     prompt_builder, input_ids, labels = LLaMa2ChatPromptBuilder(model_family="prismatic"), [], []
        
    #     for turn_idx, turn in enumerate(conversation):
    #         # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
    #         msg = prompt_builder.add_turn(turn["from"], turn["value"])
    #         msg = msg.rstrip()
            
    #         # Tokenize Input IDs
    #         turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
            
    #         # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
    #         turn_labels = (
    #             [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
    #         )
    #         # Add to Trackers
    #         input_ids.extend(turn_input_ids)
    #         labels.extend(turn_labels)
        
    #     # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
    #     input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
    
    #     # Handle Truncation (if necessary)
    #     # input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[:, : self.tokenizer.model_max_length]
    #     input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]
        
    #     # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
    #     labels[0] = IGNORE_INDEX
        
    #     # Process Image
    #     image = Image.open(self.image_dir / image_path).convert("RGB")
    #     pixel_values = self.image_transform(image)
        
    #     return dict(
    #         pixel_values=pixel_values, 
    #         input_ids=input_ids, 
    #         labels=labels,
    #         # Keep evaluation-specific fields for compatibility
    #         input_text=prompt_builder.get_prompt(),  # Get the full formatted prompt
    #         pure_question=conversation[0]["value"], 
    #         answer=conversation[-1]["value"], 
    #         image=image
    #     )

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
        input_text, pure_question, answer, input_ids, labels, image = tuple([instance[key] for instance in instances] for key in ("input_text", "pure_question", "answer", "input_ids", "labels", "image"))
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
            pure_question=pure_question,
            answer=answer,
            input_ids=input_ids,
            image=image,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
        )


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


def identify_relation(question: str) -> str:
    """Identify the spatial relation in a question."""
    spatial_relations = ["right", "left", "behind", "in front of"]
    
    for relation in spatial_relations:
        pattern = r'\b' + re.escape(relation) + r'\b'
        if re.search(pattern, question.lower()):
            return relation
    
    return "other"
