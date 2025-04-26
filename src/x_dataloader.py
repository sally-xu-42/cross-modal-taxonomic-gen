"""
dataloader.py

Wrapper for CLEVR dataset loading and processing.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Union, Callable, Optional, Type

import torch
from PIL import Image
from prismatic_vlms.dataset import AlignDataset
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

# Constants
IGNORE_INDEX = -100  # HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)

class CLEVRDataset(AlignDataset):
    """
    A wrapper class for the CLEVR dataset that extends the AlignDataset class.
    CLEVR is a dataset for visual question answering with programmatically generated
    scenes containing 3D objects with various attributes.
    """
    def __init__(
        self,
        questions_json: Path,
        image_dir: Path,
        image_transform,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "val",
        max_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialize the CLEVR dataset wrapper.
        
        Args:
            questions_json: Path to the CLEVR questions JSON file (e.g., 'CLEVR_val_questions.json')
            image_dir: Path to the directory containing CLEVR images
            image_transform: Transformation to apply to the images
            tokenizer: Tokenizer to use for processing text
            split: Dataset split ('train', 'val', or 'test')
            max_samples: Maximum number of samples to load (None for all)
            seed: Random seed for sampling
        """
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.dataset_type = "clevr"
        self.split = split
        
        # Create Prompt Template
        self.prompt_template = "Question: {question}" + self.tokenizer.eos_token
        self.answer_template = "Answer: {answer}" + self.tokenizer.eos_token
        
        # Load CLEVR questions
        with open(questions_json, "r") as f:
            data = json.load(f)
            questions = data["questions"]
            
        # Sample data if max_samples is specified
        if max_samples is not None and max_samples < len(questions):
            random.seed(seed)
            questions = random.sample(questions, max_samples)
        
        # Convert to the format expected by __getitem__
        self.examples = []
        for q in questions:
            image_filename = q["image_filename"]
            question_text = q["question"]
            answer_text = q["answer"]
            
            # Create conversation format similar to AlignDataset
            self.examples.append({
                "image": image_filename,
                "conversations": [
                    {"from": "human", "value": f"{question_text}\n<image>"},
                    {"from": "gpt", "value": answer_text}
                ],
                # Store additional CLEVR-specific metadata
                "question_type": q.get("question_type", ""),
                "question_family_index": q.get("question_family_index", 0),
                "program": q.get("program", [])
            })
            
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the CLEVR dataset.
        
        Args:
            idx: Index to retrieve from the dataset.
            
        Returns:
            Dictionary containing pixel_values, input_ids, and labels
        """
        example = self.examples[idx]
        image_path = example["image"]
        conversation = example["conversations"]
        
        # Format Question
        question = conversation[0]["value"].replace("<image>", "").strip()
        answer = conversation[1]["value"].strip()
        
        # For training: predict the answer given the question and image
        caption = self.answer_template.format(answer=answer)
                
        # Tokenize the caption (answer)
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)
        
        # Set the <BOS> token's label to IGNORE_INDEX
        labels[0] = IGNORE_INDEX
        
        # Process Image
        img_path = self.image_dir / image_path
        pixel_values = self.image_transform(Image.open(img_path).convert("RGB"))
        
        # Create question embedding as well (for conditioning)
        question_ids = self.tokenizer(
            self.prompt_template.format(question=question), 
            truncation=True, 
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "pixel_values": pixel_values, 
            "input_ids": input_ids, 
            "labels": labels,
            "question_ids": question_ids,
            "metadata": {
                "question": question,
                "answer": answer,
                "question_type": example.get("question_type", ""),
                "question_family_index": example.get("question_family_index", 0)
            }
        }
    
    def get_program(self, idx: int) -> List[Dict]:
        """
        Get the functional program for a given example.
        CLEVR includes functional programs that can be executed to derive the answer.
        
        Args:
            idx: Index of the example
            
        Returns:
            The functional program as a list of operations
        """
        return self.examples[idx].get("program", [])
    
    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            # All CLEVR examples are multimodal
            is_multimodal = True
            question = example["conversations"][0]["value"].replace("<image>", "")
            answer = example["conversations"][1]["value"]
            n_words = len(question.split()) + len(answer.split())
            modality_lengths.append((is_multimodal, n_image_patches + n_words))
        return modality_lengths