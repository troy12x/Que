"""
Data utilities for InfinityFormer.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    Dataset for text data with caching support.
    """
    def __init__(
        self,
        tokenizer,
        file_path: str,
        block_size: int = 1024,
        overwrite_cache: bool = False,
        cache_dir: Optional[str] = None,
        split: str = "train",
        **kwargs
    ):
        """
        Args:
            tokenizer: Tokenizer to use for encoding text
            file_path: Path to the text file or directory containing text files
            block_size: Maximum sequence length
            overwrite_cache: Whether to overwrite the cached data
            cache_dir: Directory to store cached data
            split: Dataset split ('train', 'validation', 'test')
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.split = split
        
        # Set up cache
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(file_path), "cache")
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Get cache file path
        cached_features_file = os.path.join(
            cache_dir,
            f"cached_{split}_{block_size}_{os.path.basename(file_path)}.pt"
        )
        
        # Load from cache or process the data
        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.examples = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {file_path}")
            
            # Read the input file
            if os.path.isdir(file_path):
                # If directory, read all .txt files
                file_list = [
                    os.path.join(file_path, f) for f in os.listdir(file_path)
                    if f.endswith(".txt") and os.path.isfile(os.path.join(file_path, f))
                ]
                texts = []
                for f in file_list:
                    with open(f, "r", encoding="utf-8") as f:
                        texts.append(f.read())
                text = "\n".join(texts)
            else:
                # Single file
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            
            # Tokenize the text
            tokenized_text = tokenizer.encode(text)
            
            # Create training examples
            self.examples = []
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                self.examples.append(tokenized_text[i:i + block_size])
            
            # Save to cache
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.examples, cached_features_file)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return torch.tensor(self.examples[i], dtype=torch.long)


class DataCollatorForLanguageModeling:
    """
    Data collator for language modeling.
    """
    def __init__(
        self,
        tokenizer,
        mlm: bool = False,
        mlm_probability: float = 0.15,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer used for the model
            mlm: Whether to use masked language modeling
            mlm_probability: Probability of masking tokens for MLM
            pad_to_multiple_of: Pad sequences to a multiple of this value
        """
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(
        self, examples: List[Union[torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate function for language modeling.
        """
        # Handle dict-style examples
        if isinstance(examples[0], dict):
            examples = [e["input_ids"] for e in examples]
        
        # Pad sequences
        batch = self._tensorize_batch(examples)
        
        # If MLM is enabled, prepare masked inputs and labels
        if self.mlm:
            inputs, labels = self.mask_tokens(batch)
            return {"input_ids": inputs, "labels": labels}
        
        # For causal LM, shift input and create labels
        inputs = batch[:, :-1].contiguous()
        labels = batch[:, 1:].clone()
        
        return {"input_ids": inputs, "labels": labels}
    
    def _tensorize_batch(
        self, examples: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert a list of examples into a padded batch tensor.
        """
        # Find max length in the batch
        max_length = max(len(e) for e in examples)
        
        # Pad to multiple of pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None and (max_length % self.pad_to_multiple_of != 0):
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        
        # Create padded batch
        batch = torch.full(
            (len(examples), max_length),
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            dtype=torch.long,
        )
        
        for i, example in enumerate(examples):
            batch[i, :len(example)] = torch.tensor(example, dtype=torch.long)
        
        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling.
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling."
            )
        
        labels = inputs.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Don't mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        
        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with [MASK] token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of the time, replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        # The rest 10% keep the original token
        
        return inputs, labels


def load_dataset(
    tokenizer,
    data_dir: str,
    block_size: int = 1024,
    overwrite_cache: bool = False,
    cache_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Dataset]:
    """
    Load and process dataset from directory.
    
    Args:
        tokenizer: Tokenizer to use for encoding text
        data_dir: Directory containing train.txt, validation.txt, and test.txt
        block_size: Maximum sequence length
        overwrite_cache: Whether to overwrite the cached data
        cache_dir: Directory to store cached data
        
    Returns:
        Dictionary with train, validation, and test datasets
    """
    datasets = {}
    
    for split in ["train", "validation", "test"]:
        file_path = os.path.join(data_dir, f"{split}.txt")
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found, skipping {split} split")
            continue
        
        datasets[split] = TextDataset(
            tokenizer=tokenizer,
            file_path=file_path,
            block_size=block_size,
            overwrite_cache=overwrite_cache,
            cache_dir=cache_dir,
            split=split,
            **kwargs
        )
    
    return datasets


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    collate_fn = None,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    
    Args:
        dataset: Dataset to create DataLoader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        collate_fn: Function to collate samples into batches
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs
    )
