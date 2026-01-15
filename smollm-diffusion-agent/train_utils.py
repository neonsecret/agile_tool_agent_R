"""
Training utilities and helper functions.

Extracted from train.py for better code organization.
"""

import torch
import yaml
from typing import Dict, Optional
from accelerate import Accelerator


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_collate_fn(bucket_sizes, max_seq_len, pad_token_id: int):
    """Create a collate_fn that pads to the next bucket size >= batch max length.

    This keeps shapes more stable for torch.compile and CUDA graphs.
    """
    bucket_sizes = sorted(bucket_sizes)

    def collate_fn(batch):
        max_len = max(item["input_ids"].size(0) for item in batch)
        pad_to = None
        for b in bucket_sizes:
            if b >= max_len:
                pad_to = b
                break
        if pad_to is None:
            pad_to = bucket_sizes[-1]
        pad_to = min(pad_to, max_seq_len)

        def pad_tensor(t: torch.Tensor, pad_value: int):
            if t.size(0) > pad_to:
                return t[:pad_to]
            if t.size(0) == pad_to:
                return t
            pad_width = pad_to - t.size(0)
            return torch.nn.functional.pad(t, (0, pad_width), value=pad_value)

        input_ids = [pad_tensor(item["input_ids"], pad_value=pad_token_id) for item in batch]
        attention_mask = [pad_tensor(item["attention_mask"], pad_value=0) for item in batch]
        scaffold_mask = [pad_tensor(item["scaffold_mask"], pad_value=0) for item in batch]
        labels = [pad_tensor(item["labels"], pad_value=-100) for item in batch]

        batch_dict = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "scaffold_mask": torch.stack(scaffold_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

        return batch_dict

    return collate_fn


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    import os
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        accelerator.print("Loaded scheduler state from checkpoint")

    start_epoch = checkpoint['epoch'] + 1
    best_eval_loss = checkpoint['eval_loss']

    accelerator.print(f"Resumed from epoch {checkpoint['epoch']}, best eval loss: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss
