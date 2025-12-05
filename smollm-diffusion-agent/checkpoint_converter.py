"""
Utilities for converting checkpoints between PyTorch and MLX formats.

This allows migrating models between PyTorch and MLX backends.
"""

import torch
import mlx.core as mx
import numpy as np
import os
import json
import argparse
from typing import Dict, Any, Optional
from pathlib import Path


def pytorch_to_mlx_state_dict(pytorch_state: Dict[str, torch.Tensor]) -> Dict[str, mx.array]:
    """
    Convert PyTorch state dict to MLX format.
    
    Args:
        pytorch_state: PyTorch state dictionary
    
    Returns:
        MLX state dictionary
    """
    mlx_state = {}
    
    for key, value in pytorch_state.items():
        # Skip base_llm weights (loaded separately in MLX)
        if key.startswith("base_llm."):
            continue
        
        # Convert tensor to numpy then to MLX array
        if isinstance(value, torch.Tensor):
            numpy_array = value.detach().cpu().numpy()
            mlx_array = mx.array(numpy_array)
            mlx_state[key] = mlx_array
        else:
            mlx_state[key] = value
    
    return mlx_state


def mlx_to_pytorch_state_dict(mlx_state: Dict[str, mx.array]) -> Dict[str, torch.Tensor]:
    """
    Convert MLX state dict to PyTorch format.
    
    Args:
        mlx_state: MLX state dictionary
    
    Returns:
        PyTorch state dictionary
    """
    pytorch_state = {}
    
    for key, value in mlx_state.items():
        if isinstance(value, mx.array):
            numpy_array = np.array(value)
            torch_tensor = torch.from_numpy(numpy_array)
            pytorch_state[key] = torch_tensor
        else:
            pytorch_state[key] = value
    
    return pytorch_state


def convert_pytorch_checkpoint_to_mlx(
    pytorch_checkpoint_path: str,
    mlx_checkpoint_dir: str,
    model_config: Optional[Dict] = None,
):
    """
    Convert a PyTorch checkpoint to MLX format.
    
    Args:
        pytorch_checkpoint_path: Path to PyTorch .pt checkpoint
        mlx_checkpoint_dir: Directory to save MLX checkpoint
        model_config: Optional model configuration
    """
    print(f"Loading PyTorch checkpoint from {pytorch_checkpoint_path}")
    pytorch_checkpoint = torch.load(pytorch_checkpoint_path, map_location="cpu")
    
    # Convert model state dict
    pytorch_state = pytorch_checkpoint.get("model_state_dict", {})
    mlx_state = pytorch_to_mlx_state_dict(pytorch_state)
    
    # Create output directory
    os.makedirs(mlx_checkpoint_dir, exist_ok=True)
    
    # Save MLX model weights
    # Note: MLX uses .npz format, but we'll save as a dictionary for now
    model_path = os.path.join(mlx_checkpoint_dir, "model_weights.npz")
    
    # Convert MLX arrays to numpy for saving
    npz_dict = {}
    for key, value in mlx_state.items():
        if isinstance(value, mx.array):
            npz_dict[key] = np.array(value)
        else:
            npz_dict[key] = value
    
    np.savez(model_path, **npz_dict)
    print(f"Saved model weights to {model_path}")
    
    # Save metadata
    metadata = {
        "epoch": pytorch_checkpoint.get("epoch", 0),
        "eval_loss": pytorch_checkpoint.get("eval_loss", float('inf')),
        "config": pytorch_checkpoint.get("config", model_config),
    }
    
    metadata_path = os.path.join(mlx_checkpoint_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    # Note: Optimizer state conversion is more complex and may not be needed
    # if starting fresh training in MLX
    
    print(f"Conversion complete! MLX checkpoint saved to {mlx_checkpoint_dir}")


def convert_mlx_checkpoint_to_pytorch(
    mlx_checkpoint_dir: str,
    pytorch_checkpoint_path: str,
):
    """
    Convert an MLX checkpoint to PyTorch format.
    
    Args:
        mlx_checkpoint_dir: Directory containing MLX checkpoint
        pytorch_checkpoint_path: Path to save PyTorch checkpoint
    """
    print(f"Loading MLX checkpoint from {mlx_checkpoint_dir}")
    
    # Load model weights
    model_path = os.path.join(mlx_checkpoint_dir, "model_weights.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    npz_data = np.load(model_path)
    mlx_state = {key: mx.array(value) for key, value in npz_data.items()}
    
    # Convert to PyTorch
    pytorch_state = mlx_to_pytorch_state_dict(mlx_state)
    
    # Load metadata
    metadata_path = os.path.join(mlx_checkpoint_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Create PyTorch checkpoint
    pytorch_checkpoint = {
        "model_state_dict": pytorch_state,
        "epoch": metadata.get("epoch", 0),
        "eval_loss": metadata.get("eval_loss", float('inf')),
        "config": metadata.get("config", {}),
    }
    
    # Save
    torch.save(pytorch_checkpoint, pytorch_checkpoint_path)
    print(f"Saved PyTorch checkpoint to {pytorch_checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert checkpoints between PyTorch and MLX")
    parser.add_argument("--from-format", type=str, choices=["pytorch", "mlx"], required=True)
    parser.add_argument("--to-format", type=str, choices=["pytorch", "mlx"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Input checkpoint path")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    
    args = parser.parse_args()
    
    if args.from_format == "pytorch" and args.to_format == "mlx":
        convert_pytorch_checkpoint_to_mlx(args.input, args.output)
    elif args.from_format == "mlx" and args.to_format == "pytorch":
        convert_mlx_checkpoint_to_pytorch(args.input, args.output)
    else:
        print(f"Conversion from {args.from_format} to {args.to_format} not supported")

