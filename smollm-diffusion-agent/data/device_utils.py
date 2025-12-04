"""Device utility functions for cross-platform support (CUDA, MPS, CPU)."""

import torch


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def empty_cache(device: torch.device):
    """Empty cache for the given device (CUDA or MPS)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def synchronize(device: torch.device):
    """Synchronize device operations (CUDA or MPS)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def get_device_map_for_quantization(device: torch.device):
    """Get device_map parameter for model loading with quantization.
    
    For 4-bit quantization with bitsandbytes, only CUDA is supported.
    For MPS, use device_map="auto" or None and handle device placement manually.
    """
    if device.type == "cuda":
        return {"": 0}  # bitsandbytes requires this format
    elif device.type == "mps":
        # MPS doesn't support bitsandbytes, so this shouldn't be called
        # But if it is, return None to let transformers handle it
        return None
    else:
        return None

