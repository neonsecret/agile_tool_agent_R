"""
Helper functions for inference operations.

Extracted from inference.py to improve code organization.
"""

from typing import Dict, Any, List, Optional, Tuple, Sequence
import torch
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _is_cuda_graph_supported(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_available()


def _can_use_torch_compile_mps(device: torch.device) -> bool:
    """Check if torch.compile works on MPS (requires PyTorch 2.1+)."""
    if device.type != "mps" or not torch.backends.mps.is_available():
        return False

    try:
        version = torch.__version__.split('.')
        major, minor = int(version[0]), int(version[1])
        return major >= 2 and minor >= 1
    except (ValueError, IndexError):
        return False


def _default_budget_config() -> Dict[str, int]:
    from data.budget_utils import MIN_FIELD_BUDGET, DEFAULT_MAX_BUDGET
    
    config = load_config()
    data_cfg = config.get("data", {})
    dynamic_budget_cfg = data_cfg.get("dynamic_budget", {})
    max_tokens = dynamic_budget_cfg.get(
        "max_tokens",
        data_cfg.get("mask_budget", DEFAULT_MAX_BUDGET),
    )
    min_tokens = dynamic_budget_cfg.get("min_tokens", MIN_FIELD_BUDGET)
    extra_tokens = dynamic_budget_cfg.get("extra_tokens", 0)
    return {
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "extra_tokens": extra_tokens,
    }


def _default_expansion_config() -> Dict[str, Any]:
    config = load_config()
    infer_cfg = config.get("inference", {})
    expansion_cfg = infer_cfg.get("expansion", {})
    return {
        "enabled": expansion_cfg.get("enabled", False),
        "max_rounds": expansion_cfg.get("max_rounds", 0),
        "expand_tokens": expansion_cfg.get("expand_tokens", 4),
        "tail_window": expansion_cfg.get("tail_window", 4),
        "tail_null_threshold": expansion_cfg.get("tail_null_threshold", 0.5),
    }


def _apply_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply Gumbel noise for temperature-based sampling."""
    if temperature <= 0.0:
        return logits
    noise = torch.rand_like(logits)
    noise = noise.clamp_min(1e-10)
    gumbel = -torch.log(-torch.log(noise))
    return logits / temperature + gumbel
