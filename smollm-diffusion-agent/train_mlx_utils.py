"""
MLX training utilities.

Helper functions for distributed training, checkpointing, and memory management on Apple Silicon.
"""

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import os
import json
import math
from typing import Dict, Optional, Tuple


def all_reduce_grads(grads, world):
    """
    Average gradients across all processes (all nodes' GPUs).

    This synchronizes gradients across all nodes in the distributed setup.
    """
    world_size = world.size() if callable(world.size) else world.size

    if world_size == 1:
        return grads

    def reduce_fn(x):
        summed = mx.distributed.all_sum(x)
        return summed / world_size

    if isinstance(grads, dict):
        return {k: all_reduce_grads(v, world) if isinstance(v, (dict, list, tuple))
        else reduce_fn(v) for k, v in grads.items()}
    elif isinstance(grads, (list, tuple)):
        return type(grads)(all_reduce_grads(g, world) if isinstance(g, (dict, list, tuple))
                           else reduce_fn(g) for g in grads)
    else:
        return reduce_fn(grads)


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        mem_info = mx.metal.get_memory_info()
        for key in ["active_memory", "current_memory", "used_memory"]:
            if key in mem_info:
                return mem_info[key] / (1024 * 1024)
        for v in mem_info.values():
            if isinstance(v, (int, float)):
                return v / (1024 * 1024)
    except Exception:
        pass
    try:
        return mx.metal.get_active_memory() / (1024 * 1024)
    except Exception:
        return 0.0


def setup_gpu() -> bool:
    """Ensure MLX is using GPU (Metal) for computation."""
    if mx.metal.is_available():
        mx.set_default_device(mx.Device(mx.DeviceType.gpu))
        print("GPU (Metal) enabled as default device")
        return True
    print("Warning: Metal GPU not available, using CPU")
    return False


def print_device_info():
    """Print current device information."""
    print(f"MLX Device Info:")
    print(f"  Default device: {mx.default_device()}")
    print(f"  Metal (GPU) available: {mx.metal.is_available()}")
    gpu_mem = get_gpu_memory_mb()
    if gpu_mem > 0:
        print(f"  GPU memory in use: {gpu_mem:.1f} MB")


def cosine_schedule_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def clip_grad_norm(grads: Dict, max_norm: float) -> Dict:
    """Clip gradient norm."""

    def collect_norms(g):
        if isinstance(g, mx.array):
            return [float(mx.sum(g * g))]
        elif isinstance(g, dict):
            norms = []
            for v in g.values():
                norms.extend(collect_norms(v))
            return norms
        return []

    def scale_grads(g, coef):
        if isinstance(g, mx.array):
            return g * coef
        elif isinstance(g, dict):
            return {k: scale_grads(v, coef) for k, v in g.items()}
        return g

    norms = collect_norms(grads)
    total_norm = math.sqrt(sum(norms)) if norms else 0.0
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        return scale_grads(grads, clip_coef)
    return grads


def save_checkpoint(
        model,
        optimizer: optim.Optimizer,
        epoch: int,
        eval_loss: float,
        config: Dict,
        save_dir: str,
):
    """Save model checkpoint (only trainable heads)."""
    os.makedirs(save_dir, exist_ok=True)

    diffusion_weights = dict(model.diffusion_head.parameters())
    router_weights = dict(model.router_head.parameters())

    flat_weights = {}
    for name, param in diffusion_weights.items():
        flat_weights[f"diffusion_head.{name}"] = np.array(param)
    for name, param in router_weights.items():
        flat_weights[f"router_head.{name}"] = np.array(param)

    model_path = os.path.join(save_dir, "model.npz")
    np.savez(model_path, **flat_weights)

    metadata = {
        "epoch": epoch,
        "eval_loss": eval_loss,
        "config": config,
    }
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved checkpoint to {save_dir}")


def load_checkpoint(
        checkpoint_path: str,
        model,
) -> Tuple[int, float]:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")

    model_path = os.path.join(checkpoint_path, "model.npz")
    if os.path.exists(model_path):
        data = np.load(model_path)

        diff_weights = {}
        router_weights = {}

        for key in data.files:
            if key.startswith("diffusion_head."):
                name = key.replace("diffusion_head.", "")
                diff_weights[name] = mx.array(data[key])
            elif key.startswith("router_head."):
                name = key.replace("router_head.", "")
                router_weights[name] = mx.array(data[key])

        if diff_weights:
            model.diffusion_head.update(diff_weights)
        if router_weights:
            model.router_head.update(router_weights)

        print(f"Loaded {len(diff_weights)} diffusion_head weights, {len(router_weights)} router_head weights")

    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        start_epoch = metadata.get("epoch", 0) + 1
        best_eval_loss = metadata.get("eval_loss", float('inf'))
        print(f"Resumed from epoch {metadata.get('epoch', 0)}, best eval loss: {best_eval_loss:.4f}")
        return start_epoch, best_eval_loss

    return 0, float('inf')


def check_nan(arr, name):
    """Check if array contains NaN and return debug info."""
    if isinstance(arr, mx.array):
        has_nan = mx.any(mx.isnan(arr))
        mx.eval(has_nan)
        if has_nan:
            return f"{name}: NaN detected! shape={arr.shape}, dtype={arr.dtype}"
    return None
