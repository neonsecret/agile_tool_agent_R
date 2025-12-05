"""
MLX training script with quantization and distributed training support.

Usage:
    # Single machine training (GPU automatically used on Apple Silicon)
    python train_mlx.py --config config.yaml

    # Distributed training across multiple Macs
    python -m mlx.distributed.launch --hostfile hostfile.txt train_mlx.py --config config.yaml
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
import os
import json
import argparse
import torch
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import math
import wandb

from model.mlx_hybrid_model import HybridSmolLMMLX, log_softmax
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token


def all_reduce_grads(grads, world):
    """
    Average gradients across all processes (all nodes' GPUs).

    This synchronizes gradients across all nodes in the distributed setup.
    Each node computes gradients on its local GPU, then gradients are
    averaged across all nodes.

    Args:
        grads: Gradient dictionary (tree structure)
        world: Distributed world object

    Returns:
        Averaged gradients
    """
    world_size = world.size() if callable(world.size) else world.size

    if world_size == 1:
        return grads

    def reduce_fn(x):
        # all_sum collects and sums gradients from all nodes
        summed = mx.distributed.all_sum(x)
        # Average by dividing by number of nodes
        return summed / world_size

    # Apply reduction to all gradient arrays (recursively handle nested structures)
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
        # Try new API first
        mem_info = mx.metal.get_memory_info()
        # Keys may be "active_memory", "peak_memory", or similar
        for key in ["active_memory", "current_memory", "used_memory"]:
            if key in mem_info:
                return mem_info[key] / (1024 * 1024)
        # If dict has numeric values, return first one
        for v in mem_info.values():
            if isinstance(v, (int, float)):
                return v / (1024 * 1024)
    except Exception:
        pass
    # Fallback to deprecated API if new one fails
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


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def cosine_schedule_with_warmup(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup (matches PyTorch version)."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def clip_grad_norm(grads: Dict, max_norm: float) -> Dict:
    """Clip gradient norm (MLX version of torch.nn.utils.clip_grad_norm_)."""

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


def collate_fn(batch: list) -> Dict[str, mx.array]:
    """Collate function for batching (converts torch tensors from dataset to MLX)."""

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.numpy()
        return np.array(x)

    input_ids = [to_numpy(item['input_ids']) for item in batch]
    attention_mask = [to_numpy(item['attention_mask']) for item in batch]
    scaffold_mask = [to_numpy(item['scaffold_mask']) for item in batch]
    labels = [to_numpy(item['labels']) for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    def pad_sequence(seqs, pad_value):
        return mx.array(np.stack([
            np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value)
            for seq in seqs
        ]))

    batch_dict = {
        "input_ids": pad_sequence(input_ids, 0),
        "attention_mask": pad_sequence(attention_mask, 0),
        "scaffold_mask": pad_sequence(scaffold_mask, False),
        "labels": pad_sequence(labels, -100),
    }

    if 'router_label' in batch[0]:
        batch_dict["router_labels"] = mx.array([item['router_label'] for item in batch])

    return batch_dict


def evaluate(
        model: HybridSmolLMMLX,
        eval_loader: list,
        train_router: bool,
) -> Dict[str, float]:
    """Evaluate the model on validation set."""
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_router_loss = 0.0
    total_router_correct = 0
    total_router_samples = 0
    num_eval_batches = 0

    for batch in eval_loader:
        current_router_labels = batch.get("router_labels") if train_router else None

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            scaffold_mask=batch["scaffold_mask"],
            router_labels=current_router_labels,
            training=False,
        )
        mx.eval(outputs["loss"])  # Force evaluation

        if outputs["loss"] is not None:
            loss_val = float(outputs["loss"])
            total_loss += loss_val
            num_eval_batches += 1

            losses_detail = outputs.get("losses", {})
            if "diffusion" in losses_detail:
                mx.eval(losses_detail["diffusion"])
                total_diffusion_loss += float(losses_detail["diffusion"])
            if "router" in losses_detail:
                mx.eval(losses_detail["router"])
                total_router_loss += float(losses_detail["router"])

            # Calculate router accuracy
            if train_router and "router_logits" in outputs and current_router_labels is not None:
                router_preds = mx.argmax(outputs["router_logits"], axis=-1)
                correct = mx.sum(router_preds == current_router_labels)
                mx.eval(correct)
                total_router_correct += int(correct)
                total_router_samples += current_router_labels.shape[0]

    metrics = {
        "eval/total_loss": total_loss / max(num_eval_batches, 1),
        "eval/diffusion_loss": total_diffusion_loss / max(num_eval_batches, 1),
    }

    if train_router and total_router_samples > 0:
        metrics["eval/router_loss"] = total_router_loss / max(num_eval_batches, 1)
        metrics["eval/router_accuracy"] = total_router_correct / total_router_samples

    return metrics


def check_nan(arr, name):
    """Check if array contains NaN and return debug info."""
    if isinstance(arr, mx.array):
        has_nan = mx.any(mx.isnan(arr))
        mx.eval(has_nan)
        if has_nan:
            return f"{name}: NaN detected! shape={arr.shape}, dtype={arr.dtype}"
    return None


def train_step(
        model: HybridSmolLMMLX,
        batch: Dict,
        optimizer: optim.Optimizer,
        train_router: bool,
        world=None,
        max_grad_norm: float = 1.0,
        debug: bool = False,
) -> Tuple[mx.array, Optional[Dict]]:
    """Single training step with gradient clipping. Returns (loss, debug_info)."""
    hidden_states = model._get_hidden_states(batch["input_ids"], batch.get("attention_mask"))
    mx.eval(hidden_states)

    # Debug: check hidden states
    if debug or True:  # Always check for now
        hs_check = check_nan(hidden_states, "hidden_states")
        if hs_check:
            return mx.array(float('nan')), {"reason": "hidden_states_nan", "details": hs_check}

    local_debug = {}

    def loss_fn(diff_params, router_params):
        model.diffusion_head.update(diff_params)
        model.router_head.update(router_params)

        total_loss = mx.array(0.0, dtype=mx.float32)
        labels = batch["labels"]
        scaffold_mask = batch["scaffold_mask"]

        if labels is not None and scaffold_mask is not None:
            diff_loss = model.diffusion_head.training_step(
                tokens=labels,
                hidden_states=hidden_states,
                scaffold_mask=scaffold_mask,
            )
            total_loss = total_loss + diff_loss
            if float(diff_loss) == 0.0:
                local_debug["diffusion_zero_loss"] = {
                    "scaffold_sum": float(mx.sum(scaffold_mask.astype(mx.float32))),
                    "reason": "no_mask_or_no_valid",
                }

        if train_router:
            router_labels = batch.get("router_labels")
            if router_labels is not None:
                router_logits = model.router_head(hidden_states)
                log_probs = log_softmax(router_logits, axis=-1)
                nll = -mx.take_along_axis(
                    log_probs,
                    mx.expand_dims(router_labels.astype(mx.int32), -1),
                    axis=-1
                ).squeeze(-1)
                total_loss = total_loss + mx.mean(nll)

        return total_loss

    diff_params = dict(model.diffusion_head.parameters())
    router_params = dict(model.router_head.parameters())

    loss_and_grad_fn = mx.value_and_grad(loss_fn, argnums=(0, 1))
    loss, (diff_grads, router_grads) = loss_and_grad_fn(diff_params, router_params)
    mx.eval(loss)

    # Check loss for NaN before continuing
    loss_val = float(loss)
    if np.isnan(loss_val):
        # Debug: find which part has NaN
        debug_msg = f"NaN in loss computation. hidden_states: min={float(mx.min(hidden_states)):.4f}, max={float(mx.max(hidden_states)):.4f}"
        return loss, debug_msg

    # Distributed gradient sync
    if world is not None and world.size() > 1:
        diff_grads = all_reduce_grads(diff_grads, world)
        if train_router:
            router_grads = all_reduce_grads(router_grads, world)
        loss = mx.distributed.all_sum(loss) / world.size()
        mx.eval(loss)

    # Gradient clipping
    diff_grads = clip_grad_norm(diff_grads, max_grad_norm)
    if train_router:
        router_grads = clip_grad_norm(router_grads, max_grad_norm)

    # Apply optimizer updates
    new_diff_params = optimizer.apply_gradients(diff_grads, diff_params)
    model.diffusion_head.update(new_diff_params)

    if train_router:
        new_router_params = optimizer.apply_gradients(router_grads, router_params)
        model.router_head.update(new_router_params)

    return loss, local_debug


def save_checkpoint(
        model: HybridSmolLMMLX,
        optimizer: optim.Optimizer,
        epoch: int,
        eval_loss: float,
        config: Dict,
        save_dir: str,
):
    """Save model checkpoint (only trainable heads)."""
    os.makedirs(save_dir, exist_ok=True)

    # Save only trainable head weights (not base model)
    diffusion_weights = dict(model.diffusion_head.parameters())
    router_weights = dict(model.router_head.parameters())

    # Flatten to npz format
    flat_weights = {}
    for name, param in diffusion_weights.items():
        flat_weights[f"diffusion_head.{name}"] = np.array(param)
    for name, param in router_weights.items():
        flat_weights[f"router_head.{name}"] = np.array(param)

    model_path = os.path.join(save_dir, "model.npz")
    np.savez(model_path, **flat_weights)

    # Save metadata
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
        model: HybridSmolLMMLX,
) -> Tuple[int, float]:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")

    # Load model weights
    model_path = os.path.join(checkpoint_path, "model.npz")
    if os.path.exists(model_path):
        data = np.load(model_path)

        # Separate diffusion_head and router_head weights
        diff_weights = {}
        router_weights = {}

        for key in data.files:
            if key.startswith("diffusion_head."):
                name = key.replace("diffusion_head.", "")
                diff_weights[name] = mx.array(data[key])
            elif key.startswith("router_head."):
                name = key.replace("router_head.", "")
                router_weights[name] = mx.array(data[key])

        # Load weights into model
        if diff_weights:
            model.diffusion_head.update(diff_weights)
        if router_weights:
            model.router_head.update(router_weights)

        print(f"Loaded {len(diff_weights)} diffusion_head weights, {len(router_weights)} router_head weights")

    # Load metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        start_epoch = metadata.get("epoch", 0) + 1
        best_eval_loss = metadata.get("eval_loss", float('inf'))
        print(f"Resumed from epoch {metadata.get('epoch', 0)}, best eval loss: {best_eval_loss:.4f}")
        return start_epoch, best_eval_loss

    return 0, float('inf')


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (disable GPU)")
    args = parser.parse_args()

    # Setup GPU (Metal) - MLX uses Metal GPU by default on Apple Silicon
    print("=" * 60)
    print("MLX Training - Device Setup")
    print("=" * 60)

    if args.cpu:
        mx.set_default_device(mx.Device(mx.DeviceType.cpu))
        print("CPU mode forced via --cpu flag")
    else:
        gpu_available = setup_gpu()
        if not gpu_available:
            print("Warning: Training will run on CPU (slower)")

    print_device_info()
    print("=" * 60)

    # Initialize distributed training if available
    world = None
    try:
        world = mx.distributed.init()
        if world.size() > 1:
            print(f"Distributed training initialized:")
            print(f"  Rank: {world.rank()}/{world.size()}")
            print(f"  Device on this node: {mx.default_device()}")
    except Exception as e:
        print(f"Distributed training not available: {e}")
        print("Continuing with single-machine training")

    # Load config
    config = load_config(args.config)
    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    diff_cfg = config["diffusion"]
    quant_cfg = config.get("quantization", {})

    # Initialize wandb (only on main process for distributed)
    use_wandb = training_cfg.get("use_wandb", False)
    is_main_process = world is None or world.rank() == 0

    if use_wandb and is_main_process:
        wandb.init(
            project=training_cfg.get("wandb_project", "smollm-diffusion-mlx"),
            config={
                **config,
                "backend": "mlx",
                "device": str(mx.default_device()),
                "distributed": world is not None and world.size() > 1,
                "world_size": world.size() if world else 1,
            },
            name=f"mlx-{model_cfg['base_model_id'].split('/')[-1]}",
        )
        print("Weights & Biases logging enabled")

    # Set random seed
    mx.random.seed(training_cfg["seed"])
    np.random.seed(training_cfg["seed"])

    # Load model
    print("\nLoading model and tokenizer...")

    # Determine quantization setting from unified config
    quantize_enabled = quant_cfg.get("enabled", False)
    quantize_bits = quant_cfg.get("bits", 4) if quantize_enabled else None

    if quantize_enabled:
        if quantize_bits in [4, 8]:
            print(f"Quantization: {quantize_bits}-bit (MLX native)")
        else:
            print(f"Warning: {quantize_bits}-bit quantization not supported, using {4}-bit")
            quantize_bits = 4
    else:
        print("Quantization: disabled")
        quantize_bits = None

    model = HybridSmolLMMLX(
        base_model_id=model_cfg["base_model_id"],
        quantize_bits=quantize_bits,
        diffusion_config=diff_cfg,
        vocab_size=model_cfg.get("vocab_size"),
    )

    tokenizer = model.tokenizer

    # Resolve mask and null tokens
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    null_token_config = data_cfg.get("null_token", None)
    null_token_str, null_token_id = None, None
    if null_token_config is not None or data_cfg.get("use_null_token", True):
        try:
            null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)
            print(f"NULL token: {null_token_str} (ID: {null_token_id})")
        except ValueError as e:
            print(f"Warning: NULL token not available: {e}")

    print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")

    # Set mask/null token IDs in diffusion head
    model.diffusion_head.set_mask_token_id(mask_token_id)
    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)

    # Setup dataset
    print("Loading dataset...")
    full_dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        mask_token=mask_token_str,
        null_token=null_token_str,
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48),
    )

    # Split dataset
    eval_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - eval_size

    indices = np.arange(len(full_dataset))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:]

    # For distributed training, each rank gets a subset
    if world and world.size() > 1:
        rank_train_size = train_size // world.size()
        start_idx = world.rank() * rank_train_size
        end_idx = start_idx + rank_train_size if world.rank() < world.size() - 1 else train_size
        train_indices = train_indices[start_idx:end_idx]
        print(f"Rank {world.rank()}: Using {len(train_indices)} training samples")

    train_dataset = [full_dataset[i] for i in train_indices]
    eval_dataset = [full_dataset[i] for i in eval_indices]

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    # Create data loaders
    def create_loader(dataset, batch_size, shuffle=True):
        batches = []
        dataset_copy = list(dataset)
        if shuffle:
            np.random.shuffle(dataset_copy)
        for i in range(0, len(dataset_copy), batch_size):
            batch_data = dataset_copy[i:i + batch_size]
            batches.append(collate_fn(batch_data))
        return batches

    train_loader = create_loader(train_dataset, training_cfg["batch_size"], shuffle=True)
    eval_loader = create_loader(eval_dataset, training_cfg["batch_size"], shuffle=False)

    # Setup optimizer with LR scheduling (matches train.py)
    train_router = training_cfg["train_router"]
    base_lr = float(training_cfg["learning_rate"])
    num_epochs = training_cfg["num_epochs"]
    total_steps = num_epochs * len(train_loader)
    warmup_steps = min(training_cfg.get("warmup_steps", 2500), int(0.1 * total_steps))

    optimizer = optim.AdamW(learning_rate=base_lr)
    print(f"LR Schedule: Cosine with warmup ({warmup_steps} warmup / {total_steps} total steps)")

    # Load checkpoint if resuming
    start_epoch = 0
    best_eval_loss = float('inf')
    global_step = 0

    if args.resume or training_cfg.get("resume_from_checkpoint", False):
        checkpoint_path = args.resume or training_cfg.get("checkpoint_path", "checkpoints/best_model")
        start_epoch, best_eval_loss = load_checkpoint(checkpoint_path, model)
        global_step = start_epoch * len(train_loader)
        print(f"Starting from epoch {start_epoch}, global step {global_step}")

    print("Starting training loop...")

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        train_loader = create_loader(train_dataset, training_cfg["batch_size"], shuffle=True)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            disable=(world is not None and world.rank() != 0),
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Update learning rate with schedule
            current_lr = cosine_schedule_with_warmup(global_step, warmup_steps, total_steps, base_lr)
            optimizer.learning_rate = current_lr

            loss, debug_info = train_step(model, batch, optimizer, train_router, world)
            loss_val = float(loss)

            if np.isnan(loss_val):
                print(f"\nNaN loss at batch {batch_idx}!")
                print(f"Batch: input_ids={batch['input_ids'].shape}, scaffold_sum={float(mx.sum(batch['scaffold_mask']))}")
                if debug_info:
                    print(f"Debug: {debug_info}")
                labels = batch['labels'][0][:30].astype(mx.int32)
                mx.eval(labels)
                print(f"First 30 labels: {labels.tolist()}")
                raise ValueError("NaN loss - stopping training")

            if loss_val == 0.0 and debug_info:
                print(f"[Zero loss debug] batch {batch_idx}: {debug_info}")

            epoch_loss += loss_val
            num_batches += 1
            global_step += 1

            gpu_mem = get_gpu_memory_mb()
            postfix = {"loss": f"{loss_val:.4f}", "lr": f"{current_lr:.2e}"}
            if gpu_mem > 0:
                postfix["GPU_MB"] = f"{gpu_mem:.0f}"
            progress_bar.set_postfix(postfix)

            if use_wandb and is_main_process and global_step % 10 == 0:
                log_dict = {
                    "train/loss": loss_val,
                    "train/learning_rate": current_lr,
                    "train/step": global_step,
                    "train/epoch": epoch + 1,
                }
                if gpu_mem > 0:
                    log_dict["system/gpu_memory_mb"] = gpu_mem
                wandb.log(log_dict, step=global_step)

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        gpu_mem = get_gpu_memory_mb()

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        print(f"  Device: {mx.default_device()}")
        if gpu_mem > 0:
            print(f"  GPU Memory: {gpu_mem:.1f} MB")

        # Evaluation
        print("Running evaluation...")
        eval_metrics = evaluate(model, eval_loader, train_router)

        print(f"Eval Results:")
        print(f"  Eval Loss: {eval_metrics['eval/total_loss']:.4f}")
        print(f"  Eval Diffusion Loss: {eval_metrics['eval/diffusion_loss']:.4f}")
        if train_router and 'eval/router_accuracy' in eval_metrics:
            print(f"  Eval Router Accuracy: {eval_metrics['eval/router_accuracy']:.2%}")

        # Log epoch metrics to wandb
        if use_wandb and is_main_process:
            epoch_log = {
                "epoch": epoch + 1,
                "train/epoch_loss": avg_epoch_loss,
                **eval_metrics,
            }
            if gpu_mem > 0:
                epoch_log["system/gpu_memory_mb"] = gpu_mem
            wandb.log(epoch_log, step=global_step)

        # Save best model
        if eval_metrics['eval/total_loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['eval/total_loss']
            print(f"New best model! Eval loss: {best_eval_loss:.4f}")

            if world is None or world.rank() == 0:
                save_checkpoint(model, optimizer, epoch, best_eval_loss, config, "checkpoints/best_model_mlx")

        print("-" * 80)

    print(f"\nTraining complete! Best eval loss: {best_eval_loss:.4f}")

    # Finish wandb run
    if use_wandb and is_main_process:
        wandb.finish()


if __name__ == "__main__":
    train()
