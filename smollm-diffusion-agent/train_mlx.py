"""
MLX training script with quantization and distributed training support.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
import argparse
from tqdm import tqdm
import wandb

from model.mlx_hybrid_model import HybridSmolLMMLX
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token

from train_mlx_utils import (
    setup_gpu,
    print_device_info,
    cosine_schedule_with_warmup,
    save_checkpoint,
    load_checkpoint,
    get_gpu_memory_mb,
)
from train_mlx_training import collate_fn, evaluate, train_step


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    base_model_id = model_cfg.get("mlx_base_model_id", model_cfg["base_model_id"])
    model = HybridSmolLMMLX(
        base_model_id=base_model_id,
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
        data_config=config,
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
                print(
                    f"Batch: input_ids={batch['input_ids'].shape}, scaffold_sum={float(mx.sum(batch['scaffold_mask']))}")
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
