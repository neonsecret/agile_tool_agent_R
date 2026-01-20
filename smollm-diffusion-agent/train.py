import torch
import os
import logging

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "0")

logger = logging.getLogger(__name__)

try:
    import unsloth
except ImportError as e:
    logger.debug(f"unsloth not available: {e}")
import random
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import wandb
from data.device_utils import empty_cache, get_device
from data.config_utils import validate_and_adjust_config, get_model_kwargs, print_device_capabilities
from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency

from train_utils import (
    load_config,
    build_collate_fn,
    load_checkpoint,
    build_scheduler,
    build_optimizer,
)
from train_eval import (
    evaluate, _compute_null_counts, _null_metrics_from_counts,
    _sum_across_processes, _compute_diversity_counts
)
from train_functional import functional_evaluation


def train():
    # Load Config
    config = load_config()

    # Print device capabilities first (on main process only)
    if not os.environ.get("LOCAL_RANK") or int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print_device_capabilities()

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    diff_cfg = config["diffusion"]
    quant_cfg = config.get("quantization", {})
    compile_cfg = training_cfg.get("compile", {})
    bucket_sizes = data_cfg.get("bucket_sizes", [512, 1024, 1536, 2048])

    # Check if distributed training is enabled
    dist_cfg = training_cfg.get("distributed", {})
    is_distributed = dist_cfg.get("enabled", False)
    
    # 1. Initialize Accelerator with W&B FIRST (so we get proper device assignment)
    accelerator = Accelerator(
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        log_with="wandb" if training_cfg["use_wandb"] else None,
        mixed_precision="bf16"  # Use bfloat16 to match model dtype
    )
    set_seed(training_cfg["seed"])

    # NOW validate config with the accelerator's device (which respects LOCAL_RANK)
    config = validate_and_adjust_config(config, accelerator.device)

    # Disable unsloth in multi-GPU mode (not stable with DDP)
    if accelerator.num_processes > 1 and model_cfg.get("use_unsloth"):
        accelerator.print("⚠️  Multi-GPU detected: Disabling unsloth (not stable with DDP)")
        model_cfg["use_unsloth"] = False

    if training_cfg["use_wandb"]:
        accelerator.init_trackers(
            project_name=training_cfg["wandb_project"],
            config=config
        )

    accelerator.print(f"Using device: {accelerator.device} (process {accelerator.process_index}/{accelerator.num_processes})")

    # 2. Setup Tokenizer First (needed for vocab size)
    accelerator.print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve mask token from config
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    # Resolve NULL token for self-adaptive masking (variable-length fields)
    null_token_config = data_cfg.get("null_token", None)
    null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)
    accelerator.print(f"NULL token: {null_token_str} (ID: {null_token_id})")

    accelerator.print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")
    accelerator.print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # 3. Initialize Model
    accelerator.print("Loading Model...")

    # Get model kwargs using config utils (handles device-specific settings)
    model_kwargs = get_model_kwargs(config, accelerator.device)
    model_kwargs['vocab_size'] = len(tokenizer)
    model_kwargs['diffusion_config'] = diff_cfg

    model = HybridSmolLM(**model_kwargs)

    # Set mask token ID in diffusion head for proper noising
    model.diffusion_head.set_mask_token_id(mask_token_id)

    # Set NULL token ID for self-adaptive masking
    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)

    # Optional: torch.compile for training (compile the full model before prepare)
    if compile_cfg.get("enabled", False):
        compile_mode = compile_cfg.get("mode", "reduce-overhead")
        compile_fullgraph = compile_cfg.get("fullgraph", False)

        accelerator.print(f"torch.compile enabled for training (mode={compile_mode}, fullgraph={compile_fullgraph})")
        accelerator.print("Note: for best stability, pad/bucket sequences to fixed lengths per batch.")
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
            accelerator.print("Inductor CUDA graphs disabled for training.")
        except Exception as e:
            logger.warning(f"Could not configure inductor config: {e}")

        try:
            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
            )
        except Exception as e:
            accelerator.print(f"torch.compile failed, falling back to eager: {e}")

    logits_for_metrics = training_cfg.get("logits_for_metrics", True)

    # 4. Setup Dataset
    accelerator.print("Loading Dataset...")
    # Load full dataset with NULL token support for automatic budgeting
    system_message = data_cfg.get("system_message", "/no_think")
    max_history_messages = int(data_cfg.get("max_history_messages", 12))
    full_dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        mask_token=mask_token_str,
        null_token=null_token_str,  # Enables self-adaptive masking
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48),
        system_message=system_message,
        max_history_messages=max_history_messages,
        data_config=config,
    )

    # Split into train/eval (95/05 split)
    eval_size = int(0.05 * len(full_dataset))
    train_size = len(full_dataset) - eval_size
    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(training_cfg["seed"])
    )

    accelerator.print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        collate_fn=build_collate_fn(bucket_sizes, training_cfg["max_seq_len"], tokenizer.pad_token_id)
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=build_collate_fn(bucket_sizes, training_cfg["max_seq_len"], tokenizer.pad_token_id)
    )

    # 4. Optimizer
    # Only optimize diffusion head parameters
    params_to_optimize = list(model.diffusion_head.parameters())
    optimizer, optimizer_info = build_optimizer(model.diffusion_head, training_cfg)
    accelerator.print(f"Optimizer: {optimizer_info.get('name', 'optimizer')}")
    if optimizer_info.get("fallback_from"):
        accelerator.print(
            f"  Fallback from {optimizer_info['fallback_from']}: {optimizer_info['fallback_reason']}"
        )
    if optimizer_info.get("name") == "muon":
        accelerator.print(
            "  Muon lr={muon_lr:.2e}, momentum={muon_momentum:.2f}, weight_decay={muon_weight_decay:.2e}".format(
                **optimizer_info
            )
        )
        accelerator.print(
            "  AdamW lr={adamw_lr:.2e}, weight_decay={adamw_weight_decay:.2e}".format(
                **optimizer_info
            )
        )

    # 5. Prepare
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # 6. Learning Rate Scheduler
    num_epochs = training_cfg["num_epochs"]
    gradient_accumulation_steps = training_cfg["gradient_accumulation_steps"]
    num_update_steps_per_epoch = max(
        1,
        (len(train_dataloader) + gradient_accumulation_steps - 1)
        // gradient_accumulation_steps,
    )
    total_training_steps = num_epochs * num_update_steps_per_epoch
    scheduler, scheduler_info = build_scheduler(
        optimizer, training_cfg, total_training_steps
    )

    scheduler_name = scheduler_info.get("name", "scheduler")
    accelerator.print(f"LR Scheduler: {scheduler_name}")
    accelerator.print(f"  Total training steps: {total_training_steps}")
    if scheduler_name == "clamped_cosine":
        min_lr = float(training_cfg["learning_rate"]) * scheduler_info["min_lr_ratio"]
        accelerator.print(
            f"  Warmup steps: {scheduler_info['warmup_steps']}"
        )
        accelerator.print(
            f"  Min LR: {min_lr:.2e}"
        )
    if scheduler_name == "one_cycle":
        accelerator.print(
            f"  Max LR: {scheduler_info['max_lr']}"
        )
        accelerator.print(
            f"  Pct start: {scheduler_info['pct_start']:.2f}"
        )

    # 7. Load checkpoint if resuming
    start_epoch = 0
    best_eval_loss = float('inf')
    global_step = 0

    if training_cfg.get("resume_from_checkpoint", False):
        checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")
        start_epoch, best_eval_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator)
        global_step = start_epoch * num_update_steps_per_epoch
        accelerator.print(f"Starting from epoch {start_epoch}, global step {global_step}")

    # 8. Training Loop
    accelerator.print("Starting training loop...")
    model.train()

    for epoch in range(start_epoch, training_cfg["num_epochs"]):
        # Training epoch
        epoch_loss = 0
        epoch_diffusion_loss = 0
        num_batches = 0

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{training_cfg['num_epochs']}",
            disable=not accelerator.is_local_main_process
        )

        for batch in progress_bar:
            did_step = False
            # Mark CUDA graph step boundary to prevent tensor reuse issues (torch.compile feature)
            try:
                torch.compiler.cudagraph_mark_step_begin()
            except AttributeError as e:
                logger.debug(f"cudagraph_mark_step_begin not available: {e}")

            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    scaffold_mask=batch["scaffold_mask"],
                    return_logits=logits_for_metrics,
                )

                loss = outputs["loss"]

                if loss is None:
                    continue

                loss_val = loss.item()
                losses_detail = outputs.get("losses", {})

                # Track losses
                epoch_loss += loss_val
                num_batches += 1
                if "diffusion" in losses_detail:
                    epoch_diffusion_loss += losses_detail["diffusion"].item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'avg_loss': f'{epoch_loss / num_batches:.4f}'
                })

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(params_to_optimize, 1.0)

                    optimizer.step()
                    scheduler.step()  # Step LR scheduler
                    optimizer.zero_grad()
                    global_step += 1
                    did_step = True

                    # Log metrics to wandb (only on update steps)
                    logs = {"train/total_loss": loss_val, "train/step": global_step}
                    for k, v in losses_detail.items():
                        logs[f"train/{k}_loss"] = v.item() if torch.is_tensor(v) else v

                    # Add NULL token and diversity metrics every 50 steps
                    if global_step % 50 == 0 and outputs.get("logits") is not None:
                        with torch.no_grad():
                            logits = outputs["logits"]
                            predictions = torch.argmax(logits, dim=-1)
                            counts = _compute_null_counts(
                                predictions,
                                batch["labels"],
                                batch["scaffold_mask"],
                                null_token_id,
                            )
                            div_counts = _compute_diversity_counts(
                                predictions,
                                batch["scaffold_mask"],
                            )
                            if counts is not None:
                                counts.update(div_counts)
                                for key, value in counts.items():
                                    counts[key] = _sum_across_processes(value, accelerator)
                                null_metrics = _null_metrics_from_counts(counts)
                                for key, value in null_metrics.items():
                                    logs[f"train/{key}"] = value

                    if grad_norm is not None:
                        try:
                            logs["train/grad_norm"] = float(grad_norm)
                        except (TypeError, ValueError):
                            pass

                    accelerator.log(logs, step=global_step)

                    if global_step % 100 == 0:
                        current_lrs = scheduler.get_last_lr()
                        lr_logs = {"train/learning_rate": current_lrs[0]}
                        if len(current_lrs) > 1:
                            for idx, lr_val in enumerate(current_lrs):
                                lr_logs[f"train/learning_rate_group_{idx}"] = lr_val
                        accelerator.log(lr_logs, step=global_step)

            if did_step:
                # Step-based functional evaluation
                eval_every_n_steps = training_cfg.get("eval_every_n_steps", 1000)
                eval_num_samples = training_cfg.get("eval_num_samples", 10)

                if global_step % eval_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        accelerator.print(f"\n{'=' * 80}")
                        accelerator.print(f"Running functional evaluation at step {global_step}")
                        accelerator.print(f"{'=' * 80}")

                        functional_metrics = functional_evaluation(
                            model=model,
                            eval_dataset=eval_dataset,
                            tokenizer=tokenizer,
                            accelerator=accelerator,
                            num_examples=eval_num_samples
                        )

                        if functional_metrics and training_cfg["use_wandb"]:
                            # Add step prefix to metrics
                            step_metrics = {f"step_{k}": v for k, v in functional_metrics.items()}
                            step_metrics["train/step"] = global_step
                            accelerator.log(step_metrics, step=global_step)

                        model.train()
                    accelerator.wait_for_everyone()

                # Optional: Regular cleanup
                if global_step % 100 == 0:
                    empty_cache(accelerator.device)

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_diffusion_loss = epoch_diffusion_loss / max(num_batches, 1)

        accelerator.print(f"\nEpoch {epoch + 1} Summary:")
        accelerator.print(f"  Avg Train Loss: {avg_epoch_loss:.4f}")
        accelerator.print(f"  Avg Diffusion Loss: {avg_diffusion_loss:.4f}")

        # Evaluation
        accelerator.print("\nRunning evaluation...")
        eval_metrics = evaluate(
            model,
            eval_dataloader,
            accelerator,
            null_token_id=null_token_id,
            return_logits=logits_for_metrics,
        )

        accelerator.print(f"Eval Results:")
        accelerator.print(f"  Eval Loss: {eval_metrics['eval/total_loss']:.4f}")
        accelerator.print(f"  Eval Diffusion Loss: {eval_metrics['eval/diffusion_loss']:.4f}")

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_diffusion_loss": avg_diffusion_loss,
        }

        epoch_metrics.update(eval_metrics)
        accelerator.log(epoch_metrics, step=global_step)

        # Save best model
        if eval_metrics['eval/total_loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['eval/total_loss']
            accelerator.print(f"New best model! Eval loss: {best_eval_loss:.4f}")

            if accelerator.is_main_process:
                save_dir = f"checkpoints/best_model"
                os.makedirs(save_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)

                trainable_state_dict = unwrapped_model.get_trainable_state_dict()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainable_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'eval_loss': best_eval_loss,
                    'config': config
                }, f"{save_dir}/model.pt")
                accelerator.print(f"Saved best model to {save_dir} (trainable params only)")

        accelerator.print("-" * 80)

    # Final functional evaluation
    accelerator.print("\n")
    accelerator.print("FINAL EVALUATION - Showing Model Outputs")

    functional_metrics = functional_evaluation(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        accelerator=accelerator,
        num_examples=10  # Show 10 examples at the end
    )

    if functional_metrics and training_cfg["use_wandb"]:
        accelerator.log(functional_metrics, step=global_step)

    accelerator.end_training()
    accelerator.print(f"\nTraining complete! Best eval loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    train()
