import torch

try:
    import unsloth
except:
    pass
import random
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import os
import wandb
from data.device_utils import empty_cache, get_device
from data.config_utils import validate_and_adjust_config, get_model_kwargs, print_device_capabilities
from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency
from data.metrics import (
    calculate_null_token_metrics,
    calculate_field_level_metrics,
    calculate_parse_metrics,
    calculate_scaffold_metrics,
    extract_tool_call_json
)

from train_utils import load_config, build_collate_fn, load_checkpoint
from train_eval import evaluate, _compute_null_counts, _null_metrics_from_counts, _sum_across_processes
from train_functional import functional_evaluation

def train():
    # Load Config
    config = load_config()

    # Print device capabilities first
    print_device_capabilities()

    # Get device (before accelerator init)
    device = get_device()

    # Validate and adjust config for device
    config = validate_and_adjust_config(config, device)

    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]
    diff_cfg = config["diffusion"]
    quant_cfg = config.get("quantization", {})
    compile_cfg = training_cfg.get("compile", {})
    bucket_sizes = data_cfg.get("bucket_sizes", [512, 1024, 1536, 2048])

    # 1. Initialize Accelerator with W&B
    accelerator = Accelerator(
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        log_with="wandb" if training_cfg["use_wandb"] else None,
        mixed_precision="bf16"  # Use bfloat16 to match model dtype
    )
    set_seed(training_cfg["seed"])

    if training_cfg["use_wandb"]:
        accelerator.init_trackers(
            project_name=training_cfg["wandb_project"],
            config=config
        )

    accelerator.print(f"Using device: {accelerator.device}")

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
    if compile_cfg.get("enabled", False) and hasattr(torch, "compile"):
        compile_mode = compile_cfg.get("mode", "reduce-overhead")
        compile_fullgraph = compile_cfg.get("fullgraph", False)

        accelerator.print(f"torch.compile enabled for training (mode={compile_mode}, fullgraph={compile_fullgraph})")
        accelerator.print("Note: for best stability, pad/bucket sequences to fixed lengths per batch.")
        try:
            # Disable CUDA graphs to avoid tensor reuse issues during training
            # Note: reduce-overhead mode enables CUDA graphs by default, but we disable them here
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
            accelerator.print("Inductor CUDA graphs disabled for training.")

            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph
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
    
    optimizer = AdamW(params_to_optimize, lr=float(training_cfg["learning_rate"]))

    # 5. Prepare
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # 6. Learning Rate Scheduler (Cosine with Warmup)
    # From guide.md: warmup_steps=2500, cosine decay
    num_epochs = training_cfg["num_epochs"]
    gradient_accumulation_steps = training_cfg["gradient_accumulation_steps"]
    num_update_steps_per_epoch = max(1, len(train_dataloader) // gradient_accumulation_steps)
    total_training_steps = num_epochs * num_update_steps_per_epoch

    warmup_steps = training_cfg.get("warmup_steps", 2500)
    # Cap warmup to 10% of total steps if total steps is smaller
    warmup_steps = min(warmup_steps, int(0.1 * total_training_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )

    accelerator.print(f"LR Scheduler: Cosine with warmup")
    accelerator.print(f"  Total training steps: {total_training_steps}")
    accelerator.print(f"  Warmup steps: {warmup_steps}")

    # 7. Load checkpoint if resuming
    start_epoch = 0
    best_eval_loss = float('inf')
    global_step = 0

    if training_cfg.get("resume_from_checkpoint", False):
        checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")
        start_epoch, best_eval_loss = load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator)
        global_step = start_epoch * len(train_dataloader)
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
            # Mark CUDA graph step boundary to prevent tensor reuse issues
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()

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

                # Log metrics to wandb
                logs = {"train/total_loss": loss_val, "train/step": global_step}
                for k, v in losses_detail.items():
                    logs[f"train/{k}_loss"] = v.item() if torch.is_tensor(v) else v
                
                # Add NULL token metrics every 50 steps
                if global_step % 50 == 0 and "logits" in outputs:
                    with torch.no_grad():
                        logits = outputs["logits"]
                        predictions = torch.argmax(logits, dim=-1)
                        counts = _compute_null_counts(
                            predictions,
                            batch["labels"],
                            batch["scaffold_mask"],
                            null_token_id,
                        )
                        if counts is not None:
                            for key, value in counts.items():
                                counts[key] = _sum_across_processes(value, accelerator)
                            null_metrics = _null_metrics_from_counts(counts)
                            for key, value in null_metrics.items():
                                logs[f"train/{key}"] = value

                accelerator.log(logs, step=global_step)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)

                optimizer.step()
                scheduler.step()  # Step LR scheduler
                optimizer.zero_grad()
                global_step += 1

                # Log learning rate
                if global_step % 100 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    accelerator.log({"train/learning_rate": current_lr}, step=global_step)

            # Step-based functional evaluation
            eval_every_n_steps = training_cfg.get("eval_every_n_steps", 1000)
            eval_num_samples = training_cfg.get("eval_num_samples", 10)

            if (global_step > 0 and
                    global_step % eval_every_n_steps == 0 and
                    accelerator.is_main_process):
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

                # Only save trainable parameters (diffusion_head)
                full_state_dict = unwrapped_model.state_dict()
                trainable_state_dict = {k: v for k, v in full_state_dict.items()
                                        if k.startswith('diffusion_head.')}

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
