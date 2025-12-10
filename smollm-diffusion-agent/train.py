import torch
import yaml
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
from data.device_utils import empty_cache
from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, accelerator):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        accelerator.print("Loaded scheduler state from checkpoint")

    start_epoch = checkpoint['epoch'] + 1
    best_eval_loss = checkpoint['eval_loss']

    accelerator.print(f"Resumed from epoch {checkpoint['epoch']}, best eval loss: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss


def build_collate_fn(bucket_sizes, max_seq_len):
    """Create a collate_fn that pads to the next bucket size >= batch max length.

    This keeps shapes more stable for torch.compile and CUDA graphs.
    """
    bucket_sizes = sorted(bucket_sizes)

    def collate_fn(batch):
        # Determine target pad length: next bucket >= max length in batch
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
            return F.pad(t, (0, pad_width), value=pad_value)

        input_ids = [pad_tensor(item["input_ids"], pad_value=0) for item in batch]
        attention_mask = [pad_tensor(item["attention_mask"], pad_value=0) for item in batch]
        scaffold_mask = [pad_tensor(item["scaffold_mask"], pad_value=0) for item in batch]
        labels = [pad_tensor(item["labels"], pad_value=-100) for item in batch]

        batch_dict = {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
            "scaffold_mask": torch.stack(scaffold_mask, dim=0),
            "labels": torch.stack(labels, dim=0),
        }

        if "router_label" in batch[0]:
            router_labels = torch.tensor(
                [item["router_label"] for item in batch], dtype=torch.long
            )
            batch_dict["router_labels"] = router_labels

        return batch_dict

    return collate_fn


def evaluate(model, eval_dataloader, accelerator, train_router):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_router_loss = 0
    total_router_correct = 0
    total_router_samples = 0
    num_batches = 0
    
    # Track per-class router accuracy
    router_predictions = []
    router_labels_list = []

    eval_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in eval_bar:
            # Always get router labels for accuracy tracking (independent of train_router)
            current_router_labels = batch.get("router_labels")
            
            # Only pass router_labels to model if training router (affects loss computation)
            router_labels_for_loss = current_router_labels if train_router else None

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                scaffold_mask=batch["scaffold_mask"],
                router_labels=router_labels_for_loss
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

                losses_detail = outputs.get("losses", {})
                if "diffusion" in losses_detail:
                    total_diffusion_loss += losses_detail["diffusion"].item()
                if "router" in losses_detail:
                    total_router_loss += losses_detail["router"].item()

                # Calculate router accuracy ALWAYS (independent of train_router flag)
                # This lets us monitor router even if we're not training it yet
                if "router_logits" in outputs and current_router_labels is not None:
                    router_preds = torch.argmax(outputs["router_logits"], dim=-1)
                    total_router_correct += (router_preds == current_router_labels).sum().item()
                    total_router_samples += current_router_labels.size(0)
                    
                    # Collect for per-class analysis
                    router_predictions.extend(router_preds.cpu().tolist())
                    router_labels_list.extend(current_router_labels.cpu().tolist())

    model.train()

    metrics = {
        "eval/total_loss": total_loss / max(num_batches, 1),
        "eval/diffusion_loss": total_diffusion_loss / max(num_batches, 1),
    }

    if train_router:
        metrics["eval/router_loss"] = total_router_loss / max(num_batches, 1)
    
    # Always track router accuracy if we have predictions
    if total_router_samples > 0:
        router_accuracy = total_router_correct / total_router_samples
        metrics["eval/router_accuracy"] = router_accuracy
        
        # Per-class accuracy: Chat (0), Tool (1), Think (2)
        class_names = ["chat", "tool", "think"]
        for class_idx, class_name in enumerate(class_names):
            class_mask = [label == class_idx for label in router_labels_list]
            if sum(class_mask) > 0:
                class_preds = [pred for pred, mask in zip(router_predictions, class_mask) if mask]
                class_labels = [label for label, mask in zip(router_labels_list, class_mask) if mask]
                class_acc = sum(p == l for p, l in zip(class_preds, class_labels)) / len(class_labels)
                metrics[f"eval/router_accuracy_{class_name}"] = class_acc
        
        if accelerator.is_local_main_process:
            accelerator.print(f"Router Accuracy: {router_accuracy:.4f}")
            for class_name in class_names:
                key = f"eval/router_accuracy_{class_name}"
                if key in metrics:
                    accelerator.print(f"  {class_name.capitalize()}: {metrics[key]:.4f}")

    return metrics


def s3_denoise(model, hidden_states, labels, scaffold_mask, num_steps=4):
    """
    S3-style top-K confidence denoising (matches inference.py strategy).

    This uses the ACTUAL inference strategy:
    - Fixed t=0 (fully denoised state)
    - Top-K confidence-based token selection
    - Iterative refinement over multiple steps

    Args:
        model: The HybridSmolLM model (unwrapped)
        hidden_states: Context embeddings from base model
        labels: Ground truth tokens (used to initialize sequence)
        scaffold_mask: Boolean mask indicating positions to denoise
        num_steps: Number of S3 denoising steps

    Returns:
        final_tokens: Denoised tokens after S3 iteration
    """
    device = hidden_states.device
    mask_token_id = model.diffusion_head.mask_token_id

    # Convert hidden_states to match diffusion head dtype (bfloat16)
    diffusion_head_dtype = next(model.diffusion_head.parameters()).dtype
    hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

    current_tokens = labels.clone()
    current_tokens[scaffold_mask] = mask_token_id

    total_masks = scaffold_mask.sum().item()
    budget = max(1, int(total_masks / num_steps))

    t = torch.zeros(1, device=device)

    for step in range(num_steps):
        mask_positions = current_tokens == mask_token_id
        mask_positions = mask_positions & scaffold_mask
        mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

        if mask_indices.numel() == 0:
            break

        # Handle 0D tensor case (single mask position)
        if mask_indices.dim() == 0:
            mask_indices = mask_indices.unsqueeze(0)

        logits = model.diffusion_head.predict(hidden_states, current_tokens, t)
        predictions = torch.argmax(logits, dim=-1)

        log_probs = torch.log_softmax(logits[0, mask_indices], dim=-1)
        mask_conf = log_probs.gather(-1, predictions[0, mask_indices].unsqueeze(-1)).squeeze(-1)

        remaining = mask_indices.numel()
        remaining_steps = num_steps - step
        if remaining_steps <= 1:
            k = remaining
        else:
            k = min(budget, remaining)

        topk = torch.topk(mask_conf, k)
        selected = mask_indices[topk.indices]

        current_tokens[0, selected] = predictions[0, selected]

    return current_tokens


def functional_evaluation(model, eval_dataset, tokenizer, accelerator, num_examples=5):
    """
    Evaluate model using S3-style inference (matches actual inference.py strategy).

    This tests the REAL inference approach:
    - Fixed t=0 prediction
    - Top-K confidence-based selection
    - Iterative refinement
    """
    model.eval()

    accelerator.print("\n" + "=" * 80)
    accelerator.print("FUNCTIONAL EVALUATION - S3 Denoising Strategy")
    accelerator.print("=" * 80)

    indices = random.sample(range(len(eval_dataset)), min(num_examples, len(eval_dataset)))

    total_exact_matches = 0
    total_token_accuracy = 0
    total_tokens = 0

    unwrapped_model = accelerator.unwrap_model(model)
    diffusion_num_steps = unwrapped_model.diffusion_head.num_steps

    with torch.no_grad():
        for i, idx in enumerate(indices):
            example = eval_dataset[idx]

            input_ids = example['input_ids'].unsqueeze(0).to(accelerator.device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(accelerator.device)
            scaffold_mask = example['scaffold_mask'].unsqueeze(0).to(accelerator.device)
            labels = example['labels'].unsqueeze(0).to(accelerator.device)

            if scaffold_mask.sum() == 0:
                continue

            outputs = unwrapped_model.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]

            predicted_tokens = s3_denoise(
                unwrapped_model,
                hidden_states,
                labels,
                scaffold_mask,
                num_steps=diffusion_num_steps
            )

            masked_positions = scaffold_mask[0].cpu()
            true_tokens = labels[0][masked_positions].cpu()
            pred_tokens = predicted_tokens[0][masked_positions].cpu()

            # Filter out invalid token IDs (-100 is padding/ignore token)
            valid_mask = (true_tokens >= 0) & (true_tokens < len(tokenizer))
            true_tokens_valid = true_tokens[valid_mask]
            pred_tokens_valid = pred_tokens[valid_mask]

            # Strip NULL tokens from both true and predicted (for self-adaptive masking)
            null_token_id = unwrapped_model.diffusion_head.null_token_id
            if null_token_id is not None:
                # Remove NULL tokens from true tokens (they're padding for variable-length fields)
                true_tokens_valid = true_tokens_valid[true_tokens_valid != null_token_id]
                # Remove NULL tokens from predictions (model may predict NULL for unused slots)
                pred_tokens_valid = pred_tokens_valid[pred_tokens_valid != null_token_id]

            # Align lengths: compare only up to minimum length (handles variable-length fields)
            min_len = min(len(true_tokens_valid), len(pred_tokens_valid))
            if min_len == 0:
                continue

            true_tokens_aligned = true_tokens_valid[:min_len]
            pred_tokens_aligned = pred_tokens_valid[:min_len]

            correct_tokens = (true_tokens_aligned == pred_tokens_aligned).sum().item()
            num_tokens = min_len
            token_accuracy = correct_tokens / num_tokens if num_tokens > 0 else 0

            total_token_accuracy += correct_tokens
            total_tokens += num_tokens

            exact_match = torch.all(true_tokens_aligned == pred_tokens_aligned).item()
            total_exact_matches += exact_match

            # Convert to list of integers for tokenizer.decode
            true_tokens_list = true_tokens_aligned.tolist()
            pred_tokens_list = pred_tokens_aligned.tolist()

            true_masked_text = tokenizer.decode(true_tokens_list, skip_special_tokens=False)
            pred_masked_text = tokenizer.decode(pred_tokens_list, skip_special_tokens=False)

            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

            accelerator.print(f"\n--- Example {i + 1} ---")
            accelerator.print(f"Input (first 200 chars):\n  {input_text[:200]}...")
            accelerator.print(f"\nMasked positions: {masked_positions.sum().item()} tokens")
            accelerator.print(f"Ground Truth (masked): {true_masked_text}")
            accelerator.print(f"Predicted (masked):    {pred_masked_text}")
            accelerator.print(f"Token Accuracy: {token_accuracy:.2%} ({correct_tokens}/{num_tokens})")
            accelerator.print(f"Exact Match: {'✓' if exact_match else '✗'}")

    accelerator.print("\n" + "-" * 80)
    empty_cache(accelerator.device)
    if total_tokens > 0:
        overall_token_acc = total_token_accuracy / total_tokens
        overall_exact_match = total_exact_matches / len(indices)

        accelerator.print(f"Overall Statistics ({len(indices)} examples):")
        accelerator.print(f"  Token-level Accuracy: {overall_token_acc:.2%} ({total_token_accuracy}/{total_tokens})")
        accelerator.print(f"  Exact Match Rate: {overall_exact_match:.2%} ({total_exact_matches}/{len(indices)})")

        return {
            "functional/token_accuracy": overall_token_acc,
            "functional/exact_match_rate": overall_exact_match,
        }
    else:
        accelerator.print("No masked tokens found in examples")
        return {}


def train():
    # Load Config
    config = load_config()
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
    null_token_str, null_token_id = None, None
    if null_token_config is not None or data_cfg.get("use_null_token", True):
        try:
            null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)
            accelerator.print(f"NULL token: {null_token_str} (ID: {null_token_id})")
        except ValueError as e:
            accelerator.print(f"Warning: NULL token not available: {e}")
            null_token_str, null_token_id = None, None

    accelerator.print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")
    accelerator.print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # 3. Initialize Model with correct vocab size
    # Determine quantization setting from unified config
    quantize_enabled = quant_cfg.get("enabled", False)
    quantize_bits = quant_cfg.get("bits", 4)
    
    # For PyTorch, only 4-bit quantization is supported (and only on CUDA)
    load_in_4bit = quantize_enabled and quantize_bits == 4
    
    if quantize_enabled:
        if quantize_bits == 4:
            accelerator.print(f"Quantization: 4-bit (PyTorch bitsandbytes - CUDA only)")
        else:
            accelerator.print(f"Quantization: {quantize_bits}-bit not supported in PyTorch, using bfloat16 instead")
            load_in_4bit = False
    else:
        accelerator.print("Quantization: disabled")
    
    accelerator.print("Loading Model...")
    model = HybridSmolLM(
        base_model_id=model_cfg["base_model_id"],
        load_in_4bit=load_in_4bit,
        diffusion_config=diff_cfg,
        vocab_size=len(tokenizer)  # Use tokenizer vocab size!
    )

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

    # 4. Setup Dataset
    accelerator.print("Loading Dataset...")
    # Load full dataset with NULL token support for automatic budgeting
    full_dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        mask_token=mask_token_str,
        null_token=null_token_str,  # Enables self-adaptive masking
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48)
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
        collate_fn=build_collate_fn(bucket_sizes, training_cfg["max_seq_len"])
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=build_collate_fn(bucket_sizes, training_cfg["max_seq_len"])
    )

    # 4. Optimizer
    train_router = training_cfg["train_router"]
    if train_router:
        params_to_optimize = list(model.diffusion_head.parameters()) + list(model.router_head.parameters())
    else:
        params_to_optimize = list(model.diffusion_head.parameters())

    optimizer = AdamW(params_to_optimize, lr=float(training_cfg["learning_rate"]))

    # 5. Learning Rate Scheduler (Cosine with Warmup)
    # From guide.md: warmup_steps=2500, cosine decay
    num_epochs = training_cfg["num_epochs"]
    gradient_accumulation_steps = training_cfg["gradient_accumulation_steps"]
    num_update_steps_per_epoch = len(train_dataloader) // gradient_accumulation_steps
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

    # 6. Prepare
    model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, scheduler
    )

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
        epoch_router_loss = 0
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
                current_router_labels = batch.get("router_labels") if train_router else None

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    scaffold_mask=batch["scaffold_mask"],
                    router_labels=current_router_labels
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
                if "router" in losses_detail:
                    epoch_router_loss += losses_detail["router"].item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'avg_loss': f'{epoch_loss / num_batches:.4f}'
                })

                # Log metrics to wandb
                logs = {"train/total_loss": loss_val, "train/step": global_step}
                for k, v in losses_detail.items():
                    logs[f"train/{k}_loss"] = v.item() if torch.is_tensor(v) else v

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
        if train_router and epoch_router_loss > 0:
            avg_router_loss = epoch_router_loss / max(num_batches, 1)
            accelerator.print(f"  Avg Router Loss: {avg_router_loss:.4f}")

        # Evaluation
        accelerator.print("\nRunning evaluation...")
        eval_metrics = evaluate(model, eval_dataloader, accelerator, train_router)

        accelerator.print(f"Eval Results:")
        accelerator.print(f"  Eval Loss: {eval_metrics['eval/total_loss']:.4f}")
        accelerator.print(f"  Eval Diffusion Loss: {eval_metrics['eval/diffusion_loss']:.4f}")
        if train_router:
            if 'eval/router_loss' in eval_metrics:
                accelerator.print(f"  Eval Router Loss: {eval_metrics['eval/router_loss']:.4f}")
            if 'eval/router_accuracy' in eval_metrics:
                accelerator.print(f"  Eval Router Accuracy: {eval_metrics['eval/router_accuracy']:.2%}")

        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_diffusion_loss": avg_diffusion_loss,
        }
        if train_router and epoch_router_loss > 0:
            epoch_metrics["train/epoch_router_loss"] = avg_router_loss

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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'eval_loss': best_eval_loss,
                    'config': config
                }, f"{save_dir}/model.pt")
                accelerator.print(f"Saved best model to {save_dir}")

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
