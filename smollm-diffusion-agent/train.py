import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import os
import wandb

from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, validate_mask_token_consistency


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(checkpoint_path, model, optimizer, accelerator):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)

    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_eval_loss = checkpoint['eval_loss']

    accelerator.print(f"Resumed from epoch {checkpoint['epoch']}, best eval loss: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    scaffold_mask = [item['scaffold_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    router_labels = None
    if 'router_label' in batch[0]:
        router_labels = torch.tensor([item['router_label'] for item in batch], dtype=torch.long)

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    scaffold_mask_padded = pad_sequence(scaffold_mask, batch_first=True, padding_value=False)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    batch_dict = {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "scaffold_mask": scaffold_mask_padded,
        "labels": labels_padded
    }

    if router_labels is not None:
        batch_dict["router_labels"] = router_labels

    return batch_dict


def evaluate(model, eval_dataloader, accelerator, train_router):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_router_loss = 0
    total_router_correct = 0
    total_router_samples = 0
    num_batches = 0

    eval_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)

    with torch.no_grad():
        for batch in eval_bar:
            current_router_labels = batch.get("router_labels") if train_router else None

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                scaffold_mask=batch["scaffold_mask"],
                router_labels=current_router_labels
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                num_batches += 1

                losses_detail = outputs.get("losses", {})
                if "diffusion" in losses_detail:
                    total_diffusion_loss += losses_detail["diffusion"].item()
                if "router" in losses_detail:
                    total_router_loss += losses_detail["router"].item()

                # Calculate router accuracy if training router
                if train_router and "router_logits" in outputs and current_router_labels is not None:
                    router_preds = torch.argmax(outputs["router_logits"], dim=-1)
                    total_router_correct += (router_preds == current_router_labels).sum().item()
                    total_router_samples += current_router_labels.size(0)

    model.train()

    metrics = {
        "eval/total_loss": total_loss / max(num_batches, 1),
        "eval/diffusion_loss": total_diffusion_loss / max(num_batches, 1),
    }

    if train_router and total_router_samples > 0:
        metrics["eval/router_loss"] = total_router_loss / max(num_batches, 1)
        metrics["eval/router_accuracy"] = total_router_correct / total_router_samples

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

    import random
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

            correct_tokens = (true_tokens == pred_tokens).sum().item()
            num_tokens = len(true_tokens)
            token_accuracy = correct_tokens / num_tokens if num_tokens > 0 else 0

            total_token_accuracy += correct_tokens
            total_tokens += num_tokens

            exact_match = torch.all(true_tokens == pred_tokens).item()
            total_exact_matches += exact_match

            true_masked_text = tokenizer.decode(true_tokens, skip_special_tokens=False)
            pred_masked_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)

            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

            accelerator.print(f"\n--- Example {i + 1} ---")
            accelerator.print(f"Input (first 200 chars):\n  {input_text[:200]}...")
            accelerator.print(f"\nMasked positions: {masked_positions.sum().item()} tokens")
            accelerator.print(f"Ground Truth (masked): {true_masked_text}")
            accelerator.print(f"Predicted (masked):    {pred_masked_text}")
            accelerator.print(f"Token Accuracy: {token_accuracy:.2%} ({correct_tokens}/{num_tokens})")
            accelerator.print(f"Exact Match: {'✓' if exact_match else '✗'}")

    accelerator.print("\n" + "-" * 80)
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

    accelerator.print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")
    accelerator.print(f"Tokenizer vocabulary size: {len(tokenizer)}")

    # 3. Initialize Model with correct vocab size
    accelerator.print("Loading Model...")
    model = HybridSmolLM(
        base_model_id=model_cfg["base_model_id"],
        load_in_4bit=model_cfg["load_in_4bit"],
        diffusion_config=diff_cfg,
        vocab_size=len(tokenizer)  # Use tokenizer vocab size!
    )

    # Set mask token ID in diffusion head for proper noising
    model.diffusion_head.set_mask_token_id(mask_token_id)

    # 4. Setup Dataset
    accelerator.print("Loading Dataset...")
    # Load full dataset
    full_dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        mask_token=mask_token_str,
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
        collate_fn=collate_fn
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # 4. Optimizer
    train_router = training_cfg["train_router"]
    if train_router:
        params_to_optimize = list(model.diffusion_head.parameters()) + list(model.router_head.parameters())
    else:
        params_to_optimize = list(model.diffusion_head.parameters())

    optimizer = AdamW(params_to_optimize, lr=float(training_cfg["learning_rate"]))

    # 5. Prepare
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # 6. Load checkpoint if resuming
    start_epoch = 0
    best_eval_loss = float('inf')
    global_step = 0

    if training_cfg.get("resume_from_checkpoint", False):
        checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")
        start_epoch, best_eval_loss = load_checkpoint(checkpoint_path, model, optimizer, accelerator)
        global_step = start_epoch * len(train_dataloader)
        accelerator.print(f"Starting from epoch {start_epoch}, global step {global_step}")

    # 7. Training Loop
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
                optimizer.zero_grad()
                global_step += 1

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
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
