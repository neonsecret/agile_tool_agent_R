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
    diffusion_steps = torch.tensor([item['diffusion_steps'] for item in batch])

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
        "labels": labels_padded,
        "diffusion_steps": diffusion_steps
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
                diffusion_steps=batch["diffusion_steps"],
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


def functional_evaluation(model, eval_dataset, tokenizer, accelerator, num_examples=5):
    """Show actual model outputs for qualitative assessment"""
    model.eval()

    accelerator.print("\n" + "=" * 80)
    accelerator.print("FUNCTIONAL EVALUATION - Sample Outputs")
    accelerator.print("=" * 80)

    # Get a few random examples
    import random
    indices = random.sample(range(len(eval_dataset)), min(num_examples, len(eval_dataset)))

    total_exact_matches = 0
    total_token_accuracy = 0
    total_tokens = 0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            example = eval_dataset[idx]

            # Prepare batch (single example)
            input_ids = example['input_ids'].unsqueeze(0).to(accelerator.device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(accelerator.device)
            scaffold_mask = example['scaffold_mask'].unsqueeze(0).to(accelerator.device)
            labels = example['labels'].unsqueeze(0).to(accelerator.device)
            diffusion_steps = example['diffusion_steps'].unsqueeze(0).to(accelerator.device)

            # Get model predictions
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_mask=scaffold_mask,
                diffusion_steps=diffusion_steps,
                router_labels=None
            )

            # Get diffusion logits and predictions
            diffusion_logits = outputs.get("diffusion_logits")
            if diffusion_logits is not None:
                # Predict tokens
                predicted_ids = torch.argmax(diffusion_logits, dim=-1)

                # Decode sequences
                input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                predicted_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=False)

                # Get ground truth for masked positions
                masked_positions = scaffold_mask[0].cpu()
                if masked_positions.sum() > 0:
                    true_tokens = labels[0][masked_positions].cpu()
                    pred_tokens = predicted_ids[0][masked_positions].cpu()

                    # Calculate token-level accuracy
                    correct_tokens = (true_tokens == pred_tokens).sum().item()
                    num_tokens = len(true_tokens)
                    token_accuracy = correct_tokens / num_tokens if num_tokens > 0 else 0

                    total_token_accuracy += correct_tokens
                    total_tokens += num_tokens

                    # Check exact match
                    exact_match = torch.all(true_tokens == pred_tokens).item()
                    total_exact_matches += exact_match

                    # Decode only the masked tokens
                    true_masked_text = tokenizer.decode(true_tokens, skip_special_tokens=False)
                    pred_masked_text = tokenizer.decode(pred_tokens, skip_special_tokens=False)

                    # Print example
                    accelerator.print(f"\n--- Example {i+1} ---")
                    accelerator.print(f"Input (first 200 chars):\n  {input_text[:200]}...")
                    accelerator.print(f"\nMasked positions: {masked_positions.sum().item()} tokens")
                    accelerator.print(f"Ground Truth (masked): {true_masked_text}")
                    accelerator.print(f"Predicted (masked):    {pred_masked_text}")
                    accelerator.print(f"Token Accuracy: {token_accuracy:.2%} ({correct_tokens}/{num_tokens})")
                    accelerator.print(f"Exact Match: {'✓' if exact_match else '✗'}")

    # Overall stats
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

    accelerator.print("=" * 80 + "\n")
    model.train()


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

    # 2. Initialize Model
    accelerator.print("Loading Model...")
    model = HybridSmolLM(
        base_model_id=model_cfg["base_model_id"],
        load_in_4bit=model_cfg["load_in_4bit"]
    )

    # 3. Setup Tokenizer & Dataset
    accelerator.print("Loading Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use an existing token as mask token (works in both 4-bit and non-4bit modes)
    # We don't need to add a new token - just designate an existing one as the mask
    if tokenizer.mask_token is None:
        tokenizer.mask_token = tokenizer.eos_token
        accelerator.print(f"Using token '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id}) as mask token")

    accelerator.print(f"Mask token: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
    accelerator.print(f"Vocabulary size: {len(tokenizer)}")

    # Load full dataset
    full_dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"]
    )

    # Split into train/eval (90/10 split)
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
            desc=f"Epoch {epoch+1}/{training_cfg['num_epochs']}",
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
                    diffusion_steps=batch["diffusion_steps"],
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
                    'avg_loss': f'{epoch_loss/num_batches:.4f}'
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

            # Optional: Regular cleanup
            if global_step % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        avg_diffusion_loss = epoch_diffusion_loss / max(num_batches, 1)

        accelerator.print(f"\nEpoch {epoch+1} Summary:")
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
