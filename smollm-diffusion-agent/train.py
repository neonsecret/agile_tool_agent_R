import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed
import os
import wandb

from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset


def load_config(config_path="smollm-diffusion-agent/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
        log_with="wandb" if training_cfg["use_wandb"] else None
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

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({'mask_token': '<MASK>'})
        accelerator.print(f"Added <MASK> token, new vocab size: {len(tokenizer)}")

        if not model_cfg["load_in_4bit"]:
            model.base_llm.resize_token_embeddings(len(tokenizer))
        else:
            accelerator.print("Warning: 4-bit mode enabled. Skipping base model embedding resize.")

        # Re-init heads
        hidden_size = model.base_llm.config.hidden_size
        vocab_size = len(tokenizer)

        model.diffusion_head = model.diffusion_head.__class__(
            input_dim=hidden_size,
            vocab_size=vocab_size,
            hidden_dim=hidden_size,
            num_steps=diff_cfg["num_steps"]
        ).to(accelerator.device)

        model.router_head = model.router_head.__class__(
            hidden_size=hidden_size
        ).to(accelerator.device)

    dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
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
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # 6. Training Loop
    accelerator.print("Starting training loop...")
    model.train()

    global_step = 0
    for epoch in range(training_cfg["num_epochs"]):
        for i, batch in enumerate(dataloader):
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

                # Log metrics
                logs = {"train/total_loss": loss_val}
                for k, v in losses_detail.items():
                    logs[f"train/{k}_loss"] = v.item() if torch.is_tensor(v) else v

                accelerator.log(logs, step=global_step)
                accelerator.print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_val:.4f}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            # Optional: Regular cleanup
            if i % 100 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    accelerator.end_training()


if __name__ == "__main__":
    train()
