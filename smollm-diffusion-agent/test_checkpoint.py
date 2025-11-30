import torch
import yaml
from transformers import AutoTokenizer
from model.hybrid_model import HybridSmolLM
from data.dataset_loader import SmartScaffoldDataset
import os


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(checkpoint_path, model_cfg):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = HybridSmolLM(
        base_model_id=model_cfg["base_model_id"],
        load_in_4bit=model_cfg["load_in_4bit"]
    )

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}, eval loss: {checkpoint['eval_loss']:.4f}")

    return model, checkpoint


def test_model(num_samples=5):
    config = load_config()
    training_cfg = config["training"]
    model_cfg = config["model"]
    data_cfg = config["data"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, model_cfg)

    if not model_cfg["load_in_4bit"]:
        model = model.to(device)
    else:
        model.diffusion_head = model.diffusion_head.to(device).to(torch.float16)
        model.router_head = model.router_head.to(device).to(torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from data.utils import resolve_mask_token
    
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)
    print(f"Using mask token: {mask_token_str} (ID: {mask_token_id})")
    
    model.diffusion_head.set_mask_token_id(mask_token_id)

    print(f"\nLoading dataset...")
    dataset = SmartScaffoldDataset(
        tokenizer,
        limit=data_cfg["limit"],
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        mask_token=mask_token_str
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"\nTesting model with {num_samples} samples from dataset")
    print("=" * 100)

    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            example = dataset[i]

            input_ids = example['input_ids'].unsqueeze(0).to(device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(device)
            scaffold_mask = example['scaffold_mask'].unsqueeze(0).to(device)
            labels = example['labels'].unsqueeze(0).to(device)
            diffusion_steps = torch.tensor([example['diffusion_steps']]).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                scaffold_mask=scaffold_mask,
                diffusion_steps=diffusion_steps,
                router_labels=None
            )

            diffusion_logits = outputs.get("diffusion_logits")

            if diffusion_logits is not None:
                predicted_ids = torch.argmax(diffusion_logits, dim=-1)

                masked_positions = scaffold_mask[0].cpu()

                if masked_positions.sum() > 0:
                    true_tokens = labels[0][masked_positions].cpu()
                    pred_tokens = predicted_ids[0][masked_positions].cpu()

                    correct_tokens = (true_tokens == pred_tokens).sum().item()
                    num_tokens = len(true_tokens)
                    token_accuracy = correct_tokens / num_tokens if num_tokens > 0 else 0

                    total_correct += correct_tokens
                    total_tokens += num_tokens

                    input_ids_cpu = input_ids[0].cpu().clone()
                    output_ids_pred = input_ids_cpu.clone()
                    output_ids_true = input_ids_cpu.clone()

                    output_ids_pred[masked_positions] = pred_tokens
                    output_ids_true[masked_positions] = true_tokens

                    output_ids_pred = output_ids_pred.clamp(min=0, max=tokenizer.vocab_size - 1)
                    output_ids_true = output_ids_true.clamp(min=0, max=tokenizer.vocab_size - 1)

                    input_text = tokenizer.decode(input_ids_cpu, skip_special_tokens=False)
                    pred_full_text = tokenizer.decode(output_ids_pred, skip_special_tokens=False)
                    true_full_text = tokenizer.decode(output_ids_true, skip_special_tokens=False)

                    print(f"\nSample {i+1}:")
                    print(f"Input (with masks, first 400 chars):\n{input_text[:400]}...")
                    print(f"\nGround Truth (full output):\n{true_full_text}")
                    print(f"\nModel Output (full output):\n{pred_full_text}")
                    print(f"\nAccuracy: {token_accuracy:.2%} ({correct_tokens}/{num_tokens} tokens)")

                    exact_match = torch.all(true_tokens == pred_tokens).item()
                    print(f"Exact Match: {'✓ YES' if exact_match else '✗ NO'}")
                    print("-" * 100)

    if total_tokens > 0:
        overall_accuracy = total_correct / total_tokens
        print(f"\n{'=' * 100}")
        print(f"Overall Results:")
        print(f"  Total Samples: {min(num_samples, len(dataset))}")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Correct Tokens: {total_correct}")
        print(f"  Overall Accuracy: {overall_accuracy:.2%}")
        print(f"{'=' * 100}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test model checkpoint on dataset samples")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    args = parser.parse_args()

    test_model(num_samples=args.num_samples)