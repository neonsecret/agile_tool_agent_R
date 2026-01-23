import sys
import os

# Ensure we can import from current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import torch
import yaml
from transformers import AutoTokenizer

# Use same import style as train.py
from data.dataset_loader import SmartScaffoldDataset
from data.device_utils import get_device
from data.utils import resolve_mask_token
from model.hybrid_model import HybridSmolLM
from model.diffusion_head import SchemaDiffusionHead


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_dataset_structure(dataset, tokenizer, num_samples=10):
    """Test that dataset returns correct data structure."""
    print("=" * 80)
    print("Testing Dataset Structure")
    print("=" * 80)

    errors = []
    warnings = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        required_keys = ["input_ids", "attention_mask", "scaffold_mask", "labels", "is_tool"]
        missing_keys = [k for k in required_keys if k not in sample]
        if missing_keys:
            errors.append(f"Sample {i}: Missing keys: {missing_keys}")
            continue

        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        scaffold_mask = sample["scaffold_mask"]
        labels = sample["labels"]

        seq_len = len(input_ids)

        if len(attention_mask) != seq_len:
            errors.append(f"Sample {i}: attention_mask length mismatch: {len(attention_mask)} != {seq_len}")

        if len(scaffold_mask) != seq_len:
            errors.append(f"Sample {i}: scaffold_mask length mismatch: {len(scaffold_mask)} != {seq_len}")

        if len(labels) != seq_len:
            errors.append(f"Sample {i}: labels length mismatch: {len(labels)} != {seq_len}")

        if not isinstance(input_ids, torch.Tensor):
            errors.append(f"Sample {i}: input_ids is not a tensor")

        if not isinstance(scaffold_mask, torch.Tensor):
            errors.append(f"Sample {i}: scaffold_mask is not a tensor")

        if scaffold_mask.dtype != torch.bool:
            errors.append(f"Sample {i}: scaffold_mask dtype is {scaffold_mask.dtype}, expected bool")

        if not isinstance(labels, torch.Tensor):
            errors.append(f"Sample {i}: labels is not a tensor")

        if not isinstance(sample["is_tool"], (int, torch.Tensor)):
            errors.append(f"Sample {i}: is_tool is not int or tensor")

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("✅ All structure tests passed!")

    return len(errors) == 0


def test_mask_token_consistency(dataset, tokenizer, mask_token_id, num_samples=10):
    """Test that mask tokens in input_ids match scaffold_mask."""
    print("\n" + "=" * 80)
    print("Testing Mask Token Consistency")
    print("=" * 80)

    errors = []
    warnings = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        scaffold_mask = sample["scaffold_mask"]

        mask_positions = (input_ids == mask_token_id)

        scaffold_positions = scaffold_mask.bool()

        if not torch.equal(mask_positions, scaffold_positions):
            diff_mask = mask_positions & ~scaffold_positions
            diff_scaffold = scaffold_positions & ~mask_positions

            if diff_mask.sum() > 0:
                errors.append(
                    f"Sample {i}: Found {diff_mask.sum().item()} positions with mask token "
                    f"but scaffold_mask=False"
                )

            if diff_scaffold.sum() > 0:
                errors.append(
                    f"Sample {i}: Found {diff_scaffold.sum().item()} positions with "
                    f"scaffold_mask=True but no mask token"
                )

        if scaffold_mask.sum() == 0 and sample["is_tool"]:
            warnings.append(
                f"Sample {i}: Tool call (is_tool=1) but no scaffold_mask positions"
            )

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("✅ All mask token consistency tests passed!")

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings[:5]:
            print(f"  - {warning}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings) - 5} more warnings")

    return len(errors) == 0


def test_labels_consistency(dataset, num_samples=10):
    """Test that labels are only set where scaffold_mask is True.
    
    Note: scaffold_mask positions can have labels=-100 when mask budget exceeds
    target tokens (self-adaptive masking). This is expected behavior.
    """
    print("\n" + "=" * 80)
    print("Testing Labels Consistency")
    print("=" * 80)

    errors = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        scaffold_mask = sample["scaffold_mask"]
        labels = sample["labels"]

        valid_labels = labels != -100

        # Check that labels are ONLY set where scaffold_mask is True
        # (scaffold_mask=True with labels=-100 is OK - means mask budget > target tokens)
        diff_valid = valid_labels & ~scaffold_mask

        if diff_valid.sum() > 0:
            errors.append(
                f"Sample {i}: Found {diff_valid.sum().item()} positions with labels "
                f"but scaffold_mask=False"
            )

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    else:
        print("✅ All labels consistency tests passed!")

    return len(errors) == 0


def test_sequence_lengths(dataset, max_seq_len, num_samples=10):
    """Test that sequences don't exceed max_seq_len."""
    print("\n" + "=" * 80)
    print("Testing Sequence Lengths")
    print("=" * 80)

    errors = []
    max_found = 0

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        seq_len = len(sample["input_ids"])
        max_found = max(max_found, seq_len)

        if seq_len > max_seq_len:
            errors.append(
                f"Sample {i}: Sequence length {seq_len} exceeds max_seq_len {max_seq_len}"
            )

    print(f"Max sequence length found: {max_found}")
    print(f"Max allowed: {max_seq_len}")

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
    else:
        print("✅ All sequence length tests passed!")

    return len(errors) == 0


def show_examples(dataset, tokenizer, mask_token_id, num_examples=3):
    """Show example outputs from the dataset."""
    print("\n" + "=" * 80)
    print("Example Dataset Outputs")
    print("=" * 80)

    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        scaffold_mask = sample["scaffold_mask"]
        labels = sample["labels"]

        print(f"\n--- Example {i + 1} ---")
        print(f"Is tool: {sample['is_tool']}")
        print(f"Sequence length: {len(input_ids)}")
        print(f"Scaffold mask positions: {scaffold_mask.sum().item()}")
        print(f"Valid labels: {(labels != -100).sum().item()}")

        input_text = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"\nInput text (first 300 chars):")
        print(f"  {input_text[:300]}...")

        if scaffold_mask.sum() > 0:
            mask_indices = torch.nonzero(scaffold_mask, as_tuple=True)[0]
            mask_tokens = input_ids[mask_indices]
            label_tokens = labels[mask_indices]

            print(f"\nMask positions: {mask_indices.tolist()[:10]}...")
            print(f"Mask token IDs: {mask_tokens.tolist()[:10]}...")
            print(f"Label token IDs: {label_tokens.tolist()[:10]}...")

            valid_labels = label_tokens[label_tokens != -100]
            if len(valid_labels) > 0:
                label_text = tokenizer.decode(valid_labels, skip_special_tokens=False)
                print(f"\nLabel text (first 100 chars):")
                print(f"  {label_text[:100]}...")


def debug_model_input_output(dataset, tokenizer, mask_token_id, model, num_examples=3):
    """Debug what dataset feeds to model and what model expects to produce."""
    print("\n" + "=" * 80)
    print("DEBUGGING: Dataset -> Model Flow")
    print("=" * 80)

    device = next(model.parameters()).device
    model.eval()

    for i in range(min(num_examples, len(dataset))):
        sample = dataset[i]

        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        scaffold_mask = sample["scaffold_mask"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        is_tool = sample["is_tool"]

        print(f"\n{'=' * 80}")
        print(f"EXAMPLE {i + 1}")
        print(f"{'=' * 80}")

        print(f"\n1. DATASET OUTPUT:")
        print(f"   Is tool: {is_tool} ({'Tool Call' if is_tool else 'Chat'})")
        print(f"   Sequence length: {input_ids.shape[1]}")
        print(f"   Scaffold mask positions: {scaffold_mask.sum().item()}")
        print(f"   Valid labels (non -100): {(labels != -100).sum().item()}")

        if scaffold_mask.sum() == 0:
            print("\n   ⚠️  No scaffold mask positions - skipping diffusion analysis")
            continue

        mask_positions = scaffold_mask[0]
        mask_indices = torch.nonzero(mask_positions, as_tuple=True)[0]

        print(f"\n   Mask positions (first 20): {mask_indices[:20].tolist()}")

        input_mask_tokens = input_ids[0][mask_positions]
        expected_labels = labels[0][mask_positions]
        valid_labels_mask = expected_labels != -100

        print(f"\n   Input mask tokens at scaffold positions:")
        print(f"     Token IDs: {input_mask_tokens[:20].tolist()}")
        print(f"     All are mask_token_id ({mask_token_id})? {torch.all(input_mask_tokens == mask_token_id)}")

        print(f"\n   Expected labels at scaffold positions:")
        print(f"     Label IDs: {expected_labels[:20].tolist()}")
        print(f"     Valid labels count: {valid_labels_mask.sum().item()}/{len(expected_labels)}")

        if valid_labels_mask.sum() > 0:
            valid_label_ids = expected_labels[valid_labels_mask][:20]
            valid_label_text = tokenizer.decode(valid_label_ids, skip_special_tokens=False)
            print(f"     Valid label text (first 100 chars): {valid_label_text[:100]}")

        print(f"\n2. MODEL FORWARD PASS:")

        with torch.no_grad():
            base_outputs = model.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = base_outputs.hidden_states[-1]

            print(f"   Hidden states shape: {hidden_states.shape}")
            print(f"   Hidden states dtype: {hidden_states.dtype}")

        print(f"\n3. DIFFUSION HEAD ANALYSIS:")

        diffusion_head = model.diffusion_head

        print(f"   Mask token ID in model: {diffusion_head.mask_token_id}")
        print(f"   Vocab size: {diffusion_head.vocab_size}")
        print(f"   Hidden dim: {diffusion_head.hidden_dim}")
        print(f"   Num steps: {diffusion_head.num_steps}")

        if diffusion_head.mask_token_id != mask_token_id:
            print(f"   ⚠️  WARNING: Mask token mismatch!")
            print(f"      Dataset uses: {mask_token_id}")
            print(f"      Model uses: {diffusion_head.mask_token_id}")

        print(f"\n4. SIMULATING TRAINING STEP:")

        with torch.no_grad():
            batch_size = labels.shape[0]
            t = torch.rand(batch_size, device=device)

            print(f"   Random timestep t: {t.item():.4f}")

            noisy_tokens, mask_positions_diff = diffusion_head.forward_diffusion(
                labels, scaffold_mask, t
            )

            print(f"   After forward diffusion:")
            print(f"     Masked positions: {mask_positions_diff.sum().item()}")

            if mask_positions_diff.sum() > 0:
                noisy_at_masks = noisy_tokens[0][mask_positions_diff[0]]
                print(f"     Noisy tokens at masked positions (first 10): {noisy_at_masks[:10].tolist()}")
                print(f"     All are mask_token_id? {torch.all(noisy_at_masks == mask_token_id)}")

            prompt_mask = ~scaffold_mask
            logits = diffusion_head.predict(
                hidden_states,
                noisy_tokens,
                t,
                scaffold_mask=scaffold_mask,
                prompt_mask=prompt_mask,
            )

            print(f"   Prediction logits shape: {logits.shape}")

            valid_mask_positions = mask_positions_diff & (labels >= 0) & scaffold_mask

            if valid_mask_positions.sum() > 0:
                active_logits = logits[valid_mask_positions]
                active_labels = labels[valid_mask_positions]

                predictions = torch.argmax(active_logits, dim=-1)

                print(f"\n   Predictions vs Expected:")
                print(f"     Active positions: {valid_mask_positions.sum().item()}")
                print(f"     Predictions (first 20): {predictions[:20].tolist()}")
                print(f"     Expected (first 20): {active_labels[:20].tolist()}")

                correct = (predictions == active_labels).sum().item()
                total = len(predictions)
                accuracy = correct / total if total > 0 else 0

                print(f"     Accuracy: {accuracy:.2%} ({correct}/{total})")

                if total > 0:
                    pred_text = tokenizer.decode(predictions[:20], skip_special_tokens=False)
                    expected_text = tokenizer.decode(active_labels[:20], skip_special_tokens=False)
                    print(f"     Predicted text (first 100 chars): {pred_text[:100]}")
                    print(f"     Expected text (first 100 chars): {expected_text[:100]}")
            else:
                print(f"   ⚠️  No valid mask positions for loss calculation")

        print(f"\n5. INPUT/OUTPUT SUMMARY:")
        print(f"   Dataset provides:")
        print(f"     - input_ids: {input_ids.shape} (contains mask tokens at scaffold positions)")
        print(f"     - scaffold_mask: {scaffold_mask.shape} (marks positions to denoise)")
        print(f"     - labels: {labels.shape} (ground truth tokens, -100 for ignore)")
        print(f"     - attention_mask: {attention_mask.shape}")

        print(f"\n   Model expects to:")
        print(f"     - Receive hidden_states from base LLM: {hidden_states.shape}")
        print(f"     - Predict tokens at scaffold_mask positions")
        print(f"     - Match labels where labels != -100")

        print(f"\n   Data flow:")
        print(f"     1. Dataset puts mask_token_id ({mask_token_id}) in input_ids at scaffold positions")
        print(f"     2. Base LLM processes input_ids -> hidden_states")
        print(f"     3. Diffusion head receives labels (ground truth) and scaffold_mask")
        print(f"     4. Forward diffusion adds noise to labels at scaffold positions")
        print(f"     5. Diffusion head predicts original tokens from noisy state")
        print(f"     6. Loss computed on predictions vs labels at masked positions")

        print(f"\n6. TRAINING LOSS COMPUTATION:")

        with torch.no_grad():
            training_loss = model.diffusion_head.training_step(
                tokens=labels,
                hidden_states=hidden_states,
                scaffold_mask=scaffold_mask
            )

            print(f"   Training loss: {training_loss.item():.4f}")

            if training_loss.item() > 0:
                print(f"   ✓ Loss is computed correctly")
            else:
                print(f"   ⚠️  Loss is 0 (no valid positions for loss)")

        print(f"\n7. KEY INSIGHTS:")
        print(f"   - Dataset creates {scaffold_mask.sum().item()} scaffold positions")
        print(f"   - Only {(labels != -100).sum().item()} positions have valid labels")
        print(f"   - This is expected when mask_budget > actual target token count")
        print(f"   - Model will only learn on positions with valid labels (non -100)")
        print(f"   - Positions with labels=-100 are ignored in loss computation")


def test_chat_examples(dataset, num_samples=10):
    """Test that chat examples (is_tool=0) have no scaffold_mask."""
    print("\n" + "=" * 80)
    print("Testing Chat Examples")
    print("=" * 80)

    chat_count = 0
    tool_count = 0
    errors = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        is_tool = sample["is_tool"]
        scaffold_mask = sample["scaffold_mask"]

        if not is_tool:
            chat_count += 1
            if scaffold_mask.sum() > 0:
                errors.append(
                    f"Sample {i}: Chat example (is_tool=0) but has "
                    f"{scaffold_mask.sum().item()} scaffold_mask positions"
                )
        else:
            tool_count += 1

    print(f"Chat examples (is_tool=0): {chat_count}")
    print(f"Tool examples (is_tool=1): {tool_count}")

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
    else:
        print("✅ All chat example tests passed!")

    return len(errors) == 0


def main():
    print("Dataset Test Script with Model Debugging")
    print("=" * 80)

    config = load_config()
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    diff_cfg = config.get("diffusion", {})

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_id"])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")
    print(f"Max sequence length: {training_cfg['max_seq_len']}")
    print(f"Dataset limit: {data_cfg.get('limit', 'None')}")

    print("\nLoading dataset...")
    dataset = SmartScaffoldDataset(
        tokenizer=tokenizer,
        split="train",
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        limit=data_cfg.get("limit", 100),
        mask_token=mask_token_str,
        mask_budget=data_cfg.get("mask_budget", 48)
    )

    print(f"Dataset size: {len(dataset)}")

    num_test_samples = min(50, len(dataset))

    results = {}
    results["structure"] = test_dataset_structure(dataset, tokenizer, num_test_samples)
    results["mask_consistency"] = test_mask_token_consistency(
        dataset, tokenizer, mask_token_id, num_test_samples
    )
    results["labels_consistency"] = test_labels_consistency(dataset, num_test_samples)
    results["sequence_lengths"] = test_sequence_lengths(
        dataset, training_cfg["max_seq_len"], num_test_samples
    )
    results["chat_examples"] = test_chat_examples(dataset, num_test_samples)

    show_examples(dataset, tokenizer, mask_token_id, num_examples=3)

    print("\n" + "=" * 80)
    print("Loading Model for Debugging...")
    print("=" * 80)

    device = get_device()
    print(f"Using device: {device}")

    model = HybridSmolLM(
        base_model_id=model_cfg["base_model_id"],
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        diffusion_config=diff_cfg,
        vocab_size=len(tokenizer)
    )
    model.diffusion_head.set_mask_token_id(mask_token_id)
    model = model.to(device)

    print(f"Model loaded successfully")
    print(f"Diffusion head mask_token_id: {model.diffusion_head.mask_token_id}")

    debug_model_input_output(
        dataset, tokenizer, mask_token_id, model, num_examples=3
    )

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    if all_passed:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
