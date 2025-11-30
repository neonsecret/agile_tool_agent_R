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
from data.utils import resolve_mask_token


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
        
        required_keys = ["input_ids", "attention_mask", "scaffold_mask", "labels", "diffusion_steps", "router_label"]
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
        
        if not isinstance(sample["diffusion_steps"], (int, torch.Tensor)):
            errors.append(f"Sample {i}: diffusion_steps is not int or tensor")
        
        if not isinstance(sample["router_label"], (int, torch.Tensor)):
            errors.append(f"Sample {i}: router_label is not int or tensor")
    
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
        
        if scaffold_mask.sum() == 0 and sample["router_label"] == 1:
            warnings.append(
                f"Sample {i}: Tool call (router_label=1) but no scaffold_mask positions"
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
        
        print(f"\n--- Example {i+1} ---")
        print(f"Router label: {sample['router_label']}")
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


def test_chat_examples(dataset, num_samples=10):
    """Test that chat examples (router_label=0) have no scaffold_mask."""
    print("\n" + "=" * 80)
    print("Testing Chat Examples")
    print("=" * 80)
    
    chat_count = 0
    tool_count = 0
    errors = []
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        router_label = sample["router_label"]
        scaffold_mask = sample["scaffold_mask"]
        
        if router_label == 0:
            chat_count += 1
            if scaffold_mask.sum() > 0:
                errors.append(
                    f"Sample {i}: Chat example (router_label=0) but has "
                    f"{scaffold_mask.sum().item()} scaffold_mask positions"
                )
        else:
            tool_count += 1
    
    print(f"Chat examples (router_label=0): {chat_count}")
    print(f"Tool examples (router_label=1): {tool_count}")
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:10]:
            print(f"  - {error}")
    else:
        print("✅ All chat example tests passed!")
    
    return len(errors) == 0


def main():
    print("Dataset Test Script")
    print("=" * 80)
    
    config = load_config()
    model_cfg = config["model"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    
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
        mask_token=mask_token_str
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

