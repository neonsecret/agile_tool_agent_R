"""
Investigate training data to understand NULL token distribution and field length patterns.
"""
import torch
from transformers import AutoTokenizer
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token
from collections import defaultdict, Counter
import yaml
import numpy as np


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def analyze_dataset():
    config = load_config()
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    
    data_cfg = config.get("data", {})
    mask_token_config = data_cfg.get("mask_token", None)
    null_token_config = data_cfg.get("null_token", None)
    
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)
    null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)
    
    print("="*80)
    print("DATASET INVESTIGATION")
    print("="*80)
    print(f"Mask token: {mask_token_str} (ID: {mask_token_id})")
    print(f"NULL token: {null_token_str} (ID: {null_token_id})")
    print()
    
    dataset_name = data_cfg.get("dataset_name", "interstellarninja/hermes_reasoning_tool_use")
    mask_budget = data_cfg.get("mask_budget", 48)
    system_message = data_cfg.get("system_message", "/no_think")
    max_history = data_cfg.get("max_history_messages", 12)
    max_seq_len = config.get("training", {}).get("max_seq_len", 2048)
    chat_sampling_rate = data_cfg.get("chat_sampling_rate", 0.1)
    
    print("Loading dataset (this may take a while)...")
    dataset = SmartScaffoldDataset(
        tokenizer=tokenizer,
        split="train",
        max_seq_len=max_seq_len,
        mask_token=mask_token_str,
        null_token=null_token_str,
        mask_budget=mask_budget,
        system_message=system_message,
        max_history_messages=max_history,
        chat_sampling_rate=chat_sampling_rate,
        limit=1000,  # Sample 1000 examples for analysis
    )
    
    print(f"Loaded {len(dataset)} examples\n")
    
    # Statistics to collect
    total_mask_tokens = []
    total_null_tokens = []
    null_to_mask_ratios = []
    mask_token_positions = []
    null_token_positions = []
    tool_examples = 0
    chat_examples = 0
    
    field_lengths = defaultdict(list)  # field_name -> [length1, length2, ...]
    tool_names = Counter()
    
    print("Analyzing examples...")
    for i in range(min(len(dataset), 1000)):
        if i % 100 == 0:
            print(f"  Processing {i}/{min(len(dataset), 1000)}...")
        
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        is_tool = sample["is_tool"]
        
        if is_tool:  # Tool example
            tool_examples += 1
            
            # Count mask and NULL tokens
            mask_count = (input_ids == mask_token_id).sum().item()
            null_count = (input_ids == null_token_id).sum().item() if null_token_id else 0
            
            total_mask_tokens.append(mask_count)
            total_null_tokens.append(null_count)
            
            if mask_count > 0:
                null_to_mask_ratios.append(null_count / mask_count)
            
            # Find positions of mask and NULL tokens
            mask_positions = torch.where(input_ids == mask_token_id)[0].tolist()
            null_positions = torch.where(input_ids == null_token_id)[0].tolist() if null_token_id else []
            
            mask_token_positions.extend(mask_positions)
            null_token_positions.extend(null_positions)
            
        else:  # Chat example
            chat_examples += 1
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nExample Distribution:")
    print(f"  Tool examples: {tool_examples} ({tool_examples/(tool_examples+chat_examples)*100:.1f}%)")
    print(f"  Chat examples: {chat_examples} ({chat_examples/(tool_examples+chat_examples)*100:.1f}%)")
    
    if total_mask_tokens:
        print(f"\nMask Token Statistics (per tool example):")
        print(f"  Mean: {np.mean(total_mask_tokens):.1f}")
        print(f"  Median: {np.median(total_mask_tokens):.1f}")
        print(f"  Std: {np.std(total_mask_tokens):.1f}")
        print(f"  Min: {np.min(total_mask_tokens)}")
        print(f"  Max: {np.max(total_mask_tokens)}")
        print(f"  Percentiles:")
        print(f"    25th: {np.percentile(total_mask_tokens, 25):.1f}")
        print(f"    50th: {np.percentile(total_mask_tokens, 50):.1f}")
        print(f"    75th: {np.percentile(total_mask_tokens, 75):.1f}")
        print(f"    90th: {np.percentile(total_mask_tokens, 90):.1f}")
    
    if total_null_tokens:
        print(f"\nNULL Token Statistics (per tool example):")
        print(f"  Mean: {np.mean(total_null_tokens):.1f}")
        print(f"  Median: {np.median(total_null_tokens):.1f}")
        print(f"  Std: {np.std(total_null_tokens):.1f}")
        print(f"  Min: {np.min(total_null_tokens)}")
        print(f"  Max: {np.max(total_null_tokens)}")
        
        print(f"\nNULL-to-Mask Ratio:")
        print(f"  Mean: {np.mean(null_to_mask_ratios):.2f}")
        print(f"  Median: {np.median(null_to_mask_ratios):.2f}")
        print(f"  This means on average, {np.mean(null_to_mask_ratios)*100:.1f}% of masked positions are NULL tokens!")
    
    # Analyze a few examples in detail
    print(f"\n{'='*80}")
    print("DETAILED EXAMPLE ANALYSIS")
    print("="*80)
    
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        if not sample["is_tool"]:  # Skip chat examples
            continue
        
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        print(f"\nExample {i}:")
        
        # Decode the full sequence
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Count tokens
        mask_count = (input_ids == mask_token_id).sum().item()
        null_count = (input_ids == null_token_id).sum().item() if null_token_id else 0
        real_tokens_in_scaffold = mask_count - null_count
        
        print(f"  Total length: {len(input_ids)} tokens")
        print(f"  Mask tokens: {mask_count}")
        print(f"  NULL tokens: {null_count}")
        print(f"  Real tokens to predict: {real_tokens_in_scaffold}")
        print(f"  NULL ratio: {null_count/mask_count*100:.1f}%")
        
        # Show the scaffold portion (last ~200 tokens)
        scaffold_start = max(0, len(input_ids) - 200)
        scaffold_text = tokenizer.decode(input_ids[scaffold_start:], skip_special_tokens=False)
        print(f"  Scaffold preview (last 200 tokens):")
        for line in scaffold_text.split('\n')[:10]:
            print(f"    {line}")
        
        # Show what the model should predict (from labels)
        label_mask = labels != -100
        if label_mask.any():
            predicted_tokens = labels[label_mask]
            null_in_labels = (predicted_tokens == null_token_id).sum().item() if null_token_id else 0
            real_in_labels = len(predicted_tokens) - null_in_labels
            print(f"  Labels to predict: {len(predicted_tokens)} tokens")
            print(f"    Real tokens: {real_in_labels}")
            print(f"    NULL tokens: {null_in_labels}")
            print(f"    NULL ratio in labels: {null_in_labels/len(predicted_tokens)*100:.1f}%")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print("="*80)
    
    if null_to_mask_ratios:
        avg_null_ratio = np.mean(null_to_mask_ratios)
        if avg_null_ratio > 0.5:
            print(f"\n⚠️  CRITICAL: {avg_null_ratio*100:.1f}% of masked positions are NULL tokens!")
            print(f"   The model is learning to predict NULL more often than real tokens.")
            print(f"   Recommended fixes:")
            print(f"   1. Reduce mask_budget from {mask_budget} to 32")
            print(f"   2. Add loss weighting: penalize NULL predictions")
            print(f"   3. Mask NULL token in inference logits")
        elif avg_null_ratio > 0.3:
            print(f"\n⚠️  WARNING: {avg_null_ratio*100:.1f}% of masked positions are NULL tokens.")
            print(f"   This is moderate but could still cause issues.")
            print(f"   Consider reducing mask_budget from {mask_budget} to 36-40")
        else:
            print(f"\n✓ NULL ratio is reasonable at {avg_null_ratio*100:.1f}%")


if __name__ == "__main__":
    analyze_dataset()
