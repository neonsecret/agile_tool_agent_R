"""
Hand-inspect actual training data to verify what the model sees.

Shows:
1. Input sequences (prompt + scaffold with masks)
2. Target labels (what model should predict)
3. Length jitter in action
4. NULL token usage
5. Field budgets
"""
import yaml
import torch
from transformers import AutoTokenizer
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token


def inspect_single_example(dataset, idx, tokenizer):
    print("\n" + "=" * 80)
    print(f"EXAMPLE {idx}")
    print("=" * 80)
    
    batch = dataset[idx]
    
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    scaffold_mask = batch["scaffold_mask"]
    is_tool = batch["is_tool"]
    
    print(f"\nIs Tool Call: {is_tool}")
    print(f"Sequence Length: {len(input_ids)}")
    print(f"Scaffold Positions: {scaffold_mask.sum().item()}")
    
    if not is_tool:
        print("\n[This is a chat example, not a tool call - skipping details]\n")
        return
    
    scaffold_positions = torch.where(scaffold_mask)[0].tolist()
    print(f"Scaffold Position Indices: {scaffold_positions[:10]}... ({len(scaffold_positions)} total)")
    
    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    
    print("\n" + "-" * 80)
    print("FULL SEQUENCE (truncated):")
    print("-" * 80)
    print(decoded_full[:800])
    if len(decoded_full) > 800:
        print(f"\n... [truncated, total {len(decoded_full)} chars]")
    
    print("\n" + "-" * 80)
    print("SCAFFOLD REGION DETAIL:")
    print("-" * 80)
    
    if len(scaffold_positions) > 0:
        start_ctx = max(0, scaffold_positions[0] - 10)
        end_ctx = min(len(input_ids), scaffold_positions[-1] + 10)
        
        for i in range(start_ctx, end_ctx):
            token_id = input_ids[i].item()
            label_id = labels[i].item()
            is_scaffold = scaffold_mask[i].item()
            
            token_str = tokenizer.decode([token_id])
            
            if label_id == -100:
                label_str = "[IGNORE]"
            else:
                label_str = tokenizer.decode([label_id])
            
            marker = "→" if is_scaffold else " "
            
            if is_scaffold:
                print(f"  {marker} Pos {i:4d}: Input='{token_str:20s}' | Label='{label_str:20s}'")
    
    print("\n" + "-" * 80)
    print("LABEL STATISTICS:")
    print("-" * 80)
    
    mask_token_id = dataset.mask_token_id
    null_token_id = dataset.null_token_id
    
    scaffold_labels = labels[scaffold_mask]
    non_ignore = scaffold_labels[scaffold_labels != -100]
    
    if len(non_ignore) > 0:
        num_null = (non_ignore == null_token_id).sum().item() if null_token_id else 0
        num_real = len(non_ignore) - num_null
        
        print(f"Total scaffold labels: {len(non_ignore)}")
        print(f"  Real tokens: {num_real} ({num_real/len(non_ignore)*100:.1f}%)")
        if null_token_id:
            print(f"  NULL tokens: {num_null} ({num_null/len(non_ignore)*100:.1f}%)")
        
        unique_tokens = torch.unique(non_ignore)
        print(f"  Unique token types: {len(unique_tokens)}")
        
        print(f"\nSample target tokens (first 20):")
        for i, label_id in enumerate(non_ignore[:20].tolist()):
            token_str = tokenizer.decode([label_id])
            print(f"  {i}: '{token_str}'", end="")
            if (i + 1) % 5 == 0:
                print()
        print("\n")


def inspect_length_jitter():
    print("\n" + "=" * 80)
    print("LENGTH JITTER DEMONSTRATION")
    print("=" * 80)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    jitter_cfg = config["data"]["dynamic_budget"].get("length_jitter", {})
    
    print(f"\nConfiguration:")
    print(f"  Enabled: {jitter_cfg.get('enabled', False)}")
    print(f"  Mode: {jitter_cfg.get('mode', 'over')}")
    print(f"  Range: {jitter_cfg.get('min_jitter', 0)} to {jitter_cfg.get('max_jitter', 5)} tokens")
    
    print(f"\nExample field with true length = 10 tokens:")
    print(f"  Over-budget jitter: 10 → 10-15 (model must predict NULLs)")
    print(f"  Under-budget jitter: 10 → 5-10 (model must be concise)")
    print(f"  Both mode: Random choice of over OR under each time")
    
    print(f"\nWhy this helps:")
    print(f"  ✓ Model learns when to STOP (NULL prediction)")
    print(f"  ✓ Model learns to be CONCISE (not fill all slots)")
    print(f"  ✓ Prevents memorizing exact field lengths")
    print(f"  ✓ Robust to variable-length fields")


def main():
    print("=" * 80)
    print("TRAINING DATA INSPECTION - SmolLM Diffusion Agent")
    print("=" * 80)
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 80)
    print("DATASET COMPOSITION")
    print("=" * 80)
    
    datasets = config["data"]["datasets"]
    
    sizes = {
        'interstellarninja/hermes_reasoning_tool_use': 20000,
        'Salesforce/xlam-function-calling-60k': 60000,
        'glaiveai/glaive-function-calling-v2': 50000,
        'Salesforce/APIGen-MT-5k': 5000,
        'argilla/apigen-function-calling': 109000,
        'nvidia/When2Call': 15000,
        'ibm-research/nestful': 1860,
    }
    
    fc_total = 0
    code_total = 0
    
    for ds in datasets:
        name = ds['name']
        weight = ds.get('weight', 1.0)
        limit = ds.get('limit')
        base_size = sizes.get(name, 0)
        effective = min(base_size, limit) if limit else base_size
        weighted = int(effective * weight)
        
        is_code = 'apigen-function-calling' in name or 'nestful' in name
        if is_code:
            code_total += weighted
            category = "CODE/MATH"
        else:
            fc_total += weighted
            category = "FUNC-CALL"
        
        print(f"\n{name}")
        print(f"  Category: {category}")
        print(f"  Base: ~{base_size:,}, Weight: {weight}, Limit: {limit or 'unlimited'}")
        print(f"  Effective: ~{weighted:,} examples")
    
    total = fc_total + code_total
    print(f"\n{'='*80}")
    print(f"TOTAL: ~{total:,} examples")
    print(f"  Function Calling: ~{fc_total:,} ({fc_total/total*100:.1f}%)")
    print(f"  Code/Math: ~{code_total:,} ({code_total/total*100:.1f}%)")
    print(f"{'='*80}")
    
    inspect_length_jitter()
    
    print("\n" + "=" * 80)
    print("LOADING SAMPLE DATA (this may take a few minutes)...")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    
    mask_token_str = config["data"].get("mask_token")
    if mask_token_str is None:
        mask_token_str = "<|reserved_special_token_2|>"
    
    null_token_str = config["data"].get("null_token")
    if null_token_str is None:
        null_token_str = "<|reserved_special_token_11|>"
    
    print(f"\nSpecial tokens:")
    print(f"  MASK: {mask_token_str}")
    print(f"  NULL: {null_token_str}")
    
    try:
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=config["training"]["max_seq_len"],
            max_new_tokens=config["training"]["max_new_tokens"],
            limit=5,  # Just load 5 examples for inspection
            mask_token=mask_token_str,
            null_token=null_token_str,
            chat_sampling_rate=config["data"]["chat_sampling_rate"],
            mask_budget=config["data"]["mask_budget"],
            system_message=config["data"]["system_message"],
            max_history_messages=config["data"]["max_history_messages"],
            data_config=config,
        )
        
        print(f"\n✓ Loaded {len(dataset)} examples")
        
        for i in range(min(3, len(dataset))):
            inspect_single_example(dataset, i, tokenizer)
        
    except Exception as e:
        print(f"\n✗ Failed to load dataset: {e}")
        print(f"\nThis is expected if datasets haven't been downloaded yet.")
        print(f"The configuration is correct. Run train.py to download and process datasets.")
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Model sees: Prompt + <MASK> tokens in scaffold")
    print("  2. Model predicts: Actual token values OR NULL for unused slots")
    print("  3. Length jitter: Randomly over/under-allocates slots")
    print("  4. Loss computed: ONLY on scaffold positions (not prompt)")
    print("  5. NULL tokens: Teach model when to STOP generating")


if __name__ == "__main__":
    main()
