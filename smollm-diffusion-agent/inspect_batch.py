"""
Inspect a batch to verify padding/bucketing, masks, and label alignment.

Run with:
  python smollm-diffusion-agent/inspect_batch.py
"""

import torch
from transformers import AutoTokenizer
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token
from train import build_collate_fn, load_config


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def summarize_tensor(name, t, pad_value=None, tokenizer=None, show_decoded=False):
    print(f"\n{name}: shape={tuple(t.shape)}, dtype={t.dtype}")
    print(f"  first 32: {t[0][:32].tolist()}")
    if t.size(1) > 32:
        print(f"  last 16:  {t[0][-16:].tolist()}")
    if pad_value is not None:
        pad_count = (t[0] == pad_value).sum().item()
        print(f"  pad_value={pad_value}, count={pad_count}")
    if show_decoded and tokenizer is not None:
        valid_ids = t[0][t[0] >= 0].tolist()
        if valid_ids:
            decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)
            print(f"  decoded (first 50 valid tokens): {decoded}...")


def verify_alignment(batch, mask_token_id, null_token_id, tokenizer):
    print_section("ALIGNMENT VERIFICATION")

    input_ids = batch["input_ids"]
    scaffold_mask = batch["scaffold_mask"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    for i in range(input_ids.size(0)):
        print(f"\n--- Example {i} ---")
        seq_len = input_ids[i].size(0)

        # Count positions
        scaffold_positions = scaffold_mask[i].sum().item()
        valid_label_positions = (labels[i] != -100).sum().item()
        mask_token_positions = (input_ids[i] == mask_token_id).sum().item()
        attention_active = attention_mask[i].sum().item()

        print(f"  sequence length: {seq_len}")
        print(f"  attention_mask active positions: {attention_active}")
        print(f"  scaffold_mask=True positions: {scaffold_positions}")
        print(f"  labels != -100 positions: {valid_label_positions}")
        print(f"  input_ids == mask_token_id ({mask_token_id}): {mask_token_positions}")

        # Verify: scaffold_mask should align with valid labels
        scaffold_indices = scaffold_mask[i].nonzero(as_tuple=True)[0]
        label_valid_indices = (labels[i] != -100).nonzero(as_tuple=True)[0]

        if scaffold_positions != valid_label_positions:
            print(f"  ⚠️  MISMATCH: scaffold_mask ({scaffold_positions}) != valid labels ({valid_label_positions})")
        else:
            if torch.equal(scaffold_indices, label_valid_indices):
                print(f"  ✓ scaffold_mask and labels are aligned")
            else:
                print(f"  ⚠️  POSITION MISMATCH: indices don't match")
                print(f"      scaffold indices (first 10): {scaffold_indices[:10].tolist()}")
                print(f"      label indices (first 10): {label_valid_indices[:10].tolist()}")

        # Verify mask tokens are at scaffold positions in input_ids
        mask_positions = (input_ids[i] == mask_token_id).nonzero(as_tuple=True)[0]
        if mask_token_positions == scaffold_positions:
            if torch.equal(mask_positions, scaffold_indices):
                print(f"  ✓ mask tokens are at scaffold positions")
            else:
                print(f"  ⚠️  mask tokens NOT at scaffold positions")
        else:
            print(f"  ⚠️  mask token count ({mask_token_positions}) != scaffold positions ({scaffold_positions})")

        # Verify padding positions: where attention_mask=0, scaffold_mask should be False
        padding_positions = (attention_mask[i] == 0).nonzero(as_tuple=True)[0]
        if len(padding_positions) > 0:
            scaffold_at_padding = scaffold_mask[i][padding_positions].any().item()
            labels_at_padding = (labels[i][padding_positions] != -100).any().item()
            if scaffold_at_padding or labels_at_padding:
                print(f"  ⚠️  PADDING ERROR: scaffold_mask or labels set in padding region")
            else:
                print(f"  ✓ padding positions are correctly masked")

        # Show what labels contain (actual target tokens)
        if valid_label_positions > 0:
            target_ids = labels[i][labels[i] != -100].tolist()
            print(f"  target token IDs (first 20): {target_ids}")
            if null_token_id is not None:
                null_count = sum(1 for t in target_ids if t == null_token_id)
                print(f"  null_token_id ({null_token_id}) count in targets: {null_count}")
            # Decode target tokens
            decoded_target = tokenizer.decode(target_ids, skip_special_tokens=False)
            print(f"  decoded target: {decoded_target}...")


def verify_bucketing(items, bucket_sizes, max_seq_len, collate_fn):
    print_section("BUCKETING VERIFICATION")

    lengths = [it["input_ids"].size(0) for it in items]
    max_len = max(lengths)

    # Calculate expected bucket
    expected_bucket = None
    for b in sorted(bucket_sizes):
        if b >= max_len:
            expected_bucket = b
            break
    if expected_bucket is None:
        expected_bucket = sorted(bucket_sizes)[-1]
    expected_bucket = min(expected_bucket, max_seq_len)

    print(f"Item lengths: {lengths}")
    print(f"Max length in batch: {max_len}")
    print(f"Bucket sizes: {bucket_sizes}")
    print(f"Expected bucket: {expected_bucket}")

    batch = collate_fn(items)
    actual_len = batch["input_ids"].size(1)

    print(f"Actual padded length: {actual_len}")

    if actual_len == expected_bucket:
        print(f"✓ Bucketing is correct")
    else:
        print(f"⚠️  BUCKETING MISMATCH: expected {expected_bucket}, got {actual_len}")

    return batch


def inspect_single_example(ds, idx, tokenizer, mask_token_id, null_token_id):
    print_section(f"RAW EXAMPLE {idx} (before collate)")

    item = ds[idx]
    print(f"input_ids shape: {item['input_ids'].shape}")
    print(f"scaffold_mask True count: {item['scaffold_mask'].sum().item()}")
    print(f"labels != -100 count: {(item['labels'] != -100).sum().item()}")

    # Show the template structure
    ex = ds.processed_examples[idx]
    if ex.get("template") is not None:
        template = ex["template"]
        print(template)
        print(f"\nTemplate text:\n{template.text}...")
        print(f"\nField segments:")
        for seg in template.field_segments:
            print(f"  {seg.name}: positions {seg.start}-{seg.end}, {len(seg.value_positions)} mask slots")

        # Show target values
        target_map = ex["target_tokens_map"]
        print(f"\nTarget tokens per field:")
        for name, tokens in target_map.items():
            decoded = tokenizer.decode(tokens, skip_special_tokens=False)
            print(f"  {name}: {len(tokens)} tokens -> '{decoded}...'")
    else:
        print("  (Chat example - no template)")

    # Show decoded input
    decoded_input = tokenizer.decode(item['input_ids'].tolist(), skip_special_tokens=False)
    print(f"\nDecoded input (first 500 chars):\n{decoded_input}...")


def main():
    config = load_config()
    data_cfg = config["data"]
    training_cfg = config["training"]
    bucket_sizes = data_cfg.get("bucket_sizes", [512, 1024, 1536, 2048])

    print_section("CONFIG")
    print(f"bucket_sizes: {bucket_sizes}")
    print(f"max_seq_len: {training_cfg['max_seq_len']}")
    print(f"mask_budget: {data_cfg.get('mask_budget', 48)}")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, data_cfg.get("mask_token"))
    null_token_str, null_token_id = None, None
    try:
        null_token_str, null_token_id = resolve_null_token(tokenizer, data_cfg.get("null_token"))
    except Exception as e:
        print(f"Warning: NULL token unavailable: {e}")

    print_section("TOKENS")
    print(f"mask_token: '{mask_token_str}' (id={mask_token_id})")
    print(f"null_token: '{null_token_str}' (id={null_token_id})")
    print(f"pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    print(f"eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    # Create dataset
    ds = SmartScaffoldDataset(
        tokenizer,
        split="train",
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        limit=8,
        mask_token=mask_token_str,
        null_token=null_token_str,
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48),
    )

    print_section("DATASET")
    print(f"Dataset size: {len(ds)}")

    # Find tool and chat examples
    tool_indices = [i for i, ex in enumerate(ds.processed_examples) if ex.get("template") is not None]
    chat_indices = [i for i, ex in enumerate(ds.processed_examples) if ex.get("template") is None]
    print(f"Tool examples: {len(tool_indices)}")
    print(f"Chat examples: {len(chat_indices)}")

    # Inspect a tool example
    if tool_indices:
        inspect_single_example(ds, tool_indices[0], tokenizer, mask_token_id, null_token_id)

    # Build batch with tool examples only (to verify scaffold_mask)
    if len(tool_indices) >= 2:
        items = [ds[i] for i in tool_indices[:2]]
    else:
        items = [ds[i] for i in range(min(2, len(ds)))]

    collate_fn = build_collate_fn(bucket_sizes, training_cfg["max_seq_len"])
    batch = verify_bucketing(items, bucket_sizes, training_cfg["max_seq_len"], collate_fn)

    print_section("BATCH TENSORS")
    summarize_tensor("input_ids", batch["input_ids"], pad_value=0, tokenizer=tokenizer)
    summarize_tensor("attention_mask", batch["attention_mask"], pad_value=0)
    summarize_tensor("scaffold_mask", batch["scaffold_mask"], pad_value=0)
    summarize_tensor("labels", batch["labels"], pad_value=-100, tokenizer=tokenizer, show_decoded=True)

    verify_alignment(batch, mask_token_id, null_token_id, tokenizer)

    print_section("SUMMARY")
    print("Check the above output for any ⚠️ warnings")


if __name__ == "__main__":
    main()
