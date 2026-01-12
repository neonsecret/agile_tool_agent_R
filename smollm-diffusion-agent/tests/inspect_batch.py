"""
Inspect the first 10 items to verify padding/bucketing and masks.

Run with:
  conda run -n torch313 python smollm-diffusion-agent/tests/inspect_batch.py
"""

import torch

from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token
from train import build_collate_fn, load_config


def summarize_tensor(name, t, pad_value=None, n=32):
    print(f"\n{name}:")
    print(f"  first {n}: {t[0][:n].tolist()}")
    if pad_value is not None:
        tail = t[0][-16:].tolist()
        print(f"  last 16:  {tail}")
        print(f"  pad value count: {(t[0] == pad_value).sum().item()}")


def main():
    config = load_config()
    data_cfg = config["data"]
    training_cfg = config["training"]
    bucket_sizes = data_cfg.get("bucket_sizes", [512, 1024, 1536, 2048])

    # Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokens
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, data_cfg.get("mask_token"))
    null_token_str, null_token_id = None, None
    try:
        null_token_str, null_token_id = resolve_null_token(tokenizer, data_cfg.get("null_token"))
    except Exception as e:  # pragma: no cover
        print(f"Warning: NULL token unavailable: {e}")

    print(f"mask_token={mask_token_str} ({mask_token_id}), null_token={null_token_str} ({null_token_id})")

    # Dataset (first 10 items)
    ds = SmartScaffoldDataset(
        tokenizer,
        split="train",
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        limit=10,
        mask_token=mask_token_str,
        null_token=null_token_str,
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48),
    )

    num_items = min(10, len(ds))
    items = [ds[i] for i in range(num_items)]
    lengths = [it["input_ids"].size(0) for it in items]
    max_len = max(lengths)
    pad_to = None
    for b in sorted(bucket_sizes):
        if b >= max_len:
            pad_to = b
            break
    if pad_to is None:
        pad_to = sorted(bucket_sizes)[-1]
    pad_to = min(pad_to, training_cfg["max_seq_len"])

    collate = build_collate_fn(bucket_sizes, training_cfg["max_seq_len"])
    batch = collate(items)

    print("\n--- Batch inspection (first 10 items) ---")
    print(f"item lengths: {lengths}")
    print(f"bucket_sizes: {bucket_sizes} -> pad_to: {pad_to}")
    print("shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

    summarize_tensor("input_ids", batch["input_ids"])
    summarize_tensor("attention_mask", batch["attention_mask"], pad_value=0)
    summarize_tensor("scaffold_mask", batch["scaffold_mask"], pad_value=0)
    summarize_tensor("labels", batch["labels"], pad_value=-100)

    if null_token_id is not None:
        null_count = (batch["labels"] == null_token_id).sum().item()
        print(f"\nNULL token id {null_token_id} count in labels: {null_count}")

    if "router_labels" in batch:
        print(f"\nrouter_labels: {batch['router_labels'].tolist()}")

    # Show first available template text if present
    try:
        tpl = None
        for ex in ds.processed_examples:
            tpl = ex.get("template", None)
            if tpl is not None and getattr(tpl, "text", None):
                break
        if tpl is not None:
            print("\nSample template text (structure only; mask slots not shown):")
            print(tpl.text if getattr(tpl, "text", None) else "(empty)")
            try:
                decoded_tpl = tokenizer.decode(
                    tpl.to_tensor(tokenizer.device if hasattr(tokenizer, "device") else torch.device("cpu")),
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                print("\nDecoded template tokens (includes mask slots):")
                print(decoded_tpl[:400])
            except Exception:
                pass
    except Exception:
        pass


if __name__ == "__main__":
    main()
"""
Inspect a small batch to verify padding/bucketing and masks.

Run with:
  conda run -n torch313 python smollm-diffusion-agent/tests/inspect_batch.py
"""

import json

import torch

from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token
from train import build_collate_fn, load_config


def main():
    config = load_config()
    data_cfg = config["data"]
    training_cfg = config["training"]
    bucket_sizes = data_cfg.get("bucket_sizes", [512, 1024, 1536, 2048])

    # Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["base_model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokens
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, data_cfg.get("mask_token"))
    null_token_str, null_token_id = None, None
    try:
        null_token_str, null_token_id = resolve_null_token(tokenizer, data_cfg.get("null_token"))
    except Exception as e:  # pragma: no cover
        print(f"Warning: NULL token unavailable: {e}")

    print(f"mask_token={mask_token_str} ({mask_token_id}), null_token={null_token_str} ({null_token_id})")

    # Dataset (small slice)
    ds = SmartScaffoldDataset(
        tokenizer,
        split="train",
        max_seq_len=training_cfg["max_seq_len"],
        max_new_tokens=training_cfg["max_new_tokens"],
        limit=4,  # small sample for inspection
        mask_token=mask_token_str,
        null_token=null_token_str,
        chat_sampling_rate=data_cfg.get("chat_sampling_rate", 0.1),
        mask_budget=data_cfg.get("mask_budget", 48),
    )

    items = [ds[i] for i in range(min(2, len(ds)))]
    lengths = [it["input_ids"].size(0) for it in items]
    max_len = max(lengths)
    pad_to = None
    for b in sorted(bucket_sizes):
        if b >= max_len:
            pad_to = b
            break
    if pad_to is None:
        pad_to = sorted(bucket_sizes)[-1]
    pad_to = min(pad_to, training_cfg["max_seq_len"])

    collate = build_collate_fn(bucket_sizes, training_cfg["max_seq_len"])
    batch = collate(items)

    print("\n--- Batch inspection ---")
    print(f"item lengths: {lengths}")
    print(f"bucket_sizes: {bucket_sizes} -> pad_to: {pad_to}")
    print("shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")

    # Inspect first example tensors
    def summarize_tensor(name, t, pad_value=None):
        print(f"\n{name}:")
        print(f"  first 32: {t[0][:32].tolist()}")
        if pad_value is not None:
            tail = t[0][-16:].tolist()
            print(f"  last 16:  {tail}")
            print(f"  pad value count: {(t[0] == pad_value).sum().item()}")

    summarize_tensor("input_ids", batch["input_ids"])
    summarize_tensor("attention_mask", batch["attention_mask"], pad_value=0)
    summarize_tensor("scaffold_mask", batch["scaffold_mask"], pad_value=0)
    summarize_tensor("labels", batch["labels"], pad_value=-100)

    if null_token_id is not None:
        null_count = (batch["labels"] == null_token_id).sum().item()
        print(f"\nNULL token id {null_token_id} count in labels: {null_count}")

    if "router_labels" in batch:
        print(f"\nrouter_labels: {batch['router_labels'].tolist()}")

    # Show template text if available
    try:
        print("\nSample template text:")
        # grab template from dataset internals if present
        tpl = ds.processed_examples[0].get("template", None)
        if tpl is not None and getattr(tpl, "text", None):
            print(tpl.text)
    except Exception:
        pass


if __name__ == "__main__":
    main()

