import argparse
import torch
from transformers import AutoTokenizer
from accelerate import Accelerator
from model.hybrid_model import HybridSmolLM


def dry_run():
    print("=== DRY RUN VERIFICATION ===")

    # 1. Check Environment
    print("Checking Accelerator...")
    try:
        accelerator = Accelerator()
        print(f"Accelerator detected: {accelerator.device}")
    except Exception as e:
        print(f"Accelerator Failed: {e}")
        return

    # 2. Check Model Loading (CPU only)
    print("\nLoading Model Architecture (CPU)...")
    try:
        model = HybridSmolLM()
        print("Model loaded successfully.")
        print(f"Base Model Config: {model.base_llm.config}")

        # Verify Freezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Params: {total_params:,}")
        print(f"Trainable Params (Diffusion Head): {trainable_params:,}")
        print(f"Ratio: {trainable_params / total_params:.4%}")

        if trainable_params > 50_000_000:  # Expecting small head
            print("WARNING: Trainable params seem too high. Did we freeze base model?")
    except Exception as e:
        print(f"Model Load Failed: {e}")
        return

    # 3. Check Tokenizer
    print("\nChecking Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        vocab_size = len(tokenizer)
        print(f"Vocab Size: {vocab_size}")
    except Exception as e:
        print(f"Tokenizer Failed: {e}")
        return

    print("\n=== DRY RUN SUCCESS ===")
    print("Ready for cluster deployment.")


if __name__ == "__main__":
    dry_run()
