"""
Quick test script for BFCL evaluation using sample data.

This script demonstrates the BFCL evaluation process on a small sample dataset.
It's useful for testing and debugging before running on the full benchmark.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

from evaluate_bfcl import (
    BFCLEvaluator,
    _build_generator,
    _build_model,
    _load_config,
)
from data.config_utils import validate_and_adjust_config
from data.device_utils import get_device


def main():
    print("=" * 60)
    print("Berkeley Function Call Leaderboard - Quick Test")
    print("=" * 60)
    print()

    # Use sample data
    sample_data_path = Path(__file__).parent / "sample_bfcl_data.json"
    output_path = Path(__file__).parent / "bfcl_quick_test_results.json"

    if not sample_data_path.exists():
        print(f"Error: Sample data not found at {sample_data_path}")
        print("Please ensure sample_bfcl_data.json exists in the same directory.")
        return

    print(f"Using sample dataset: {sample_data_path}")
    print(f"Output will be saved to: {output_path}")
    print()

    # Load model + generator
    print("Loading diffusion LLM model...")
    print("(This may take a moment...)")
    config = _load_config()
    device = get_device()
    config = validate_and_adjust_config(config, device)
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("base_model_id", "HuggingFaceTB/SmolLM3-3B")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _build_model(config, tokenizer, device)
    generator, gen_config = _build_generator(config, tokenizer, device, model)
    print(f"Model loaded on device: {device}")
    print()

    # Create evaluator
    print("Creating BFCL evaluator...")
    evaluator = BFCLEvaluator(
        generator=generator,
        gen_config=gen_config,
    )
    print()

    # Load and display sample data info
    with open(sample_data_path, 'r') as f:
        sample_data = json.load(f)

    print(f"Sample dataset contains {len(sample_data)} examples:")
    for i, example in enumerate(sample_data):
        print(f"  {i + 1}. {example['question'][:60]}...")
    print()

    # Run evaluation
    print("Starting evaluation...")
    print("-" * 60)
    metrics = evaluator.evaluate_dataset(
        data_path=str(sample_data_path),
        output_path=str(output_path),
        limit=None  # Evaluate all samples
    )
    print("-" * 60)
    print()

    # Display detailed results
    print("Detailed Results:")
    print("=" * 60)

    with open(output_path, 'r') as f:
        results_data = json.load(f)

    for result in results_data['results'][:5]:  # Show first 5
        if 'error' in result:
            print(f"\nExample {result['id']} - ERROR: {result['error']}")
            continue

        print(f"\nExample {result['id']}:")
        print(f"Query: {result['query']}")
        print(f"Predicted: {result['prediction'].get('name', 'N/A')}")
        print(f"Ground Truth: {result['ground_truth'].get('name', 'N/A')}")
        print(f"Correct: {'✓' if result['correct'] else '✗'}")

        # Show arguments if correct
        if result['correct']:
            pred_args = result['prediction'].get("arguments", {})
            if pred_args:
                print(f"Arguments: {pred_args}")

    print()
    print("=" * 60)
    print("Summary:")
    print(f"  Total Examples: {metrics['total']}")
    print(f"  Correct: {metrics['correct']}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Error Rate: {metrics['error_rate']:.2%}")
    print("=" * 60)
    print()
    print(f"Full results saved to: {output_path}")
    print()
    print("To run on full BFCL dataset, use:")
    print("  python evaluate_bfcl.py --data-path /path/to/bfcl/data.json")
    print()


if __name__ == "__main__":
    main()
