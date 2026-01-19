"""
Quick test script for BFCL evaluation using sample data.

This script demonstrates the BFCL evaluation process on a small sample dataset.
It's useful for testing and debugging before running on the full benchmark.
"""

import json
from pathlib import Path
from evaluate_bfcl import BFCLEvaluator
from models import load_diffusion_llm


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

    # Load model
    print("Loading diffusion LLM model...")
    print("(This may take a moment...)")
    model, tokenizer, device = load_diffusion_llm()
    print(f"Model loaded on device: {device}")
    print()

    # Create evaluator
    print("Creating BFCL evaluator...")
    evaluator = BFCLEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_steps=16  # Using 16 steps as in test_function_calling()
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
        print(f"Predicted: {result['prediction'].get('function_name', 'N/A')}")
        print(f"Ground Truth: {result['ground_truth'].get('name', 'N/A')}")
        print(f"Correct: {'✓' if result['correct'] else '✗'}")

        # Show arguments if correct
        if result['correct']:
            pred_args = {}
            i = 1
            while f"arg_{i}_name" in result['prediction']:
                arg_name = result['prediction'].get(f"arg_{i}_name", "")
                arg_value = result['prediction'].get(f"arg_{i}_value", "")
                if arg_name and arg_value:
                    pred_args[arg_name] = arg_value
                i += 1

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
