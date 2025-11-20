"""
Berkeley Function Call Leaderboard (BFCL) Evaluation Script

This script evaluates the diffusion LLM on the Berkeley Function Call Leaderboard.
It loads the BFCL v3 dataset from HuggingFace (llamastack/bfcl_v3) and uses the
SelfAdaptiveSchemaScaffolder to extract function calls from user queries and
evaluates them against BFCL benchmarks.

Usage:
    python evaluate_bfcl.py --test-category simple --limit 10

The script filters for the specified test_category (default: "simple") and evaluates
the model's ability to generate correct function calls.
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import argparse
from datasets import load_dataset

sys.path.append("../dLLM-CtrlGen/")
from scaffolding import SelfAdaptiveSchemaScaffolder, SelfAdaptiveSchemaConfig
from models import load_diffusion_llm
from decoding.generator import SelfAdaptiveGenerator, GenerationConfig


class BFCLEvaluator:
    """Evaluator for Berkeley Function Call Leaderboard."""

    def __init__(self, model, tokenizer, device, generation_steps: int = 16):
        """
        Initialize the BFCL evaluator.

        Args:
            model: The diffusion LLM model
            tokenizer: The tokenizer
            device: Device to run on
            generation_steps: Number of generation steps for diffusion
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_steps = generation_steps
        self.generator = SelfAdaptiveGenerator(model, tokenizer, device)

    def create_function_calling_prompt(self, functions: List[Dict[str, Any]]) -> str:
        """
        Create a function calling prompt from BFCL function definitions.

        Args:
            functions: List of function definitions from BFCL

        Returns:
            Formatted prompt template string
        """
        prompt = (
            "You are a function calling assistant. Based on the user's request, select the appropriate "
            "function and extract its arguments. Return ONLY a JSON object with the following keys: "
            "{fields}. If an argument is not needed or unavailable, use the literal token '{null}' as its value.\n\n"
            "Available functions:\n"
        )

        for i, func in enumerate(functions, 1):
            func_name = func.get("name", "unknown")
            func_desc = func.get("description", "No description available")

            # Extract parameters
            params = func.get("parameters", {}).get("properties", {})
            param_strs = []
            for param_name, param_info in params.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                param_strs.append(f"{param_name}: {param_type}")

            params_signature = ", ".join(param_strs) if param_strs else ""
            prompt += f"{i}. {func_name}({params_signature}) - {func_desc}\n"

        prompt += "\nUser request: {text}"
        return prompt

    def parse_function_call(self, result_text: str) -> Dict[str, Any]:
        """
        Parse the generated function call from model output.

        Args:
            result_text: Generated text from the model

        Returns:
            Parsed function call dictionary
        """
        try:
            # Try to extract JSON from the result
            # The result might contain markdown code blocks
            text = result_text.strip()

            # Remove markdown code blocks if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            # Parse JSON
            parsed = json.loads(text)
            return parsed
        except Exception as e:
            print(f"Error parsing function call: {e}")
            print(f"Result text: {result_text}")
            return {}

    def evaluate_single(self, query: str, functions: List[Dict[str, Any]],
                        ground_truth: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Evaluate a single BFCL test case.

        Args:
            query: User query
            functions: List of available functions
            ground_truth: Ground truth function call

        Returns:
            Tuple of (prediction, is_correct)
        """
        # Create prompt template
        prompt_template = self.create_function_calling_prompt(functions)

        # Determine maximum number of arguments needed
        max_args = max(
            len(func.get("parameters", {}).get("properties", {}))
            for func in functions
        ) if functions else 2
        max_args = max(max_args, 2)  # At least 2 arguments

        # Create schema fields dynamically
        fields = ["function_name"]
        token_budgets = {"function_name": 16}

        for i in range(1, max_args + 1):
            fields.extend([f"arg_{i}_name", f"arg_{i}_value"])
            token_budgets[f"arg_{i}_name"] = 16
            token_budgets[f"arg_{i}_value"] = 32

        # Create schema config
        schema_cfg = SelfAdaptiveSchemaConfig(
            fields=tuple(fields),
            token_budgets=token_budgets,
            prompt_template=prompt_template,
        )

        # Build scaffolder and template
        scaffolder = SelfAdaptiveSchemaScaffolder(schema_cfg)
        template = scaffolder.build_template(self.tokenizer)

        # Make prompt
        prompt = scaffolder.make_prompt(query)

        # Generate
        result = self.generator.generate(
            prompt,
            template,
            config=GenerationConfig(steps=self.generation_steps),
            trace=False
        )

        # Parse result
        prediction = self.parse_function_call(result.text)
        print("prediction:", prediction)
        print("ground truth:", ground_truth)

        # Simple correctness check (can be enhanced)
        is_correct = self.check_correctness(prediction, ground_truth)

        return prediction, is_correct

    def check_correctness(self, prediction: Dict[str, Any],
                          ground_truth: Dict[str, Any]) -> bool:
        """
        Check if prediction matches ground truth.

        Args:
            prediction: Predicted function call
            ground_truth: Ground truth function call

        Returns:
            True if correct, False otherwise
        """
        if not prediction:
            return False

        # Extract function name from prediction
        pred_func_name = prediction.get("function_name", "").strip()
        # Remove parentheses and arguments if present
        if "(" in pred_func_name:
            pred_func_name = pred_func_name.split("(")[0].strip()

        # Get ground truth function name
        gt_func_name = ground_truth.get("name", "").strip()

        # Check function name match
        if pred_func_name != gt_func_name:
            return False

        # Check arguments (basic implementation)
        # In a full implementation, you'd want more sophisticated argument matching
        gt_args = ground_truth.get("arguments", {})

        # Extract arguments from prediction
        pred_args = {}
        i = 1
        while f"arg_{i}_name" in prediction:
            arg_name = prediction.get(f"arg_{i}_name", "").strip()
            arg_value = prediction.get(f"arg_{i}_value", "").strip()
            if arg_name and arg_value and arg_value != "{null}":
                pred_args[arg_name] = arg_value
            i += 1

        # Simple argument comparison
        if set(gt_args.keys()) != set(pred_args.keys()):
            return False

        for key in gt_args:
            if str(gt_args[key]).lower().strip() != str(pred_args[key]).lower().strip():
                return False

        return True

    def evaluate_dataset(self, test_category: str = "simple", output_path: str = None,
                         limit: int = None) -> Dict[str, Any]:
        """
        Evaluate on BFCL dataset from HuggingFace.

        Args:
            test_category: BFCL category to evaluate (default: "simple")
            output_path: Path to save results (optional)
            limit: Limit number of examples to evaluate (optional)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Load dataset from HuggingFace
        print(f"Loading BFCL dataset from HuggingFace...")
        ds = load_dataset("llamastack/bfcl_v3")

        # Filter for specific test category
        print(f"Filtering for test_category='{test_category}'...")
        dataset = [ex for ex in ds["train"] if ex["test_category"] == test_category]

        if limit:
            dataset = dataset[:limit]

        print(f"Evaluating {len(dataset)} examples from category '{test_category}'...")

        results = []
        correct = 0
        total = 0
        bar = tqdm(dataset)
        for idx, example in enumerate(bar):
            try:
                # Parse JSON strings from dataset
                functions = json.loads(example["functions"])
                ground_truth_list = json.loads(example["ground_truth"])
                turns = json.loads(example["turns"])

                # Extract user query from turns (last user message in first turn)
                query = ""
                for turn in turns[0]:
                    if turn.get("role") == "user":
                        query = turn.get("content", "")

                # Convert ground truth format from BFCL to our format
                # BFCL format: [{"function_name": {"param": ["value1", "value2"]}}]
                # Our format: {"name": "function_name", "arguments": {"param": "value"}}
                ground_truth = self._parse_ground_truth(ground_truth_list)

                prediction, is_correct = self.evaluate_single(query, functions, ground_truth)

                results.append({
                    "id": example["id"],
                    "query": query,
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                    "correct": is_correct
                })
                bar.set_description_str(f"correct: {is_correct} correct/{total}")
                if is_correct:
                    correct += 1
                total += 1

            except Exception as e:
                print(f"Error evaluating example {example.get('id', idx)}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "id": example.get("id", idx),
                    "error": str(e)
                })

        # Compute metrics
        accuracy = correct / total if total > 0 else 0
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_rate": 1 - accuracy
        }

        print("\n=== Evaluation Results ===")
        print(f"Total examples: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

        # Save results if output path provided
        if output_path:
            output_data = {
                "metrics": metrics,
                "results": results
            }
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {output_path}")

        return metrics

    def _parse_ground_truth(self, ground_truth_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse BFCL ground truth format to our format.

        BFCL format: [{"function_name": {"param": ["value1", "value2"]}}]
        Our format: {"name": "function_name", "arguments": {"param": "value"}}

        Args:
            ground_truth_list: Ground truth in BFCL format

        Returns:
            Ground truth in our format
        """
        if not ground_truth_list or len(ground_truth_list) == 0:
            return {"name": "", "arguments": {}}

        # Get first function call
        first_call = ground_truth_list[0]

        # Extract function name and arguments
        func_name = list(first_call.keys())[0]
        params = first_call[func_name]

        # Convert parameter values (take first value from list)
        arguments = {}
        for param_name, values in params.items():
            if values and len(values) > 0:
                arguments[param_name] = values[0]

        return {
            "name": func_name,
            "arguments": arguments
        }


def load_bfcl_dataset(category: str = "simple") -> str:
    """
    Helper to construct BFCL dataset path.

    Args:
        category: BFCL category (simple, parallel, multiple, etc.)

    Returns:
        Path to dataset file
    """
    # Common BFCL dataset locations
    possible_paths = [
        f"data/bfcl/{category}.json",
        f"berkeley-function-call-leaderboard/data/{category}.json",
        f"../data/bfcl/{category}.json",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"BFCL dataset not found. Please download it from "
        f"https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard "
        f"and place it in one of: {possible_paths}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion LLM on Berkeley Function Call Leaderboard (HuggingFace dataset)"
    )
    parser.add_argument(
        "--test-category",
        type=str,
        default="simple",
        help="BFCL test category to evaluate (default: simple)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="bfcl_results.json",
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of diffusion generation steps"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing)"
    )

    args = parser.parse_args()

    print("=== Berkeley Function Call Leaderboard Evaluation ===\n")

    # Load model
    print("Loading diffusion LLM model...")
    model, tokenizer, device = load_diffusion_llm()

    # Create evaluator
    evaluator = BFCLEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        generation_steps=args.steps
    )

    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        test_category=args.test_category,
        output_path=args.output_path,
        limit=args.limit
    )

    print("\nEvaluation complete!")
    return metrics


if __name__ == "__main__":
    main()
