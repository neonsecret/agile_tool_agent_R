"""
Multi-dataset loader that combines multiple function calling datasets.

Supports weighted sampling from multiple sources with format unification.
"""
from typing import List, Dict, Any, Optional
from datasets import load_dataset, concatenate_datasets
import random
from .dataset_formats import get_adapter


class MultiDatasetConfig:
    """Configuration for a single dataset source."""

    def __init__(self, name: str, split: str = "train", weight: float = 1.0, limit: Optional[int] = None):
        self.name = name
        self.split = split
        self.weight = weight
        self.limit = limit


def load_and_unify_dataset(config: MultiDatasetConfig) -> List[Dict[str, Any]]:
    """
    Load a dataset and convert it to unified format.
    
    Args:
        config: Dataset configuration
    
    Returns:
        List of unified examples
    """
    print(f"Loading {config.name} (split={config.split})...")

    # Load dataset
    ds = load_dataset(config.name, split=config.split)

    # Apply limit if specified
    if config.limit:
        ds = ds.select(range(min(config.limit, len(ds))))

    print(f"  Loaded {len(ds)} examples")

    # Get adapter
    adapter = get_adapter(config.name)

    # Convert to unified format
    unified_examples = []
    failed = 0

    for ex in ds:
        try:
            unified = adapter.convert(ex)
            unified_examples.append(unified.to_dict())
        except Exception as e:
            failed += 1
            if failed <= 5:  # Only print first 5 failures
                print(f"  Warning: Failed to convert example: {e}")

    if failed > 0:
        print(f"  Failed to convert {failed}/{len(ds)} examples")

    print(f"  Successfully converted {len(unified_examples)} examples")

    return unified_examples


def create_multi_dataset(
        dataset_configs: List[MultiDatasetConfig],
        shuffle: bool = True,
        seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Create a combined dataset from multiple sources with weighted sampling.
    
    Args:
        dataset_configs: List of dataset configurations
        shuffle: Whether to shuffle the combined dataset
        seed: Random seed for reproducibility
    
    Returns:
        Combined list of examples
    """
    all_examples = []

    # Load each dataset
    for config in dataset_configs:
        examples = load_and_unify_dataset(config)

        # Apply weighting by duplicating/subsampling
        if config.weight != 1.0:
            n_samples = int(len(examples) * config.weight)
            if n_samples > len(examples):
                # Oversample: duplicate with replacement
                random.seed(seed)
                examples = random.choices(examples, k=n_samples)
            elif n_samples < len(examples):
                # Undersample: random selection
                random.seed(seed)
                examples = random.sample(examples, n_samples)
            print(f"  Applied weight {config.weight}: {len(examples)} examples")

        all_examples.extend(examples)

    print(f"\nTotal examples: {len(all_examples)}")

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        random.shuffle(all_examples)
        print(f"Shuffled combined dataset")

    return all_examples


def load_multi_dataset_from_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load multiple datasets from config.yaml.
    
    Config format:
        data:
          datasets:
            - name: "interstellarninja/hermes_reasoning_tool_use"
              split: "train"
              weight: 0.5
            - name: "Salesforce/xlam-function-calling-60k"
              split: "train"
              weight: 0.5
              limit: 10000  # Optional
    
    Or fallback to single dataset:
        data:
          dataset_name: "..."
    """
    data_cfg = config.get("data", {})

    global_limit = data_cfg.get("limit")

    # Check if multi-dataset config exists
    if "datasets" in data_cfg and isinstance(data_cfg["datasets"], list):
        dataset_configs = [
            MultiDatasetConfig(
                name=ds.get("name"),
                split=ds.get("split", "train"),
                weight=ds.get("weight", 1.0),
                limit=ds.get("limit")
            )
            for ds in data_cfg["datasets"]
        ]
        examples = create_multi_dataset(
            dataset_configs,
            shuffle=True,
            seed=config.get("training", {}).get("seed", 42),
        )
        if global_limit is not None:
            return examples[:global_limit]
        return examples

    # Fallback: single dataset (backwards compatible)
    elif "dataset_name" in data_cfg:
        dataset_name = data_cfg["dataset_name"]
        limit = data_cfg.get("limit")
        config_obj = MultiDatasetConfig(
            name=dataset_name,
            split="train",
            weight=1.0,
            limit=limit,
        )
        return load_and_unify_dataset(config_obj)

    else:
        raise ValueError("Config must specify either 'data.datasets' (list) or 'data.dataset_name' (string)")
