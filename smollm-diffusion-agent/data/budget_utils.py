"""
Budget calculation utilities for automatic field sizing.

Matches training's budget calculation logic to ensure consistency
between training and inference.
"""

from typing import List, Tuple, Dict, Any
from transformers import PreTrainedTokenizerBase


# Budget configuration - matches guide.md recommendations and training defaults
MIN_FIELD_BUDGET = 32  # Minimum tokens per field for consistency
DEFAULT_MAX_BUDGET = 48  # Default maximum (from config.yaml mask_budget)


def estimate_field_length(
    field_name: str,
    field_spec: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase
) -> int:
    """
    Estimate reasonable token length for a field based on its specification.
    
    Args:
        field_name: Name of the field
        field_spec: Field specification from schema (type, description, enum, etc.)
        tokenizer: Tokenizer for counting tokens
    
    Returns:
        Estimated number of tokens needed for typical values
    """
    field_type = field_spec.get("type", "string")
    enum_values = field_spec.get("enum", None)
    
    if enum_values:
        # For enums, use longest enum value
        max_enum_len = max(
            len(tokenizer.encode(str(v), add_special_tokens=False))
            for v in enum_values
        )
        return max_enum_len
    
    elif field_type in ["integer", "number"]:
        # Numbers are typically short (e.g., "123", "45.67")
        return 5
    
    elif field_type == "boolean":
        # Booleans: "true" or "false"
        return 2
    
    elif field_type == "array":
        # Arrays can vary - use conservative estimate
        return 25
    
    elif field_type == "object":
        # Objects can be complex - use larger estimate
        return 30
    
    elif field_type == "string":
        # Strings vary widely - check description for hints
        description = field_spec.get("description", "").lower()
        
        # Heuristics based on common patterns
        if any(kw in description for kw in ["id", "code", "key"]):
            return 10
        elif any(kw in description for kw in ["name", "title"]):
            return 15
        elif any(kw in description for kw in ["description", "details", "message"]):
            return 30
        elif any(kw in description for kw in ["city", "location", "address"]):
            return 20
        else:
            # Default string estimate
            return 20
    
    else:
        # Unknown type - use safe default
        return 20


def calculate_field_budget(
    estimated_length: int,
    min_budget: int = MIN_FIELD_BUDGET,
    max_budget: int = DEFAULT_MAX_BUDGET,
    extra_budget: int = 0
) -> int:
    """
    Calculate actual budget for a field.
    
    Matches training's calculation:
    budget = min(max(len(val_ids), MIN_FIELD_BUDGET), mask_budget)
    
    Args:
        estimated_length: Estimated token count for the field
        min_budget: Minimum budget (ensures consistency)
        max_budget: Maximum budget (prevents excessive memory)
    
    Returns:
        Budget size in tokens
    """
    budget = estimated_length + extra_budget
    return min(max(budget, min_budget), max_budget)


def build_fields_from_schema(
    schema: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    min_budget: int = MIN_FIELD_BUDGET,
    max_budget: int = DEFAULT_MAX_BUDGET,
    extra_budget: int = 0
) -> List[Tuple[str, int]]:
    """
    Automatically build fields list from tool schema with proper budgeting.
    
    This function:
    1. Extracts field definitions from schema
    2. Estimates token count for each field
    3. Applies budget calculation (matches training logic)
    
    Args:
        schema: Tool schema dict with 'parameters' -> 'properties'
        tokenizer: Tokenizer for estimating token counts
        min_budget: Minimum tokens per field (default: 32)
        max_budget: Maximum tokens per field (default: 48)
    
    Returns:
        List of (field_name, budget) tuples ready for schema template building
    
    Example:
        >>> schema = {
        ...     "name": "get_weather",
        ...     "parameters": {
        ...         "properties": {
        ...             "location": {"type": "string"},
        ...             "units": {"type": "string", "enum": ["C", "F"]}
        ...         }
        ...     }
        ... }
        >>> fields = build_fields_from_schema(schema, tokenizer)
        >>> # Returns: [("location", 32), ("units", 32)]
    """
    fields = []
    
    # Extract properties from schema (handle different formats)
    params = schema.get("parameters", {})
    if "properties" in params:
        props = params["properties"]
    else:
        props = params
    
    # Iterate through fields in order (preserves dict order in Python 3.7+)
    for field_name, field_spec in props.items():
        # Estimate length
        estimated_len = estimate_field_length(field_name, field_spec, tokenizer)
        
        # Calculate budget (matches training)
        budget = calculate_field_budget(
            estimated_len,
            min_budget=min_budget,
            max_budget=max_budget,
            extra_budget=extra_budget,
        )
        
        fields.append((field_name, budget))
    
    return fields


def print_budget_info(fields: List[Tuple[str, int]]) -> None:
    """Print human-readable budget information for debugging."""
    print("Field budgets:")
    total_budget = 0
    for field_name, budget in fields:
        print(f"  {field_name}: {budget} tokens")
        total_budget += budget
    print(f"Total budget: {total_budget} tokens")
