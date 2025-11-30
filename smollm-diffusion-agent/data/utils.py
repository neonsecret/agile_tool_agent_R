from transformers import PreTrainedTokenizerBase


def resolve_mask_token(tokenizer: PreTrainedTokenizerBase, mask_token_config=None):
    """
    Centralized mask token resolution with consistent fallback logic.
    
    Uses <|reserved_special_token_2|> as default (matches original design intent).
    This avoids ambiguity with EOS tokens that appear naturally in prompts.
    
    Args:
        tokenizer: Tokenizer instance
        mask_token_config: Config value (None, string, or "eos")
            - None: Use <|reserved_special_token_2|> (default, recommended)
            - "eos": Use EOS token (not recommended due to ambiguity)
            - string: Use specified token string
        
    Returns:
        tuple: (mask_token_string, mask_token_id)
    """
    if mask_token_config is None:
        mask_token_str = "<|reserved_special_token_2|>"
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token_str)
        if mask_token_id is None:
            mask_token_str = tokenizer.eos_token
            mask_token_id = tokenizer.eos_token_id
    elif mask_token_config == "eos":
        mask_token_str = tokenizer.eos_token
        mask_token_id = tokenizer.eos_token_id
    else:
        mask_token_str = mask_token_config
        mask_token_id = tokenizer.convert_tokens_to_ids(mask_token_str)
        if mask_token_id is None:
            mask_token_str = "<|reserved_special_token_2|>"
            mask_token_id = tokenizer.convert_tokens_to_ids(mask_token_str)
            if mask_token_id is None:
                mask_token_str = tokenizer.eos_token
                mask_token_id = tokenizer.eos_token_id
    
    if tokenizer.mask_token is None:
        tokenizer.mask_token = mask_token_str
    
    return mask_token_str, mask_token_id


def validate_mask_token_consistency(model_mask_token_id, template_mask_token_id, context=""):
    """
    Validate that mask token IDs match between model and template.
    
    Args:
        model_mask_token_id: Mask token ID set in the model
        template_mask_token_id: Mask token ID from the template
        context: Optional context string for error message
        
    Raises:
        ValueError: If mask token IDs don't match
    """
    if model_mask_token_id != template_mask_token_id:
        raise ValueError(
            f"Mask token ID mismatch{context}: "
            f"model has {model_mask_token_id}, template has {template_mask_token_id}. "
            "Ensure both use the same mask token configuration."
        )

