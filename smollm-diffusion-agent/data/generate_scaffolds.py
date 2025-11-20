import json
import torch
from transformers import AutoTokenizer


class ScaffoldGenerator:
    def __init__(self, tokenizer_id="HuggingFaceTB/SmolLM3-3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        # Ensure we have a mask token
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '<MASK>'})

    def create_scaffold_example(self, query, function_schema, target_args):
        """
        query: "Weather in London"
        function_schema: {"name": "weather", "parameters": {"loc": "string"}}
        target_args: {"loc": "London"}
        """

        # 1. Construct the prompt (AR part)
        # This is simplified. Real training needs chat template.
        prompt_text = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<|decision:use_tool|>\n<|tool_name:{function_schema['name']}|>"

        # 2. Construct the Scaffold (Python Template)
        # We iterate over schema keys to build the deterministic part
        # And insert <MASK> for values

        # Example: {"loc": "<MASK>"}
        scaffold_structure = {}
        for key in function_schema['parameters']:
            scaffold_structure[key] = "<MASK>"

        scaffold_str = json.dumps(scaffold_structure)

        # 3. Construct the Target (Ground Truth)
        # Example: {"loc": "London"}
        target_str = json.dumps(target_args)

        # 4. Tokenization alignment is tricky here.
        # We need the scaffold tokens to align with the target tokens except at mask positions.
        # A simple strategy:
        # Tokenize scaffold_str -> find IDs of <MASK> -> these are mask positions
        # Tokenize target_str -> these are the labels

        # Note: This simple strategy assumes length matches, which is NOT true if <MASK> is 1 token and "London" is 1 token.
        # But "Los Angeles" is 3 tokens.
        # So we need DYNAMIC masking length or Padding.
        # The markdown suggested: "location": "<MASK>" + " <NULL>" * (max_len - 1)

        # For this MVP, let's assume 1 mask token per value token for simplicity, 
        # or just show the structure.

        input_text = prompt_text + "\n" + scaffold_str
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")

        # Create mask
        # We find the token ID of <MASK>
        mask_token_id = self.tokenizer.mask_token_id
        scaffold_mask = (input_ids == mask_token_id)

        # Labels need to be the same length as input_ids
        # For the mask positions, we put the target token IDs.
        # For non-mask positions, we put -100 (ignore).

        labels = torch.full_like(input_ids, -100)

        # NAIVE FILLING for MVP:
        # In reality, we need a robust aligner that knows "London" corresponds to the 5th mask.

        return {
            "input_ids": input_ids,
            "scaffold_mask": scaffold_mask,
            "labels": labels  # Placeholder labels
        }


if __name__ == "__main__":
    generator = ScaffoldGenerator()
    schema = {"name": "weather", "parameters": {"loc": "string"}}
    example = generator.create_scaffold_example(
        "Weather in London",
        schema,
        {"loc": "London"}
    )
    print("Scaffold Mask Shape:", example['scaffold_mask'].shape)
    print("Has Masks:", example['scaffold_mask'].any())
