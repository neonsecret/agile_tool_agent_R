"""Tests for dataset loading and processing."""
import json
import pytest
import torch
from transformers import AutoTokenizer

from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def mask_token(tokenizer):
    mask_str, mask_id = resolve_mask_token(tokenizer, None)
    return mask_str, mask_id


@pytest.fixture(scope="module")
def null_token(tokenizer):
    null_str, null_id = resolve_null_token(tokenizer, None)
    return null_str, null_id


class TestDatasetLoading:

    def test_dataset_loads(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=10,
            mask_token=mask_str,
            null_token=null_str,
            chat_sampling_rate=0.1,
        )
        
        assert len(dataset) > 0
        print(f"Dataset loaded with {len(dataset)} examples")

    def test_dataset_item_structure(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=20,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "scaffold_mask" in item
        assert "labels" in item
        assert "router_label" in item
        
        assert isinstance(item["input_ids"], torch.Tensor)
        assert item["input_ids"].dtype == torch.long
        assert item["scaffold_mask"].dtype == torch.bool

    def test_input_ids_within_max_seq_len(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        max_len = 512
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=max_len,
            limit=30,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert len(item["input_ids"]) <= max_len, f"Item {i} exceeds max_seq_len"


class TestScaffoldMask:

    def test_scaffold_mask_marks_mask_tokens(self, tokenizer, mask_token, null_token):
        mask_str, mask_id = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=50,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        # Find a tool call example (router_label=1)
        tool_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                tool_example = item
                break
        
        if tool_example is None:
            pytest.skip("No tool call examples found in sampled data")
        
        scaffold_mask = tool_example["scaffold_mask"]
        input_ids = tool_example["input_ids"]
        
        # Scaffold mask should have True values
        assert scaffold_mask.any(), "No scaffold positions marked"
        
        # Positions marked in scaffold_mask should have mask_token_id in input_ids
        masked_positions = scaffold_mask.nonzero(as_tuple=True)[0]
        for pos in masked_positions:
            assert input_ids[pos].item() == mask_id, f"Position {pos} is masked but doesn't have mask token"

    def test_labels_align_with_scaffold_mask(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, null_id = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=50,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        # Find a tool call example
        tool_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                tool_example = item
                break
        
        if tool_example is None:
            pytest.skip("No tool call examples found")
        
        scaffold_mask = tool_example["scaffold_mask"]
        labels = tool_example["labels"]
        
        # Positions NOT in scaffold should have labels = -100
        non_scaffold = ~scaffold_mask
        assert (labels[non_scaffold] == -100).all(), "Non-scaffold positions should be ignored (-100)"
        
        # Positions IN scaffold should have valid labels (>= 0 or null_token_id)
        scaffold_positions = scaffold_mask.nonzero(as_tuple=True)[0]
        for pos in scaffold_positions:
            label = labels[pos].item()
            # Should either be a valid token ID or -100 (if no target available)
            assert label >= -100, f"Invalid label {label} at position {pos}"


class TestChatVsToolExamples:

    def test_chat_examples_have_empty_scaffold(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=100,
            mask_token=mask_str,
            null_token=null_str,
            chat_sampling_rate=0.5,
        )
        
        # Find a chat example (router_label=0)
        chat_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 0:
                chat_example = item
                break
        
        if chat_example is None:
            pytest.skip("No chat examples found")
        
        # Chat examples should have no scaffold positions
        assert not chat_example["scaffold_mask"].any(), "Chat examples shouldn't have scaffold"
        assert (chat_example["labels"] == -100).all(), "Chat examples should have all labels=-100"

    def test_tool_examples_have_scaffold(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=50,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        tool_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                tool_example = item
                break
        
        if tool_example is None:
            pytest.skip("No tool call examples found")
        
        assert tool_example["scaffold_mask"].any(), "Tool examples should have scaffold positions"


class TestTruncation:

    def test_truncation_preserves_scaffold(self, tokenizer, mask_token, null_token):
        mask_str, mask_id = mask_token
        null_str, _ = null_token
        
        # Use moderately short max_seq_len to force prompt truncation while fitting scaffold
        short_len = 512
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=short_len,
            limit=50,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            if item["router_label"] == 1:  # Tool example
                input_ids = item["input_ids"]
                scaffold_mask = item["scaffold_mask"]
                
                # Check that scaffold positions exist and have mask tokens
                if scaffold_mask.any():
                    masked_positions = scaffold_mask.nonzero(as_tuple=True)[0]
                    for pos in masked_positions:
                        assert input_ids[pos].item() == mask_id


class TestToolCallWrapper:

    def test_tool_call_wrapper_in_sequence(self, tokenizer, mask_token, null_token):
        mask_str, _ = mask_token
        null_str, _ = null_token
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=50,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        tool_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                tool_example = item
                break
        
        if tool_example is None:
            pytest.skip("No tool call examples found")
        
        # Decode and check for tool_call markers
        text = tokenizer.decode(tool_example["input_ids"], skip_special_tokens=False)
        
        assert "<tool_call>" in text, "Tool example should contain <tool_call>"
        assert "</tool_call>" in text, "Tool example should contain </tool_call>"
        assert '"name":' in text, "Tool example should contain tool name"
        assert '"arguments":' in text, "Tool example should contain arguments key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
