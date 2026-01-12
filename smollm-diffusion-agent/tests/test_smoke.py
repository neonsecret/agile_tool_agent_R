"""Quick smoke tests to verify basic functionality before training.

These tests are designed to be FAST - they don't load full models or run inference.
They just verify that the data pipeline, components, and formatting work correctly.
"""
import pytest
import torch
from transformers import AutoTokenizer

from data.dataset_loader import SmartScaffoldDataset
from data.schema import build_schema_template
from data.utils import resolve_mask_token, resolve_null_token
from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    encode_tool_call_wrapper,
    parse_first_tool_call,
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def mask_and_null(tokenizer):
    mask_str, mask_id = resolve_mask_token(tokenizer, None)
    null_str, null_id = resolve_null_token(tokenizer, None)
    return mask_str, mask_id, null_str, null_id


class TestDataPipeline:
    """Test that data loads and formats correctly."""

    def test_dataset_creates_valid_examples(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=10,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        # Get a tool example
        tool_example = None
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                tool_example = item
                break
        
        assert tool_example is not None, "No tool examples found"
        
        # Verify structure
        assert tool_example["input_ids"].shape[0] <= 1024
        assert tool_example["scaffold_mask"].any()
        
        # Verify mask tokens are in the right places
        masked_positions = tool_example["scaffold_mask"].nonzero(as_tuple=True)[0]
        for pos in masked_positions:
            assert tool_example["input_ids"][pos].item() == mask_id
        
        # Verify labels are correct
        for pos in masked_positions:
            label = tool_example["labels"][pos].item()
            # Should be a valid token ID or null_token_id
            assert label >= -100
            if label >= 0:
                assert label < len(tokenizer)

    def test_tool_call_format_in_dataset(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        dataset = SmartScaffoldDataset(
            tokenizer=tokenizer,
            split="train",
            max_seq_len=1024,
            limit=20,
            mask_token=mask_str,
            null_token=null_str,
        )
        
        for i in range(len(dataset)):
            item = dataset[i]
            if item["router_label"] == 1:
                text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)
                
                # Should contain SmolLM3's tool call format
                assert "<tool_call>" in text
                assert "</tool_call>" in text
                assert '"name":' in text
                assert '"arguments":' in text
                break


class TestSmolLM3Compatibility:
    """Test that our formatting matches SmolLM3's expectations."""

    def test_tool_schema_injection(self, tokenizer):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's the weather?"},
        ]
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "properties": {"location": {"type": "string"}}
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        # Verify tool appears in prompt
        assert "get_weather" in text
        assert "location" in text

    def test_tool_call_wrapper_parseable(self, tokenizer):
        tool_name = "get_weather"
        parts = encode_tool_call_wrapper(tokenizer, tool_name)
        
        # Simulate filled arguments
        args_text = '{"location": "London"}'
        args_ids = tokenizer.encode(args_text, add_special_tokens=False)
        
        full_ids = parts.prefix_ids + args_ids + parts.suffix_ids
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        
        # Should parse correctly
        parsed = parse_first_tool_call(full_text)
        assert parsed is not None
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"]["location"] == "London"


class TestModelComponents:
    """Test that model components initialize and forward correctly."""

    def test_router_head(self, tokenizer):
        from model.hybrid_model import RouterHead
        
        device = torch.device("cpu")  # Use CPU for speed
        hidden_size = 3072
        router = RouterHead(hidden_size, num_classes=3).to(device)
        
        hidden_states = torch.randn(2, 10, hidden_size)
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        
        logits = router(hidden_states, attention_mask)
        
        assert logits.shape == (2, 3)
        assert not torch.isnan(logits).any()

    def test_diffusion_head(self, tokenizer, mask_and_null):
        from model.diffusion_head import SchemaDiffusionHead
        
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        device = torch.device("cpu")
        head = SchemaDiffusionHead(
            input_dim=3072,
            vocab_size=len(tokenizer),
            hidden_dim=512,
            num_layers=1,
            num_steps=2,
        ).to(device)
        head.set_mask_token_id(mask_id)
        
        hidden_states = torch.randn(2, 32, 3072)
        current_tokens = torch.randint(0, len(tokenizer), (2, 32))
        t = torch.rand(2)
        
        logits = head.predict(hidden_states, current_tokens, t)
        
        assert logits.shape == (2, 32, len(tokenizer))
        assert not torch.isnan(logits).any()


class TestSchemaScaffolding:
    """Test that schema scaffolding works correctly."""

    def test_schema_template_has_correct_masks(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        fields = [("city", 20), ("units", 10)]
        
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        # Should have 30 mask tokens (20 + 10)
        mask_count = sum(1 for t in template.tokens if t == mask_id)
        assert mask_count == 30
        
        # Should have 2 field segments
        assert len(template.field_segments) == 2
        assert template.field_segments[0].name == "city"
        assert template.field_segments[1].name == "units"

    def test_schema_template_text_structure(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        fields = [("location", 16)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        # Should be valid JSON structure
        assert "{" in template.text
        assert "}" in template.text
        assert '"location"' in template.text


class TestEndToEndFormat:
    """Test that the complete format (prompt + scaffold + wrapper) works."""

    def test_complete_format(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        # 1. Create prompt with tools
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Weather in Paris?"},
        ]
        tools = [{"name": "get_weather", "parameters": {"properties": {"location": {"type": "string"}}}}]
        
        prompt_ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        
        # 2. Create tool call wrapper
        parts = encode_tool_call_wrapper(tokenizer, "get_weather")
        
        # 3. Create scaffold
        fields = [("location", 16)]
        template = build_schema_template(tokenizer, fields, mask_str, null_str, include_codeblock=False)
        
        # 4. Combine
        full_ids = list(prompt_ids) + list(parts.prefix_ids) + list(template.tokens) + list(parts.suffix_ids)
        
        # Verify it's under max_seq_len
        assert len(full_ids) <= 1024
        
        # Decode and verify structure
        text = tokenizer.decode(full_ids, skip_special_tokens=False)
        assert "<tool_call>" in text
        assert "</tool_call>" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
