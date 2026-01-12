"""Tests for SmolLM3 prompting helpers."""
import json
import pytest
from transformers import AutoTokenizer

from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    encode_tool_call_wrapper,
    parse_first_tool_call,
)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")


class TestParseFirstToolCall:

    def test_valid_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "London"}}\n</tool_call>'
        result = parse_first_tool_call(text)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"] == {"city": "London"}

    def test_tool_call_with_surrounding_text(self):
        text = 'Some preamble <tool_call>\n{"name": "search", "arguments": {"q": "test"}}\n</tool_call> and more'
        result = parse_first_tool_call(text)
        assert result is not None
        assert result["name"] == "search"

    def test_no_tool_call(self):
        assert parse_first_tool_call("Hello, how can I help?") is None

    def test_unclosed_tool_call(self):
        assert parse_first_tool_call('<tool_call>\n{"name": "test"}') is None

    def test_empty_tool_call(self):
        assert parse_first_tool_call("<tool_call></tool_call>") is None

    def test_invalid_json(self):
        assert parse_first_tool_call("<tool_call>not json</tool_call>") is None

    def test_multiple_tool_calls_returns_first(self):
        text = (
            '<tool_call>\n{"name": "first", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "second", "arguments": {}}\n</tool_call>'
        )
        result = parse_first_tool_call(text)
        assert result["name"] == "first"


class TestEncodeToolCallWrapper:

    def test_wrapper_structure(self, tokenizer):
        parts = encode_tool_call_wrapper(tokenizer, "get_weather")
        
        prefix_str = tokenizer.decode(parts.prefix_ids, skip_special_tokens=False)
        suffix_str = tokenizer.decode(parts.suffix_ids, skip_special_tokens=False)
        
        assert "<tool_call>" in prefix_str
        assert '"name": "get_weather"' in prefix_str
        assert '"arguments":' in prefix_str
        assert "</tool_call>" in suffix_str
        assert "}" in suffix_str

    def test_wrapper_different_tool_names(self, tokenizer):
        parts1 = encode_tool_call_wrapper(tokenizer, "search")
        parts2 = encode_tool_call_wrapper(tokenizer, "calculate")
        
        prefix1 = tokenizer.decode(parts1.prefix_ids, skip_special_tokens=False)
        prefix2 = tokenizer.decode(parts2.prefix_ids, skip_special_tokens=False)
        
        assert '"name": "search"' in prefix1
        assert '"name": "calculate"' in prefix2

    def test_wrapper_with_special_chars_in_name(self, tokenizer):
        parts = encode_tool_call_wrapper(tokenizer, "get_weather_v2")
        prefix = tokenizer.decode(parts.prefix_ids, skip_special_tokens=False)
        assert '"name": "get_weather_v2"' in prefix


class TestApplySmolLM3ChatTemplate:

    def test_basic_messages(self, tokenizer):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        ids = apply_smollm3_chat_template(tokenizer, messages, add_generation_prompt=True)
        
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_decodes_correctly(self, tokenizer):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's 2+2?"},
        ]
        ids = apply_smollm3_chat_template(tokenizer, messages, add_generation_prompt=True)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        assert "system" in text.lower() or "<|im_start|>" in text
        assert "2+2" in text

    def test_with_tools(self, tokenizer):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's the weather?"},
        ]
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools, add_generation_prompt=True)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        # Should contain tool schema
        assert "get_weather" in text
        assert "location" in text

    def test_no_generation_prompt(self, tokenizer):
        messages = [{"role": "user", "content": "Hi"}]
        ids_with = apply_smollm3_chat_template(tokenizer, messages, add_generation_prompt=True)
        ids_without = apply_smollm3_chat_template(tokenizer, messages, add_generation_prompt=False)
        
        # With generation prompt should be longer
        assert len(ids_with) >= len(ids_without)


class TestEndToEndToolCallFormat:

    def test_complete_tool_call_roundtrip(self, tokenizer):
        """Test that we can encode a prompt, add a tool call wrapper, and parse it back."""
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Get weather in Paris"},
        ]
        
        prompt_ids = apply_smollm3_chat_template(tokenizer, messages, add_generation_prompt=True)
        parts = encode_tool_call_wrapper(tokenizer, "get_weather")
        
        # Simulate argument scaffold (just the structure)
        args_json = '{"location": "Paris"}'
        args_ids = tokenizer.encode(args_json, add_special_tokens=False)
        
        full_ids = prompt_ids + parts.prefix_ids + args_ids + parts.suffix_ids
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        
        # Parse the tool call back
        parsed = parse_first_tool_call(full_text)
        assert parsed is not None
        assert parsed["name"] == "get_weather"
        assert parsed["arguments"]["location"] == "Paris"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
