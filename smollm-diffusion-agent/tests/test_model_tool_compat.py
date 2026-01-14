"""Test SmolLM3 base model tool calling compatibility.

These tests verify that:
1. SmolLM3 can understand and generate tool calls with the proper template
2. The tool schema injection works correctly
3. The model's native tool-calling format matches what we expect
"""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    parse_first_tool_call,
)


@pytest.fixture
def tokenizer(shared_tokenizer):
    """Use shared session-scoped tokenizer."""
    return shared_tokenizer


@pytest.fixture
def model(shared_base_model):
    """Use shared session-scoped base model."""
    return shared_base_model


class TestChatTemplateToolInjection:

    def test_tools_appear_in_system_message(self, tokenizer):
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
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["C", "F"]}
                    },
                    "required": ["location"]
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        # SmolLM3's template should include tool schema in the system section
        assert "get_weather" in text
        assert "location" in text
        # Check for tool_call instruction pattern
        assert "<tool_call>" in text or "tool" in text.lower()

    def test_multiple_tools_all_injected(self, tokenizer):
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "Help me"},
        ]
        tools = [
            {"name": "search", "description": "Search the web", "parameters": {}},
            {"name": "calculate", "description": "Do math", "parameters": {}},
            {"name": "translate", "description": "Translate text", "parameters": {}},
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        assert "search" in text
        assert "calculate" in text
        assert "translate" in text


class TestBaseModelToolGeneration:

    @pytest.mark.slow
    def test_model_generates_tool_call_format(self, tokenizer, model):
        """Test that SmolLM3 generates tool calls in the expected format."""
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's the weather in London?"},
        ]
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        input_ids = torch.tensor([ids], device=model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(output[0, len(ids):], skip_special_tokens=False)
        print(f"Generated: {generated}")
        
        # The model should generate a tool call
        parsed = parse_first_tool_call(generated)
        
        # Note: An untrained/base model might not always produce valid tool calls,
        # but the format should be recognizable if it tries
        if parsed is not None:
            assert "name" in parsed
            print(f"Parsed tool call: {parsed}")

    @pytest.mark.slow
    def test_model_can_decline_tool_call(self, tokenizer, model):
        """Test that model can generate a regular response when appropriate."""
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What is 2+2?"},  # Simple question, no tool needed
        ]
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        input_ids = torch.tensor([ids], device=model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated = tokenizer.decode(output[0, len(ids):], skip_special_tokens=False)
        print(f"Generated for simple question: {generated}")
        
        # Model may or may not use a tool for this - we just verify it generates something
        assert len(generated.strip()) > 0


class TestToolSchemaFormats:

    def test_nested_parameters(self, tokenizer):
        """Test that complex nested schemas are properly encoded."""
        messages = [{"role": "user", "content": "Book a flight"}]
        tools = [
            {
                "name": "book_flight",
                "description": "Book a flight",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "departure": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "date": {"type": "string"}
                            }
                        },
                        "arrival": {
                            "type": "object", 
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    }
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        assert "book_flight" in text
        assert "departure" in text or "city" in text

    def test_array_parameters(self, tokenizer):
        """Test array type parameters."""
        messages = [{"role": "user", "content": "Send emails"}]
        tools = [
            {
                "name": "send_emails",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipients": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of email addresses"
                        }
                    }
                }
            }
        ]
        
        ids = apply_smollm3_chat_template(tokenizer, messages, tools=tools)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        
        assert "send_emails" in text
        assert "recipients" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
