"""Test that base SmolLM3 generates tool calls with our dataset format.

This is critical to verify before training - if the base model doesn't naturally
generate tool calls with our prompt format, the training won't work.
"""
import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from data.dataset_loader import SmartScaffoldDataset
from data.smollm3_prompting import apply_smollm3_chat_template, parse_first_tool_call
from data.utils import resolve_mask_token, resolve_null_token


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def base_model():
    """Load base model once for all tests."""
    # Use float16 for MPS (Mac), bfloat16 for CUDA
    if torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
    elif torch.cuda.is_available():
        dtype = torch.bfloat16
        device = "cuda"
    else:
        dtype = torch.float32
        device = "cpu"
    
    print(f"\nLoading SmolLM3-3B on {device} with {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM3-3B",
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def dataset_examples(tokenizer):
    """Load a few real examples from the dataset."""
    mask_str, _ = resolve_mask_token(tokenizer, None)
    null_str, _ = resolve_null_token(tokenizer, None)
    
    dataset = SmartScaffoldDataset(
        tokenizer=tokenizer,
        split="train",
        max_seq_len=1024,
        limit=50,  # Load more to find good tool examples
        mask_token=mask_str,
        null_token=null_str,
    )
    
    # Get the raw processed examples (before tokenization)
    # We want the original messages + tools to reconstruct clean prompts
    return dataset.processed_examples


class TestBaseModelToolGeneration:
    """Test that SmolLM3 generates tool calls with our format."""

    def test_model_generates_tool_call_with_dataset_format(self, tokenizer, base_model, dataset_examples):
        """
        Critical test: Take a real tool-calling example from our dataset,
        reconstruct just the prompt (without the tool call answer),
        and verify the base model generates a tool call.
        """
        # Find a tool example
        tool_example = None
        for ex in dataset_examples:
            if ex["router_label"] == 1 and ex["tools_schema"]:
                tool_example = ex
                break
        
        if tool_example is None:
            pytest.skip("No tool examples found in dataset")
        
        messages = tool_example["messages"]
        tools = tool_example["tools_schema"]
        tool_name = tool_example["tool_name"]
        
        print(f"\n{'='*80}")
        print(f"Testing tool: {tool_name}")
        print(f"Messages: {messages[-1]['content'][:100]}...")  # Last user message
        print(f"Available tools: {[t.get('name', 'unnamed') for t in tools]}")
        
        # Encode prompt with tools (same as training)
        prompt_ids = apply_smollm3_chat_template(
            tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )
        
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        print(f"\nPrompt length: {len(prompt_ids)} tokens")
        print(f"Prompt preview (last 200 chars):\n...{prompt_text[-200:]}")
        
        # Generate with base model
        input_ids = torch.tensor([prompt_ids], device=base_model.device)
        
        with torch.no_grad():
            output = base_model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=False,  # Greedy for deterministic output
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_ids = output[0, len(prompt_ids):]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        print(f"\n{'='*80}")
        print(f"GENERATED OUTPUT:")
        print(generated_text)
        print(f"{'='*80}\n")
        
        # Try to parse tool call
        parsed = parse_first_tool_call(generated_text)
        
        # Assertions
        assert "<tool_call>" in generated_text, \
            f"Model did not generate <tool_call> tag. Generated: {generated_text[:200]}"
        
        assert "</tool_call>" in generated_text, \
            f"Model did not close <tool_call> tag. Generated: {generated_text[:200]}"
        
        assert parsed is not None, \
            f"Tool call did not parse as valid JSON. Generated: {generated_text[:300]}"
        
        assert "name" in parsed, \
            f"Parsed tool call missing 'name' field. Parsed: {parsed}"
        
        assert "arguments" in parsed, \
            f"Parsed tool call missing 'arguments' field. Parsed: {parsed}"
        
        print(f"✅ Model generated valid tool call!")
        print(f"   Tool name: {parsed['name']}")
        print(f"   Arguments: {parsed['arguments']}")
        
        # Check if it selected the right tool (soft check - might select different tool)
        available_tool_names = [t.get('name', '') for t in tools]
        assert parsed['name'] in available_tool_names, \
            f"Model selected tool '{parsed['name']}' not in available tools: {available_tool_names}"

    def test_model_generates_multiple_tool_calls(self, tokenizer, base_model, dataset_examples):
        """Test on multiple examples to ensure consistency."""
        
        tool_examples = [ex for ex in dataset_examples if ex["router_label"] == 1 and ex["tools_schema"]][:3]
        
        if len(tool_examples) < 2:
            pytest.skip("Not enough tool examples in dataset")
        
        success_count = 0
        total_count = 0
        
        for i, ex in enumerate(tool_examples):
            total_count += 1
            messages = ex["messages"]
            tools = ex["tools_schema"]
            
            prompt_ids = apply_smollm3_chat_template(tokenizer, messages=messages, tools=tools, add_generation_prompt=True)
            input_ids = torch.tensor([prompt_ids], device=base_model.device)
            
            with torch.no_grad():
                output = base_model.generate(
                    input_ids,
                    max_new_tokens=100,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            generated_text = tokenizer.decode(output[0, len(prompt_ids):], skip_special_tokens=False)
            parsed = parse_first_tool_call(generated_text)
            
            if parsed and "name" in parsed and "arguments" in parsed:
                success_count += 1
                print(f"  Example {i+1}: ✅ Valid tool call: {parsed['name']}")
            else:
                print(f"  Example {i+1}: ❌ Failed to generate valid tool call")
                print(f"     Generated: {generated_text[:150]}")
        
        success_rate = success_count / total_count
        print(f"\nSuccess rate: {success_count}/{total_count} ({success_rate:.1%})")
        
        # We expect at least 50% success rate (base model might not always use tools)
        assert success_rate >= 0.5, \
            f"Base model only generated valid tool calls {success_rate:.1%} of the time"

    def test_tool_schema_affects_generation(self, tokenizer, base_model):
        """Verify that including tools in the prompt actually changes behavior."""
        
        messages = [
            {"role": "system", "content": "/no_think"},
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ]
        
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    }
                }
            }
        ]
        
        # Generate WITH tools
        prompt_with_tools = apply_smollm3_chat_template(tokenizer, messages, tools=tools, add_generation_prompt=True)
        input_ids_with = torch.tensor([prompt_with_tools], device=base_model.device)
        
        with torch.no_grad():
            output_with = base_model.generate(
                input_ids_with,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_with = tokenizer.decode(output_with[0, len(prompt_with_tools):], skip_special_tokens=False)
        
        # Generate WITHOUT tools
        prompt_without_tools = apply_smollm3_chat_template(tokenizer, messages, tools=None, add_generation_prompt=True)
        input_ids_without = torch.tensor([prompt_without_tools], device=base_model.device)
        
        with torch.no_grad():
            output_without = base_model.generate(
                input_ids_without,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        generated_without = tokenizer.decode(output_without[0, len(prompt_without_tools):], skip_special_tokens=False)
        
        print(f"\n{'='*80}")
        print(f"WITH TOOLS:\n{generated_with}")
        print(f"\n{'='*80}")
        print(f"WITHOUT TOOLS:\n{generated_without}")
        print(f"{'='*80}\n")
        
        # The outputs should be different
        assert generated_with != generated_without, \
            "Tool injection had no effect on model output"
        
        # With tools, more likely to contain tool call
        has_tool_call_with = "<tool_call>" in generated_with
        has_tool_call_without = "<tool_call>" in generated_without
        
        print(f"With tools: {'✅ Generated tool call' if has_tool_call_with else '❌ No tool call'}")
        print(f"Without tools: {'✅ Generated tool call' if has_tool_call_without else '❌ No tool call'}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
