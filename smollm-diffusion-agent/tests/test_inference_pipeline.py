"""Tests for the inference pipeline (without trained weights)."""
import pytest
import torch
from transformers import AutoTokenizer

from data.schema import build_schema_template
from data.utils import resolve_mask_token, resolve_null_token
from data.smollm3_prompting import parse_first_tool_call


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


@pytest.fixture(scope="module")
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def mask_and_null(tokenizer):
    mask_str, mask_id = resolve_mask_token(tokenizer, None)
    null_str, null_id = resolve_null_token(tokenizer, None)
    return mask_str, mask_id, null_str, null_id


class TestSchemaTemplateBuilding:

    def test_simple_template(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, _ = mask_and_null
        
        fields = [
            ("location", 32),
            ("units", 16),
        ]
        
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        assert template is not None
        assert len(template.field_segments) == 2
        assert template.mask_token_id == mask_id
        
        # Check that mask tokens are in the template
        mask_count = (torch.tensor(template.tokens) == mask_id).sum().item()
        assert mask_count == 32 + 16, f"Expected 48 mask tokens, got {mask_count}"

    def test_template_text_is_valid_json_structure(self, tokenizer, mask_and_null):
        mask_str, _, null_str, _ = mask_and_null
        
        fields = [("city", 20)]
        
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        # Template text should have JSON-like structure
        assert "{" in template.text
        assert "}" in template.text
        assert '"city"' in template.text

    def test_template_field_positions(self, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, _ = mask_and_null
        
        fields = [("name", 10), ("age", 10)]
        
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        for segment in template.field_segments:
            assert len(segment.value_positions) > 0
            for pos in segment.value_positions:
                assert template.tokens[pos] == mask_id


class TestToolSelectionParsing:

    def test_parse_valid_tool_call(self):
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        result = parse_first_tool_call(text)
        
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"]["city"] == "Paris"

    def test_parse_tool_call_with_think_block(self):
        text = """<think>
Let me check the weather.
</think>
<tool_call>
{"name": "get_weather", "arguments": {"location": "London"}}
</tool_call>"""
        
        result = parse_first_tool_call(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_parse_no_tool_call(self):
        text = "I can help you with that! The answer is 42."
        result = parse_first_tool_call(text)
        assert result is None


class TestGeneratorInitialization:

    @pytest.mark.slow
    def test_generator_initializes(self, tokenizer, device, mask_and_null):
        from model.hybrid_model import HybridSmolLM
        from inference import FunctionCallGenerator
        
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        # Use 4-bit only on CUDA
        use_4bit = torch.cuda.is_available()
        
        model = HybridSmolLM(
            base_model_id="HuggingFaceTB/SmolLM3-3B",
            load_in_4bit=use_4bit,
            diffusion_config={
                "hidden_dim": 512,
                "num_layers": 1,
                "num_steps": 2,
            },
            vocab_size=len(tokenizer),
            use_unsloth=False,
        )
        model.diffusion_head.set_mask_token_id(mask_id)
        if null_id:
            model.diffusion_head.set_null_token_id(null_id)
        
        # Get actual device where model is loaded
        if not model.use_mlx:
            if device.type != "cpu":
                model = model.to(device)
        
        actual_device = next(model.parameters()).device
        
        generator = FunctionCallGenerator(
            model=model,
            tokenizer=tokenizer,
            device=actual_device,
            use_torch_compile=False,
            use_cuda_graph=False,
        )
        
        assert generator is not None
        assert generator.model is model

    @pytest.mark.slow
    def test_generator_generate_runs(self, tokenizer, device, mask_and_null):
        """Test that generate() runs without crashing (even with untrained weights)."""
        from model.hybrid_model import HybridSmolLM
        from inference import FunctionCallGenerator, GenerationConfig
        
        mask_str, mask_id, null_str, null_id = mask_and_null
        
        # Use 4-bit only on CUDA
        use_4bit = torch.cuda.is_available()
        
        model = HybridSmolLM(
            base_model_id="HuggingFaceTB/SmolLM3-3B",
            load_in_4bit=use_4bit,
            diffusion_config={
                "hidden_dim": 512,
                "num_layers": 1,
                "num_steps": 2,
            },
            vocab_size=len(tokenizer),
            use_unsloth=False,
        )
        model.diffusion_head.set_mask_token_id(mask_id)
        if null_id:
            model.diffusion_head.set_null_token_id(null_id)
        
        if not model.use_mlx:
            if device.type != "cpu":
                model = model.to(device)
        model.eval()
        
        actual_device = next(model.parameters()).device
        
        generator = FunctionCallGenerator(
            model=model,
            tokenizer=tokenizer,
            device=actual_device,
            use_torch_compile=False,
            use_cuda_graph=False,
        )
        
        fields = [("location", 16)]  # Smaller budget for faster test
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )
        
        tools = [
            {
                "name": "get_weather",
                "parameters": {
                    "properties": {"location": {"type": "string"}}
                }
            }
        ]
        
        config = GenerationConfig(
            steps=1,  # Just 1 step for quick test
            temperature=0.0,
            show_steps=False,
        )
        
        output = generator.generate(
            prompt="Tokyo weather?",  # Shorter prompt
            template=template,
            config=config,
            tool_name="get_weather",
            tools=tools,
        )
        
        assert output is not None
        assert output.steps_executed > 0
        assert len(output.text) > 0


class TestBuildFieldsFromSchema:

    def test_build_fields_simple(self, tokenizer):
        from data.budget_utils import build_fields_from_schema
        
        schema = {
            "name": "test_tool",
            "parameters": {
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                }
            }
        }
        
        fields = build_fields_from_schema(schema, tokenizer, min_budget=16, max_budget=48)
        
        assert len(fields) == 2
        field_names = [f[0] for f in fields]
        assert "query" in field_names
        assert "limit" in field_names
        
        for name, budget in fields:
            assert budget >= 16
            assert budget <= 48

    def test_build_fields_empty_schema(self, tokenizer):
        from data.budget_utils import build_fields_from_schema
        
        schema = {"name": "empty_tool", "parameters": {}}
        fields = build_fields_from_schema(schema, tokenizer, min_budget=16, max_budget=48)
        
        assert fields == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
