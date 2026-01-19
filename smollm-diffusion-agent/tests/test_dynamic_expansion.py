"""Tests for dynamic budget expansion during inference."""
import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from data.schema import build_schema_template, SchemaTemplate
from data.utils import resolve_mask_token, resolve_null_token
from inference import FunctionCallGenerator, GenerationState, GenerationOutput


@pytest.fixture
def tokenizer(shared_tokenizer):
    return shared_tokenizer


@pytest.fixture
def mask_and_null(shared_tokenizer):
    mask_str, mask_id = resolve_mask_token(shared_tokenizer, None)
    null_str, null_id = resolve_null_token(shared_tokenizer, None)
    return mask_str, mask_id, null_str, null_id


@pytest.fixture
def mock_model():
    model = Mock()
    model.eval = Mock()
    model.diffusion_head = Mock()
    model.diffusion_head.mask_token_id = 32002
    model.diffusion_head.null_token_id = 32011
    return model


@pytest.fixture
def generator(mock_model, tokenizer):
    device = torch.device("cpu")
    gen = FunctionCallGenerator(
        model=mock_model,
        tokenizer=tokenizer,
        device=device,
        use_torch_compile=False,
        use_cuda_graph=False,
        max_seq_len=2048,
    )
    gen.expansion_config = {
        "enabled": True,
        "max_rounds": 2,
        "expand_tokens": 4,
        "tail_window": 4,
        "tail_null_threshold": 0.5,
    }
    return gen


class TestOverflowDetection:

    def test_detect_overflow_no_null_token(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, _, _ = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=None,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 20), dtype=torch.long)
        state = GenerationState(
            sequence=sequence,
            prompt_length=10,
            prefix_length=2,
            template=template,
        )

        overflow_fields = generator._detect_overflow_fields(state)
        assert overflow_fields == [], "Should detect no overflow without NULL token"

    def test_detect_overflow_field_full(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 50), dtype=torch.long)
        prompt_length = 10
        prefix_length = 2

        segment = template.field_segments[0]
        tail_window = generator.expansion_config.get("tail_window", 4)
        for pos_idx in range(len(segment.value_positions)):
            abs_pos = prompt_length + prefix_length + segment.value_positions[pos_idx]
            if pos_idx < len(segment.value_positions) - 1:
                sequence[0, abs_pos] = 100
            else:
                sequence[0, abs_pos] = null_id

        state = GenerationState(
            sequence=sequence,
            prompt_length=prompt_length,
            prefix_length=prefix_length,
            template=template,
        )

        overflow_fields = generator._detect_overflow_fields(state)
        assert "location" in overflow_fields, "Should detect overflow when tail has few NULLs"

    def test_detect_overflow_field_partial(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 50), dtype=torch.long)
        prompt_length = 10
        prefix_length = 2

        segment = template.field_segments[0]
        for pos_idx in range(len(segment.value_positions)):
            abs_pos = prompt_length + prefix_length + segment.value_positions[pos_idx]
            if pos_idx < 2:
                sequence[0, abs_pos] = 100
            else:
                sequence[0, abs_pos] = null_id

        state = GenerationState(
            sequence=sequence,
            prompt_length=prompt_length,
            prefix_length=prefix_length,
            template=template,
        )

        overflow_fields = generator._detect_overflow_fields(state)
        assert "location" not in overflow_fields, "Should not detect overflow when tail has many NULLs"


class TestTemplateExpansion:

    def test_expand_template_single_field(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8), ("units", 4)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        original_location_len = template.field_segments[0].length

        new_template = generator._expand_template(template, ["location"])

        assert new_template is not None
        assert len(new_template.field_segments) == 2

        new_location_seg = next(s for s in new_template.field_segments if s.name == "location")
        new_units_seg = next(s for s in new_template.field_segments if s.name == "units")

        expand_by = generator.expansion_config.get("expand_tokens", 4)
        assert new_location_seg.length == original_location_len + expand_by
        assert new_units_seg.length == 4, "Non-expanded field should keep same length"

    def test_expand_template_multiple_fields(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8), ("units", 4), ("format", 6)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        new_template = generator._expand_template(template, ["location", "format"])

        assert new_template is not None

        new_location_seg = next(s for s in new_template.field_segments if s.name == "location")
        new_units_seg = next(s for s in new_template.field_segments if s.name == "units")
        new_format_seg = next(s for s in new_template.field_segments if s.name == "format")

        expand_by = generator.expansion_config.get("expand_tokens", 4)
        assert new_location_seg.length == 8 + expand_by
        assert new_units_seg.length == 4
        assert new_format_seg.length == 6 + expand_by

    def test_expand_template_respects_max(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        generator.budget_config = {
            "min_tokens": 0,
            "max_tokens": 10,
            "extra_tokens": 0,
        }

        fields = [("location", 9)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        new_template = generator._expand_template(template, ["location"])

        new_location_seg = next(s for s in new_template.field_segments if s.name == "location")
        assert new_location_seg.length == 10, "Should cap at max_tokens"


class TestWarmStart:

    def test_warm_start_copies_tokens(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        old_fields = [("location", 8)]
        old_template = build_schema_template(
            tokenizer=tokenizer,
            fields=old_fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        old_sequence = torch.zeros((1, 50), dtype=torch.long)
        old_prompt_length = 10
        old_prefix_length = 2

        old_seg = old_template.field_segments[0]
        for idx in range(len(old_seg.value_positions)):
            abs_pos = old_prompt_length + old_prefix_length + old_seg.value_positions[idx]
            old_sequence[0, abs_pos] = 1000 + idx

        warm_state = GenerationState(
            sequence=old_sequence,
            prompt_length=old_prompt_length,
            prefix_length=old_prefix_length,
            template=old_template,
        )

        new_fields = [("location", 12)]
        new_template = build_schema_template(
            tokenizer=tokenizer,
            fields=new_fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        new_sequence = torch.full((1, 60), fill_value=mask_id, dtype=torch.long)
        new_prompt_length = 10
        new_prefix_length = 2

        generator._apply_warm_start(
            warm_state=warm_state,
            new_sequence=new_sequence,
            new_template=new_template,
            new_prompt_length=new_prompt_length,
            new_prefix_length=new_prefix_length,
        )

        new_seg = new_template.field_segments[0]
        for idx in range(8):
            abs_pos = new_prompt_length + new_prefix_length + new_seg.value_positions[idx]
            expected = 1000 + idx
            assert new_sequence[0, abs_pos] == expected, f"Position {idx} should copy old value {expected}"

        for idx in range(8, 12):
            abs_pos = new_prompt_length + new_prefix_length + new_seg.value_positions[idx]
            assert new_sequence[0, abs_pos] == mask_id, f"Position {idx} should remain masked"


class TestExpansionLoop:

    def test_expansion_disabled_without_null_token(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, _, _ = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=None,
            include_codeblock=False,
        )

        enabled = generator._expansion_enabled(template)
        assert not enabled, "Expansion should be disabled without NULL token"

    def test_expansion_disabled_by_config(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        generator.expansion_config = {"enabled": False}

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        enabled = generator._expansion_enabled(template)
        assert not enabled, "Expansion should be disabled by config"

    def test_expansion_enabled(self, generator, tokenizer, mask_and_null):
        mask_str, mask_id, null_str, null_id = mask_and_null

        generator.expansion_config = {
            "enabled": True,
            "max_rounds": 2,
            "expand_tokens": 4,
            "tail_window": 4,
            "tail_null_threshold": 0.5,
        }

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        enabled = generator._expansion_enabled(template)
        assert enabled, "Expansion should be enabled with NULL token and config=True"

    @patch.object(FunctionCallGenerator, '_generate_single_pass')
    def test_expansion_stops_when_no_overflow(
            self,
            mock_single_pass,
            generator,
            tokenizer,
            mask_and_null
    ):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 50), dtype=torch.long)
        prompt_length = 10
        prefix_length = 2

        segment = template.field_segments[0]
        for pos_idx in range(len(segment.value_positions)):
            abs_pos = prompt_length + prefix_length + segment.value_positions[pos_idx]
            if pos_idx < 2:
                sequence[0, abs_pos] = 100
            else:
                sequence[0, abs_pos] = null_id

        state = GenerationState(
            sequence=sequence,
            prompt_length=prompt_length,
            prefix_length=prefix_length,
            template=template,
        )

        output = GenerationOutput(
            text="test output",
            token_ids=[100, 101],
            steps_executed=4,
        )

        mock_single_pass.return_value = (output, state)

        result = generator.generate(
            prompt="test",
            template=template,
            config=None,
            tool_name="test_tool",
            tools=[],
        )

        assert mock_single_pass.call_count == 1, "Should only run once when no overflow detected"
        assert result.text == "test output"

    @patch.object(FunctionCallGenerator, '_generate_single_pass')
    @patch.object(FunctionCallGenerator, '_detect_overflow_fields')
    def test_expansion_runs_multiple_rounds(
            self,
            mock_detect_overflow,
            mock_single_pass,
            generator,
            tokenizer,
            mask_and_null
    ):
        mask_str, mask_id, null_str, null_id = mask_and_null

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 50), dtype=torch.long)
        state = GenerationState(
            sequence=sequence,
            prompt_length=10,
            prefix_length=2,
            template=template,
        )

        output = GenerationOutput(
            text="test output",
            token_ids=[100, 101],
            steps_executed=4,
        )

        mock_single_pass.return_value = (output, state)
        mock_detect_overflow.side_effect = [["location"], []]

        result = generator.generate(
            prompt="test",
            template=template,
            config=None,
            tool_name="test_tool",
            tools=[],
        )

        assert mock_single_pass.call_count == 2, "Should run twice: initial + one expansion"
        assert mock_detect_overflow.call_count == 2

    @patch.object(FunctionCallGenerator, '_generate_single_pass')
    @patch.object(FunctionCallGenerator, '_detect_overflow_fields')
    def test_expansion_respects_max_rounds(
            self,
            mock_detect_overflow,
            mock_single_pass,
            generator,
            tokenizer,
            mask_and_null
    ):
        mask_str, mask_id, null_str, null_id = mask_and_null

        generator.expansion_config["max_rounds"] = 2

        fields = [("location", 8)]
        template = build_schema_template(
            tokenizer=tokenizer,
            fields=fields,
            mask_token=mask_str,
            null_token=null_str,
            include_codeblock=False,
        )

        sequence = torch.zeros((1, 50), dtype=torch.long)
        state = GenerationState(
            sequence=sequence,
            prompt_length=10,
            prefix_length=2,
            template=template,
        )

        output = GenerationOutput(
            text="test output",
            token_ids=[100, 101],
            steps_executed=4,
        )

        mock_single_pass.return_value = (output, state)
        mock_detect_overflow.return_value = ["location"]

        result = generator.generate(
            prompt="test",
            template=template,
            config=None,
            tool_name="test_tool",
            tools=[],
        )

        max_calls = 1 + generator.expansion_config["max_rounds"]
        assert mock_single_pass.call_count == max_calls, f"Should run {max_calls} times (initial + max_rounds)"
