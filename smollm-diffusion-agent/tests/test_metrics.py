"""
Tests for specialized function calling metrics.
"""
import pytest
import torch
import json
from data.metrics import (
    calculate_null_token_metrics,
    calculate_field_level_metrics,
    calculate_parse_metrics,
    calculate_scaffold_metrics,
    extract_tool_call_json
)


class TestNullTokenMetrics:
    def test_null_token_metrics_basic(self):
        """Test NULL token metrics calculation."""
        # Setup: 10 positions, 5 should be NULL, 5 should be real tokens
        predictions = torch.tensor([[1, 2, 999, 999, 999, 3, 4, 999, 5, 6]])  # 999 = NULL
        labels = torch.tensor([[1, 2, 999, 999, 888, 3, 4, 999, 5, 888]])  # Mix of NULL and real
        mask_positions = torch.tensor([[True] * 10])
        null_token_id = 999

        metrics = calculate_null_token_metrics(predictions, labels, mask_positions, null_token_id)

        assert "null_prediction_rate" in metrics
        assert "null_accuracy" in metrics
        assert "real_token_accuracy" in metrics
        assert "null_precision" in metrics
        assert "null_recall" in metrics

        # 4 predictions are NULL out of 10 = 40%
        assert metrics["null_prediction_rate"] == pytest.approx(0.4, abs=0.01)

        # Of 3 actual NULLs, 3 predicted correctly = 100%
        assert metrics["null_accuracy"] == pytest.approx(1.0, abs=0.01)

        # Of 4 predicted NULLs, 3 are correct = 75% precision
        assert metrics["null_precision"] == pytest.approx(0.75, abs=0.01)

    def test_null_token_metrics_no_null(self):
        """Test when NULL token is None."""
        predictions = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        mask_positions = torch.tensor([[True] * 5])

        metrics = calculate_null_token_metrics(predictions, labels, mask_positions, None)

        assert metrics == {}  # Should return empty dict


class TestFieldLevelMetrics:
    def test_field_level_accuracy(self):
        """Test field-level accuracy calculation."""
        predicted = [
            {"name": "get_weather", "arguments": {"location": "NYC", "units": "C"}},
            {"name": "get_weather", "arguments": {"location": "LA", "units": "F"}},
        ]
        ground_truth = [
            {"name": "get_weather", "arguments": {"location": "NYC", "units": "F"}},  # units wrong
            {"name": "get_weather", "arguments": {"location": "LA", "units": "F"}},  # perfect
        ]

        metrics = calculate_field_level_metrics(predicted, ground_truth)

        assert "tool_name_accuracy" in metrics
        assert "field_exact_match_rate" in metrics

        # Both tool names correct
        assert metrics["tool_name_accuracy"] == 1.0

        # First: 1/2 fields correct, Second: 2/2 fields correct = avg 0.75
        assert metrics["field_exact_match_rate"] == pytest.approx(0.75, abs=0.01)

    def test_per_field_tracking(self):
        """Test per-field accuracy tracking."""
        predicted = [
            {"name": "tool1", "arguments": {"location": "NYC", "units": "C"}},
            {"name": "tool1", "arguments": {"location": "NYC", "units": "C"}},
            {"name": "tool1", "arguments": {"location": "NYC", "units": "F"}},
        ]
        ground_truth = [
            {"name": "tool1", "arguments": {"location": "NYC", "units": "C"}},
            {"name": "tool1", "arguments": {"location": "SF", "units": "C"}},
            {"name": "tool1", "arguments": {"location": "NYC", "units": "F"}},
        ]

        metrics = calculate_field_level_metrics(predicted, ground_truth)

        # location: 2/3 correct, units: 3/3 correct
        assert "field_location_accuracy" in metrics
        assert "field_units_accuracy" in metrics
        assert metrics["field_location_accuracy"] == pytest.approx(0.666, abs=0.01)
        assert metrics["field_units_accuracy"] == 1.0


class TestParseMetrics:
    def test_parse_success_rate(self):
        """Test JSON parse success rate."""
        texts = [
            '<tool_call>\n{"name": "tool1", "arguments": {}}\n</tool_call>',  # Valid
            '<tool_call>\n{"name": "tool2"}\n</tool_call>',  # Valid
            '<tool_call>\n{invalid json}\n</tool_call>',  # Invalid JSON
            'Just text',  # No tool call
        ]
        has_tool_calls = [True, True, True, False]

        metrics = calculate_parse_metrics(texts, has_tool_calls)

        assert "json_parse_success_rate" in metrics
        assert "tool_call_format_rate" in metrics
        assert "false_positive_rate" in metrics
        assert "false_negative_rate" in metrics

        # 3 have format out of 4 = 75%
        assert metrics["tool_call_format_rate"] == 0.75

        # 2 parse successfully out of 4 = 50%
        assert metrics["json_parse_success_rate"] == 0.5

        # No false negatives (all tool calls have format)
        assert metrics["false_negative_rate"] == 0.0

    def test_extract_tool_call_json(self):
        """Test tool call JSON extraction."""
        text = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "NYC"}}\n</tool_call>'
        result = extract_tool_call_json(text)

        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"]["location"] == "NYC"

        # Test invalid JSON
        invalid_text = '<tool_call>\n{invalid}\n</tool_call>'
        result = extract_tool_call_json(invalid_text)
        assert result is None


class TestScaffoldMetrics:
    def test_scaffold_statistics(self):
        """Test scaffold statistics calculation."""
        scaffold_sizes = [32, 48, 40, 36]
        mask_counts = [32, 48, 40, 36]
        null_counts = [10, 20, 15, 12]

        metrics = calculate_scaffold_metrics(scaffold_sizes, mask_counts, null_counts)

        assert "avg_scaffold_size" in metrics
        assert "avg_mask_count" in metrics
        assert "avg_null_ratio" in metrics
        assert "scaffold_size_std" in metrics

        # Average scaffold size = (32+48+40+36)/4 = 39
        assert metrics["avg_scaffold_size"] == pytest.approx(39.0, abs=0.1)

        # Average NULL ratio should be calculated
        assert 0.0 <= metrics["avg_null_ratio"] <= 1.0

        assert metrics["max_scaffold_size"] == 48
        assert metrics["min_scaffold_size"] == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
