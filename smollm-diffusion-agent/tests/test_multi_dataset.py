"""
Tests for multi-dataset loading and format conversion.
"""
from pathlib import Path

import pytest
import yaml
from data.dataset_formats import (
    HermesReasoningAdapter,
    XLAMAdapter,
    GlaiveV2Adapter,
    NousHermesAdapter,
    get_adapter,
)
from data.multi_dataset_loader import MultiDatasetConfig, load_and_unify_dataset
from data.dataset_loader import SmartScaffoldDataset
from data.utils import resolve_mask_token, resolve_null_token
import json


class TestFormatAdapters:
    """Test each dataset format adapter."""

    def test_hermes_adapter(self):
        """Test Hermes reasoning adapter (pass-through)."""
        example = {
            "conversations": [
                {"from": "human", "value": "What's the weather?"},
                {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_weather\"}\n</tool_call>"}
            ],
            "tools": "[{\"name\": \"get_weather\"}]"
        }

        unified = HermesReasoningAdapter.convert(example)
        assert len(unified.conversations) == 2
        assert unified.conversations[0]["from"] == "human"
        assert unified.tools == example["tools"]

    def test_xlam_adapter(self):
        """Test XLAM dataset adapter."""
        example = {
            "id": 0,
            "query": "Where can I find live giveaways?",
            "answers": '[{"name": "live_giveaways", "arguments": {"type": "beta"}}]',
            "tools": '[{"name": "live_giveaways", "description": "..."}]'
        }

        unified = XLAMAdapter.convert(example)

        # Should have 2 turns: user + assistant
        assert len(unified.conversations) >= 1
        assert unified.conversations[0]["from"] == "human"
        assert unified.conversations[0]["value"] == example["query"]

        # Assistant response should contain <tool_call>
        if len(unified.conversations) > 1:
            assert "<tool_call>" in unified.conversations[1]["value"]
            assert "live_giveaways" in unified.conversations[1]["value"]

    def test_xlam_adapter_multiple_calls(self):
        """Test XLAM with multiple tool calls."""
        example = {
            "id": 1,
            "query": "Test query",
            "answers": '[{"name": "tool1", "arguments": {"arg": 1}}, {"name": "tool2", "arguments": {"arg": 2}}]',
            "tools": "[]"
        }

        unified = XLAMAdapter.convert(example)
        assert len(unified.conversations) == 2

        # Should have both tool calls
        response = unified.conversations[1]["value"]
        assert response.count("<tool_call>") == 2
        assert "tool1" in response
        assert "tool2" in response

    def test_glaive_adapter(self):
        """Test Glaive v2 adapter."""
        example = {
            "system": "You are a helpful assistant.",
            "chat": "USER: Hello\nASSISTANT: Hi there!",
            "functions": [{"name": "test_function"}]
        }

        unified = GlaiveV2Adapter.convert(example)

        # Should parse conversations
        assert len(unified.conversations) >= 2

        # Tools should be JSON string
        tools = json.loads(unified.tools)
        assert len(tools) == 1
        assert tools[0]["name"] == "test_function"

    def test_nous_hermes_adapter_conversations(self):
        """Test Nous Hermes adapter with conversations field."""
        example = {
            "conversations": [
                {"from": "human", "value": "Test"},
                {"from": "gpt", "value": "Response"}
            ],
            "tools": "[]"
        }

        unified = NousHermesAdapter.convert(example)
        assert len(unified.conversations) == 2

    def test_nous_hermes_adapter_messages(self):
        """Test Nous Hermes adapter with messages field."""
        example = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"}
            ],
            "tools": "[]"
        }

        unified = NousHermesAdapter.convert(example)
        assert len(unified.conversations) == 2
        assert unified.conversations[0]["from"] == "human"
        assert unified.conversations[1]["from"] == "gpt"

    def test_get_adapter(self):
        """Test adapter registry."""
        assert get_adapter("Salesforce/xlam-function-calling-60k") == XLAMAdapter
        assert get_adapter("glaiveai/glaive-function-calling-v2") == GlaiveV2Adapter
        assert get_adapter("unknown_dataset") == HermesReasoningAdapter  # Fallback


@pytest.mark.slow
class TestMultiDatasetLoader:
    """Test multi-dataset loading (requires network)."""

    def test_load_xlam_sample(self):
        """Load a small sample from XLAM dataset."""
        config = MultiDatasetConfig(
            name="Salesforce/xlam-function-calling-60k",
            split="train",
            weight=1.0,
            limit=10  # Only 10 examples for testing
        )

        examples = load_and_unify_dataset(config)

        assert len(examples) == 10
        assert all("conversations" in ex for ex in examples)
        assert all("tools" in ex for ex in examples)

    def test_load_hermes_sample(self):
        """Load a small sample from Hermes dataset."""
        config = MultiDatasetConfig(
            name="interstellarninja/hermes_reasoning_tool_use",
            split="train",
            weight=1.0,
            limit=10
        )

        examples = load_and_unify_dataset(config)

        assert len(examples) == 10
        assert all("conversations" in ex for ex in examples)

    @pytest.mark.slow
    def test_smart_scaffold_uses_config_datasets(self, shared_tokenizer):
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        config = yaml.safe_load(config_path.read_text())

        data_cfg = config.get("data", {})
        datasets_cfg = data_cfg.get("datasets", [])
        for ds_cfg in datasets_cfg:
            if ds_cfg.get("limit") is None:
                ds_cfg["limit"] = 20
        if data_cfg.get("limit") is None:
            data_cfg["limit"] = 40

        config["data"] = data_cfg

        mask_str, _ = resolve_mask_token(shared_tokenizer, None)
        null_str, _ = resolve_null_token(shared_tokenizer, None)

        dataset = SmartScaffoldDataset(
            tokenizer=shared_tokenizer,
            max_seq_len=1024,
            limit=data_cfg.get("limit"),
            mask_token=mask_str,
            null_token=null_str,
            data_config=config,
        )

        assert len(dataset) > 0
        assert all("conversations" in ex and "tools" in ex for ex in dataset.ds)


class TestManualInspection:
    """Manual inspection helpers (run with -s to see output)."""

    @pytest.mark.slow
    def test_print_xlam_samples(self):
        """Print XLAM samples for manual review."""
        config = MultiDatasetConfig(
            name="Salesforce/xlam-function-calling-60k",
            split="train",
            limit=3
        )

        examples = load_and_unify_dataset(config)

        print("\n" + "=" * 80)
        print("XLAM SAMPLES")
        print("=" * 80)
        for i, ex in enumerate(examples):
            print(f"\nExample {i + 1}:")
            print(f"Conversations: {len(ex['conversations'])} turns")
            for turn in ex['conversations']:
                print(f"  {turn['from']}: {turn['value'][:100]}...")
            print(f"Tools: {ex['tools'][:200]}...")

    @pytest.mark.slow
    def test_compare_formats(self):
        """Compare formats side-by-side."""
        xlam_config = MultiDatasetConfig(
            name="Salesforce/xlam-function-calling-60k",
            split="train",
            limit=2
        )
        hermes_config = MultiDatasetConfig(
            name="interstellarninja/hermes_reasoning_tool_use",
            split="train",
            limit=2
        )

        xlam_examples = load_and_unify_dataset(xlam_config)
        hermes_examples = load_and_unify_dataset(hermes_config)

        print("\n" + "=" * 80)
        print("FORMAT COMPARISON")
        print("=" * 80)

        print("\nXLAM Example:")
        print(json.dumps(xlam_examples[0], indent=2))

        print("\nHermes Example:")
        print(json.dumps(hermes_examples[0], indent=2))


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_multi_dataset.py -v -s -m slow
    pytest.main([__file__, "-v", "-s"])
