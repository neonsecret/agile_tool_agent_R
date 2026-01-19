from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class ToolCallParts:
    prefix_ids: List[int]
    suffix_ids: List[int]


def apply_smollm3_chat_template(
        tokenizer: PreTrainedTokenizerBase,
        messages: Sequence[Dict[str, str]],
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
) -> List[int]:
    """
    Apply SmolLM3's tokenizer chat template with optional tool injection.

    SmolLM3's `chat_template.jinja` expects `tools` (xml tool mode) to be passed
    so it can render the <tools>...</tools> list in the system header.
    """
    apply_fn = tokenizer.apply_chat_template
    sig = inspect.signature(apply_fn)

    kwargs: Dict[str, Any] = {
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
    }

    if tools is not None:
        if "tools" not in sig.parameters:
            raise ValueError(
                "This transformers version does not support `tools=` in "
                "tokenizer.apply_chat_template(), but SmolLM3 tool calling requires it. "
                "Upgrade transformers or inject tools manually."
            )
        kwargs["tools"] = list(tools)

    chat_ids = apply_fn(list(messages), **kwargs)
    if isinstance(chat_ids, list):
        return chat_ids
    return list(chat_ids)


def encode_tool_call_wrapper(
        tokenizer: PreTrainedTokenizerBase,
        tool_name: str,
) -> ToolCallParts:
    """
    Return token IDs for a SmolLM3-compatible tool call wrapper.

    The wrapper is:
      <tool_call>
      {"name": "<tool_name>", "arguments":  <ARGS_JSON_OBJECT> }
      </tool_call>

    We provide prefix up to `"arguments": ` and suffix to close the outer object.
    """
    prefix = f'<tool_call>\n{{"name": {json.dumps(tool_name)}, "arguments": '
    suffix = "}\n</tool_call>"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    return ToolCallParts(prefix_ids=prefix_ids, suffix_ids=suffix_ids)


def parse_first_tool_call(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("<tool_call>")
    if start < 0:
        return None
    end = text.find("</tool_call>", start)
    if end < 0:
        return None

    payload = text[start + len("<tool_call>"):end].strip()
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None
