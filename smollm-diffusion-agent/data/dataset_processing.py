"""
Dataset processing utilities.

Handles tool schema parsing, tool call extraction, and message building.
"""

import json
import re
from typing import Any, Dict, List, Optional, Sequence


def parse_tools_schema(tools_schema_raw):
    """Parse tools schema from various input formats."""
    if tools_schema_raw is None:
        return []
    if isinstance(tools_schema_raw, list):
        return tools_schema_raw
    if isinstance(tools_schema_raw, dict):
        return [tools_schema_raw]
    if isinstance(tools_schema_raw, str):
        try:
            parsed = json.loads(tools_schema_raw)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return [parsed]
        return parsed
    return None


def map_role(role: str) -> str:
    """Map alternative role names to standard format."""
    if role == "gpt":
        return "assistant"
    if role == "human":
        return "user"
    return role


def build_messages(
        conversations: Sequence[Dict[str, Any]],
        msg_idx: int,
        system_message: str,
        max_history_messages: int,
) -> List[Dict[str, str]]:
    """
    Build a SmolLM3-compatible `messages` list for tokenizer.apply_chat_template.

    The returned messages ALWAYS start with a system message, and end with the
    last message before `msg_idx` (typically a user message). We truncate older
    history so the scaffold fits under `max_seq_len`.
    """
    history: List[Dict[str, str]] = [{"role": "system", "content": system_message}]

    upto = conversations[:msg_idx]
    mapped = [
        {"role": map_role(m.get("from", "")), "content": m.get("value", "")}
        for m in upto
        if m.get("from") is not None and m.get("value") is not None
    ]
    if max_history_messages > 0:
        mapped = mapped[-max_history_messages:]
    history.extend(mapped)
    return history


def extract_tool_call_from_message(msg_value: str) -> Optional[Dict[str, Any]]:
    """Extract and parse first tool call from a message."""
    tool_call_matches = list(re.finditer(r'<tool_call>(.*?)</tool_call>', msg_value, re.DOTALL))

    if not tool_call_matches:
        return None

    match = tool_call_matches[0]
    tool_call_json_str = match.group(1).strip()

    try:
        tool_call_json = json.loads(tool_call_json_str)
    except json.JSONDecodeError:
        return None

    return tool_call_json
