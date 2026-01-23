"""
Format adapters for multiple function calling datasets.

Each adapter converts a dataset's native format to a unified schema that
SmartScaffoldDataset can process.
"""
import json
import re
from typing import Any, Dict, List, Optional


def _coerce_arguments(arguments: Any) -> Dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if not isinstance(arguments, dict):
        return {}
    return arguments


def _format_tool_call(name: str, arguments: Any) -> str:
    tool_call = {
        "name": name,
        "arguments": _coerce_arguments(arguments),
    }
    return f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>"


def _extract_json_object(text: str, start_idx: int) -> Optional[str]:
    idx = text.find("{", start_idx)
    if idx == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for pos in range(idx, len(text)):
        ch = text[pos]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[idx:pos + 1]
    return None


def _extract_glaive_tool_calls(text: str) -> List[str]:
    calls = []
    for match in re.finditer(r"<<functioncall>>", text):
        json_blob = _extract_json_object(text, match.end())
        if not json_blob:
            continue
        try:
            parsed = json.loads(json_blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            name = parsed.get("name", "")
            arguments = parsed.get("arguments", {})
            calls.append(_format_tool_call(name, arguments))
    return calls


def _clean_glaive_text(text: str) -> str:
    text = text.replace("<<|endoftext|>>", "")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("FUNCTION RESPONSE:"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _normalize_tool_call_blocks(text: str) -> str:
    def _replace(match: re.Match) -> str:
        payload = match.group(1).strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return match.group(0)
        if not isinstance(parsed, dict):
            return match.group(0)
        name = parsed.get("name", "")
        arguments = parsed.get("arguments", {})
        return _format_tool_call(name, arguments)

    return re.sub(r"<tool_call>(.*?)</tool_call>", _replace, text, flags=re.DOTALL)


def _normalize_tool_calls_in_conversations(
    conversations: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    normalized = []
    for msg in conversations:
        value = msg.get("value", "")
        if "<tool_call>" in value:
            normalized.append(
                {"from": msg.get("from", ""), "value": _normalize_tool_call_blocks(value)}
            )
        else:
            normalized.append(msg)
    return normalized


class UnifiedExample:
    """
    Unified format for all datasets.
    
    Structure:
        conversations: List[Dict[str, str]] with keys "from" and "value"
        tools: List[Dict] or str (JSON string of tool schemas)
    """

    def __init__(self, conversations: List[Dict[str, str]], tools: Any):
        self.conversations = conversations
        self.tools = tools

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversations": self.conversations,
            "tools": self.tools
        }


class HermesReasoningAdapter:
    """
    Adapter for interstellarninja/hermes_reasoning_tool_use dataset.
    
    Already in the correct format, just pass through.
    """

    @staticmethod
    def convert(example: Dict[str, Any]) -> UnifiedExample:
        conversations = example.get("conversations", [])
        conversations = _normalize_tool_calls_in_conversations(conversations)
        return UnifiedExample(
            conversations=conversations,
            tools=example.get("tools", "[]")
        )


class XLAMAdapter:
    """
    Adapter for Salesforce/xlam-function-calling-60k dataset.
    
    Format:
        - id: int
        - query: str (user query)
        - answers: str (JSON list of tool calls)
        - tools: str (JSON list of tool schemas)
    
    Converts to:
        - conversations: [{"from": "human", "value": query}, {"from": "gpt", "value": <tool_call>...}]
        - tools: tools (JSON string)
    """

    @staticmethod
    def convert(example: Dict[str, Any]) -> UnifiedExample:
        query = example.get("query", "")
        answers_str = example.get("answers", "[]")
        tools = example.get("tools", "[]")

        # Parse answers
        try:
            answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
        except json.JSONDecodeError:
            answers = []

        # Build conversations
        conversations = [
            {"from": "human", "value": query}
        ]

        # Format tool calls in SmolLM3 format
        if answers and len(answers) > 0:
            tool_calls_text = []
            for answer in answers:
                if isinstance(answer, dict):
                    name = answer.get("name", "")
                    arguments = answer.get("arguments", {})
                    tool_calls_text.append(_format_tool_call(name, arguments))

            assistant_response = "\n\n".join(tool_calls_text)
            conversations.append({"from": "gpt", "value": assistant_response})

        return UnifiedExample(
            conversations=conversations,
            tools=tools
        )


class GlaiveV2Adapter:
    """
    Adapter for glaiveai/glaive-function-calling-v2 dataset.
    
    Format (typical):
        - system: str
        - chat: str (serialized conversation)
        - functions: List[Dict] (tool definitions)
    
    Converts to unified format.
    """

    @staticmethod
    def convert(example: Dict[str, Any]) -> UnifiedExample:
        system = example.get("system", "")
        chat = example.get("chat", "")
        functions = example.get("functions")
        if functions is None:
            functions = example.get("tools", [])

        # Parse chat string into conversations
        conversations = []

        # Add system message if present
        if system:
            conversations.append({"from": "system", "value": system})

        # Try to parse chat (format varies, handle gracefully)
        if chat:
            parts = re.split(r"(USER:|ASSISTANT:)", chat)
            current_role = None
            current_text = []

            for part in parts:
                if part == "USER:":
                    if current_role and current_text:
                        text = " ".join(current_text).strip()
                        conversations.append({"from": current_role, "value": text})
                    current_role = "human"
                    current_text = []
                elif part == "ASSISTANT:":
                    if current_role and current_text:
                        text = " ".join(current_text).strip()
                        conversations.append({"from": current_role, "value": text})
                    current_role = "gpt"
                    current_text = []
                else:
                    if part.strip():
                        current_text.append(part.strip())

            if current_role and current_text:
                text = " ".join(current_text).strip()
                conversations.append({"from": current_role, "value": text})

        formatted = []
        for msg in conversations:
            if msg.get("from") != "gpt":
                formatted.append(msg)
                continue
            raw_text = msg.get("value", "")
            tool_calls = _extract_glaive_tool_calls(raw_text)
            if tool_calls:
                formatted.append({"from": "gpt", "value": "\n\n".join(tool_calls)})
            else:
                cleaned = _clean_glaive_text(raw_text)
                if cleaned:
                    formatted.append({"from": "gpt", "value": cleaned})

        # Convert functions to JSON string
        if isinstance(functions, str):
            tools_str = functions
        else:
            tools_str = json.dumps(functions) if functions else "[]"

        return UnifiedExample(
            conversations=formatted,
            tools=tools_str
        )


class APIGenMTAdapter:
    """
    Adapter for Salesforce/APIGen-MT-5k dataset.

    Format:
        - conversations: list of dicts with "from" and "value"
        - tools: JSON string or list of tool schemas
        - system: optional system prompt
    """

    @staticmethod
    def convert(example: Dict[str, Any]) -> UnifiedExample:
        system = example.get("system", "")
        conversations_raw = example.get("conversations", [])
        tools = example.get("tools", "[]")

        conversations = []
        if system:
            conversations.append({"from": "system", "value": system})

        for msg in conversations_raw:
            role = msg.get("from", "")
            value = msg.get("value", "")
            if role in {"human", "user"}:
                conversations.append({"from": "human", "value": value})
            elif role in {"gpt", "assistant"}:
                conversations.append({"from": "gpt", "value": value})
            elif role == "function_call":
                payload = value
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        payload = {}
                if isinstance(payload, dict):
                    name = payload.get("name", "")
                    arguments = payload.get("arguments", {})
                    conversations.append({
                        "from": "gpt",
                        "value": _format_tool_call(name, arguments),
                    })
            elif role == "observation":
                continue

        if isinstance(tools, str):
            tools_str = tools
        else:
            tools_str = json.dumps(tools) if tools else "[]"

        return UnifiedExample(conversations=conversations, tools=tools_str)


class NousHermesAdapter:
    """
    Adapter for NousResearch/hermes-function-calling-v1 dataset.
    
    Similar to Hermes reasoning but may have slight format differences.
    """

    @staticmethod
    def convert(example: Dict[str, Any]) -> UnifiedExample:
        # Check if it has the standard format
        if "conversations" in example:
            return UnifiedExample(
                conversations=_normalize_tool_calls_in_conversations(
                    example.get("conversations", [])
                ),
                tools=example.get("tools", "[]")
            )

        # Alternative format: messages field
        if "messages" in example:
            messages = example.get("messages", [])
            conversations = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                # Map role
                if role == "user":
                    conversations.append({"from": "human", "value": content})
                elif role == "assistant":
                    conversations.append({"from": "gpt", "value": content})
                elif role == "system":
                    conversations.append({"from": "system", "value": content})

            return UnifiedExample(
                conversations=_normalize_tool_calls_in_conversations(conversations),
                tools=example.get("tools", "[]")
            )

        # Fallback: empty conversations
        return UnifiedExample(conversations=[], tools="[]")


# Registry of adapters
DATASET_ADAPTERS = {
    "interstellarninja/hermes_reasoning_tool_use": HermesReasoningAdapter,
    "Salesforce/xlam-function-calling-60k": XLAMAdapter,
    "glaiveai/glaive-function-calling-v2": GlaiveV2Adapter,
    "Salesforce/APIGen-MT-5k": APIGenMTAdapter,
    "NousResearch/hermes-function-calling-v1": NousHermesAdapter,
}


def get_adapter(dataset_name: str):
    """Get the appropriate adapter for a dataset."""
    return DATASET_ADAPTERS.get(dataset_name, HermesReasoningAdapter)
