"""
Format adapters for multiple function calling datasets.

Each adapter converts a dataset's native format to a unified schema that
SmartScaffoldDataset can process.
"""
import json
from typing import Any, Dict, List, Optional


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
        return UnifiedExample(
            conversations=example.get("conversations", []),
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
                    tool_call = {
                        "name": name,
                        "arguments": arguments
                    }
                    tool_calls_text.append(
                        f"<tool_call>\n{json.dumps(tool_call, indent=2)}\n</tool_call>"
                    )
            
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
        functions = example.get("functions", [])
        
        # Parse chat string into conversations
        conversations = []
        
        # Add system message if present
        if system:
            conversations.append({"from": "system", "value": system})
        
        # Try to parse chat (format varies, handle gracefully)
        if chat:
            # Simple heuristic: split by "USER:" and "ASSISTANT:" markers
            import re
            parts = re.split(r'(USER:|ASSISTANT:)', chat)
            current_role = None
            current_text = []
            
            for part in parts:
                if part == "USER:":
                    if current_role and current_text:
                        conversations.append({
                            "from": current_role,
                            "value": " ".join(current_text).strip()
                        })
                    current_role = "human"
                    current_text = []
                elif part == "ASSISTANT:":
                    if current_role and current_text:
                        conversations.append({
                            "from": current_role,
                            "value": " ".join(current_text).strip()
                        })
                    current_role = "gpt"
                    current_text = []
                else:
                    if part.strip():
                        current_text.append(part.strip())
            
            # Add final turn
            if current_role and current_text:
                conversations.append({
                    "from": current_role,
                    "value": " ".join(current_text).strip()
                })
        
        # Convert functions to JSON string
        tools_str = json.dumps(functions) if functions else "[]"
        
        return UnifiedExample(
            conversations=conversations,
            tools=tools_str
        )


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
                conversations=example.get("conversations", []),
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
                conversations=conversations,
                tools=example.get("tools", "[]")
            )
        
        # Fallback: empty conversations
        return UnifiedExample(conversations=[], tools="[]")


# Registry of adapters
DATASET_ADAPTERS = {
    "interstellarninja/hermes_reasoning_tool_use": HermesReasoningAdapter,
    "Salesforce/xlam-function-calling-60k": XLAMAdapter,
    "glaiveai/glaive-function-calling-v2": GlaiveV2Adapter,
    "NousResearch/hermes-function-calling-v1": NousHermesAdapter,
}


def get_adapter(dataset_name: str):
    """Get the appropriate adapter for a dataset."""
    return DATASET_ADAPTERS.get(dataset_name, HermesReasoningAdapter)
