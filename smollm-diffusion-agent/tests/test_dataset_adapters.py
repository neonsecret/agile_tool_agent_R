from data.dataset_formats import APIGenMTAdapter, GlaiveV2Adapter, HermesReasoningAdapter
from data.dataset_processing import extract_tool_call_from_message


def test_glaive_adapter_parses_tool_call():
    example = {
        "system": "You are a helpful assistant.",
        "chat": (
            "USER: What is the weather?\n"
            "ASSISTANT: <<functioncall>> "
            "{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"SF\\\"}\"}\n"
            "FUNCTION RESPONSE: {\"temp\": 60}\n"
        ),
        "functions": [
            {
                "name": "get_weather",
                "parameters": {"properties": {"location": {"type": "string"}}},
            }
        ],
    }

    converted = GlaiveV2Adapter.convert(example).to_dict()
    conversations = converted["conversations"]
    assert any("<tool_call>" in msg["value"] for msg in conversations)

    tool_msg = next(msg for msg in conversations if "<tool_call>" in msg["value"])
    tool_call = extract_tool_call_from_message(tool_msg["value"])
    assert tool_call is not None
    assert tool_call["name"] == "get_weather"
    assert isinstance(tool_call["arguments"], dict)
    assert tool_call["arguments"]["location"] == "SF"


def test_apigen_adapter_parses_function_call():
    example = {
        "system": "Use tools when necessary.",
        "conversations": [
            {"from": "human", "value": "Book a hotel."},
            {"from": "function_call",
             "value": "{\"name\": \"book_hotel\", \"arguments\": {\"city\": \"NY\"}}"},
            {"from": "observation", "value": "{\"status\": \"ok\"}"},
            {"from": "gpt", "value": "Done."},
        ],
        "tools": [
            {
                "name": "book_hotel",
                "parameters": {"properties": {"city": {"type": "string"}}},
            }
        ],
    }

    converted = APIGenMTAdapter.convert(example).to_dict()
    conversations = converted["conversations"]
    tool_msg = next(msg for msg in conversations if "<tool_call>" in msg["value"])
    tool_call = extract_tool_call_from_message(tool_msg["value"])
    assert tool_call is not None
    assert tool_call["name"] == "book_hotel"
    assert isinstance(tool_call["arguments"], dict)
    assert tool_call["arguments"]["city"] == "NY"


def test_hermes_adapter_normalizes_arguments():
    example = {
        "conversations": [
            {"from": "human", "value": "Find flights"},
            {"from": "gpt", "value": (
                "<tool_call>\n"
                "{\"name\": \"search_flights\", \"arguments\": \"{\\\"from\\\": \\\"SFO\\\"}\"}\n"
                "</tool_call>"
            )},
        ],
        "tools": [
            {
                "name": "search_flights",
                "parameters": {"properties": {"from": {"type": "string"}}},
            }
        ],
    }

    converted = HermesReasoningAdapter.convert(example).to_dict()
    tool_msg = next(msg for msg in converted["conversations"] if "<tool_call>" in msg["value"])
    tool_call = extract_tool_call_from_message(tool_msg["value"])
    assert tool_call is not None
    assert tool_call["name"] == "search_flights"
    assert isinstance(tool_call["arguments"], dict)
    assert tool_call["arguments"]["from"] == "SFO"
