"""
System prompt variations for training augmentation.

These prompts are randomly sampled during training to prevent overfitting
to a single system prompt pattern. 10% of the time, no system prompt is used.

Usage:
    from data.system_prompts import get_random_system_prompt
    
    # Returns None 10% of the time, otherwise a random prompt
    system_prompt = get_random_system_prompt()
"""

import random

SYSTEM_PROMPTS = [
    # Standard helpful assistant
    "You are a helpful assistant with access to tools.",

    # Function-focused
    "You are an AI assistant that can use tools to help users. When appropriate, use available tools to provide accurate information.",

    # Concise
    "You are a helpful AI assistant. You have access to various tools and functions.",

    # Detailed
    "You are an advanced AI assistant designed to help users with their questions and tasks. You have access to a variety of tools and functions that you can use when needed to provide accurate and helpful responses.",

    # Professional
    "You are a professional AI assistant with tool-calling capabilities. Use available tools when they can help provide better answers.",

    # Conversational
    "Hi! I'm an AI assistant here to help you. I can chat with you or use tools to get information when needed.",

    # Technical
    "You are an AI assistant equipped with function calling capabilities. Analyze user queries and determine whether to respond directly or utilize available tools.",

    # Minimal
    "You are an assistant with tool access.",

    # Instruction-focused
    "You are a helpful assistant. When a user asks a question, first decide if you need to use a tool. If you do, select the appropriate tool and provide the necessary parameters.",

    # Friendly
    "You're a friendly AI assistant! I can chat with you or use special tools to help answer your questions.",

    # No-nonsense
    "AI assistant with tool capabilities. Use tools when necessary.",

    # Detailed with reasoning
    "You are an intelligent AI assistant with access to tools. Think carefully about each query to decide if you should answer directly, explain your reasoning process, or use tools to gather information.",
]


def get_random_system_prompt(none_probability: float = 0.1) -> str:
    """
    Get a random system prompt for training augmentation.
    
    Args:
        none_probability: Probability of returning None (no system prompt)
    
    Returns:
        Random system prompt string, or None
    """
    if random.random() < none_probability:
        return None
    return random.choice(SYSTEM_PROMPTS)


def format_system_message(system_prompt: str = None) -> str:
    """
    Format system prompt with proper chat template markers.
    
    Args:
        system_prompt: System prompt text, or None to omit
    
    Returns:
        Formatted string with <|im_start|> markers, or empty string if None
    """
    if system_prompt is None:
        return ""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
