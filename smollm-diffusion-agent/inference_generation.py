"""
Generation-specific operations for inference.

Handles autoregressive generation, tool selection, and chat mode.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from transformers import AutoTokenizer

from data.smollm3_prompting import (
    apply_smollm3_chat_template,
    encode_tool_call_wrapper,
    parse_first_tool_call,
)


class GenerationOperations:
    """Handles all generation operations for the inference pipeline."""

    def __init__(
            self,
            model,
            tokenizer: AutoTokenizer,
            device: torch.device,
            system_message_tool: str = "/no_think",
            system_message_chat: str = "/think",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._system_message_tool = system_message_tool
        self._system_message_chat = system_message_chat

    def _build_messages(self, prompt: str, system_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    def _generate_base(
            self,
            prompt: str,
            tools: Optional[Sequence[Dict[str, Any]]] = None,
            system_message: Optional[str] = None,
            max_new_tokens: int = 256,
            do_sample: bool = False,
            temperature: float = 0.0,
            top_p: float = 1.0,
    ) -> Tuple[str, str]:
        """
        Generate using the frozen base model with SmolLM3's official chat template.
        
        Returns:
            (generated_text_raw, generated_text_clean)
        """
        if system_message is None:
            system_message = self._system_message_chat

        messages = self._build_messages(prompt, system_message=system_message)
        prompt_ids_list = apply_smollm3_chat_template(
            self.tokenizer,
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
        )
        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_len = prompt_ids.shape[1]
        attention_mask = torch.ones_like(prompt_ids, dtype=torch.long, device=self.device)

        generate_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        output_ids = self.model.base_llm.generate(**generate_kwargs)

        gen_ids = output_ids[0, prompt_len:]
        generated_raw = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generated_clean = self.tokenizer.decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        del attention_mask
        return generated_raw, generated_clean

    def generate_chat(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Standard autoregressive generation for chat mode.
        
        Args:
            prompt: User query
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated response
        """
        raw, clean = self._generate_base(
            prompt,
            tools=None,
            system_message=self._system_message_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        del raw
        return clean.strip()

    def generate_think(self, prompt: str, max_new_tokens: int = 512) -> str:
        raw, clean = self._generate_base(
            prompt,
            tools=None,
            system_message=self._system_message_chat,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        del raw
        return clean.strip()

    def select_tool_call(
            self, prompt: str, tool_registry: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use SmolLM3's official tool calling template to decide whether to call a tool.

        Returns parsed tool call dict (with "name" and "arguments") or None.
        """
        tools_list = list(tool_registry.values())
        generated_raw, _ = self._generate_base(
            prompt,
            tools=tools_list,
            system_message=self._system_message_tool,
            max_new_tokens=256,
            do_sample=False,
        )
        return parse_first_tool_call(generated_raw)
