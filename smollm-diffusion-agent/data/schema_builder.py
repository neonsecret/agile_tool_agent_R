from dataclasses import dataclass
from typing import List, Sequence, Tuple
from transformers import PreTrainedTokenizerBase

from .utils import resolve_mask_token


@dataclass(frozen=True)
class FieldSegment:
    """Metadata describing the token range allocated to a schema field."""
    name: str
    start: int
    end: int
    value_positions: Tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.value_positions)


@dataclass(frozen=True)
class SchemaTemplate:
    """
    Encapsulates the tokenised scaffold ready to be injected into the
    diffusion generation process.
    """
    tokens: Tuple[int, ...]
    field_segments: Tuple[FieldSegment, ...]
    text: str
    mask_token_id: int
    mask_token: str

    def num_variable_tokens(self) -> int:
        return sum(segment.length for segment in self.field_segments)


def _encode_append(
        tokenizer: PreTrainedTokenizerBase, buffer: List[int], text_parts: List[str], text: str
) -> None:
    if not text:
        return
    text_parts.append(text)
    buffer.extend(tokenizer.encode(text, add_special_tokens=False))


def build_schema_template(
        tokenizer: PreTrainedTokenizerBase,
        fields: Sequence[Tuple[str, int]],  # (field_name, token_budget)
        mask_token: str,
        include_codeblock: bool = True,
        indent: str = "  ",
) -> SchemaTemplate:
    """
    Construct a schema template for the provided fields.
    """
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token)

    tokens: List[int] = []
    text_parts: List[str] = []
    field_segments: List[FieldSegment] = []

    if include_codeblock:
        _encode_append(tokenizer, tokens, text_parts, "```json\n")

    _encode_append(tokenizer, tokens, text_parts, "{\n")

    for index, (field, budget) in enumerate(fields):
        prefix = f'{indent}"{field}": "'
        suffix = '"' + (",\n" if index < len(fields) - 1 else "\n")

        _encode_append(tokenizer, tokens, text_parts, prefix)

        start_index = len(tokens)
        value_positions: List[int] = []

        for _ in range(budget):
            value_positions.append(len(tokens))
            tokens.append(mask_token_id)

        end_index = len(tokens)

        _encode_append(tokenizer, tokens, text_parts, suffix)

        field_segments.append(
            FieldSegment(
                name=field,
                start=start_index,
                end=end_index,
                value_positions=tuple(value_positions),
            )
        )

    _encode_append(tokenizer, tokens, text_parts, "}")

    if include_codeblock:
        _encode_append(tokenizer, tokens, text_parts, "\n```")

    template_text = "".join(text_parts)

    return SchemaTemplate(
        tokens=tuple(tokens),
        field_segments=tuple(field_segments),
        text=template_text,
        mask_token_id=mask_token_id,
        mask_token=mask_token_str,
    )
