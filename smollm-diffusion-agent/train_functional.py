"""
S3 denoising strategy and functional evaluation.

Implements the actual inference strategy used during evaluation.
"""

import torch
import random
from typing import Dict, List, Optional, Tuple
from accelerate import Accelerator
from torch.utils.data import Subset

from data.device_utils import empty_cache
from data.smollm3_prompting import encode_tool_call_wrapper
from train_utils import safe_unwrap_model


def s3_denoise(model, hidden_states, input_ids, labels, scaffold_mask, num_steps=4):
    """
    S3-style top-K confidence denoising (matches inference.py strategy).

    This uses the ACTUAL inference strategy:
    - Timestep based on remaining masks
    - Top-K confidence-based token selection
    - Iterative refinement over multiple steps
    """
    device = hidden_states.device
    mask_token_id = model.diffusion_head.mask_token_id

    diffusion_head_dtype = next(model.diffusion_head.parameters()).dtype
    hidden_states = hidden_states.to(dtype=diffusion_head_dtype)

    current_tokens = input_ids.clone()
    current_tokens[scaffold_mask] = mask_token_id

    total_masks = scaffold_mask.sum().item()
    budget = max(1, int(total_masks / num_steps))
    initial_variable_count = int(total_masks)
    prompt_mask = ~scaffold_mask

    for step in range(num_steps):
        mask_positions = current_tokens == mask_token_id
        mask_positions = mask_positions & scaffold_mask
        mask_indices = torch.nonzero(mask_positions[0], as_tuple=False).squeeze(-1)

        if mask_indices.numel() == 0:
            break

        if mask_indices.dim() == 0:
            mask_indices = mask_indices.unsqueeze(0)

        remaining = int(mask_indices.numel())
        t_val = float(remaining) / float(initial_variable_count)
        t = torch.full((current_tokens.shape[0],), t_val, device=device, dtype=torch.float)

        logits = model.diffusion_head.predict(
            hidden_states,
            current_tokens,
            t,
            scaffold_mask=scaffold_mask,
            prompt_mask=prompt_mask,
        )
        predictions = torch.argmax(logits, dim=-1)

        log_probs = torch.log_softmax(logits[0, mask_indices], dim=-1)
        mask_conf = log_probs.gather(-1, predictions[0, mask_indices].unsqueeze(-1)).squeeze(-1)

        remaining = mask_indices.numel()
        remaining_steps = num_steps - step
        if remaining_steps <= 1:
            k = remaining
        else:
            k = min(budget, remaining)

        topk = torch.topk(mask_conf, k)
        selected = mask_indices[topk.indices]

        current_tokens[0, selected] = predictions[0, selected]

    return current_tokens


def _get_processed_example(eval_dataset, idx: int) -> Optional[Dict]:
    base_dataset = eval_dataset
    base_idx = idx
    if isinstance(eval_dataset, Subset):
        base_dataset = eval_dataset.dataset
        base_idx = eval_dataset.indices[idx]

    processed = getattr(base_dataset, "processed_examples", None)
    if processed is None or base_idx >= len(processed):
        return None
    return processed[base_idx]


def _truncate_text(text: str, max_len: int = 160) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _extract_field_texts(
    tokenizer,
    token_seq: torch.Tensor,
    template,
    prompt_len: int,
    prefix_len: int,
    null_token_id: Optional[int],
) -> List[Tuple[str, str]]:
    field_texts = []
    for segment in template.field_segments:
        token_ids = []
        for pos in segment.value_positions:
            idx = prompt_len + prefix_len + pos
            if idx >= token_seq.numel():
                continue
            token_id = int(token_seq[idx].item())
            if token_id < 0:
                continue
            if null_token_id is not None and token_id == null_token_id:
                continue
            token_ids.append(token_id)
        text = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        field_texts.append((segment.name, text))
    return field_texts


def _compute_prompt_len(total_len: int, template_len: int, prefix_len: int, suffix_len: int) -> int:
    return max(0, total_len - (template_len + prefix_len + suffix_len))


def functional_evaluation(model, eval_dataset, tokenizer, accelerator, num_examples=5):
    """
    Evaluate model using S3-style inference (matches actual inference.py strategy).

    This tests the REAL inference approach:
    - Timestep based on remaining masks
    - Top-K confidence-based selection
    - Iterative refinement
    """
    model.eval()

    accelerator.print("\n" + "=" * 80)
    accelerator.print("FUNCTIONAL EVALUATION - S3 Denoising Strategy")
    accelerator.print("=" * 80)

    indices = random.sample(range(len(eval_dataset)), min(num_examples, len(eval_dataset)))

    total_exact_matches = 0
    total_token_accuracy = 0
    total_tokens = 0

    unwrapped_model = safe_unwrap_model(model)
    diffusion_num_steps = unwrapped_model.diffusion_head.num_steps

    with torch.no_grad():
        for i, idx in enumerate(indices):
            example = eval_dataset[idx]

            input_ids = example['input_ids'].unsqueeze(0).to(accelerator.device)
            attention_mask = example['attention_mask'].unsqueeze(0).to(accelerator.device)
            scaffold_mask = example['scaffold_mask'].unsqueeze(0).to(accelerator.device)
            labels = example['labels'].unsqueeze(0).to(accelerator.device)

            if scaffold_mask.sum() == 0:
                continue

            outputs = unwrapped_model.base_llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]

            predicted_tokens = s3_denoise(
                unwrapped_model,
                hidden_states,
                input_ids,
                labels,
                scaffold_mask,
                num_steps=diffusion_num_steps
            )

            masked_positions = scaffold_mask[0].cpu()
            true_tokens = labels[0][masked_positions].cpu()
            pred_tokens = predicted_tokens[0][masked_positions].cpu()

            valid_mask = (true_tokens >= 0) & (true_tokens < len(tokenizer))
            true_tokens_valid = true_tokens[valid_mask]
            pred_tokens_valid = pred_tokens[valid_mask]

            null_token_id = unwrapped_model.diffusion_head.null_token_id
            if null_token_id is not None:
                true_tokens_valid = true_tokens_valid[true_tokens_valid != null_token_id]
                pred_tokens_valid = pred_tokens_valid[pred_tokens_valid != null_token_id]

            min_len = min(len(true_tokens_valid), len(pred_tokens_valid))
            if min_len == 0:
                continue

            true_tokens_aligned = true_tokens_valid[:min_len]
            pred_tokens_aligned = pred_tokens_valid[:min_len]

            correct_tokens = (true_tokens_aligned == pred_tokens_aligned).sum().item()
            num_tokens = min_len
            token_accuracy = correct_tokens / num_tokens if num_tokens > 0 else 0

            total_token_accuracy += correct_tokens
            total_tokens += num_tokens

            exact_match = torch.all(true_tokens_aligned == pred_tokens_aligned).item()
            total_exact_matches += exact_match

            true_tokens_list = true_tokens_aligned.tolist()
            pred_tokens_list = pred_tokens_aligned.tolist()

            true_masked_text = tokenizer.decode(true_tokens_list, skip_special_tokens=False)
            pred_masked_text = tokenizer.decode(pred_tokens_list, skip_special_tokens=False)

            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)

            accelerator.print(f"\n--- Example {i + 1} ---")
            accelerator.print(f"Input (first 200 chars):\n  {input_text[:200]}...")
            accelerator.print(f"\nMasked positions: {masked_positions.sum().item()} tokens")
            accelerator.print(f"Ground Truth (masked): {true_masked_text}")
            accelerator.print(f"Predicted (masked):    {pred_masked_text}")
            accelerator.print(f"Token Accuracy: {token_accuracy:.2%} ({correct_tokens}/{num_tokens})")
            accelerator.print(f"Exact Match: {'✓' if exact_match else '✗'}")

            processed = _get_processed_example(eval_dataset, idx)
            if processed is not None:
                template = processed.get("template")
                tool_name = processed.get("tool_name")
                if template is not None and tool_name:
                    wrapper = encode_tool_call_wrapper(tokenizer, tool_name)
                    prefix_len = len(wrapper.prefix_ids)
                    suffix_len = len(wrapper.suffix_ids)
                    total_len = int(input_ids.shape[1])
                    prompt_len = _compute_prompt_len(
                        total_len,
                        len(template.tokens),
                        prefix_len,
                        suffix_len,
                    )
                    labels_cpu = labels[0].detach().cpu()
                    pred_cpu = predicted_tokens[0].detach().cpu()
                    true_fields = _extract_field_texts(
                        tokenizer,
                        labels_cpu,
                        template,
                        prompt_len,
                        prefix_len,
                        null_token_id,
                    )
                    pred_fields = _extract_field_texts(
                        tokenizer,
                        pred_cpu,
                        template,
                        prompt_len,
                        prefix_len,
                        null_token_id,
                    )
                    accelerator.print("Field values:")
                    for (field_name, gt_text), (_, pred_text) in zip(true_fields, pred_fields):
                        gt_clean = _truncate_text(gt_text)
                        pred_clean = _truncate_text(pred_text)
                        accelerator.print(f"  {field_name}: gt={gt_clean} | pred={pred_clean}")

    accelerator.print("\n" + "-" * 80)
    empty_cache(accelerator.device)
    if total_tokens > 0:
        overall_token_acc = total_token_accuracy / total_tokens
        overall_exact_match = total_exact_matches / len(indices)

        accelerator.print(f"Overall Statistics ({len(indices)} examples):")
        accelerator.print(f"  Token-level Accuracy: {overall_token_acc:.2%} ({total_token_accuracy}/{total_tokens})")
        accelerator.print(f"  Exact Match Rate: {overall_exact_match:.2%} ({total_exact_matches}/{len(indices)})")

        return {
            "functional/token_accuracy": overall_token_acc,
            "functional/exact_match_rate": overall_exact_match,
        }
    else:
        accelerator.print("No masked tokens found in examples")
        return {}
