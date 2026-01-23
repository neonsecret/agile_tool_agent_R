"""
Demo for diffusion-based function call generation.
"""

import os
import torch
from transformers import AutoTokenizer

from inference import FunctionCallGenerator, GenerationConfig
from model.hybrid_model import HybridSmolLM
from data.schema import build_schema_template
from data.utils import resolve_mask_token, resolve_null_token, validate_mask_token_consistency
from data.budget_utils import build_fields_from_schema, print_budget_info, MIN_FIELD_BUDGET, DEFAULT_MAX_BUDGET
from data.config_utils import validate_and_adjust_config, get_model_kwargs, get_inference_kwargs, \
    print_device_capabilities
from data.device_utils import get_device
from inference_utils import load_config


def demo_inference():
    """Demo function showing how to use the generator with automatic budgeting."""
    print_device_capabilities()
    device = get_device()
    print(f"Using device: {device}")

    config = load_config()
    config = validate_and_adjust_config(config, device)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    inference_cfg = config.get("inference", {})
    inference_kwargs = get_inference_kwargs(config, device)

    steps = inference_cfg.get("steps", 4)
    temperature = inference_cfg.get("temperature", 0.0)
    cfg_scale = inference_cfg.get("cfg_scale", 0.0)
    reencode_every = inference_cfg.get("reencode_hidden_states_every", 0)
    max_seq_length = inference_kwargs.get("max_seq_len", 2048)
    use_torch_compile = inference_kwargs["use_torch_compile"]
    use_cuda_graph = inference_kwargs["use_cuda_graph"]

    data_cfg = config.get("data", {})
    mask_token_config = data_cfg.get("mask_token", None)
    mask_token_str, mask_token_id = resolve_mask_token(tokenizer, mask_token_config)

    null_token_config = data_cfg.get("null_token", None)
    null_token_str, null_token_id = resolve_null_token(tokenizer, null_token_config)

    dynamic_budget_cfg = data_cfg.get("dynamic_budget", {})
    max_field_budget = dynamic_budget_cfg.get(
        "max_tokens",
        data_cfg.get("mask_budget", DEFAULT_MAX_BUDGET),
    )
    min_field_budget = dynamic_budget_cfg.get("min_tokens", MIN_FIELD_BUDGET)
    extra_field_budget = dynamic_budget_cfg.get("extra_tokens", 0)
    budget_config = {
        "min_tokens": min_field_budget,
        "max_tokens": max_field_budget,
        "extra_tokens": extra_field_budget,
    }

    print(f"Using mask token: {mask_token_str} (ID: {mask_token_id})")
    print(f"Budget range: {min_field_budget}-{max_field_budget} tokens per field")
    print(f"Budget extra tokens: {extra_field_budget}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    training_cfg = config.get("training", {})
    checkpoint_path = training_cfg.get("checkpoint_path", "checkpoints/best_model/model.pt")

    model_kwargs = get_model_kwargs(config, device)
    model_kwargs['vocab_size'] = len(tokenizer)

    model = HybridSmolLM(**model_kwargs)

    try:
        model.to(device)
    except NotImplementedError:
        print("Model has meta/offloaded tensors, skipping explicit .to() call")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        checkpoint_state = checkpoint['model_state_dict']
        checkpoint_state = {k: v.to(device) for k, v in checkpoint_state.items()}
        
        if hasattr(model, 'load_trainable_state_dict'):
            missing_keys, unexpected_keys = model.load_trainable_state_dict(
                checkpoint_state, strict=False
            )
            num_loaded = len([k for k in checkpoint_state.keys() if k.startswith('diffusion_head.')])
            print(f"Loaded {num_loaded} trainable parameters from checkpoint")
        else:
            model_state = model.state_dict()
            filtered_state = {}
            loaded_keys = []
            
            for key, value in checkpoint_state.items():
                if key.startswith('diffusion_head.'):
                    if key in model_state and model_state[key].shape == value.shape:
                        filtered_state[key] = value
                        loaded_keys.append(key)
            
            if filtered_state:
                model.load_state_dict(filtered_state, strict=False)
                print(f"Loaded {len(loaded_keys)} diffusion head weights from checkpoint")
            else:
                print("Warning: No compatible weights found, using untrained model")

        if 'epoch' in checkpoint:
            print(f"Checkpoint info: epoch {checkpoint['epoch']}, eval loss: {checkpoint.get('eval_loss', 'N/A')}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using untrained model")

    model.eval()

    model.diffusion_head.set_mask_token_id(mask_token_id)

    if null_token_id is not None:
        model.diffusion_head.set_null_token_id(null_token_id)

    generator = FunctionCallGenerator(
        model,
        tokenizer,
        device,
        use_torch_compile=use_torch_compile,
        use_cuda_graph=use_cuda_graph,
        max_seq_len=max_seq_length,
        budget_config=budget_config,
    )

    print(f"Optimizations: torch.compile={use_torch_compile}, cuda_graph={use_cuda_graph}")

    prompt = "What's the weather in London?"

    tool_registry = {
        "get_weather": {
            "name": "get_weather",
            "parameters": {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["C", "F"],
                        "description": "Temperature units"
                    }
                }
            }
        },
        "search_web": {
            "name": "search_web",
            "parameters": {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                }
            }
        }
    }

    gen_config = GenerationConfig(
        steps=steps,
        temperature=temperature,
        cfg_scale=cfg_scale,
        show_steps=True,
        use_cuda_graph=use_cuda_graph,
        reencode_hidden_states_every=reencode_every,
    )

    print("\n" + "=" * 80)
    print("TOOL CALL GENERATION")
    print("=" * 80)

    tool_schema = tool_registry["get_weather"]
    fields = build_fields_from_schema(
        tool_schema,
        tokenizer,
        min_budget=min_field_budget,
        max_budget=max_field_budget,
        extra_budget=extra_field_budget,
    )

    print("\nAutomatic budget calculation:")
    print_budget_info(fields)

    template = build_schema_template(
        tokenizer=tokenizer,
        fields=fields,
        mask_token=mask_token_str,
        null_token=null_token_str,
        include_codeblock=False
    )

    validate_mask_token_consistency(
        model.diffusion_head.mask_token_id,
        template.mask_token_id,
        context=" in demo_inference()"
    )

    print(f"Scaffold template: {template.text}")

    output = generator.generate(
        prompt=prompt,
        template=template,
        config=gen_config,
        trace=True,
        tool_name="get_weather"
    )

    print(f"\n{'=' * 80}")
    print("RESULT")
    print("=" * 80)
    print(f"Generated text: {output.text}")
    print(f"Steps executed: {output.steps_executed}")


if __name__ == "__main__":
    demo_inference()
