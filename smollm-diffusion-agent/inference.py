import torch
from model.hybrid_model import HybridSmolLM
from transformers import AutoTokenizer
import json


def inference():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load Model
    model = HybridSmolLM()
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

    # 2. Simulate User Query & AR Decision (Pre-computed for this demo)
    # Query: "Weather in London"
    # Model (AR) says: <|tool_name:weather|>

    # 3. Construct Scaffold (Python side)
    schema = {"loc": "<MASK>"}  # Simplified
    scaffold_str = json.dumps(schema)

    prompt = "<|im_start|>user\nWeather in London<|im_end|>\n<|im_start|>assistant\n" + scaffold_str

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Identify mask positions
    # In a real app, you'd map these robustly.
    # Here we just pretend the last few tokens are the mask.
    # Let's say we want to predict the token at index -2

    # 4. Diffusion Generation
    # Reverse diffusion: Start from random noise (or masking) and denoise?
    # The code in diffusion_head assumes we pass in a step `t` and it predicts the token.
    # Usually diffusion inference starts at T=Max and goes to T=0.

    print("Running Diffusion Inference...")

    # We iterate 4 -> 3 -> 2 -> 1 -> 0
    current_tokens = input_ids.clone()  # Start with masks

    mask_idx = -2  # Pretend mask is at -2

    with torch.no_grad():
        # Get base embeddings ONCE (if we assume they don't change much, or re-run if they do)
        # The hybrid model does this internally.
        pass

    # Simplest inference loop:
    for t in reversed(range(4)):
        print(f"Denoising step {t}...")
        t_tensor = torch.tensor([t], device=device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            diffusion_steps=t_tensor
        )

        logits = outputs["logits"]
        predicted_id = logits[0, mask_idx].argmax().item()
        print(f"Step {t} prediction: {tokenizer.decode([predicted_id])}")

        # In a real diffusion loop, you'd mix this prediction with the previous state.


if __name__ == "__main__":
    inference()
