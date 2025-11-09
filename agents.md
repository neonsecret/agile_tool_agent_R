# Bi-Modal Reasoning SLM: Technical Development Guide

## Executive Summary

This document outlines the technical architecture, implementation strategy, and best practices for developing a **reasoning bi-modal SLM with tool modality**—a small language model (e.g., Gemma-270M) that reasons jointly across textual and tool-specific modalities. The model will learn to select, invoke, and reason about tools natively through a dedicated tokenizer and multimodal reasoning framework.

---

## 1. Project Architecture Overview

### 1.1 Core Modalities

**Textual Modality:**
- Standard LLM token sequence modeling (conversation, reasoning chains, CoT)
- Leverages pre-trained embedding and transformer layers
- Responsible for natural language understanding and generation

**Tool Modality:**
- **Dedicated tokenizer** for tool schemas and specifications
- Tokens represent: function names, parameter types, constraints, return types, descriptions
- Enables structured, interpretable reasoning about API/tool capabilities
- **Key difference from ToolGen**: Tokenizes schema components (not single opaque token per tool)
- Supports zero-shot generalization to unseen tools via grammar/structure understanding

### 1.2 Unified Reasoning

- Joint embedding space where text tokens and tool tokens interact
- Cross-modal attention layers enable semantic reasoning across modalities
- Model learns to compose tool calls within natural language flow
- Single forward pass generates both reasoning (text) and actions (tool calls)

---

## 2. Technology Stack & Libraries

### 2.1 Core Libraries

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Base Model** | `transformers` (HuggingFace) | ≥4.40 | Model loading, tokenization, inference |
| **Fine-tuning** | `trl` (TRL) | ≥0.7.0 | SFTTrainer, GRPO trainer for RL |
| **Training Acceleration** | `unsloth` | ≥2024.11 | 2x faster training, memory optimization for SLMs |
| **Parameter Efficiency** | `peft` (PEFT) | ≥0.7.0 | LoRA adapters (7B+ models only) |
| **Inference Server** | `vllm` | ≥0.4.0+ | High-throughput serving, quantization |
| **Quantization** | `bitsandbytes` | ≥0.41 | 4-bit/8-bit quantization (7B+ models) |
| **Data Processing** | `datasets` | ≥2.14 | Dataset loading, preprocessing, caching |
| **RL Training** | `torch` | ≥2.1.0 | Core training loop, distributed training |
| **Constrained Decoding** | `outlines` or `xgrammar` | Latest | Token masking, grammar-based structured output |
| **Long Context** | Custom + `flash-attn` | ≥2.4 | Sliding window, FlashAttention-2 optimization |

### 2.2 Key Framework Choices

**Why TRL (Transformer Reinforcement Learning):**
- Native GRPO support without separate critic model
- Integrated SFTTrainer for supervised fine-tuning
- Built-in reward computation for verifiable outcomes
- Production-ready on diverse hardware

**Why vLLM for Inference:**
- Fastest open-source inference engine (PagedAttention)
- Quantization-aware (GPTQ, AWQ, bitsandbytes support)
- Logit bias / token masking support for constrained decoding
- Optimized for both small (270M) and large (7B+) models

**Training Strategy by Model Size:**
- **270M-2B models**: Full fine-tuning with Unsloth (recommended)
- **7B-8B models**: QLoRA (LoRA rank r=8-16) with bitsandbytes quantization
- Reason: Small models train fast enough; LoRA only needed for large models on limited hardware

---

## 3. Programming Guidelines

### 3.1 Code Style & Structure

**PEP 8 Compliance:**
- Follow PEP 8 style guide for all Python code
- Use meaningful variable/function names (descriptive, not abbreviated)
- Maximum line length: 100 characters (not 79 for readability with modern screens)
- Use type hints for function signatures

**Function Design:**
- **Keep functions small**: Max 50 lines per function
- **Single responsibility**: Each function does ONE thing
- **Decompose complexity**: Split large functions into smaller, testable units
- Example: Instead of 200-line `train()`, split into `load_data()`, `setup_model()`, `run_training_loop()`, `save_checkpoint()`

**Code Organization:**
- Group related functions into classes when appropriate
- Use modules to separate concerns (training, inference, data, evaluation)
- Avoid monolithic files (>500 lines suggests need for splitting)

### 3.2 Error Handling & Robustness

**Minimal try/except:**
- Only catch exceptions where you can meaningfully handle them
- Avoid bare `except:` clauses
- Don't silence errors—log them properly
- Prefer validation and early returns over try/except

```python
# BAD: Overly broad exception handling
try:
    data = load_data(path)
    model = train_model(data)
    save_model(model)
except:
    pass

# GOOD: Specific, meaningful handling
def load_data(path: str) -> Dataset:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return datasets.load_from_disk(path)
```

**Validation:**
- Validate inputs at function entry
- Use assertions for internal invariants (development only)
- Use explicit checks for user-facing code

### 3.3 Readability Priorities

**Clarity > Cleverness:**
- Prefer explicit over implicit
- Write self-documenting code (clear names, simple logic)
- Add comments only for "why", not "what"
- Avoid nested comprehensions beyond 2 levels

**Code Bloat:**
- Avoid premature abstraction
- Don't create classes/interfaces unless complexity justifies it
- Remove dead code immediately
- Use simple data structures (dicts, dataclasses) over custom classes when possible

**Example Structure:**
```python
def train_sft_phase1(
    model: PreTrainedModel,
    tokenizer: ToolTokenizer,
    dataset: Dataset,
    config: TrainingConfig
) -> TrainingMetrics:
    """Train phase 1a: tool selection.
    
    Args:
        model: Pre-trained base model
        tokenizer: Tool tokenizer with extended vocab
        dataset: Phase 1a dataset (tool selection examples)
        config: Training hyperparameters
        
    Returns:
        Training metrics (loss, accuracy per epoch)
    """
    trainer = setup_trainer(model, tokenizer, config)
    train_dataloader = prepare_dataloader(dataset, config.batch_size)
    
    metrics = run_training_loop(trainer, train_dataloader, config.epochs)
    save_checkpoint(model, config.output_dir)
    
    return metrics
```

---

## 4. Repository Structure Pattern

Reference: TinyAgent, Granite Function Calling projects (see idea_recap.md for full list)

```
bi-modal-slm/
├── README.md
├── agents.md (this file)
├── LICENSE
├── requirements.txt
├── setup.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py (Model wrapper, multimodal config)
│   │   ├── tool_tokenizer.py (Custom tool tokenizer implementation)
│   │   └── multimodal_fusion.py (Cross-modal attention layers)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py (SFTTrainer wrapper with curriculum)
│   │   ├── grpo_trainer.py (GRPO RL trainer)
│   │   ├── reward.py (Reward functions: tool accuracy, arg match, etc.)
│   │   └── datasets.py (Data loading, curriculum scheduling)
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py (vLLM-based inference wrapper)
│   │   ├── logit_bias.py (Token masking utilities)
│   │   ├── constrained_decoding.py (Outlines/XGrammar integration)
│   │   └── tool_executor.py (Execute parsed tool calls)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py (Tool accuracy, hallucination rate, compositionality)
│   │   ├── benchmarks.py (BFCL, custom tool calling benchmarks)
│   │   └── eval_loop.py (Evaluation pipeline)
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tool_schemas.py (Tool definition, schema parsing)
│   │   ├── dataset_builder.py (Synthetic data generation, curriculum)
│   │   └── augmentation.py (Data augmentation, system prompt variance)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py (Centralized config management)
│       ├── logging.py (Experiment tracking, wandb integration)
│       └── constants.py (Token mappings, special tokens)
│
├── configs/
│   ├── model_config.yaml (Model size, quantization, context)
│   ├── training_config.yaml (SFT + GRPO hyperparameters)
│   ├── tool_vocab.json (Tool tokenizer vocabulary)
│   └── system_prompts.json (5+ system prompt templates)
│
├── data/
│   ├── tool_schemas/ (Function definitions, API specs)
│   ├── training/ (SFT dataset phases)
│   │   ├── phase1_selection.jsonl
│   │   ├── phase2_arguments.jsonl
│   │   └── phase3_joint.jsonl
│   └── evaluation/ (Benchmark data, hold-out test sets)
│
├── scripts/
│   ├── prepare_data.py (Generate SFT datasets, curriculum)
│   ├── train_sft.py (Supervised fine-tuning pipeline)
│   ├── train_grpo.py (GRPO reinforcement learning)
│   ├── evaluate.py (Comprehensive evaluation suite)
│   ├── infer.py (Inference with masking options)
│   └── export_model.py (Merge LoRA, quantize, export)
│
├── notebooks/
│   ├── 01_eda.ipynb (Data exploration, tool schema analysis)
│   ├── 02_training_curves.ipynb (Training monitoring, ablations)
│   └── 03_inference_benchmark.ipynb (Latency, throughput analysis)
│
├── tests/
│   ├── test_tokenizer.py (Tool tokenizer correctness)
│   ├── test_training.py (Training loop, loss computation)
│   ├── test_inference.py (Inference + masking logic)
│   └── test_evaluation.py (Metric computation)
│
└── docs/
    ├── tool_modality.md (Detailed tool tokenizer design)
    ├── multimodal_fusion.md (Cross-modal architecture)
    ├── training_pipeline.md (SFT curriculum + GRPO details)
    ├── inference_guide.md (vLLM setup, masking, serving)
    └── paper_roadmap.md (Research contribution path)
```

---

## 4. Tool Tokenizer Implementation

### 4.1 Design Principles

**Goal**: Represent tool schemas as interpretable token sequences, NOT as single opaque tokens (key difference from ToolGen).

**Core Tokens:**
```
FUNCTION_NAME: "get_weather"
PARAMETER: "location"
TYPE: STRING, INT, BOOL, ENUM, ARRAY, OBJECT, FLOAT
CONSTRAINT: REQUIRED, OPTIONAL, DEFAULT
DESCRIPTION: "A city name"
RETURN_TYPE: JSON, STRING, ARRAY
ENUM_VALUES: ["C", "F", "K"]
```

### 4.2 Tokenizer Architecture

```python
# Pseudo-code structure
class ToolTokenizer:
    def __init__(self, tool_vocab_size=500):
        # Base vocab: 32k-50k (standard LLM tokens)
        # Tool vocab: 500 tokens for tool-specific concepts
        # Total: 32.5k-50.5k vocab size
        
        self.special_tokens = {
            '<tool_start>': token_id,
            '<tool_end>': token_id,
            '<func_name>': token_id,
            '<param>': token_id,
            '<type>': token_id,
            '<required>': token_id,
            '<optional>': token_id,
            # ... etc
        }
        
        self.tool_vocab = {
            # Predefined tool concepts
            'get_weather': token_id,
            'string': token_id,
            'location': token_id,
            # ... etc
        }
    
    def tokenize_tool_schema(self, tool_def):
        """
        Input: {"name": "get_weather", "params": [...], "returns": "json"}
        Output: [token_1, token_2, ..., token_n]
        """
        tokens = [self.special_tokens['<tool_start>']]
        tokens.append(self.encode_function_name(tool_def['name']))
        # ... encode parameters, types, constraints
        tokens.append(self.special_tokens['<tool_end>'])
        return tokens
```

### 4.3 Vocabulary Extension Strategy

```
Original Gemma-270M vocab: ~250K tokens
After extension: ~250.5K tokens (minimal impact)

Tool tokens added: 500 (reserved block in embedding table)
```

**Implementation in HuggingFace:**
```python
from transformers import AutoTokenizer

# Load base tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Add tool-specific tokens
tool_tokens = [
    '<tool_start>', '<tool_end>',
    '<func_name>', '<param>', '<type>',
    '<required>', '<optional>', ...
]
tokenizer.add_tokens(tool_tokens)

# Fine-tune embeddings for new tokens during training
model.resize_token_embeddings(len(tokenizer))
```

### 4.4 Schema to Tokens Mapping Example

```
Tool Definition (JSON):
{
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "location": {"type": "string", "description": "City name", "required": true},
        "unit": {"type": "enum", "enum": ["C", "F", "K"], "default": "C"}
    },
    "returns": "json"
}

Token Sequence:
<tool_start> 
<func_name> get_weather
<description> Get weather for a city
<param> location <type> string <required> true <description> City name
<param> unit <type> enum [C, F, K] <default> C
<return_type> json
<tool_end>
```

---

## 5. Training Pipeline

### 5.1 Phase 1: Supervised Fine-Tuning (SFT) - Curriculum Learning

#### Stage 1a: Tool Selection
- **Objective**: Given query + tool list, output correct tool(s)
- **Data**: 10-15k diverse examples
- **Format**:
  ```
  System: "Select the appropriate tool(s) for this task."
  User: "What's the weather in Boston?"
  Tools: [get_weather, get_news, search_web]
  Model Output: get_weather
  ```
- **Loss**: Cross-entropy (tool classification)
- **Hyperparameters**: 
  - 270M-2B: LR=2e-4, Epochs=3, Full fine-tuning with Unsloth
  - 7B+: LR=1e-4, Epochs=3, QLoRA (r=8-16)

#### Stage 1b: Argument Generation
- **Objective**: Given query + selected tool schema, generate correct arguments
- **Data**: 10-15k examples (tool selection already correct)
- **Format**:
  ```
  System: "Generate function arguments matching the schema."
  User: "Get weather in Boston in Fahrenheit"
  Tool Schema: <tokenized get_weather schema>
  Model Output: get_weather(location="Boston", unit="F")
  ```
- **Loss**: Token-level cross-entropy
- **Key**: Model learns argument type matching, value formatting

#### Stage 1c: Joint Reasoning
- **Objective**: Full end-to-end prompt → tool call
- **Data**: 30-40k mixed examples (combining 1a + 1b + new complex cases)
- **Format**:
  ```
  System: "You are a helpful assistant with access to tools."
  User: "I need weather for Boston and news about AI."
  Available Tools: <tokenized schemas for get_weather, get_news>
  
  Model Output:
  "I'll get the weather and news for you."
  [get_weather(location="Boston", unit="F")]
  "The weather is sunny."
  [get_news(topic="AI")]
  "Here's the latest AI news..."
  ```
- **Loss Weighting**: 
  - Initial (epoch 1-3): tool_loss=0.7, args_loss=0.3
  - Mid (epoch 4-6): tool_loss=0.5, args_loss=0.5
  - Final (epoch 7-10): tool_loss=0.4, args_loss=0.6

### 5.2 Phase 2: Reinforcement Learning (GRPO)

#### GRPO Setup

```python
# TRL GRPO Trainer with Unsloth (270M-2B models)
from trl import GRPOTrainer, GRPOConfig
from unsloth import FastLanguageModel

# Load model with Unsloth acceleration
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-2b",
    max_seq_length=2048,
    dtype=None,  # Auto-detect optimal dtype
    load_in_4bit=False,  # Full precision for small models
)

grpo_config = GRPOConfig(
    output_dir="./grpo_checkpoint",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    num_train_epochs=5,
    max_prompt_length=512,
    max_completion_length=256,
    
    # GRPO-specific
    num_generations_per_prompt=4,  # rollouts per prompt
    temperature=0.7,
    top_p=0.95,
    
    # KL penalty to stay near SFT checkpoint
    kl_penalty=0.05,  # 0.05-0.1 recommended
    
    # Logging
    logging_steps=10,
    report_to=["wandb"],
)

trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=rl_dataset,
    reward_fn=compute_reward,
)

trainer.train()

# For 7B+ models, use QLoRA:
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,  # LoRA rank
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_alpha=16,
#     lora_dropout=0.05,
# )
```

#### Reward Function Design

```python
def compute_reward(output, ground_truth):
    """
    Compute structured reward for tool calling.
    
    Components:
    - Tool selection correctness
    - Argument correctness
    - Format validity
    - Reasoning quality (CoT before action)
    """
    
    score = 0.0
    
    # 1. Tool Selection Reward (60% weight)
    predicted_tools = parse_tools(output)
    correct_tools = ground_truth['tools']
    
    tool_match = len(set(predicted_tools) & set(correct_tools)) / max(len(correct_tools), 1)
    score += 0.6 * (2.0 * tool_match - 1.0)  # Scale to [-1, 1]
    
    # 2. Argument Correctness (40% weight)
    for i, tool in enumerate(predicted_tools):
        pred_args = output['tools'][i]['args']
        true_args = ground_truth['tools'][i]['args']
        
        arg_match = compute_arg_similarity(pred_args, true_args)
        score += 0.4 * arg_match / len(predicted_tools) if predicted_tools else 0
    
    # 3. Penalty for Hallucination
    if has_invalid_tools(output):
        score -= 0.5
    
    # 4. Bonus for Correct Format
    if is_valid_json(output):
        score += 0.1
    
    return score
```

#### Training Dynamics

- **Batch 1-10**: Model learns to avoid hallucination
- **Batch 11-30**: Model improves argument type matching
- **Batch 31-50+**: Refinement, edge case handling, multi-step reasoning

**Convergence Criteria:**
- Validation tool accuracy: ≥95% (without masking)
- Argument accuracy: ≥85%
- Zero hallucination on valid tool list: <2%

### 5.3 Data Curation Best Practices

**Quality > Quantity:**
- 30-60k curated examples better than 500k noisy
- Diverse tool categories (math, web, database, time, etc.)
- Realistic parameter combinations
- Edge cases: optional parameters, defaults, type mismatches

**Curriculum Strategy:**
- Easy tasks (1-2 parameters, simple types) → Hard tasks (5+ params, nested objects)
- Low tool count (3 tools) → High tool count (20+ tools)
- System prompt variations (add at 10-20% of data)

---

## 6. Inference & Deployment

### 6.1 vLLM Serving Setup

```python
# inference.py
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="path/to/finetuned/model",  # Your trained model
    quantization="awq",  # Optional: "gptq", "awq", or None
    dtype="float16",
    max_num_seqs=32,
    tensor_parallel_size=1,
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
)

# Inference
response = llm.generate(
    prompts=[prompt],
    sampling_params=sampling_params,
)

print(response[0].outputs[0].text)

# For 7B+ models with LoRA adapters:
# from vllm.lora.request import LoRARequest
# llm = LLM(model="base/model", enable_lora=True, max_loras=4)
# lora_request = LoRARequest("adapter-name", 1, "/path/to/lora")
# response = llm.generate(prompts=[prompt], lora_request=lora_request, ...)
```

### 6.2 Logit Bias / Token Masking

**Use Case**: Restrict model to output valid tool names only.

```python
from vllm import SamplingParams

# Option 1: Soft Masking (Logit Bias)
valid_tool_tokens = [token_id_1, token_id_2, ...]

sampling_params = SamplingParams(
    temperature=0.7,
    logits_processors=[
        lambda logits: apply_logit_bias(logits, valid_tool_tokens, bias=5.0)
    ]
)

# Option 2: Hard Masking (Constrained Decoding with Outlines)
from outlines import generate

generator = generate.text(
    llm, 
    regex="^(tool1|tool2|tool3)\\(.*\\)$"  # Constrain to valid format + tools
)

output = generator(prompt)
```

### 6.3 Evaluation Protocol (Masked vs. Unmasked)

```python
# Evaluation loop
results_unmasked = []
results_masked = []

for test_example in test_set:
    # Without masking
    output_unmasked = llm.generate(test_example['prompt'])
    results_unmasked.append(evaluate_output(output_unmasked))
    
    # With masking
    output_masked = llm.generate(
        test_example['prompt'],
        logits_processors=[mask_invalid_tools]
    )
    results_masked.append(evaluate_output(output_masked))

# Metrics
print("Without masking:")
print(f"  Tool accuracy: {compute_tool_accuracy(results_unmasked)}")
print(f"  Hallucination rate: {compute_hallucination(results_unmasked)}")

print("With masking:")
print(f"  Tool accuracy: {compute_tool_accuracy(results_masked)}")
print(f"  Argument accuracy: {compute_arg_accuracy(results_masked)}")

# Analysis: Delta reveals model learning
delta = results_unmasked['tool_acc'] - results_masked['tool_acc']
print(f"Model dependency on masking: {delta:.2%}")
```

---

## 7. Context Window Extension

### 7.1 Sliding Window Attention

**Goal**: Extend context from 8K (Gemma baseline) → 32K

```python
# In model config
{
    "context_window": 32768,
    "attention_type": "sliding_window",
    "sliding_window_size": 4096,  # Local window per token
    "attention_sinks": 8,  # Preserve first 8 tokens for global attention
}
```

**Benefits**: O(n·w) memory vs O(n²), works with full fine-tuning, no special training

### 7.2 LongLoRA (Optional, 7B+ models)

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2b",
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# Train with extended sequences using shift-short attention
model.config.use_s2_attn = True
```

---

## 8. System Prompt Following

**Training Data Composition:**
- 80% with primary format
- 20% with alternate formats

**System Prompt Variants** (include 5-10 in training, see configs/system_prompts.json):
1. "You are a helpful assistant with access to tools. Output: tool_name(param1=value1, ...)"
2. "Generate API calls in JSON: {\"tool\": \"name\", \"args\": {...}}"
3. "Think step-by-step. Format: [tool_name(args)]"
4. "Be concise. Direct function calls without explanation."
5. "Ensure all required parameters present."

---

## 9. Evaluation Metrics & Benchmarks

### 9.1 Core Metrics

| Metric | Target |
|--------|--------|
| Tool Accuracy (no mask) | ≥95% |
| Argument Accuracy | ≥85% |
| Hallucination Rate | <2% |
| Format Compliance | ≥98% |
| Zero-shot Generalization | ≥60-70% |
| Compositionality (2-3 step) | ≥50-60% |

### 9.2 Benchmarks

**Primary:**
- Berkeley Function Calling Leaderboard (BFCL)
- ComplexFuncBench

**Custom:**
- Tool selection: 1k examples
- Argument generation: 1k examples
- Unseen tools: 200 examples
- Tool chains: 500 examples (2-3 step)

---

## 10. Key Technical Decisions

### 10.1 Why Full Fine-tuning for SLMs?
- 270M-2B models: Full training is fast (<12 hours on single GPU)
- Better parameter utilization: All weights learn tool modality
- No adapter overhead at inference
- LoRA only justified for 7B+ models on limited hardware

### 10.2 Why Unsloth?
- 2x faster than standard HuggingFace training
- Memory-efficient full fine-tuning on consumer GPUs
- Zero code changes to training loop

### 10.3 Why Sliding Window (not 1M+ immediately)?
- 32K sufficient for tool calling (most queries <2K)
- Minimal complexity increase
- Extend to 100K+ in Phase 2 if needed

---

## 11. Common Pitfalls & Mitigation

| Pitfall | Mitigation |
|---------|-----------|
| Poor dataset quality | Curate 30-60k examples, manual review 10% |
| Curriculum too aggressive | Gradual weighting schedule |
| Reward hacking in GRPO | Robust reward design, evaluate unmasked |
| Overfitting to train tools | Hold-out 20-30% tools, test unseen schemas |
| Masking becomes crutch | Always evaluate unmasked; track delta |
| System prompt brittleness | Include 5-10 variants in training |

---

## 12. Expected Results

| Task | Target (270M) |
|------|---------------|
| Single-turn tool calling (no mask) | 85-90% |
| Argument accuracy | 80-85% |
| Zero-shot tools | 55-65% |
| Tool chaining (2-step) | 50-60% |
| Latency (on-device) | <50ms |
| Memory footprint (4-bit) | <1GB |

---

## 13. Resources & References

See `idea_recap.md` for comprehensive research references.

**Key Libraries:**
- HuggingFace Transformers: https://huggingface.co/transformers/
- TRL: https://huggingface.co/docs/trl/
- Unsloth: https://github.com/unslothai/unsloth
- vLLM: https://docs.vllm.ai/
- Outlines: https://outlines-dev.github.io/

**Model Checkpoints:**
- Base: `google/gemma-2b` or `google/gemma-2-2b`
- Alternative: `meta-llama/Llama-2-7b` (with LoRA)

---

**Status**: Phase 1 - Architecture & Data Preparation