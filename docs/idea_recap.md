<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# So to recap, I want to create a reasoning bi-modal SLM, which will reason in both it's modalities, where besides textual modality there will be a tool one (with a separate tokenizer built for tools). It could use Logit Bias for Token Masking during inference as an option to increase accuracy. Also there will be sft+grpo (with nuances I mentioned). Also strict system prompt following. Also potentially use sliding window for a bigger context (not necessarily 1m+ but just bigger then vanilla).

Summarize our conversation into a document, include all relevant research as sources, and all other potential improvements we mentioned. Also include your tips and nuances too.

Here’s a comprehensive summary document outlining your proposed research and implementation plan for a reasoning bi-modal Small Language Model (SLM) with dual modalities (text and tool), covering research context, technical pipeline, best practices, implementation tips, and recommended evaluation strategies. All critical nuances and references are included.

***

# Bi-modal Reasoning SLM with Tool Modality: Proposal and Research Summary

## Concept Overview

You aim to build a **reasoning SLM (e.g., Gemma-270M) with two modalities:**

- **Textual modality** (standard LLM sequence modeling, conversation, reasoning).
- **Tool modality**, modeled and tokenized specifically to represent function signatures/schemas, arguments, and tool capabilities as structured tokens with a dedicated tokenizer/vocab.

The model should *reason* across and within both modalities. This is a more general treatment than prior work like ToolGen, enabling native multi-modal reasoning and robust generalization to unseen tools and complex tool compositions.
The end goal is to use this model with MCP, so it should use some format easily converted to MCP.
***

## Related Research

- **Multimodal LLMs:** mPLUG-Owl, AnyGPT, u-LLaVA for vision/text/audio but rarely for tool schemas as true modalities.[^1][^2][^3][^4][^5]
- **Function Calling and Tool-Augmented LLMs:** ToolGen (single-token tool retrieval/generation), Toolformer, Gorilla, Granite, TinyAgent, Octopus, etc..[^6][^7][^8][^9][^10][^11][^12][^13]
- **Function Schema/Grammar Modeling:** XGrammar, Outlines, Constrained Decoding for structured outputs.[^14][^15][^16]
- **Instruction Tuning/Prompt Optimization:** SPRIG, PILLOW, Modular/Adaptive Prompt learning.[^17][^18][^19][^20]
- **Reinforcement Learning:** PPO, GRPO, TemplateRL, Scaf-GRPO, reward design for reasoning tasks with verifiable structure.[^21][^22][^23][^24][^25][^26][^27]
- **Parallel Tool Calling:** GAP, Divide-Then-Aggregate (DTA-Llama), DynTaskMAS, NaviAgent for DAG-based tool scheduling (recommended as future work).[^28][^29][^30][^31][^32]
- **Long Context Models:** FlashAttention, LongLoRA, sliding window and memory-efficient context scaling for extended queries and multi-turn reasoning.[^33][^34][^35][^36]

***

## Architecture \& Pipeline Recap

### 1. **Model Architecture**

- **Text Modality:** Standard LLM tokenization, embedding, reasoning, and decoding.
- **Tool Modality:** Dedicated tokenizer for structured tool specifications:
    - Function names, parameter types, constraints, return types—tokenized into flexible, compositional tokens.
    - Enables zero-shot generalization to unseen tools by modeling tool schemas as interpretable grammar, not opaque tokens.[^3][^1][^14]
- **Unified Multimodal Reasoning:** Joint embedding/fusion layers learn cross-modal semantic space—model *reasons* jointly over text and tool schemas during sequence generation.[^2][^37][^3]


### 2. **System Prompt Following**

- Strict system prompt formats included in training via SFT curriculum.
- Multiple system prompt variants (3–5 recommended)—ensures robust adherence and generalizable formatting.[^18][^38]
- System prompt just part of input context—no architecture change required.


### 3. **Training Pipeline**

**Supervised Fine-tuning (SFT):**

- Three-stage curriculum:
    - **Tool selection** (given query + tool list, select correct tool).
    - **Argument generation** (given query and tool, produce correctly structured arguments).
    - **Joint reasoning** (handle full prompt-to-tool-call output).
- Loss weighting schedules (start with selection-heavy, then transition toward balanced objectives).[^23][^25][^39]
- Diverse tool schemas in training; curriculum should cover parameter variation, compositionality, edge cases.

**Reinforcement Learning (GRPO + Variants):**

- GRPO (Group Relative Policy Optimization) as main RL procedure—no critic requirement, verifiable reward for function calling.[^22][^26][^27][^21]
- Train for multi-step/chain-of-thought reasoning, compositional tool use, strict format compliance.
- Consider step-wise or template guidance methods if learning stagnates.[^24][^25][^23]
- Reward is sparse or dense (binary: exact match, partial: semantic correctness for arguments).[^26][^21]


### 4. **Token Masking / Logit Bias (Inference Only)**

- Optionally apply logit bias/token masking during inference to restrict tool name/token outputs to valid/seen tool set.[^13][^40][^41]
- Masks can be soft (bias up probability) or hard (exclude unrelated functions)—for robustness in deployment, NOT during training.
- Evaluation *must* compare masked vs. unmasked inferences to measure genuine learning and hallucination rate.


### 5. **Sliding Window / Extended Context**

- Integrate sliding window attention or LongLoRA for context expansion—enables longer inputs, multi-turn tool reasoning, and cross-modal references.[^34][^35][^36][^33]
- Start with context window 16–32K as practical middle ground.
- Further research can scale to 100K–1M tokens as resources allow.

***

## Evaluation \& Metrics

**Standard Metrics:**

- Tool selection accuracy (without masking).
- Argument generation accuracy (schema correctness, semantic matching).
- Hallucination rate (invalid tools called, arguments mismatched).
- Instruction following: format adherence rate relative to system prompt.
- Zero-shot generalization (tools unseen during training).
- Compositionality (correct chaining and argument transfer between tools).

**Advanced Metrics:**

- Parallelization correctness (for future pipeline work, dependency graph analysis).[^29][^28]
- Efficiency (latency, throughput, context size effects).
- Turn-level and chain-of-thought rewards (for multi-turn and compositional chains).[^42][^43]

***

## Implementation Tips \& Nuances

### 1. **Robust Curriculum**

- Carefully construct curriculum for SFT—simple → complex tasks, with explicit negative and edge cases.
- Vary tool schema complexity and parameter mixture for broad generalization.


### 2. **Prompt Robustness**

- Use prompt/format augmentation in data; intentionally vary system instructions and task phrasing.[^20][^17][^18]
- Test with out-of-distribution instructions for generalization.


### 3. **Dataset Quality**

- Fewer, higher-quality examples are preferable to sheer quantity—curation and diversity matters.[^10][^44][^45]
- Combine manual schemas with LLM-generated tool docs for maximum coverage.


### 4. **GRPO Reward Engineering**

- Design reward functions to incentivize not just syntactic success, but also semantic and compositional reasoning.
- Stepwise rewards can accelerate learning for multi-turn/chain-of-thought outputs.


### 5. **Parallel/Compositional Tool Use**

- Treat parallel invocation planning as a Phase 2 research direction—focus first on perfecting joint bi-modal reasoning.[^28][^29]
- If parallel calls are required, build explicit dependency graphs and label data to avoid spurious parallelizations.


### 6. **Masking Use**

- Always use logit bias/masking sparingly—as a deployment/robustness tool, not as a learning crutch.
- Track learning without masking to validate model's true reasoning ability.[^40][^13]


### 7. **Long Context Integration**

- Start with 16–32K windows; profile attention efficiency and empirical gains before targeting extreme contexts.

***

## Potential Improvements \& Future Work

- **Parallel tool scheduling (DAG-based planning):** Integrate tool dependency graphs, annotate dataset, compare synchronous vs. asynchronous execution.[^31][^29][^28]
- **Multi-turn scenario modeling:** Extend curriculum to long dialogue chains, incremental tool calling, result referencing.[^43][^42]
- **Efficiency optimizations:** Explore newer memory-efficient attention mechanisms and context handling for better edge deployment.[^46][^33][^34]
- **Modality fusion:** Enhance embedding fusion and modality-specific adapters for agile reasoning across text and tool modalities.[^37][^47][^2]
- **Zero-shot schema generalization:** Experiment with grammar/structure augmentation and contrastive training to boost unseen tool handling.[^48][^1][^3]

***

## Recommended Next Steps

1. **Design curriculum learning datasets for SFT covering text and fully tokenized tool schemas.**
2. **Develop reward functions for GRPO with support for multi-step, compositional, and strict system prompt assessment.**
3. **Pilot context window scaling using sliding window or LongLoRA approaches.**
4. **Evaluate masking vs. unmasked tool calling outcomes to track actual model learning.**
5. **Iterate system prompt variants for robust instruction following.**
6. **Review state-of-the-art in parallel tool calling for potential future expansions.**

***

## Key References

- ToolGen: Unified Tool Retrieval and Calling via Generation.[^11][^12][^49][^50][^51]
- TinyAgent: Function Calling at the Edge.[^7][^9][^45][^52][^10]
- GRPO and RL for LLM Reasoning.[^25][^27][^21][^22][^26]
- Curriculum and multi-task SFT for tool reasoning.[^39][^23][^25]
- Parallel tool scheduling and graph-based planning.[^29][^31][^28]
- Constrained decoding for structured outputs.[^15][^16][^14]
- Instruction tuning and system prompt variance.[^38][^17][^18][^20]
- Long context scaling and efficient attention mechanisms.[^35][^36][^33][^34][^46]
- Modality fusion and multimodal LLM architectures.[^4][^47][^1][^2][^3][^37]

***

## Conclusion

Your plan for a reasoning bi-modal SLM with a dedicated tool modality, curriculum-based SFT, robust instruction/system prompt following, optional inference-time masking, and context scalability is **scientifically sound, innovative, and well aligned with active research**. The phased approach, measured evaluation metrics, and nuanced strategy for future innovation are all highly recommended for a successful NeurIPS paper and a strong foundation for subsequent extensions.

---

<div align="center">⁂</div>

[^1]: https://arxiv.org/abs/2402.12226

[^2]: https://openreview.net/pdf?id=CppEmee0u6

[^3]: https://www.open-moss.com/en/anygpt/

[^4]: https://huggingface.co/papers/2402.12226

[^5]: http://arxiv.org/pdf/2312.03700v1.pdf

[^6]: https://arxiv.org/pdf/2302.04761.pdf

[^7]: http://arxiv.org/pdf/2409.00608.pdf

[^8]: https://arxiv.org/abs/2404.01549

[^9]: https://arxiv.org/abs/2410.18890

[^10]: https://arxiv.org/abs/2407.00121

[^11]: https://arxiv.org/abs/2410.03439

[^12]: https://arxiv.org/html/2410.03439

[^13]: https://aclanthology.org/2025.naacl-industry.27.pdf

[^14]: https://arxiv.org/abs/2411.15100

[^15]: https://arxiv.org/pdf/2502.05111.pdf

[^16]: https://huggingface.co/blog/vivien/llm-decoding-with-regex-constraints

[^17]: http://arxiv.org/pdf/2406.11301.pdf

[^18]: https://arxiv.org/pdf/2310.00492.pdf

[^19]: https://arxiv.org/pdf/2410.14826.pdf

[^20]: https://aclanthology.org/2023.emnlp-industry.45.pdf

[^21]: https://arxiv.org/pdf/2503.06639.pdf

[^22]: https://arxiv.org/pdf/2503.12937.pdf

[^23]: https://www.semanticscholar.org/paper/34035d548b6cc7b6448db4f24064b185231038de

[^24]: https://arxiv.org/abs/2505.21178

[^25]: https://www.semanticscholar.org/paper/39427ea2c4b5783a96b96bb1abbf6a8f1f1f5524

[^26]: https://www.deeplearning.ai/short-courses/reinforcement-fine-tuning-llms-grpo/

[^27]: https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training

[^28]: https://www.semanticscholar.org/paper/c65cbed8dabd71cdbfcb3310305c40171fd17a65

[^29]: https://arxiv.org/abs/2501.12432

[^30]: http://arxiv.org/pdf/2503.07675.pdf

[^31]: https://arxiv.org/html/2510.25320v1

[^32]: https://arxiv.org/abs/2506.19500

[^33]: https://www.abhik.xyz/concepts/attention/sliding-window-attention

[^34]: https://hkaift.com/the-remedy-for-fine-tuning-llms-in-long-context-longlora/

[^35]: https://huggingface.co/papers/2309.12307

[^36]: https://arxiv.org/abs/2309.12307

[^37]: https://www.emergentmind.com/topics/multimodal-tokenization

[^38]: https://arxiv.org/html/2402.18540v2

[^39]: https://aclanthology.org/2024.emnlp-industry.85.pdf

[^40]: https://neptune.ai/blog/customizing-llm-output-post-processing-techniques

[^41]: https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api

[^42]: https://arxiv.org/pdf/2505.11821.pdf

[^43]: https://arxiv.org/html/2505.11821v1

[^44]: https://github.com/Applied-Machine-Learning-Lab/Awesome-Function-Callings

[^45]: https://arxiv.org/abs/2409.00608

[^46]: https://www.metriccoders.com/post/efficient-attention-mechanisms-powering-the-next-generation-of-large-language-models

[^47]: https://arxiv.org/abs/2304.01933

[^48]: https://arxiv.org/pdf/2311.08066.pdf

[^49]: https://www.kdjingpai.com/en/toolgen/

[^50]: https://aisharenet.com/en/toolgen/

[^51]: https://arxiv.org/html/2410.03439v1

[^52]: https://arxiv.org/abs/2504.19277

