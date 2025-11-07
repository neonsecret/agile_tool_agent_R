"""
Theoretical Demonstration: Bi-modal Reasoning SLM with Text and Tool Modalities

This script demonstrates the conceptual architecture and inference flow of a
bi-modal Small Language Model that reasons jointly over text and tool schemas.

Key Components:
1. Dual tokenizers (text + tool)
2. Unified embedding space for cross-modal reasoning
3. Curriculum-based training simulation
4. Optional logit bias for inference-time masking
5. Sliding window attention for extended context
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# Tool Schema Representation
# ============================================================================

@dataclass
class ToolParameter:
    """Represents a function parameter in tool modality"""
    name: str
    type: str
    required: bool
    description: str


@dataclass
class ToolSchema:
    """Structured tool representation for tool modality"""
    name: str
    parameters: List[ToolParameter]
    return_type: str
    description: str


# ============================================================================
# Tool Tokenizer - Specialized tokenizer for tool modality
# ============================================================================

class ToolTokenizer:
    """
    Dedicated tokenizer for tool modality that converts function schemas
    into compositional tokens for reasoning
    """

    def __init__(self):
        # Tool-specific vocabulary
        self.vocab = {
            # Structural tokens
            '<TOOL_START>': 0, '<TOOL_END>': 1,
            '<FUNC_NAME>': 2, '<PARAM>': 3, '<TYPE>': 4,
            '<REQUIRED>': 5, '<OPTIONAL>': 6,
            '<RETURN>': 7, '<DESC>': 8,

            # Type tokens
            'str': 100, 'int': 101, 'float': 102, 'bool': 103,
            'list': 104, 'dict': 105, 'object': 106,

            # Common function name fragments for compositional understanding
            'get': 200, 'set': 201, 'create': 202, 'delete': 203,
            'update': 204, 'search': 205, 'calculate': 206,

            # Padding and special
            '<PAD>': 999, '<UNK>': 1000
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}

    def encode_tool(self, tool: ToolSchema) -> List[int]:
        """Convert tool schema to token sequence"""
        tokens = [self.vocab['<TOOL_START>']]

        # Encode function name compositionally
        tokens.append(self.vocab['<FUNC_NAME>'])
        tokens.extend(self._encode_function_name(tool.name))

        # Encode parameters
        for param in tool.parameters:
            tokens.append(self.vocab['<PARAM>'])
            tokens.extend(self._encode_param_name(param.name))
            tokens.append(self.vocab['<TYPE>'])
            tokens.append(self.vocab.get(param.type, self.vocab['<UNK>']))
            tokens.append(self.vocab['<REQUIRED>'] if param.required else self.vocab['<OPTIONAL>'])

        # Encode return type
        tokens.append(self.vocab['<RETURN>'])
        tokens.append(self.vocab.get(tool.return_type, self.vocab['<UNK>']))

        tokens.append(self.vocab['<TOOL_END>'])
        return tokens

    def _encode_function_name(self, name: str) -> List[int]:
        """Encode function name as compositional tokens"""
        # Simple demonstration: split by underscore and map to vocab
        parts = name.lower().split('_')
        tokens = []
        for part in parts:
            tokens.append(self.vocab.get(part, self.vocab['<UNK>']))
        return tokens

    def _encode_param_name(self, name: str) -> List[int]:
        """Encode parameter name"""
        # Simplified for demo - hash to consistent token ID
        return [hash(name) % 1000 + 2000]


# ============================================================================
# Bi-modal Model Architecture
# ============================================================================

class BiModalSLM:
    """
    Bi-modal Small Language Model with text and tool reasoning capabilities
    """

    def __init__(
        self,
        text_vocab_size: int = 50000,
        tool_vocab_size: int = 3000,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 12,
        context_window: int = 16384,
        use_sliding_window: bool = True
    ):
        self.text_vocab_size = text_vocab_size
        self.tool_vocab_size = tool_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.context_window = context_window
        self.use_sliding_window = use_sliding_window

        # Initialize tokenizers
        self.text_tokenizer = self._init_text_tokenizer()
        self.tool_tokenizer = ToolTokenizer()

        # Initialize embeddings (simulated)
        self.text_embeddings = np.random.randn(text_vocab_size, embed_dim) * 0.02
        self.tool_embeddings = np.random.randn(tool_vocab_size, embed_dim) * 0.02

        # Fusion layer for cross-modal reasoning
        self.fusion_weights = np.random.randn(embed_dim, embed_dim) * 0.02

        # Training state
        self.training_phase = "sft_tool_selection"  # sft_tool_selection, sft_argument_gen, sft_joint, grpo

    def _init_text_tokenizer(self):
        """Initialize standard text tokenizer (simulated)"""
        class SimpleTextTokenizer:
            def encode(self, text: str) -> List[int]:
                # Simplified: hash words to token IDs
                words = text.lower().split()
                return [hash(w) % 50000 for w in words]

            def decode(self, tokens: List[int]) -> str:
                # Simplified decoding
                return f"<decoded_text_{len(tokens)}_tokens>"

        return SimpleTextTokenizer()

    def encode_multimodal_input(
        self,
        text: str,
        tools: List[ToolSchema]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode both text and tool modalities into unified embedding space
        """
        # Encode text modality
        text_tokens = self.text_tokenizer.encode(text)
        text_embeds = np.array([self.text_embeddings[t % self.text_vocab_size] for t in text_tokens])

        # Encode tool modality
        tool_embeds_list = []
        tool_metadata = []
        for tool in tools:
            tool_tokens = self.tool_tokenizer.encode_tool(tool)
            tool_embeds = np.array([self.tool_embeddings[t % self.tool_vocab_size] for t in tool_tokens])
            tool_embeds_list.append(tool_embeds)
            tool_metadata.append({
                'name': tool.name,
                'tokens': tool_tokens,
                'length': len(tool_tokens)
            })

        # Concatenate all tool embeddings
        all_tool_embeds = np.vstack(tool_embeds_list) if tool_embeds_list else np.array([])

        # Joint embedding: [text_embeds, tool_embeds]
        if all_tool_embeds.size > 0:
            joint_embeds = np.vstack([text_embeds, all_tool_embeds])
        else:
            joint_embeds = text_embeds

        metadata = {
            'text_length': len(text_tokens),
            'tool_count': len(tools),
            'tool_metadata': tool_metadata,
            'total_length': joint_embeds.shape[0]
        }

        return joint_embeds, metadata

    def cross_modal_fusion(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply cross-modal fusion for joint reasoning
        Enables the model to reason across text and tool modalities
        """
        # Simulate attention-based fusion
        fused = embeddings @ self.fusion_weights
        # Add residual connection
        fused = embeddings + 0.1 * fused
        return fused

    def apply_sliding_window_attention(
        self,
        embeddings: np.ndarray,
        window_size: int = 4096
    ) -> np.ndarray:
        """
        Simulate sliding window attention for extended context
        """
        if not self.use_sliding_window or embeddings.shape[0] <= window_size:
            return embeddings

        # Simplified simulation: process in windows with overlap
        # In real implementation, this would be efficient attention mechanism
        print(f"  [Attention] Using sliding window (size={window_size}) for {embeddings.shape[0]} tokens")
        return embeddings  # Return as-is for demo

    def forward_pass(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Simulate forward pass through the model
        """
        # Apply cross-modal fusion
        fused_embeds = self.cross_modal_fusion(embeddings)

        # Apply sliding window attention if needed
        attended_embeds = self.apply_sliding_window_attention(fused_embeds)

        # Simulate reasoning (in reality: transformer layers)
        hidden_states = self._simulate_transformer_layers(attended_embeds)

        # Generate output logits for both modalities
        output = {
            'hidden_states': hidden_states,
            'text_logits': self._generate_text_logits(hidden_states),
            'tool_logits': self._generate_tool_logits(hidden_states),
            'metadata': metadata
        }

        return output

    def _simulate_transformer_layers(self, embeddings: np.ndarray) -> np.ndarray:
        """Simulate transformer processing"""
        # Simplified: just normalize and add noise to simulate processing
        hidden = embeddings.copy()
        for layer in range(self.num_layers):
            # Simulate layer processing
            hidden = hidden + np.random.randn(*hidden.shape) * 0.01
            # Normalize
            hidden = hidden / (np.linalg.norm(hidden, axis=1, keepdims=True) + 1e-8)
        return hidden

    def _generate_text_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        """Generate logits for text modality"""
        # Last hidden state projected to text vocab
        last_hidden = hidden_states[-1]
        logits = last_hidden @ self.text_embeddings.T
        return logits

    def _generate_tool_logits(self, hidden_states: np.ndarray) -> np.ndarray:
        """Generate logits for tool modality"""
        # Last hidden state projected to tool vocab
        last_hidden = hidden_states[-1]
        logits = last_hidden @ self.tool_embeddings.T
        return logits

    def decode_with_logit_bias(
        self,
        logits: np.ndarray,
        allowed_tools: Optional[List[str]] = None,
        bias_strength: float = 10.0
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Decode with optional logit bias for token masking (inference only)
        """
        text_logits = logits.copy()

        if allowed_tools is not None:
            print(f"  [Logit Bias] Applying masking for {len(allowed_tools)} allowed tools (bias={bias_strength})")
            # Simulate masking by boosting allowed tool tokens
            # In reality: identify tool name token ranges and apply bias
            mask = np.random.choice([0, 1], size=text_logits.shape, p=[0.8, 0.2])
            text_logits = text_logits + bias_strength * mask

        # Decode (simulated)
        predicted_token = np.argmax(text_logits)

        result = {
            'predicted_token': int(predicted_token),
            'confidence': float(np.max(text_logits)),
            'bias_applied': allowed_tools is not None
        }

        return self.text_tokenizer.decode([predicted_token]), result

    def generate_tool_call(
        self,
        text_query: str,
        available_tools: List[ToolSchema],
        use_logit_bias: bool = False,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate tool call given text query and available tools
        Demonstrates full inference pipeline
        """
        print(f"\n{'='*70}")
        print(f"GENERATING TOOL CALL")
        print(f"{'='*70}")
        print(f"Query: {text_query}")
        print(f"Available tools: {len(available_tools)}")
        print(f"Training phase: {self.training_phase}")
        print(f"Logit bias enabled: {use_logit_bias}")

        # Prepare input with optional system prompt
        if system_prompt:
            full_input = f"{system_prompt}\n\nUser query: {text_query}"
        else:
            full_input = text_query

        # Encode multimodal input
        print(f"\n[1] Encoding multimodal input...")
        embeddings, metadata = self.encode_multimodal_input(full_input, available_tools)
        print(f"  - Text tokens: {metadata['text_length']}")
        print(f"  - Tool tokens: {sum(t['length'] for t in metadata['tool_metadata'])}")
        print(f"  - Total sequence length: {metadata['total_length']}")

        # Forward pass
        print(f"\n[2] Forward pass through bi-modal model...")
        output = self.forward_pass(embeddings, metadata)
        print(f"  - Processed through {self.num_layers} layers")
        print(f"  - Hidden state shape: {output['hidden_states'].shape}")

        # Decode with optional logit bias
        print(f"\n[3] Decoding...")
        allowed_tools = [t.name for t in available_tools] if use_logit_bias else None
        decoded_text, decode_info = self.decode_with_logit_bias(
            output['text_logits'],
            allowed_tools=allowed_tools
        )

        # Simulate tool selection and argument generation
        print(f"\n[4] Tool reasoning...")
        selected_tool = self._reason_tool_selection(output, available_tools)
        arguments = self._reason_arguments(output, selected_tool)

        result = {
            'selected_tool': selected_tool.name,
            'arguments': arguments,
            'confidence': decode_info['confidence'],
            'reasoning_trace': {
                'text_length': metadata['text_length'],
                'tools_considered': len(available_tools),
                'bias_applied': decode_info['bias_applied']
            }
        }

        print(f"\n{'='*70}")
        print(f"RESULT: {selected_tool.name}({arguments})")
        print(f"{'='*70}\n")

        return result

    def _reason_tool_selection(
        self,
        output: Dict[str, Any],
        tools: List[ToolSchema]
    ) -> ToolSchema:
        """Simulate tool selection reasoning"""
        # In reality: model predicts tool through tool modality tokens
        # For demo: random selection weighted by tool complexity
        print(f"  - Reasoning over {len(tools)} tools in tool modality...")
        return np.random.choice(tools)

    def _reason_arguments(
        self,
        output: Dict[str, Any],
        tool: ToolSchema
    ) -> Dict[str, Any]:
        """Simulate argument generation reasoning"""
        # In reality: model generates structured arguments through joint reasoning
        print(f"  - Generating arguments for {len(tool.parameters)} parameters...")
        args = {}
        for param in tool.parameters:
            if param.required:
                args[param.name] = f"<generated_{param.type}>"
        return args


# ============================================================================
# Training Pipeline Simulation
# ============================================================================

class TrainingPipeline:
    """Demonstrates the SFT + GRPO training pipeline"""

    def __init__(self, model: BiModalSLM):
        self.model = model
        self.curriculum_stages = [
            "sft_tool_selection",
            "sft_argument_generation",
            "sft_joint_reasoning",
            "grpo_reinforcement"
        ]
        self.current_stage = 0

    def run_sft_curriculum(self):
        """Simulate supervised fine-tuning with curriculum learning"""
        print(f"\n{'#'*70}")
        print(f"# SUPERVISED FINE-TUNING CURRICULUM")
        print(f"{'#'*70}\n")

        for stage in self.curriculum_stages[:3]:  # SFT stages only
            print(f"\n--- Stage: {stage.upper()} ---")
            self.model.training_phase = stage

            if stage == "sft_tool_selection":
                print("Training: Given query + tools → select correct tool")
                print("  - Loss: Cross-entropy on tool selection")
                print("  - Curriculum: Simple → Complex tool sets")

            elif stage == "sft_argument_generation":
                print("Training: Given query + selected tool → generate arguments")
                print("  - Loss: Cross-entropy + schema validation loss")
                print("  - Curriculum: Simple types → Complex nested structures")

            elif stage == "sft_joint_reasoning":
                print("Training: Full pipeline - query → tool + arguments")
                print("  - Loss: Balanced weighted loss across both subtasks")
                print("  - Curriculum: Single tool → Multi-step composition")

            # Simulate training
            self._simulate_training_epoch(stage)

    def run_grpo_reinforcement(self):
        """Simulate GRPO (Group Relative Policy Optimization) training"""
        print(f"\n{'#'*70}")
        print(f"# GRPO REINFORCEMENT LEARNING")
        print(f"{'#'*70}\n")

        self.model.training_phase = "grpo_reinforcement"

        print("Training: Multi-step reasoning and compositional tool use")
        print("  - Reward: Verifiable function execution + semantic correctness")
        print("  - Group-relative policy optimization (no critic needed)")
        print("  - Focus: Format compliance + compositional reasoning")
        print("  - Exploration: Temperature-based sampling with reward feedback")

        self._simulate_grpo_epoch()

    def _simulate_training_epoch(self, stage: str):
        """Simulate one training epoch"""
        print(f"\n  Epoch simulation:")
        print(f"    - Batch size: 32")
        print(f"    - Learning rate: 1e-4")
        print(f"    - Loss: 0.{np.random.randint(100, 500)}")
        print(f"    - Accuracy: 0.{np.random.randint(70, 95)}")

    def _simulate_grpo_epoch(self):
        """Simulate GRPO training epoch"""
        print(f"\n  GRPO Epoch simulation:")
        print(f"    - Group size: 8")
        print(f"    - Reward samples: 4 per group")
        print(f"    - Average reward: {np.random.uniform(0.6, 0.9):.3f}")
        print(f"    - Policy improvement: +{np.random.uniform(0.01, 0.05):.3f}")


# ============================================================================
# Evaluation Framework
# ============================================================================

class EvaluationFramework:
    """Demonstrates evaluation metrics and procedures"""

    @staticmethod
    def evaluate(model: BiModalSLM, test_cases: List[Dict[str, Any]]):
        """Run comprehensive evaluation"""
        print(f"\n{'#'*70}")
        print(f"# MODEL EVALUATION")
        print(f"{'#'*70}\n")

        metrics = {
            'tool_selection_accuracy': [],
            'argument_accuracy': [],
            'hallucination_rate': [],
            'format_compliance': [],
            'zero_shot_performance': []
        }

        for i, test_case in enumerate(test_cases):
            print(f"\nTest case {i+1}/{len(test_cases)}: {test_case['name']}")

            # Simulate evaluation
            result = model.generate_tool_call(
                test_case['query'],
                test_case['tools'],
                use_logit_bias=test_case.get('use_bias', False),
                system_prompt=test_case.get('system_prompt')
            )

            # Compute metrics (simulated)
            metrics['tool_selection_accuracy'].append(np.random.uniform(0.85, 0.98))
            metrics['argument_accuracy'].append(np.random.uniform(0.80, 0.95))
            metrics['hallucination_rate'].append(np.random.uniform(0.02, 0.10))
            metrics['format_compliance'].append(np.random.uniform(0.90, 0.99))

            if test_case.get('zero_shot', False):
                metrics['zero_shot_performance'].append(np.random.uniform(0.70, 0.85))

        # Print summary
        print(f"\n{'='*70}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*70}")
        for metric, values in metrics.items():
            if values:
                print(f"{metric:.<40} {np.mean(values):.3f} (±{np.std(values):.3f})")
        print(f"{'='*70}\n")


# ============================================================================
# Main Demonstration
# ============================================================================

def main():
    """Run the complete demonstration"""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      BI-MODAL REASONING SLM DEMONSTRATION                           ║
║      Text + Tool Modalities with Joint Reasoning                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize model
    print("Initializing Bi-modal SLM...")
    model = BiModalSLM(
        text_vocab_size=50000,
        tool_vocab_size=3000,
        embed_dim=512,
        hidden_dim=1024,
        num_layers=12,
        context_window=16384,
        use_sliding_window=True
    )
    print(f"  ✓ Model initialized with {model.num_layers} layers")
    print(f"  ✓ Context window: {model.context_window} tokens")
    print(f"  ✓ Sliding window attention: {'enabled' if model.use_sliding_window else 'disabled'}")

    # Define example tools
    example_tools = [
        ToolSchema(
            name="search_database",
            parameters=[
                ToolParameter("query", "str", True, "Search query string"),
                ToolParameter("limit", "int", False, "Maximum results")
            ],
            return_type="list",
            description="Search database for matching records"
        ),
        ToolSchema(
            name="calculate_statistics",
            parameters=[
                ToolParameter("data", "list", True, "Input data array"),
                ToolParameter("metric", "str", True, "Statistic to calculate")
            ],
            return_type="float",
            description="Calculate statistical metrics on data"
        ),
        ToolSchema(
            name="create_report",
            parameters=[
                ToolParameter("title", "str", True, "Report title"),
                ToolParameter("content", "dict", True, "Report content"),
                ToolParameter("format", "str", False, "Output format")
            ],
            return_type="str",
            description="Generate formatted report"
        )
    ]

    # Demonstrate training pipeline
    print("\n" + "="*70)
    trainer = TrainingPipeline(model)
    trainer.run_sft_curriculum()
    trainer.run_grpo_reinforcement()

    # Demonstrate inference
    print("\n" + "="*70)
    print("INFERENCE DEMONSTRATIONS")
    print("="*70)

    # Test case 1: Without logit bias
    model.generate_tool_call(
        text_query="Find all users who registered in the last 30 days",
        available_tools=example_tools,
        use_logit_bias=False,
        system_prompt="You are a helpful assistant. Follow the instructions carefully and select appropriate tools."
    )

    # Test case 2: With logit bias
    model.generate_tool_call(
        text_query="Compute the average rating from the dataset",
        available_tools=example_tools,
        use_logit_bias=True,
        system_prompt="You are a helpful assistant. Follow the instructions carefully and select appropriate tools."
    )

    # Run evaluation
    test_cases = [
        {
            'name': 'Standard tool selection',
            'query': 'Search for products matching "laptop"',
            'tools': example_tools,
            'use_bias': False,
            'zero_shot': False
        },
        {
            'name': 'Zero-shot generalization',
            'query': 'Generate summary report of quarterly sales',
            'tools': example_tools,
            'use_bias': False,
            'zero_shot': True
        },
        {
            'name': 'With logit bias masking',
            'query': 'Calculate median from dataset',
            'tools': example_tools,
            'use_bias': True,
            'zero_shot': False,
            'system_prompt': 'Use only the provided tools. Be precise.'
        }
    ]

    EvaluationFramework.evaluate(model, test_cases)

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  DEMONSTRATION COMPLETE                                             ║
║                                                                      ║
║  Key Features Demonstrated:                                         ║
║  ✓ Dual modality tokenization (text + tool)                        ║
║  ✓ Cross-modal fusion for joint reasoning                          ║
║  ✓ Curriculum-based SFT training                                   ║
║  ✓ GRPO reinforcement learning                                     ║
║  ✓ Optional logit bias for inference-time masking                  ║
║  ✓ Sliding window attention for extended context                   ║
║  ✓ System prompt compliance                                        ║
║  ✓ Comprehensive evaluation metrics                                ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
