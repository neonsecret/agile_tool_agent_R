"""
CUDA graph optimization for diffusion head inference.

Provides a wrapper for capturing and executing CUDA graphs to reduce kernel launch overhead.
"""

from typing import Dict, Optional
import torch

from data.device_utils import synchronize


class CUDAGraphRunner:
    """Manages CUDA graph capture and execution for diffusion head."""

    def __init__(self, model, device: torch.device, use_cuda_graph: bool = True):
        self.model = model
        self.device = device
        self._cuda_graph_enabled = use_cuda_graph and self._is_cuda_graph_supported()
        self._cuda_graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_inputs: Dict[str, torch.Tensor] = {}
        self._graph_outputs: Optional[torch.Tensor] = None

    def _is_cuda_graph_supported(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def is_enabled(self) -> bool:
        return self._cuda_graph_enabled

    def setup(self, hidden_states: torch.Tensor,
              current_tokens: torch.Tensor, t: torch.Tensor,
              scaffold_mask: Optional[torch.Tensor] = None,
              prompt_mask: Optional[torch.Tensor] = None):
        """Capture CUDA graph for diffusion head forward pass."""
        if not self._cuda_graph_enabled:
            return

        for _ in range(3):
            _ = self.model.diffusion_head.predict(
                hidden_states,
                current_tokens,
                t,
                scaffold_mask=scaffold_mask,
                prompt_mask=prompt_mask,
            )

        synchronize(self.device)

        self._graph_inputs = {
            'hidden_states': hidden_states.clone(),
            'current_tokens': current_tokens.clone(),
            't': t.clone()
        }
        if scaffold_mask is not None:
            self._graph_inputs['scaffold_mask'] = scaffold_mask.clone()
        if prompt_mask is not None:
            self._graph_inputs['prompt_mask'] = prompt_mask.clone()

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_outputs = self.model.diffusion_head.predict(
                self._graph_inputs['hidden_states'],
                self._graph_inputs['current_tokens'],
                self._graph_inputs['t'],
                scaffold_mask=self._graph_inputs.get('scaffold_mask'),
                prompt_mask=self._graph_inputs.get('prompt_mask'),
            )

        print("CUDA graph captured for diffusion head")

    def run(self, hidden_states: torch.Tensor,
            current_tokens: torch.Tensor, t: torch.Tensor,
            scaffold_mask: Optional[torch.Tensor] = None,
            prompt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute captured CUDA graph with new inputs."""
        if self._cuda_graph is None:
            self.setup(hidden_states, current_tokens, t, scaffold_mask, prompt_mask)

        if self._cuda_graph is not None:
            if (hidden_states.shape == self._graph_inputs['hidden_states'].shape and
                    current_tokens.shape == self._graph_inputs['current_tokens'].shape):
                self._graph_inputs['hidden_states'].copy_(hidden_states)
                self._graph_inputs['current_tokens'].copy_(current_tokens)
                self._graph_inputs['t'].copy_(t)
                if scaffold_mask is not None and 'scaffold_mask' in self._graph_inputs:
                    self._graph_inputs['scaffold_mask'].copy_(scaffold_mask)
                if prompt_mask is not None and 'prompt_mask' in self._graph_inputs:
                    self._graph_inputs['prompt_mask'].copy_(prompt_mask)

                self._cuda_graph.replay()

                return self._graph_outputs.clone()
            else:
                return self.model.diffusion_head.predict(
                    hidden_states,
                    current_tokens,
                    t,
                    scaffold_mask=scaffold_mask,
                    prompt_mask=prompt_mask,
                )
        else:
            return self.model.diffusion_head.predict(
                hidden_states,
                current_tokens,
                t,
                scaffold_mask=scaffold_mask,
                prompt_mask=prompt_mask,
            )

    def clear(self):
        """Clear CUDA graph state."""
        self._cuda_graph = None
        self._graph_inputs = {}
        self._graph_outputs = None
