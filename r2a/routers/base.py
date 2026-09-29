"""Common interface for every router used as an ensemble member or as a target.

A router maps a query (plus an optional adversarial suffix) to one score per
candidate model in ``get_model_list()``. Routers that take part in suffix
optimization additionally expose an embedding-level forward pass:

* ``get_internal_encoder()`` returns the tokenizer / embedding table whose
  embeddings the router consumes;
* ``assemble_gcg_input()`` splices differentiable suffix embeddings into the
  query embeddings, in the router's own input format;
* ``forward_embeds()`` runs the router on those embeddings and returns scores
  that carry gradients back to the suffix.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch


class Router(ABC):
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self._model_list: List[str] = []

    @abstractmethod
    def route(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        """Return scores (one per candidate model) for ``prompt + suffix``."""

    def get_model_list(self) -> List[str]:
        return list(self._model_list)

    def modules(self) -> List[torch.nn.Module]:
        """nn.Modules owned by the router (frozen during the attack)."""
        return [v for v in vars(self).values() if isinstance(v, torch.nn.Module)]

    def freeze(self) -> None:
        for module in self.modules():
            module.requires_grad_(False)

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} does not support embedding-level forward")

    def assemble_gcg_input(
        self,
        messages: str,
        suffix_embeds: torch.Tensor,
        encoder=None,
        prefix_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{type(self).__name__} does not implement assemble_gcg_input")


class HFEncoder:
    """Tokenizer + embedding table of a router, as seen by the suffix optimizer."""

    def __init__(self, name: str, model: torch.nn.Module, tokenizer):
        self.name = name
        self._model = model
        self._tokenizer = tokenizer

    def get_model(self) -> torch.nn.Module:
        return self._model

    def get_tokenizer(self):
        return self._tokenizer

    def embed_matrix(self) -> torch.Tensor:
        return self._model.get_input_embeddings().weight

    def embedding_dim(self) -> int:
        return self.embed_matrix().shape[1]
