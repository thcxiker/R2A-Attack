"""Attack objective on the surrogate's target-pool logits (Eq. 7):
L_A = 1 - sum_{M in M_strong} p(M | q + s).
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class StrongModelPromotionLoss(nn.Module):
    def __init__(self, strong_indices: Iterable[int], weak_indices: Iterable[int]):
        super().__init__()
        self.strong_indices = sorted(strong_indices)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits.float(), dim=-1)
        return 1.0 - probs[self.strong_indices].sum()


LOSSES = {"strong_promotion": StrongModelPromotionLoss}


def create_loss(name: str, strong_indices, weak_indices, **kwargs) -> nn.Module:
    if name not in LOSSES:
        raise ValueError(f"Unknown attack loss '{name}'. Available: {sorted(LOSSES)}")
    return LOSSES[name](strong_indices, weak_indices, **kwargs)
