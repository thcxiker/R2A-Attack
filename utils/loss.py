import torch
import torch.nn as nn
import torch.nn.functional as F


def create_loss_function(device: torch.device, loss_type: str, **kwargs):
    """Create loss function"""
    print("kwargs:", kwargs)
    
    if loss_type == "strong_promotion":
        strong_indices = kwargs.get('strong_indices')
        weak_indices = kwargs.get('weak_indices')
        
        if strong_indices is None or weak_indices is None:
            raise ValueError("strong_promotion loss requires strong_indices and weak_indices")
        
        loss_fn = StrongModelPromotionLoss(
            strong_indices=strong_indices,
            weak_indices=weak_indices
        )
        return lambda logits, target: loss_fn(logits, target)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
class StrongModelPromotionLoss(nn.Module):
    """
    Directly maximize the overall probability of strong models.
    
    Strategy:
    Minimize (1 - total_strong_prob) to maximize strong model probability.
    
    Args:
        strong_indices: Strong model indices in union space
        weak_indices: Weak model indices in union space
    """
    
    def __init__(
        self,
        strong_indices: set,
        weak_indices: set
    ):
        super().__init__()
        self.strong_indices = list(strong_indices)
        self.weak_indices = list(weak_indices)
    
    def forward(
        self,
        logits: torch.Tensor,
        target: int = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [num_classes] predicted logits
            target: (unused) for interface compatibility
        
        Returns:
            loss: scalar
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Maximize total probability of strong models
        strong_probs = probs[self.strong_indices]
        total_strong_prob = strong_probs.sum()
        
        # Loss = 1 - total_strong_prob (minimize to maximize probability)
        loss = 1.0 - total_strong_prob
        
        return loss
