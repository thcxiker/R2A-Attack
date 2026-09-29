"""Ensemble GCG on the hybrid surrogate router (Sec. 3.2).

One optimization step:

1. For every trainable member router k, embed the suffix with that router's
   own tokenizer, run the router on (query + suffix) embeddings, plug its
   differentiable scores into the surrogate (the other members are constants),
   and backpropagate L_A to the suffix embeddings.
2. ``VStarBuilder`` turns the per-router gradients into a top-k candidate set
   per suffix position (Eq. 9-11).
3. ``search_width`` candidates are sampled by replacing ``n_replace`` random
   positions with random candidates, every candidate is scored with the full
   surrogate on the active queries, and the lowest-loss candidate becomes the
   new suffix.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from r2a.attack.losses import create_loss
from r2a.attack.vstar import VStarBuilder
from r2a.classifier import ModelClassifier, routing_outcome

logger = logging.getLogger(__name__)


@dataclass
class GCGConfig:
    search_width: int = 64          # B: candidates evaluated per step
    topk: int = 256                 # |C_i|: candidates kept per position
    n_replace: int = 1              # positions changed per candidate
    max_suffix_tokens: int = 30     # Delta, in tokens of the primary tokenizer
    max_common_words: int = 10000   # cap on |V*|
    prefix_mode: bool = False       # optimize a prefix instead of a suffix


@dataclass
class Member:
    """An ensemble member: router + the encoder whose embeddings it consumes."""

    name: str
    router: object
    encoder: object
    trainable: bool = True


class EnsembleGCG:
    def __init__(
        self,
        members: List[Member],
        lowrank: Member,
        adapter,
        unified_model_names: List[str],
        target_model_names: List[str],
        classifier: ModelClassifier,
        config: GCGConfig,
        loss: str = "strong_promotion",
        device: str = "cuda",
    ):
        self.members = members                  # open-source routers, in adapter order
        self.lowrank = lowrank                  # lightweight router R_l (always last)
        self.adapter = adapter
        self.unified_model_names = unified_model_names
        self.target_model_names = target_model_names
        self.classifier = classifier
        self.config = config
        self.device = device
        self._u_map = {n: i for i, n in enumerate(unified_model_names)}

        strong_idx, weak_idx = classifier.strong_weak_indices(target_model_names)
        if not strong_idx or not weak_idx:
            raise ValueError("target pool must contain both strong and weak models")
        self.loss_fn = create_loss(loss, strong_idx, weak_idx)

        for m in self.members + [self.lowrank]:
            m.router.freeze()
        self.adapter.requires_grad_(False)

        self.trainable = [m for m in self.members + [self.lowrank] if m.trainable]
        if not self.trainable:
            raise ValueError("at least one ensemble member must be trainable")
        self.vstar = VStarBuilder(
            [m.encoder for m in self.trainable],
            k=config.topk,
            max_common_words=config.max_common_words,
            device=device,
        )

    # ------------------------------------------------------------------ surrogate
    def _order(self, message: str, suffix: str) -> Tuple[str, str]:
        return (suffix, message) if self.config.prefix_mode else (message, suffix)

    def _router_stack(self, prompt: str, suffix: str, task_type: Optional[str]):
        R, U = len(self.members), len(self.unified_model_names)
        stack = torch.zeros((1, R, U), device=self.device)
        mask = torch.zeros((1, R, U), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            for r, m in enumerate(self.members):
                scores = m.router.route(prompt, suffix, type=task_type).float().to(self.device)
                for local, name in enumerate(m.router.get_model_list()):
                    if name in self._u_map:
                        stack[0, r, self._u_map[name]] = scores[local]
                        mask[0, r, self._u_map[name]] = True
        return stack, mask

    @torch.no_grad()
    def surrogate_logits(self, message: str, suffix: str, task_type: Optional[str] = None) -> torch.Tensor:
        """Surrogate scores over the target pool for ``message`` with ``suffix`` attached."""
        prompt, sfx = self._order(message, suffix)
        stack, mask = self._router_stack(prompt, sfx, task_type)
        return self.adapter(stack, mask, prompts=prompt + sfx).squeeze(0)

    def outcome(self, message: str, suffix: str, task_type: Optional[str] = None) -> dict:
        return routing_outcome(self.surrogate_logits(message, suffix, task_type), self.target_model_names, self.classifier)

    def loss(self, message: str, suffix: str, task_type: Optional[str] = None) -> float:
        return self.loss_fn(self.surrogate_logits(message, suffix, task_type)).item()

    # ------------------------------------------------------------------ gradients
    def _token_gradients(self, message: str, suffix: str, task_type: Optional[str]) -> List[torch.Tensor]:
        prompt, sfx = self._order(message, suffix)
        base_stack, mask = self._router_stack(prompt, sfx, task_type)
        with torch.no_grad():
            lowrank_logits = self.lowrank.router.route(prompt, sfx).float()

        grads = []
        for member in self.trainable:
            tokenizer = member.encoder.get_tokenizer()
            embed_layer = member.encoder.get_model().get_input_embeddings()
            ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
            ids = ids.to(embed_layer.weight.device)
            suffix_embeds = embed_layer(ids).squeeze(0).detach().requires_grad_(True)

            with torch.enable_grad():
                embeds, attn = member.router.assemble_gcg_input(
                    message, suffix_embeds, encoder=member.encoder, prefix_mode=self.config.prefix_mode
                )
                scores = member.router.forward_embeds(embeds, attention_mask=attn, task_type=task_type)
                scores = scores.float().to(self.device)
                if member is self.lowrank:
                    logits = self.adapter(base_stack, mask, lowrank_logits=scores)
                else:
                    r = self.members.index(member)
                    local, uni = [], []
                    for li, name in enumerate(member.router.get_model_list()):
                        if name in self._u_map:
                            local.append(li)
                            uni.append(self._u_map[name])
                    row = torch.zeros(len(self.unified_model_names), device=self.device).scatter(
                        0, torch.tensor(uni, device=self.device), scores[torch.tensor(local, device=self.device)]
                    )
                    onehot = torch.zeros(len(self.members), device=self.device)
                    onehot[r] = 1.0
                    others = base_stack * (1 - onehot).view(1, -1, 1)
                    stack = others + onehot.view(1, -1, 1) * row.view(1, 1, -1)
                    logits = self.adapter(stack, mask, lowrank_logits=lowrank_logits)
                loss = self.loss_fn(logits.squeeze(0))
                loss.backward()
            grad = suffix_embeds.grad
            grads.append(torch.zeros_like(suffix_embeds) if grad is None else grad.detach())
        return grads

    # ------------------------------------------------------------------ candidates
    def _candidates(self, current: str, grads: List[torch.Tensor]) -> List[str]:
        tokenizer = self.vstar.primary_encoder.get_tokenizer()
        current_ids = tokenizer.encode(current, add_special_tokens=False)
        n_pos = min(len(tokenizer.tokenize(current)), min(g.shape[0] for g in grads))
        pos_candidates = {
            pos: self.vstar.find_topk_substitutions([g[pos] for g in grads]) for pos in range(n_pos)
        }
        texts = []
        for _ in range(self.config.search_width):
            new_ids = list(current_ids)
            for pos in torch.randperm(n_pos)[: self.config.n_replace].tolist():
                if pos_candidates.get(pos):
                    new_ids[pos] = int(np.random.choice(pos_candidates[pos]))
            texts.append(tokenizer.decode(new_ids[: self.config.max_suffix_tokens], skip_special_tokens=True))
        return texts

    def joint_step(self, queries: List[Tuple[str, Optional[str]]], current: str) -> Tuple[str, float]:
        """One step of Algorithm 1 on the active queries [(message, task_type), ...].

        Token gradients are summed over the queries (gradient of the summed
        L_A), and every candidate is scored by its mean loss over the queries.
        """
        grads = None
        for message, task in queries:
            g = self._token_gradients(message, current, task)
            grads = g if grads is None else [a + b for a, b in zip(grads, g)]
        with torch.no_grad():
            candidates = self._candidates(current, grads)
            cand_losses = [float(np.mean([self.loss(m, c, t) for m, t in queries])) for c in candidates]
        best = int(np.argmin(cand_losses))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return candidates[best], cand_losses[best]
