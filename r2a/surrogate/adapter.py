"""Hybrid ensemble surrogate router (Sec. 3.1).

``LowRankSemanticAdapter`` implements Eq. 3-5 of the paper:

* open-source routers: each router's scores are standardized (z-score over its
  own candidates), zero-padded into the union pool M_uni and mapped onto the
  target pool M_t with the trainable matrix W_o (``projection_matrix``,
  initialized from the similarity of model-name embeddings): z_o^(k) = W_o z_uni^(k);
* lightweight router: z_l = E(q) W_l^1 W_l^2 with E = all-MiniLM-L6-v2 and rank r;
* surrogate: y = softmax(alpha_0 z_l + sum_k alpha_k z_o^(k)) with
  alpha = softmax(router_weights) over the K + 1 routers (alpha >= 0, sum 1).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class HFTextEmbedder(nn.Module):
    """Frozen sentence embedder (mean pooling + L2 norm)."""

    def __init__(self, model_name: str, device, local_files_only: bool = False):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: Union[str, List[str]]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.encoder(**toks)
        last = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).expand(last.size()).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        return F.normalize(emb, p=2, dim=-1)


class LowRankSemanticAdapter(nn.Module):
    def __init__(
        self,
        unified_model_names: List[str],
        target_model_names: List[str],
        text_embedder_name: str,
        device,
        temperature: float = 1.0,
        local_files_only: bool = False,
        num_routers: int = 0,
        embedding_dim: int = 384,
        rank: int = 16,
    ):
        super().__init__()
        self.device = device
        self.num_routers = num_routers
        self.num_unified = len(unified_model_names)
        self.num_target = len(target_model_names)
        self.embedder = HFTextEmbedder(text_embedder_name, device, local_files_only=local_files_only)

        # W_o (U x L), initialized from the similarity of model-name embeddings
        with torch.no_grad():
            emb_u = self.embedder.encode_texts(unified_model_names)
            emb_l = self.embedder.encode_texts(target_model_names)
            init_w = F.softmax(emb_u @ emb_l.T / temperature, dim=0)
        self.projection_matrix = nn.Parameter(init_w)

        # alpha_0 (lightweight router) and alpha_1..K; zeros -> uniform after softmax
        self.router_weights = nn.Parameter(torch.zeros(num_routers + 1, device=device))

        # lightweight router z_l = E(q) W_l^1 W_l^2
        self.semantic_encoder = nn.Sequential(
            nn.Linear(embedding_dim, rank, bias=False),          # W_l^1
            nn.Linear(rank, self.num_target, bias=False),        # W_l^2
        )
        nn.init.normal_(self.semantic_encoder[0].weight, std=0.01)
        nn.init.zeros_(self.semantic_encoder[-1].weight)

    @staticmethod
    def _normalize_router_logits(stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Z-normalize each router's scores over its own (valid) candidates."""
        eps = 1e-6
        mask_f = mask.float()
        valid_cnt = mask_f.sum(dim=2, keepdim=True).clamp(min=1.0)
        mean = (stack * mask_f).sum(dim=2, keepdim=True) / valid_cnt
        var = ((stack - mean) ** 2 * mask_f).sum(dim=2, keepdim=True) / valid_cnt
        return (stack - mean) / torch.sqrt(var + eps) * mask_f

    def get_router_weights(self) -> torch.Tensor:
        """alpha = [alpha_0, alpha_1..K]."""
        return F.softmax(self.router_weights, dim=0)

    def _check_members(self, R: int) -> None:
        if R != self.num_routers:
            raise ValueError(
                f"adapter was trained with {self.num_routers} open-source routers but got {R}; "
                "the ensemble members (and their order) must match the checkpoint"
            )

    def lowrank(self, embedding: torch.Tensor) -> torch.Tensor:
        """z_l from sentence embeddings E(q)."""
        return self.semantic_encoder(embedding)

    def combine(self, aligned_stack: torch.Tensor, mask_stack: torch.Tensor, lowrank_logits: torch.Tensor) -> torch.Tensor:
        """Surrogate logits (B, L) from member scores (B, R, U) and z_l (B, L)."""
        B, R, _ = aligned_stack.shape
        self._check_members(R)
        projected = self._normalize_router_logits(aligned_stack, mask_stack) @ self.projection_matrix  # (B, R, L)
        lowrank_logits = lowrank_logits.to(projected.device).reshape(B, -1)
        valid = (mask_stack.sum(dim=2) > 0).float()  # members with no candidate in M_uni get no weight
        w = torch.cat([torch.ones(B, 1, device=valid.device), valid], dim=1) * self.get_router_weights().view(1, R + 1)
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
        return w[:, :1] * lowrank_logits + (projected * w[:, 1:].unsqueeze(-1)).sum(dim=1)

    def forward(
        self,
        aligned_stack: torch.Tensor,
        mask_stack: torch.Tensor,
        prompts: Optional[Union[str, List[str]]] = None,
        lowrank_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Surrogate logits on the target pool.

        ``lowrank_logits`` lets the attack pass z_l computed with gradients
        w.r.t. the suffix; otherwise it is computed from ``prompts``.
        """
        if lowrank_logits is None:
            with torch.no_grad():
                embedding = self.embedder.encode_texts(prompts)
            lowrank_logits = self.lowrank(embedding)
        return self.combine(aligned_stack, mask_stack, lowrank_logits)


def build_router_stack(
    prompts: List[str],
    routers: List,
    unified_model_names: List[str],
    device,
    types: Optional[List[Optional[str]]] = None,
    suffix: str = "",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Query every open-source router and align its scores into M_uni (zero padding)."""
    u_map = {n: i for i, n in enumerate(unified_model_names)}
    B, R, U = len(prompts), len(routers), len(unified_model_names)
    aligned = torch.zeros((B, R, U), device=device)
    mask = torch.zeros((B, R, U), dtype=torch.bool, device=device)
    with torch.no_grad():
        for r_idx, router in enumerate(routers):
            scores = torch.stack([
                router.route(p, suffix=suffix, type=(types[i] if types else None)).float().to(device)
                for i, p in enumerate(prompts)
            ])
            for local_idx, m in enumerate(router.get_model_list()):
                if m in u_map:
                    aligned[:, r_idx, u_map[m]] = scores[:, local_idx]
                    mask[:, r_idx, u_map[m]] = True
    return aligned, mask


def save_adapter_checkpoint(
    adapter: LowRankSemanticAdapter,
    path: Union[str, Path],
    unified_model_names: List[str],
    target_model_names: List[str],
    router_names: List[str],
    text_embedder_name: str,
    history: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in adapter.state_dict().items() if not k.startswith("embedder.")}
    torch.save(
        {
            "state_dict": state,
            "unified_model_names": unified_model_names,
            "target_model_names": target_model_names,
            "router_names": router_names,
            "history": history or {},
            "config": {
                "text_embedder_name": text_embedder_name,
                "embedding_dim": adapter.semantic_encoder[0].in_features,
                "rank": adapter.semantic_encoder[0].out_features,
                "num_routers": adapter.num_routers,
                "adapter_class": type(adapter).__name__,
            },
        },
        path,
    )
    logger.info("Saved surrogate checkpoint to %s", path)


def load_adapter_checkpoint(
    path: Union[str, Path], device, text_embedder_name: Optional[str] = None
) -> Tuple[LowRankSemanticAdapter, List[str], List[str], List[str]]:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt.get("config", {})
    adapter = LowRankSemanticAdapter(
        unified_model_names=ckpt["unified_model_names"],
        target_model_names=ckpt["target_model_names"],
        text_embedder_name=text_embedder_name or cfg.get("text_embedder_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=device,
        embedding_dim=cfg.get("embedding_dim", 384),
        rank=cfg.get("rank", 16),
        num_routers=cfg.get("num_routers", 0),
    ).to(device)
    missing, unexpected = adapter.load_state_dict(ckpt["state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith("embedder.")]
    if missing or unexpected:
        logger.warning("checkpoint mismatch: missing=%s unexpected=%s", missing, unexpected)
    adapter.eval()
    return adapter, ckpt["unified_model_names"], ckpt["target_model_names"], ckpt.get("router_names", [])
