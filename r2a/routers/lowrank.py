"""The trainable lightweight router R_l of the surrogate, exposed as a router so
that the suffix optimizer can take gradients through it like any other member."""

from typing import Optional

import torch
import torch.nn.functional as F

from r2a.routers.base import HFEncoder, Router


class LowRankRouter(Router):
    def __init__(
        self,
        adapter,
        target_model_names,
        name: str = "lowrank",
        device: str = "cuda",
    ):
        super().__init__(name, device)
        self.tokenizer = adapter.embedder.tokenizer
        self.encoder = adapter.embedder.encoder
        self.semantic_encoder = adapter.semantic_encoder
        self._model_list = list(target_model_names)
        self.encoder.eval()

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).float()
        pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp_min(1e-8)
        return F.normalize(pooled, p=2, dim=-1)

    def route(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        toks = self.tokenizer([prompt + suffix], padding=True, truncation=True, max_length=128, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            hidden = self.encoder(**toks).last_hidden_state
            return self.semantic_encoder(self._pool(hidden, toks["attention_mask"])).squeeze(0)

    def get_internal_encoder(self) -> HFEncoder:
        return HFEncoder(f"{self.name}_encoder", self.encoder, self.tokenizer)

    def forward_embeds(self, embeds, attention_mask=None, task_type=None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(embeds.size(0), device=embeds.device)
        attention_mask = attention_mask.float()
        position_ids = torch.arange(embeds.size(0), device=embeds.device).unsqueeze(0)
        hidden = self.encoder(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            position_ids=position_ids,
        ).last_hidden_state
        return self.semantic_encoder(self._pool(hidden, attention_mask.unsqueeze(0))).squeeze(0)

    def assemble_gcg_input(self, messages, suffix_embeds, encoder=None, prefix_mode=False):
        """[CLS] [message] [suffix] [SEP]"""
        toks = self.tokenizer(messages, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            msg = self.encoder.get_input_embeddings()(toks["input_ids"].to(self.device)).squeeze(0)
        suffix_embeds = suffix_embeds.to(msg.device)
        if prefix_mode:
            full = torch.cat([msg[:1], suffix_embeds, msg[1:-1], msg[-1:]], dim=0)
        else:
            full = torch.cat([msg[:-1], suffix_embeds, msg[-1:]], dim=0)
        return full, torch.ones(full.size(0), dtype=torch.long, device=full.device)
