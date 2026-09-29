"""RouterDC (Chen et al., 2024): dual-contrastive routing with an mDeBERTa-v3
backbone. Scores are cosine similarities between the [CLS] embedding of the
query and one learned embedding per candidate LLM.
"""

from typing import Optional

import torch
import torch.nn as nn

from r2a.routers.base import HFEncoder, Router

ROUTERDC_MODELS = [
    "mistralai/Mistral-7B-v0.1",
    "meta-math/MetaMath-Mistral-7B",
    "itpossible/Chinese-Mistral-7B-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "cognitivecomputations/dolphin-2.6-mistral-7b",
    "meta-llama/Meta-Llama-3-8B",
    "cognitivecomputations/dolphin-2.9-llama3-8b",
]


class RouterDCModule(nn.Module):
    """Same parameter layout as the official RouterDC checkpoint."""

    def __init__(self, backbone, hidden_state_dim: int = 768, node_size: int = 7):
        super().__init__()
        self.backbone = backbone
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)


def cosine_scores(hidden: torch.Tensor, llm_embeddings: torch.Tensor) -> torch.Tensor:
    return (hidden @ llm_embeddings.T) / (
        torch.norm(hidden, dim=1).unsqueeze(1) * torch.norm(llm_embeddings, dim=1).unsqueeze(0)
    )


class RouterDC(Router):
    def __init__(
        self,
        checkpoint_path: str,
        name: str = "routerdc",
        device: str = "cuda",
        backbone_name: str = "microsoft/mdeberta-v3-base",
        max_length: int = 512,
    ):
        super().__init__(name, device)
        from transformers import AutoTokenizer, DebertaV2Model

        backbone = DebertaV2Model.from_pretrained(backbone_name)
        module = RouterDCModule(backbone, node_size=len(ROUTERDC_MODELS))
        module.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.backbone = module.backbone.to(device)
        self.embeddings = module.embeddings.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, truncation_side="left", padding=True)
        self.max_length = max_length
        self.backbone_name = backbone_name
        self._model_list = list(ROUTERDC_MODELS)

    @torch.no_grad()
    def route(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt + suffix, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        hidden = self.backbone(**tokens)["last_hidden_state"][:, 0, :]
        return cosine_scores(hidden, self.embeddings.weight).squeeze(0)

    def get_internal_encoder(self) -> HFEncoder:
        return HFEncoder(f"{self.name}_encoder", self.backbone, self.tokenizer)

    def forward_embeds(self, embeds, attention_mask=None, task_type=None) -> torch.Tensor:
        mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
        hidden = self.backbone(inputs_embeds=embeds.unsqueeze(0), attention_mask=mask).last_hidden_state[:, 0, :]
        return cosine_scores(hidden, self.embeddings.weight).squeeze(0)

    def assemble_gcg_input(self, messages, suffix_embeds, encoder=None, prefix_mode=False):
        """[CLS] [message] [suffix] [SEP] [PAD ...], padded to max_length like the tokenizer does."""
        encoder = encoder or self.get_internal_encoder()
        L = self.max_length
        encoded = self.tokenizer(
            messages, max_length=L, padding="max_length", truncation=True, return_tensors="pt"
        )
        embed_layer = encoder.get_model().get_input_embeddings()
        device = embed_layer.weight.device
        input_ids = encoded["input_ids"].to(device)
        valid_len = int(encoded["attention_mask"].sum().item())
        with torch.no_grad():
            input_embeds = embed_layer(input_ids).squeeze(0)
        cls_and_message = input_embeds[: valid_len - 1]
        sep = input_embeds[valid_len - 1].unsqueeze(0)
        pad = input_embeds[valid_len].unsqueeze(0) if valid_len < L else torch.zeros_like(sep)
        suffix_embeds = suffix_embeds.to(device)
        if prefix_mode:
            active = torch.cat([cls_and_message[:1], suffix_embeds, cls_and_message[1:], sep], dim=0)
        else:
            active = torch.cat([cls_and_message, suffix_embeds, sep], dim=0)
        n = active.size(0)
        if n > L:
            return active[:L], torch.ones(L, dtype=torch.long, device=device)
        full = torch.cat([active, pad.repeat(L - n, 1)], dim=0)
        mask = torch.cat([
            torch.ones(n, dtype=torch.long, device=device),
            torch.zeros(L - n, dtype=torch.long, device=device),
        ])
        return full, mask
