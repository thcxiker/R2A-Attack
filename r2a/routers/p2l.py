"""Prompt-to-Leaderboard (P2L) routers (Frick et al., 2025).

P2L predicts a Bradley-Terry coefficient per model from the hidden state of a
trailing CLS token. With ``consider_cost=True`` the coefficient of each model
is reduced by ``cost_lambda * cost(model)`` (per-token price from
``configs/p2l_model_cost.json``), which is the cost-aware routing rule used for
the P2L target in the paper.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from r2a.routers.base import HFEncoder, Router

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COST_FILE = REPO_ROOT / "configs" / "p2l_model_cost.json"


def _get_p2l_model():
    """Import P2L's model factory from https://github.com/lmarena/p2l.

    The P2L code is not redistributed with R2A; run scripts/setup_p2l.sh (clones
    it into external/p2l) or point R2A_P2L_PATH to a checkout.
    """
    for path in (os.environ.get("R2A_P2L_PATH"), REPO_ROOT / "external" / "p2l"):
        if path and Path(path).exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from p2l.model import get_p2l_model
    except ImportError as e:
        raise ImportError("P2L code not found: run `bash scripts/setup_p2l.sh` or set R2A_P2L_PATH") from e
    return get_p2l_model


class P2LRouter(Router):
    def __init__(
        self,
        name: str = "p2l",
        model_path: str = "lmarena-ai/p2l-7b-grk-02222025",
        model_type: str = "qwen2",
        head_type: str = "bt",
        loss_type: str = "bt",
        device: str = "cuda",
        max_length: int = 8192,
        local_files_only: bool = False,
        consider_cost: bool = False,
        cost_file: Optional[str] = None,
        cost_lambda: float = 0.001,
    ):
        super().__init__(name, device)
        from transformers import AutoTokenizer

        get_p2l_model = _get_p2l_model()

        self.model_path = model_path
        self.max_length = max_length
        self.consider_cost = consider_cost
        self.cost_lambda = cost_lambda

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        local_list = Path(model_path) / "model_list.json"
        if local_list.exists():
            list_file = local_list
        else:
            from huggingface_hub import hf_hub_download

            list_file = hf_hub_download(model_path, "model_list.json", local_files_only=local_files_only)
        with open(list_file, "r", encoding="utf-8") as f:
            model_list = json.load(f)
        self._model_list = model_list.get("models", []) if isinstance(model_list, dict) else model_list

        model_cls = get_p2l_model(model_type=model_type, loss_type=loss_type, head_type=head_type)
        self.model = model_cls.from_pretrained(
            model_path,
            CLS_id=self.tokenizer.cls_token_id,
            num_models=len(self._model_list),
            torch_dtype=torch.bfloat16,
            device_map=device,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.model.eval()

        with open(cost_file or DEFAULT_COST_FILE, "r", encoding="utf-8") as f:
            costs = json.load(f)["model_configs"]
        cost_dict = {item["model_name"]: item["cost"] for item in costs}
        # models without a listed price get a default cost of 1.0
        self.model_costs = torch.tensor(
            [cost_dict.get(m, 1.0) for m in self._model_list], dtype=torch.float32
        )

    def _apply_cost(self, coefs: torch.Tensor) -> torch.Tensor:
        if not self.consider_cost:
            return coefs
        # computed in fp32 and rounded once to the model dtype
        penalty = self.cost_lambda * self.model_costs.to(device=coefs.device)
        return (coefs.float() - penalty).to(coefs.dtype)

    def _format(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
        )
        return formatted + self.tokenizer.cls_token

    @torch.no_grad()
    def route(self, prompt, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        text = (" ".join(prompt) if isinstance(prompt, list) else prompt) + suffix
        inputs = self.tokenizer(
            self._format(text),
            return_tensors="pt",
            max_length=self.max_length,
            padding="longest",
            truncation=True,
        ).to(self.model.device)
        coefs = self.model(**inputs).coefs.squeeze(0)
        return self._apply_cost(coefs)

    def get_internal_encoder(self) -> HFEncoder:
        return HFEncoder(f"{self.name}_encoder", self.model, self.tokenizer)

    def forward_embeds(self, embeds, attention_mask=None, task_type=None) -> torch.Tensor:
        """P2L forward from input embeddings; the CLS token is the last position."""
        mask = attention_mask.unsqueeze(0) if attention_mask is not None else None
        hidden = self.model.model(
            inputs_embeds=embeds.unsqueeze(0).to(dtype=self.model.model.dtype),
            attention_mask=mask,
            output_hidden_states=False,
        ).last_hidden_state
        coefs = self.model.head(hidden[:, -1, :]).coefs
        return self._apply_cost(coefs.squeeze(0))

    def assemble_gcg_input(self, messages, suffix_embeds, encoder=None, prefix_mode=False):
        """[3 template tokens] [message ...] [suffix] [CLS]: the CLS token stays last."""
        target_device = suffix_embeds.device
        embed_layer = self.model.get_input_embeddings()
        encoded = self.tokenizer(
            self._format(messages),
            return_tensors="pt",
            max_length=self.max_length,
            padding="longest",
            truncation=True,
        )
        with torch.no_grad():
            input_embeds = embed_layer(encoded["input_ids"].to(embed_layer.weight.device)).squeeze(0)
        last = input_embeds[-1:]
        head = input_embeds[:3]
        body = input_embeds[3:-1] if input_embeds.size(0) > 4 else input_embeds
        suffix_embeds = suffix_embeds.to(input_embeds.device)
        if prefix_mode:
            parts = [head, suffix_embeds, body, last]
        else:
            parts = [head, body, suffix_embeds, last]
        full = torch.cat(parts, dim=0).to(target_device)
        return full, torch.ones(full.size(0), dtype=torch.long, device=target_device)
