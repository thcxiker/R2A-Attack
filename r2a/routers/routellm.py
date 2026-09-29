"""RouteLLM routers (Ong et al., 2024): BERT classifier, causal-LLM classifier,
matrix factorization (MF) and similarity-weighted ranking (SW).

Scores are ``[strong_win_rate, 0]`` over the pool ``[strong_model, weak_model]``.
With ``consider_cost=True`` (the deployed, cost-aware router) the win rate is
shifted by the calibrated routing threshold, so the router picks the strong
model exactly when ``win_rate >= threshold``.
"""

import json
from pathlib import Path
from typing import Optional

import torch

from r2a.routers.base import HFEncoder, Router

# Checkpoints released by RouteLLM (trained on GPT-4-augmented Arena data).
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}

DEFAULT_THRESHOLDS = Path(__file__).resolve().parents[2] / "configs" / "routellm_thresholds.json"


class RouteLLMRouter(Router):
    def __init__(
        self,
        name: str,
        type: str = "bert",
        strong_model: str = "gpt-4",
        weak_model: str = "mixtral-8x7b",
        device: str = "cuda",
        consider_cost: bool = False,
        checkpoint_path: Optional[str] = None,
        model_id: Optional[str] = None,
        threshold: Optional[float] = None,
        thresholds_file: Optional[str] = None,
        device_map: Optional[str] = None,
    ):
        super().__init__(name, device)
        from r2a.third_party.routellm.routers.routers import ROUTER_CLS

        router_kwargs = dict(GPT_4_AUGMENTED_CONFIG.get(type, {}))
        if checkpoint_path:
            router_kwargs["checkpoint_path"] = checkpoint_path
        if model_id and type == "causal_llm":
            router_kwargs["model_id"] = model_id
        if device_map and type == "causal_llm":
            router_kwargs["device_map"] = device_map  # default: spread over all visible GPUs
        self.router = ROUTER_CLS[type](**router_kwargs)
        self.type = type
        self._model_list = [strong_model, weak_model]
        self.consider_cost = consider_cost

        if threshold is None:
            with open(thresholds_file or DEFAULT_THRESHOLDS, "r", encoding="utf-8") as f:
                threshold = json.load(f).get(type, 0.5)
        self.threshold = float(threshold)

        if type == "causal_llm":
            self.router_model = self.router.router_model
            self.model = self.router_model.model
            self.tokenizer = self.router_model.tokenizer
        elif type == "bert":
            self.model = self.router.model.to(device)
            self.tokenizer = self.router.tokenizer

    def route(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        if self.type == "bert":
            win_rate = self._bert_strong_win_rate(prompt + suffix)
        else:
            win_rate = self.router.calculate_strong_win_rate(prompt + suffix)
        if self.consider_cost:
            win_rate -= self.threshold
        return torch.tensor([win_rate, 0.0], dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _bert_strong_win_rate(self, text: str) -> float:
        # Same computation as RouteLLM's BERTRouter, but on the router's device.
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        probs = torch.softmax(self.model(**inputs).logits.float()[0], dim=-1)
        return 1.0 - probs[-2:].sum().item()

    def get_internal_encoder(self) -> HFEncoder:
        if self.type not in ("bert", "causal_llm"):
            raise ValueError(f"RouteLLM '{self.type}' has no differentiable encoder (API embeddings)")
        return HFEncoder(f"{self.name}_encoder", self.model, self.tokenizer)

    def forward_embeds(self, embeds, attention_mask=None, task_type=None) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)
        input_embeds = embeds.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        if self.type == "causal_llm":
            orig_vocab_size = self.router_model.orig_vocab_size
            logits = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
            score_logits = logits[0, -1, orig_vocab_size:]
            binary_prob, _ = self.router_model.compute_routing_prob(score_logits)
            win_rate = 1 - binary_prob
        elif self.type == "bert":
            logits = self.model(inputs_embeds=input_embeds, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)
            win_rate = 1 - probs[:, -2:].sum(dim=-1).squeeze(0)
        else:
            raise ValueError(f"Unsupported RouteLLM type for gradients: {self.type}")
        if self.consider_cost:
            win_rate = win_rate - self.threshold
        zero = torch.zeros((), device=win_rate.device, dtype=win_rate.dtype)
        return torch.stack([win_rate, zero])

    def assemble_gcg_input(self, messages, suffix_embeds, encoder=None, prefix_mode=False):
        target_device = suffix_embeds.device
        embed_layer = self.model.get_input_embeddings()
        if self.type == "causal_llm":
            row = self.router_model.preprocess({"messages": self.router.to_openai_messages([messages])})
            input_ids = torch.as_tensor(row["input_ids"]).to(embed_layer.weight.device).reshape(1, -1)
            input_embeds = embed_layer(input_ids).squeeze(0)
            # Insert the suffix right after the user question, i.e. before the
            # "Prediction" part of the classifier prompt.
            prediction_id = self.tokenizer.convert_tokens_to_ids("Prediction")
            positions = [i for i, t in enumerate(input_ids.squeeze(0).tolist()) if t == prediction_id]
            cut = positions[-1] - 1
            head, tail = input_embeds[:cut], input_embeds[cut:]
            # The classifier first generates the assistant header and then the
            # rating token [[1]]..[[5]]; appending the header makes the last
            # position predict the rating, as in route().
            header_ids = self.tokenizer.encode(
                "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
            )
            header = embed_layer(torch.tensor(header_ids, device=embed_layer.weight.device))
            tail = torch.cat([tail, header], dim=0)
            suffix_part = suffix_embeds[1:].to(head.device)
            if prefix_mode:
                parts = [suffix_part, head, tail]
            else:
                parts = [head, suffix_part, tail]
        elif self.type == "bert":
            inputs = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True)
            input_embeds = embed_layer(inputs["input_ids"].to(embed_layer.weight.device)).squeeze(0)
            # message tokens without the trailing </s>, followed by the suffix
            parts = [input_embeds[:-1], suffix_embeds[1:].to(input_embeds.device)]
        else:
            raise ValueError(f"Unsupported RouteLLM type for gradients: {self.type}")
        full = torch.cat(parts, dim=0).to(target_device)
        return full, torch.ones(full.size(0), dtype=torch.long, device=target_device)
