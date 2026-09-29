"""GraphRouter (Feng et al., 2025): a GNN over query / task / LLM nodes.

A new query is scored by adding it to a fixed context graph of 240 historical
queries (``third_party/graphrouter/configs/context.pt``) and predicting the
query-LLM edges. Queries are embedded with all-MiniLM-L6-v2 and the task node
is the embedded description of the query's source dataset.

Note: the GNN is run in training mode, i.e. BatchNorm uses the statistics of
the current graph (context + new query).
"""

import pickle
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from r2a.routers.base import HFEncoder, Router

GRAPH_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "graphrouter"

DEFAULT_CONFIG = {
    "embedding_dim": 32,
    "edge_dim": 3,
    "query_dim": 384,
    "sentence_encoder": "sentence-transformers/all-MiniLM-L6-v2",
}


class GraphRouter(Router):
    def __init__(
        self,
        name: str = "graphrouter",
        device: str = "cuda",
        model_path: Optional[str] = None,
        context_path: Optional[str] = None,
        sentence_encoder: Optional[str] = None,
        default_task: str = "mmlu",
    ):
        super().__init__(name, device)
        from sentence_transformers import SentenceTransformer

        from r2a.third_party.graphrouter.model.graph_nn import EncoderDecoderNet

        self.config = dict(DEFAULT_CONFIG)
        cfg_dir = GRAPH_ROOT / "configs"
        with open(cfg_dir / "LLM_Descriptions.json", "r", encoding="utf-8") as f:
            self._model_list = list(json.load(f).keys())
        with open(cfg_dir / "Task_Descriptions.json", "r", encoding="utf-8") as f:
            self.task_description = json.load(f)
        with open(cfg_dir / "llm_description_embedding.pkl", "rb") as f:
            llm_embeddings = pickle.load(f)
        self.num_llms = len(self._model_list)
        self.default_task = default_task

        self.model = EncoderDecoderNet(
            query_feature_dim=self.config["query_dim"],
            llm_feature_dim=llm_embeddings.shape[1],
            hidden_features=self.config["embedding_dim"],
            in_edges=self.config["edge_dim"],
        ).to(device)
        state = torch.load(model_path or GRAPH_ROOT / "model_path" / "best_model.pth", map_location=device)
        self.model.load_state_dict(state)

        self.sentence_encoder = SentenceTransformer(
            sentence_encoder or self.config["sentence_encoder"], device=device
        )
        self.sentence_encoder.eval()
        self.llm_features = torch.tensor(llm_embeddings, dtype=torch.float, device=device)
        self._load_context(context_path or cfg_dir / "context.pt")
        self._task_cache = {}

    def _load_context(self, path) -> None:
        ctx = torch.load(path, map_location=self.device)
        self._ctx_query = ctx["query_embeddings"].to(self.device)
        self._ctx_task = ctx["task_embeddings"].to(self.device)
        edge_weight = torch.zeros(ctx["effect"].numel(), self.config["edge_dim"], device=self.device)
        edge_weight[:, 0] = ctx["cost"].to(self.device)
        edge_weight[:, 1] = ctx["effect"].to(self.device)
        self._ctx_edge_weight = edge_weight

    def _task_embedding(self, task_type: Optional[str]) -> torch.Tensor:
        key = task_type if task_type in self.task_description else self.default_task
        if key not in self._task_cache:
            with torch.no_grad():
                emb = self.sentence_encoder.encode(
                    [self.task_description[key]["feature"]],
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                )
            self._task_cache[key] = emb.squeeze(0).clone()
        return self._task_cache[key]

    def predict_new_query(self, query_emb: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        """Score one new query against the fixed context graph."""
        batch_q = torch.cat([self._ctx_query, query_emb.unsqueeze(0)], dim=0)
        batch_t = torch.cat([self._ctx_task, task_emb.unsqueeze(0)], dim=0)
        total_queries = batch_q.shape[0]
        src = torch.arange(total_queries, device=self.device).repeat_interleave(self.num_llms)
        dst = torch.arange(total_queries, total_queries + self.num_llms, device=self.device).repeat(total_queries)
        edge_index = torch.stack([src, dst], dim=0)

        ctx_weight = self._ctx_edge_weight.clone()
        ctx_weight[:, 0] = ctx_weight[:, 0] * 0.01  # down-weight the cost feature of context edges
        new_weight = torch.zeros(self.num_llms, self.config["edge_dim"], device=self.device)
        edge_weight = torch.cat([ctx_weight, new_weight], dim=0)

        n_ctx = ctx_weight.shape[0]
        edge_mask = torch.cat([
            torch.zeros(n_ctx, dtype=torch.bool, device=self.device),
            torch.ones(self.num_llms, dtype=torch.bool, device=self.device),
        ])
        edge_can_see = ~edge_mask
        return self.model(
            task_id=batch_t,
            query_features=batch_q,
            llm_features=self.llm_features,
            edge_index=edge_index,
            edge_mask=edge_mask,
            edge_can_see=edge_can_see,
            edge_weight=edge_weight,
        )

    def route(self, prompt: str, suffix: str = "", type: Optional[str] = None) -> torch.Tensor:
        text = prompt + " " + suffix if suffix else prompt
        with torch.no_grad():
            query_emb = self.sentence_encoder.encode(
                [text], convert_to_tensor=True, device=self.device, show_progress_bar=False
            )
            return self.predict_new_query(query_emb.squeeze(0).clone(), self._task_embedding(type))

    def get_internal_encoder(self) -> HFEncoder:
        transformer = self.sentence_encoder[0]
        return HFEncoder(f"{self.name}_encoder", transformer.auto_model, transformer.tokenizer)

    def forward_embeds(self, embeds, attention_mask=None, task_type=None) -> torch.Tensor:
        # Mean pooling + L2 normalization, as in SentenceTransformer.encode.
        mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)
        hidden = self.sentence_encoder[0].auto_model(
            inputs_embeds=embeds.unsqueeze(0), attention_mask=mask.unsqueeze(0)
        ).last_hidden_state
        m = mask.unsqueeze(0).unsqueeze(-1).float()
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
        query_emb = F.normalize(pooled, p=2, dim=1).squeeze(0)
        return self.predict_new_query(query_emb, self._task_embedding(task_type))

    def assemble_gcg_input(self, messages, suffix_embeds, encoder=None, prefix_mode=False):
        """[CLS] [message] [suffix] [SEP]"""
        encoder = encoder or self.get_internal_encoder()
        tokens = encoder.get_tokenizer()(messages, add_special_tokens=True, truncation=True, return_tensors="pt")
        embed_layer = encoder.get_model().get_input_embeddings()
        msg = embed_layer(tokens["input_ids"].to(embed_layer.weight.device)).squeeze(0)
        suffix_embeds = suffix_embeds.to(msg.device)
        if prefix_mode:
            full = torch.cat([suffix_embeds, msg[:-1], msg[-1:]], dim=0)
        else:
            full = torch.cat([msg[:-1], suffix_embeds, msg[-1:]], dim=0)
        return full, torch.ones(full.size(0), dtype=torch.long, device=full.device)
