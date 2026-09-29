"""Candidate-token construction for the ensemble (Eq. 9-11).

Different routers use different tokenizers, so candidates are restricted to the
vocabulary V* of words that are a *single* token for every trainable router.
For each suffix position, every router scores all words in V* by the
first-order change of the loss (-grad . e_w); scores are min-max normalized per
router so that no router dominates, averaged, and the top-k words form the
candidate set C_i.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class VStarBuilder:
    def __init__(
        self,
        encoders: List,
        k: int = 256,
        max_common_words: int = 10000,
        device: str = "cuda",
    ):
        self.encoders = encoders
        self.k = k
        self.device = device
        self.max_common_words = max_common_words
        self.primary_encoder = encoders[0]
        self.common_words, self.common_token_ids, self.common_word_to_ids = self._find_common_single_token_words()
        self.embedding_pools = self._build_embedding_pools()
        logger.info(
            "V*: %d words that are single tokens for all %d encoders", len(self.common_words), len(encoders)
        )

    def _find_common_single_token_words(self) -> Tuple[List[str], List[int], Dict[str, List[int]]]:
        per_encoder = []
        for encoder in self.encoders:
            tokenizer = encoder.get_tokenizer()
            special = set(tokenizer.all_special_ids)
            words = {}
            for word, token_id in tokenizer.get_vocab().items():
                if token_id in special or word.startswith(("##", "Ġ", "▁")):
                    continue
                if len(tokenizer.encode(word, add_special_tokens=False)) == 1:
                    words[word] = token_id
            per_encoder.append(words)
            logger.info("  %s: %d single-token words", encoder.name, len(words))

        common = set.intersection(*(set(w) for w in per_encoder))
        if len(common) >= self.max_common_words:
            common = set(sorted(common)[: self.max_common_words])
        word_to_ids = {w: [vocab[w] for vocab in per_encoder] for w in common}

        # Row r of every embedding pool corresponds to words[r].
        words = sorted(word_to_ids)
        primary_vocab = per_encoder[0]
        token_ids = [primary_vocab[w] for w in words]
        return words, token_ids, word_to_ids

    def _build_embedding_pools(self) -> List[torch.Tensor]:
        pools = []
        for enc_idx, encoder in enumerate(self.encoders):
            ids = [self.common_word_to_ids[w][enc_idx] for w in self.common_words]
            pools.append(encoder.embed_matrix().detach()[ids].to(self.device))
        return pools

    def compute_score(self, gradient: torch.Tensor, encoder_idx: int) -> torch.Tensor:
        direction = -gradient
        dtype = direction.dtype
        if direction.device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        pool = self.embedding_pools[encoder_idx].to(device=direction.device, dtype=dtype)
        return pool @ direction.to(dtype)

    def find_topk_substitutions(
        self, gradients_per_encoder: List[torch.Tensor], current_token_id: Optional[int] = None
    ) -> List[int]:
        """Top-k primary-tokenizer ids for one position, given one gradient per encoder."""
        assert len(gradients_per_encoder) == len(self.encoders)
        normalized = []
        for enc_idx, gradient in enumerate(gradients_per_encoder):
            scores = self.compute_score(gradient, enc_idx).to(self.device).float()
            lo, hi = scores.min(), scores.max()
            normalized.append((scores - lo) / (hi - lo) if hi > lo else torch.zeros_like(scores))
        aggregated = torch.stack(normalized).mean(dim=0)
        if current_token_id is not None and current_token_id in self.common_token_ids:
            aggregated[self.common_token_ids.index(current_token_id)] = -float("inf")
        k = min(self.k, aggregated.numel())
        top = torch.topk(aggregated, k).indices.cpu().tolist()
        return [self.common_token_ids[i] for i in top]
