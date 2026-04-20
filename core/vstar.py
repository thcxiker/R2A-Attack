import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import json
from pathlib import Path

from reroute.core.encoder import EncoderSpecBase, EncoderFactory, TransformerEncoderSpec
from reroute.core.attack_config import AttackConfig, RouterAttackConfig


class VStarBuilder:
    """
    V* Builder: Build Top-k candidate set for each position based on gradients
    
    Core functionality:
    1. Find common single-token words across all encoders (intersection)
    2. Get embeddings for each token in the intersection from each encoder
    3. Calculate score for each token (based on gradient direction)
    4. Normalize and take Top-k
    """
    
    def __init__(
        self,
        encoders: List[EncoderSpecBase],
        k: int = 256,
        Max_Common_Words: int = 10000,
        device: str = "cuda"
    ):
        """
        Args:
            encoders: List of multiple EncoderSpecBase objects
            k: Top-k size
            device: Device
        """
        self.encoders = encoders
        self.k = k
        self.device = device
        print("self.device is:", self.device)
        self.Max_Common_Words = Max_Common_Words
        # ✅ Use first encoder as primary encoder
        self.primary_encoder = encoders[0]
        
        print(f"\n🔍 Building V* embedding pool from {len(encoders)} encoders...")
        
        # ✅ Step 1: Find common single-token words across all encoders
        self.common_single_token_ids, self.common_word_to_ids = self._find_common_single_token_words()
        
        print(f"✓ Found {len(self.common_single_token_ids)} common single-token words")
        
        # ✅ Step 2: Get embeddings of common vocabulary for each encoder
        self.embedding_pools = self._build_embedding_pools()
        print(f"  Each pool has shape: {[pool.shape for pool in self.embedding_pools]}")
        
        print(f"✓ Built embedding pools for {len(self.encoders)} encoders")
    
    def _find_common_single_token_words(self) -> Tuple[List[int], Dict[str, List[int]]]:
        """
        Find common single-token words across all encoders
        
        Returns:
            common_token_ids: List of token IDs in primary encoder
            word_to_ids: {word: [encoder0_id, encoder1_id, ...]}
        """
        single_token_words_per_encoder = []
        vocab_per_encoder = []  # Store {word: token_id} for each encoder
        
        for encoder in self.encoders:
            tokenizer = encoder.get_tokenizer()
            single_token_words = {}  # word -> token_id
            vocab = tokenizer.get_vocab()
            
            for word, token_id in vocab.items():
                tokens = tokenizer.encode(word, add_special_tokens=False)
                
                if len(tokens) == 1:
                    if (token_id not in tokenizer.all_special_ids and
                        not word.startswith('##') and
                        not word.startswith('Ġ') and
                        not word.startswith('▁')):
                        single_token_words[word] = token_id
            
            single_token_words_per_encoder.append(set(single_token_words.keys()))
            vocab_per_encoder.append(single_token_words)
            print(f"  {encoder.name}: {len(single_token_words)} single-token words")
        
        # ✅ Take intersection
        common_words = set.intersection(*single_token_words_per_encoder)
        print(f"  Common intersection: {len(common_words)} words")
        
        if(len(common_words) >= self.Max_Common_Words):
            common_words = set(list(common_words)[:self.Max_Common_Words])
            print(f"  Trimmed common words to {self.Max_Common_Words}")
        # ✅ Build mapping: word -> [enc0_id, enc1_id, ...]
        word_to_ids = {}
        for word in common_words:
            ids = [vocab[word] for vocab in vocab_per_encoder]
            word_to_ids[word] = ids
        
        # ✅ Primary encoder's token IDs
        primary_tokenizer = self.primary_encoder.get_tokenizer()
        primary_vocab = primary_tokenizer.get_vocab()
        common_token_ids = sorted([primary_vocab[word] for word in common_words])
        
        return common_token_ids, word_to_ids
    
    def _build_embedding_pools(self) -> List[torch.Tensor]:
        """
        Build embedding pool for each encoder
        
        Returns:
            embedding_pools: [
                encoder0_embeddings: [num_common_tokens, dim0],
                encoder1_embeddings: [num_common_tokens, dim1],
                ...
            ]
        """
        pools = []
        
        for encoder_idx, encoder in enumerate(self.encoders):
            embed_matrix = encoder.embed_matrix()  # [vocab_size, embed_dim]
            
            # Extract embeddings corresponding to common vocabulary
            token_ids_for_this_encoder = [
                self.common_word_to_ids[word][encoder_idx]
                for word in sorted(self.common_word_to_ids.keys())
            ]
            
            # Ensure order is consistent with common_single_token_ids
            pool = embed_matrix[token_ids_for_this_encoder]  # [num_common, dim]
            # ✅ Uniformly convert to float32 and move to device
            pool = pool.to(device=self.device)
            pools.append(pool)
            
            print(f"  {encoder.name}: pool shape {pool.shape}")
        
        return pools
    
    def compute_gradient_per_position(self, optim_ids, encoder_idx):
        """
        Compute gradient for each position (embedding space)
        
        Returns:
            grads: [L, embed_dim]
        """
        encoder = self.encoders[encoder_idx]
        embed_layer = encoder.get_model().get_input_embeddings()
        
        # ✅ Directly get embeddings (Method B)
        optim_embeds = embed_layer(optim_ids)  # [1, L, embed_dim]
        optim_embeds.requires_grad_()
        
        # Forward
        loss = self._compute_loss(optim_embeds, encoder_idx)
        
        # Backward
        grad = torch.autograd.grad(loss, optim_embeds)[0]  # [1, L, embed_dim]
        
        return grad.squeeze(0)  # [L, embed_dim]

    def compute_score(self, gradient, encoder_idx):
        """
        gradient: [embed_dim] gradient at a certain position
        """
        opt_direction = -gradient  # Optimization direction
        
        # Embeddings of all candidate tokens
        pool = self.embedding_pools[encoder_idx]  # [num_common, embed_dim]
        # print("pool dtype is:", pool.dtype)
        # print("opt_direction dtype is:", opt_direction.dtype)
        # Calculate score using dot product
        # pool = pool.to(dtype=opt_direction.dtype)
        pool = pool.to(device=opt_direction.device, dtype=opt_direction.dtype)

        scores = pool @ opt_direction  # [num_common]
        
        return scores
    
    def find_topk_substitutions(
        self,
        gradients_per_encoder: List[torch.Tensor],
        current_token_id: Optional[int] = None,
        verbose: bool = False  # ✅ Add verbose parameter
    ) -> List[int]:
        """
        Find Top-k candidates based on gradients from multiple encoders
        
        Args:
            gradients_per_encoder: [
                encoder0_gradient: [embed_dim0],
                encoder1_gradient: [embed_dim1],
                ...
            ]
            current_token_id: Current token at this position (for exclusion)
            verbose: Whether to print detailed information
            
        Returns:
            topk_tokens: k token IDs (from primary encoder)
        """
        # ✅ Step 1: Calculate scores for each encoder
        all_scores = []
        
        if verbose:
            print("\n" + "="*60)
            print("Computing scores for each encoder:")
            print("="*60)
        assert len(gradients_per_encoder) == len(self.encoders)
        for encoder_idx, gradient   in enumerate(gradients_per_encoder):
            scores = self.compute_score(gradient, encoder_idx)  # [num_common]
            scores = scores.to(self.device)
            all_scores.append(scores)
            
            # ✅ Print score statistics for each encoder
            if verbose:
                encoder_name = self.encoders[encoder_idx].name
                print(f"\nEncoder {encoder_idx} ({encoder_name}):")
                print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
                print(f"  Score mean: {scores.mean().item():.4f}")
                print(f"  Score std: {scores.std().item():.4f}")
                
                # Print Top-5 candidates
                top5_vals, top5_indices = torch.topk(scores, min(5, len(scores)))
                print(f"  Top-5 candidates:")
                tokenizer = self.primary_encoder.get_tokenizer()
                for rank, (val, idx) in enumerate(zip(top5_vals, top5_indices)):
                    token_id = self.common_single_token_ids[idx]
                    word = tokenizer.decode([token_id])
                    print(f"    {rank+1}. score={val.item():.4f}, token_id={token_id}, word='{word}'")
        
        # ✅ Step 2: Normalize scores for each encoder
        normalized_scores = []
        
        if verbose:
            print("\n" + "="*60)
            print("Normalized scores:")
            print("="*60)
        
        for encoder_idx, scores in enumerate(all_scores):
            # Min-Max normalization to [0, 1]
            min_val = scores.min()
            max_val = scores.max()
            if max_val > min_val:
                norm_scores = (scores - min_val) / (max_val - min_val)
            else:
                norm_scores = torch.zeros_like(scores)
            normalized_scores.append(norm_scores)
            
            if verbose:
                encoder_name = self.encoders[encoder_idx].name
                print(f"\nEncoder {encoder_idx} ({encoder_name}):")
                print(f"  Normalized range: [{norm_scores.min().item():.4f}, {norm_scores.max().item():.4f}]")
                print(f"  Normalized mean: {norm_scores.mean().item():.4f}")
        
        # ✅ Step 3: Aggregate (average)
        aggregated_scores = torch.stack(normalized_scores).mean(dim=0)  # [num_common]
        
        if verbose:
            print("\n" + "="*60)
            print("Aggregated scores:")
            print("="*60)
            print(f"  Range: [{aggregated_scores.min().item():.4f}, {aggregated_scores.max().item():.4f}]")
            print(f"  Mean: {aggregated_scores.mean().item():.4f}")
            print(f"  Std: {aggregated_scores.std().item():.4f}")
        
        # ✅ Step 4: Exclude current token
        if current_token_id is not None and current_token_id in self.common_single_token_ids:
            idx = self.common_single_token_ids.index(current_token_id)
            aggregated_scores[idx] = -float('inf')
        
        # ✅ Step 5: Top-k
        k = min(self.k, len(aggregated_scores))
        topk_vals, topk_indices = torch.topk(aggregated_scores, k)
        topk_tokens = [self.common_single_token_ids[idx] for idx in topk_indices.cpu().tolist()]
        # words = [self.primary_encoder.get_tokenizer().decode([tid]) for tid in topk_tokens]
        if verbose:
            print(f"\n" + "="*60)
            print(f"Final Top-{k} candidates:")
            print("="*60)
            tokenizer = self.primary_encoder.get_tokenizer()
            for rank, (val, token_id) in enumerate(zip(topk_vals[:10], topk_tokens[:10])):
                word = tokenizer.decode([token_id])
                print(f"  {rank+1}. score={val.item():.4f}, token_id={token_id}, word='{word}'")
        
        return topk_tokens
    
    def build_vstar(
        self,
        gradients_per_position: List[List[torch.Tensor]],
        current_suffix_ids: torch.Tensor
    ) -> List[List[int]]:
        """
        Build V*: Top-k candidate set for each position
        
        Args:
            gradients_per_position: [
                position_0: [encoder0_grad, encoder1_grad, ...],
                position_1: [encoder0_grad, encoder1_grad, ...],
                ...
            ]
            current_suffix_ids: [L] current suffix
            
        Returns:
            vstar: [[k], [k], ...] Top-k tokens for each position
        """
        L = len(gradients_per_position)
        vstar = []
        
        for i in range(L):
            topk_tokens = self.find_topk_substitutions(
                gradients_per_position[i],
                current_token_id=current_suffix_ids[i].item()
            )
            vstar.append(topk_tokens)
        
        return vstar
    
    def get_embedding_from_ids(
        self,
        token_ids: torch.Tensor,
        encoder_idx: int = 0
    ) -> torch.Tensor:
        """
        Get embeddings from token IDs
        
        Args:
            token_ids: [L] or [batch_size, L]
            encoder_idx: Which encoder to use (default primary encoder)
            
        Returns:
            embeddings: [L, embed_dim] or [batch_size, L, embed_dim]
        """
        encoder = self.encoders[encoder_idx]
        embed_matrix = encoder.embed_matrix()
        return embed_matrix[token_ids]
    
    def get_common_vocab(self) -> Dict[str, int]:
        """Return common vocabulary (word -> primary encoder's token ID)"""
        primary_tokenizer = self.primary_encoder.get_tokenizer()
        primary_vocab = primary_tokenizer.get_vocab()
        
        return {
            word: primary_vocab[word]
            for word in self.common_word_to_ids.keys()
        }
    
    def get_common_words(self) -> List[str]:
        """Return list of common single-token words"""
        return sorted(self.common_word_to_ids.keys())


# ============================================================
# Usage Example
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("V* Builder: Multi-Encoder Score Aggregation")
    print("=" * 60)
    
    # Load all encoders
    print("\nLoading encoders from YAML...")
    encoders = EncoderFactory.create_all_from_yaml(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loaded {len(encoders)} encoders")
    
    # Create VStarBuilder
    vstar_builder = VStarBuilder(
        encoders=encoders,
        k=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Test score computation
    print("\n" + "=" * 60)
    print("Testing score computation...")
    print("=" * 60)
    
    # Simulate gradients (on correct device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gradients_per_encoder = [
        torch.randn(enc.embedding_dim(), device=device, dtype=torch.float32) 
        for enc in encoders
    ]
    
    print(f"\nSimulated gradients:")
    for i, g in enumerate(gradients_per_encoder):
        print(f"  Encoder {i}: shape {g.shape}, device {g.device}, dtype {g.dtype}")
    
    # ✅ Find Top-k (enable verbose output)
    topk = vstar_builder.find_topk_substitutions(gradients_per_encoder, verbose=True)
    
    print(f"\n{'='*60}")
    print(f"Summary: Top-{len(topk)} candidates selected")
    print(f"{'='*60}")
    
    # Save common vocabulary
    common_words = vstar_builder.get_common_words()
    output_path = Path("./common_vocab.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(common_words, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(common_words)} common words to {output_path}")