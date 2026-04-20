import warnings

from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple,Union
from tqdm import tqdm
from dataclasses import dataclass
import json
from reroute.utils.build import *
from reroute.utils.loss import *
from reroute.utils.utils import *
from reroute.utils.model_classifier import ModelClassifier, classify_model_lists
from reroute.core.encoder import EncoderSpecBase, EncoderFactory
from reroute.core.router import *
from reroute.core.vstar import VStarBuilder
from reroute.core.pair import RouterEncoderPair
from reroute.core.ensemble  import *
@dataclass
class GCGTrainerConfig:
    """GCG training configuration."""
    num_steps: int = 100
    search_width: int = 512
    #  the number of candidate sequences to test in each GCG iteration
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    
    vstar_k: int = 256
    
    suffix_length: int = 20
    log_interval: int = 10
    device: str = "cuda"
    verbose: bool = False


@dataclass
class GCGResult:
    """GCG optimization result."""
    best_loss: float
    best_string: str
    best_step: int
    final_string: str
    losses: List[float]
    strings: List[str]


class RouterGCG:
    """GCG attack tailored for Router ensembles."""
    

    def __init__(
        self,
        router_encoder_pairs: Union[RouterEncoderPair, List[RouterEncoderPair]],
        config: GCGTrainerConfig,
        adapter: Optional[LowRankSemanticAdapter] = None,
        unified_model_names: Optional[List[str]] = None,  # optional: pre-sorted union of model names
        target_model_names: Optional[List[str]] = None,   # adapter's target/output model names
        vstar_builder: Optional[VStarBuilder] = None,
        loss_type: str = "mse_adapter",                  # default: mse_adapter
        target_type: str = "strong",
        model_classifier: Optional[ModelClassifier] = None,
        classifier_config_path: Optional[str] = None,

    ):
        # ============================================================
        # 1. Normalize the pairs list
        # ============================================================
        if not isinstance(router_encoder_pairs, list):
            router_encoder_pairs = [router_encoder_pairs]
        
        self.pairs = router_encoder_pairs
        
        # ============================================================
        # 2. Extract routers and encoders
        # ============================================================
        self.routers = [pair.router for pair in self.pairs]
        self.encoders = [pair.encoder for pair in self.pairs]
        self.router_trainable = [pair.trainable for pair in self.pairs]
        
        # ============================================================
        # 3. Print configuration
        # ============================================================
        print(f"\n{'='*60}")
        print("Router-Encoder Pairs Configuration")
        print(f"{'='*60}")
        
        for i, pair in enumerate(self.pairs):
            status = "✓ Trainable" if pair.trainable else "✗ Frozen"
            print(f"\n{i+1}. {pair.name}")
            print(f"   Router: {pair.router.name}")
            print(f"   Encoder: {pair.encoder.name} ({pair.encoder_source})")
            print(f"   Embedding dim: {pair.encoder.embedding_dim()}")
            print(f"   Status: {status}")
         # ============================================================
        # Model classification (new)
        # ============================================================
        print(f"\n{'='*60}")
        print("Model Classification")
        print(f"{'='*60}")
        
        # create classifier
        if model_classifier is None:
            if classifier_config_path is None:
                # use default rules
                classifier_config_path = Path(__file__).parent.parent / "config" / "model_classification.yaml"
                print("Using default classifier config:", classifier_config_path)
                print("Exists:", Path(classifier_config_path).exists())
            model_classifier = ModelClassifier(
                config_path=str(classifier_config_path) if Path(classifier_config_path).exists() else None,
                default_classification="weak"
            )
        
        self.model_classifier = model_classifier
        
        # collect model lists from all Routers
        # router_model_lists = {
        #     pair.router.name: pair.router.get_model_list()
        #     for pair in self.pairs
        # }
        if unified_model_names is None:
            Routers_model_list = {p.router.name: p.router.get_model_list() for p in self.pairs}
            unified_model_names = sorted(get_model_list_union(Routers_model_list))
        self._union_models =  unified_model_names
        self.target_model_names = target_model_names
        self.unified_model_names = unified_model_names
        print("union models length:", len(self._union_models))

        # classify all models
        self.classified_models = classify_model_lists(
            {"target": self.target_model_names},
            self.model_classifier
        )
        
        # display classification results
        print(f"\nClassification Results:")
        for router_name, classification in self.classified_models.items():
            strong = classification["strong"]
            weak = classification["weak"]
            total = len(strong) + len(weak)
            
            print(f"\n  {router_name}:")
            print(f"    Total: {total}")
            print(f"    Strong: {len(strong)} ({len(strong)/total*100:.1f}%)")
            print(f"    Weak: {len(weak)} ({len(weak)/total*100:.1f}%)")
            
            # show first 3 strong and weak examples
            if len(strong) > 0:
                print(f"    Strong examples: {strong[:3]}")
            if len(weak) > 0:
                print(f"    Weak examples: {weak[:3]}")
        
        # ============================================================
        # Filter union models according to target_type
        # ============================================================
        # print(f"\n{'='*60}")
        # print("Building Model List Mappings")
        # print(f"{'='*60}")
        # Routers_model_list = {pair.router.name: pair.router.get_model_list() for pair in self.pairs}

        self.target_type = target_type
        self.adapter = adapter  # adapter provided/trained externally

        if target_type == "all":
            # attack all models
            self.target_model_indices = set(range(len(self._union_models)))
            target_models_list = self._union_models
        
        elif target_type == "strong":
            # attack only strong models
            all_strong = set()
            for classification in self.classified_models.values():
                all_strong.update(classification["strong"])
            
            # get indices of strong models in the union space
            self.target_model_indices = {
                self.target_model_names.index(model)
                for model in all_strong
                if model in self._union_models
            }
            target_models_list = sorted(list(all_strong))
        
        elif target_type == "weak":
            # attack only weak models
            all_weak = set()
            for classification in self.classified_models.values():
                all_weak.update(classification["weak"])
            
            self.target_model_indices = {
                self.target_model_indices.index(model)
                for model in all_weak
                if model in self._union_models
            }
            target_models_list = sorted(list(all_weak))
        
        else:
            raise ValueError(f"Invalid target_type: {target_type}")
        
        # ============================================================
        # Display statistics
        # ============================================================
        print(f"\nTarget Type: {target_type}")
        print(f"  Total union models: {len(self._union_models)}")
        print(f"  Target models: {len(self.target_model_indices)}")
        print(f"  Coverage: {len(self.target_model_indices)/len(self._union_models)*100:.1f}%")
        
        if len(target_models_list) <= 10:
            print(f"  Models: {target_models_list}")
        else:
            print(f"  First 5: {target_models_list[:5]}")
            print(f"  Last 5: {target_models_list[-5:]}")
        
        # display classification results
        print(f"\nClassification Results:")
        # collect all strong and weak models
        all_strong_models = set()
        all_weak_models = set()
        
        for classification in self.classified_models.values():
            all_strong_models.update(classification["strong"])
            all_weak_models.update(classification["weak"])
        
        # convert to indices in the union space
        self.strong_model_indices = {
            self.target_model_names.index(model)
            for model in all_strong_models
            if model in self.target_model_names
        }
        
        # weak_indices = all indices - strong_indices
        all_indices = set(range(len(self.target_model_names)))
        self.weak_model_indices = all_indices - self.strong_model_indices
        self.encoders = self.encoders
        self.config = config
        self.device = config.device
        self.loss_type = loss_type
        print("len of target_model_indices:", len(self.target_model_names))
        assert(max(self.strong_model_indices) < len(self.target_model_names))
        assert(max(self.weak_model_indices) < len(self.target_model_names))
        self.loss_fn = create_loss_function(
            device=self.device,
            loss_type=self.loss_type,  # or other types
            strong_indices=self.strong_model_indices,
            weak_indices=self.weak_model_indices,
            target_strong_prob=0.7,
            margin=2.0
        )        
        print(f"  Loss function: {self.loss_type}")
    # ============================================================
    # Added: model dtype conversion
    # ============================================================
        print(f"\n{'='*60}")
        print("Initializing RouterGCG")
        print(f"{'='*60}")
        print(f"Converting Encoders to bfloat16...")
        
        
        
        if vstar_builder is None:
            trainable_encoders = [
                pair.encoder for pair in self.pairs if pair.trainable  # use pair.trainable
            ]
            
            if len(trainable_encoders) == 0:
                raise ValueError("At least one RouterEncoderPair must be trainable!")
            
            print(f"\nCreating VStarBuilder:")
            print(f"  Using {len(trainable_encoders)} trainable encoder(s)")
            for enc in trainable_encoders:
                print(f"    - {enc.name}")
            print(f"  Top-k: {config.vstar_k}")
            
            self.vstar_builder = VStarBuilder(
                encoders=trainable_encoders,
                k=config.vstar_k,
                device=config.device
            )
        else:
            print(f"\nUsing external VStarBuilder:")
            print(f"  Top-k: {vstar_builder.k}")
            self.vstar_builder = vstar_builder
        self._pair_to_encoder_idx = {}
        encoder_idx = 0
        for pair in self.pairs:
            if pair.trainable:
                self._pair_to_encoder_idx[pair.name] = encoder_idx
                encoder_idx += 1
        
        print(f"\n[Mapping] pair_name -> encoder_idx:")
        for name, idx in self._pair_to_encoder_idx.items():
            print(f"  {name} -> {idx}")
        
        self.history = {
            'losses': [],
            'best_loss': float('inf'),
            'best_suffix': None,
            'best_step': 0
        }
        
        # ============================================================
        # 9. Final summary
        # ============================================================
        trainable_count = sum(pair.trainable for pair in self.pairs)
        print(f"\n✓ RouterGCG initialized:")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Trainable pairs: {trainable_count}")
        print(f"  Frozen pairs: {len(self.pairs) - trainable_count}")
        print(f"  Device: {self.device}")
    def _collect_unified_logits(self, messages: str, suffix_text: str, task_type: Optional[str] = None) -> torch.Tensor:
        """
            Collect aggregated logits from all routers in the unified space: (1, U)
        """
        U = len(self.unified_model_names)
        u_map = {name: i for i, name in enumerate(self._union_models)}
        all_router_logits = []
        # align each router's logits into the union space U
        aligned_sum = torch.zeros(U, device=self.device)
        count = torch.zeros(U, device=self.device)
        for pair in self.pairs:
            router_logits = pair.router.route(messages, suffix_text, type=task_type).to(self.device)  # (R_dim,)
            router_models = pair.router.get_model_list()
            for local_idx, m in enumerate(router_models):
                if m in u_map:
                    u_idx = u_map[m]
                    aligned_sum[u_idx] += router_logits[local_idx]
                    count[u_idx] += 1
        count = count.clamp_min(1)
        unified_logits = (aligned_sum / count).unsqueeze(0)  # (1, U)
        return unified_logits
    def _collect_unified_logits_with_gradient(
    self,
    messages: str,
    suffix_text: str,
    cur_pair_idx: int,
    cur_pair_logits: torch.Tensor,
    task_type: Optional[str] = None
) -> torch.Tensor:
        """
        Collect aggregated logits from all routers in the unified space, but
        skip `cur_pair_idx` and use the provided `cur_pair_logits` (which
        contain gradients) for that pair.

        Args:
            messages: user message string
            suffix_text: suffix text
            cur_pair_idx: index of the pair currently computing gradients (to skip)
            cur_pair_logits: gradient-bearing logits for the current pair (aligned to union space)
            task_type: optional task type

        Returns:
            unified_logits: (1, U) unified-space logits (gradients present at current pair positions)
        """
        with torch.no_grad():
            other_aligned_stack = torch.zeros((1, len(self.pairs), len(self.unified_model_names)), device=self.device)
            other_mask_stack = torch.zeros((1, len(self.pairs), len(self.unified_model_names)), dtype=torch.bool, device=self.device)
            
            u_map = {n: i for i, n in enumerate(self.unified_model_names)}
            
            for r_idx, other_pair in enumerate(self.pairs):
                if r_idx == cur_pair_idx:
                    continue  # skip the current pair
                
                other_logits = other_pair.router.route(messages, suffix_text, type=task_type).to(self.device)
                other_models = other_pair.router.get_model_list()
                
                for local_idx, m in enumerate(other_models):
                    if m in u_map:
                        u_idx = u_map[m]
                        other_aligned_stack[0, r_idx, u_idx] = other_logits[local_idx]
                        other_mask_stack[0, r_idx, u_idx] = True

        # insert gradient-bearing logits for the current pair
        cur_pair = self.pairs[cur_pair_idx]
        cur_router_models = cur_pair.router.get_model_list()
        for local_idx, m in enumerate(cur_router_models):
            if m in u_map:
                u_idx = u_map[m]
                other_aligned_stack[0, cur_pair_idx, u_idx] = cur_pair_logits[local_idx]  # gradient-bearing
                other_mask_stack[0, cur_pair_idx, u_idx] = True
        target_logits = self.adapter.forward_with_router_stack(
            other_aligned_stack.to(self.adapter.device),
            other_mask_stack.to(self.adapter.device),
            messages
                ).squeeze(0)  # (L,)
        return target_logits
        # Note: Legacy implementation (gathering logits and manually averaging)
        # was removed for clarity. Current implementation builds a router stack
        # with the gradient-bearing logits inserted for `cur_pair_idx`, and then
        # calls the adapter to obtain target logits. This preserves gradient flow
        # for the current pair while avoiding in-place or manual averaging logic.
    def _collect_router_stack(
    self,
    messages: str,
    suffix_text: str,
    task_type: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collect outputs from all routers and align to the unified space.
        Returns: (aligned_stack: (1,R,U), mask_stack: (1,R,U))
        """
        U = len(self.unified_model_names)
        R = len(self.pairs)
        u_map = {n: i for i, n in enumerate(self.unified_model_names)}
        
        aligned_stack = torch.zeros((1, R, U), device=self.device)
        mask_stack = torch.zeros((1, R, U), dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            for r_idx, pair in enumerate(self.pairs):
                router_logits = pair.router.route(messages, suffix_text, type=task_type)
                router_logits = router_logits.to(self.device)
                router_models = pair.router.get_model_list()
                
                for local_idx, m in enumerate(router_models):
                    if m in u_map:
                        u_idx = u_map[m]
                        aligned_stack[0, r_idx, u_idx] = router_logits[local_idx]
                        mask_stack[0, r_idx, u_idx] = True
        
        return aligned_stack, mask_stack
    def adapter_target_logits(
        self,
        messages: str,
        suffix: str,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        End-to-end: Router Stack -> Adapter -> Target Logits
        Returns: (1, L) target space logits
        """
        if self.adapter is None:
            raise RuntimeError("Adapter not initialized!")
        
        # collect router stack
        
        aligned_stack, mask_stack = self._collect_router_stack(messages, suffix, task_type)
        
        # use adapter's end-to-end forward (internal aggregation)
        target_logits = self.adapter.forward_with_router_stack(
            aligned_stack.to(self.adapter.device),
            mask_stack.to(self.adapter.device),
            messages
        )
        
        return target_logits  # (1, L)
        
    def _assemble_input(
        self,
        pair: RouterEncoderPair,
        # encoder: Optional[EncoderSpecBase],  
        messages: str,
        suffix_embeds: torch.Tensor

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble input (delegates to Router.assemble_gcg_input when available).

        Args:
            encoder: optional; some Routers require encoder info
            messages: user message string
            suffix_embeds: [L, hidden_dim] suffix embeddings

        Returns:
            input_embeds: [total_len, hidden_dim]
            attention_mask: [total_len]
        """
        # Check whether Router implements assemble_gcg_input
        if hasattr(pair.router, 'assemble_gcg_input'):
            # use Router-provided implementation
            return pair.router.assemble_gcg_input(
                messages=messages,
                suffix_embeds=suffix_embeds,
                attention_mask=None, # can be extended to support custom mask
                encoder=pair.encoder
            )
        else:
            # Fallback: generic implementation (suitable for most Routers)
            warnings.warn(
                f"Router '{self.router.name}' does not implement assemble_gcg_input(). "
                "Using generic implementation."
            )
            return
    def _generate_candidate_texts(
        self,
        current_text: str,
        gradients_per_pair: List[Dict],
        search_width: int,
        verbose: bool = False
    ) -> List[str]:
        """
        Generate candidate texts using VStarBuilder
        
        Strategy:
        1. For each position, collect gradients from all pairs
        2. Use VStarBuilder.find_topk_substitutions() to get candidates
        3. Sample from candidates to generate new texts
        """
        
        if len(gradients_per_pair) == 0:
            print("WARNING: No gradients available!")
            return [current_text] * search_width
        
        # ============================================================
        # 1. Use primary tokenizer as reference
        # ============================================================
        primary_tokenizer = self.vstar_builder.primary_encoder.get_tokenizer()
        current_tokens = primary_tokenizer.tokenize(current_text)
        num_positions = len(current_tokens)
        
        if verbose:
            print(f"\n1. Current text tokenization:")
            print(f"   Text: '{current_text}'")
            print(f"   Tokens: {current_tokens}")
            print(f"   Num positions: {num_positions}")
        
        # ============================================================
        # 2. For each position, use VStarBuilder to get top-k
        # ============================================================
        position_candidates = {}  # pos -> list of token IDs
        len_grad = [len(grad_info['gradients']) for grad_info in gradients_per_pair]
        min_len_grad = min(len_grad)
        for pos in range(min(num_positions,min_len_grad)):
            # Collect gradients for this position from all pairs
            gradients_for_this_position = []
            if len(gradients_per_pair) != len(self.encoders):
                print(f"error: gradients_per_pair length {len(gradients_per_pair)} does not match number of encoders {len(self.encoders)}")
                for grad_info in gradients_per_pair:
                    print("grad_info keys:", grad_info.keys())
            for grad_info in gradients_per_pair:
                gradients = grad_info['gradients']  # [num_tokens, hidden_dim]
                
                # Check if this pair has this position
                if pos >= gradients.shape[0]:
                    print("encoder name",grad_info['pair_name'])
                    print("⚠ Position out of range for this pair, skipping.")
                    print(gradients.shape, pos)
                    continue
                
                grad_at_pos = gradients[pos]  # [hidden_dim]
                gradients_for_this_position.append(grad_at_pos)
            # for gd in gradients_for_this_position: 
            #     print("dimensions of gradients_for_this_position:",gd.shape)
            
            if len(gradients_for_this_position) == 0:
                # No gradients for this position, skip
                position_candidates[pos] = []
                continue
            
            # Use VStarBuilder to find top-k
            topk_token_ids = self.vstar_builder.find_topk_substitutions(
                gradients_per_encoder=gradients_for_this_position,
                current_token_id=None,  # TODO: get current token ID
                verbose=(verbose and pos == 0)  # Only verbose for first position
            )
            
            position_candidates[pos] = topk_token_ids
        
        if verbose:
            print(f"\n2. Position candidates (via VStarBuilder):")
            for pos in range(min(3, num_positions)):
                num_candidates = len(position_candidates.get(pos, []))
                print(f"   Pos {pos}: {num_candidates} candidates")
                
                # Show top 3
                if num_candidates > 0:
                    top3_ids = position_candidates[pos][:3]
                    top3_words = [primary_tokenizer.decode([tid]) for tid in top3_ids]
                    print(f"     Top-3: {top3_words}")
        
        # ============================================================
        # 3. Generate candidates by random sampling
        # ============================================================
        candidate_texts = []
        
        # Get current token IDs
        current_token_ids = primary_tokenizer.encode(
            current_text,
            add_special_tokens=False
        )
        
        for _ in range(search_width):
            new_token_ids = current_token_ids.copy()
            
            # Randomly select positions to replace
            # positions_to_replace = np.random.choice(
            #     num_positions,
            #     size=min(self.config.n_replace, num_positions),
            #     replace=False
            # )
            # print("postore",positions_to_replace)
            positions = torch.randperm(num_positions)[:self.config.n_replace].tolist()
            # print("positions",positions)
            # Replace tokens
            for pos in positions:
                candidates = position_candidates.get(pos, [])
                
                if len(candidates) == 0:
                    print(f"⚠ No candidates for position {pos}, skipping replacement.")
                    continue
                
                # Random sample from top-k
                new_token_id = np.random.choice(candidates)
                new_token_ids[pos] = new_token_id
            
            # Decode to text
            candidate_text = primary_tokenizer.decode(
                new_token_ids[:self.config.suffix_length],  # truncated to max suffix length
                skip_special_tokens=True
            )
            candidate_texts.append(candidate_text)
        
        if verbose:
            print(f"\n3. Generated {len(candidate_texts)} candidates")
            for i, text in enumerate(candidate_texts[:3]):
                print(f"   {i+1}. '{text}'")
        
        return candidate_texts
    def _evaluate_candidate_texts(
        self,
        messages: str,
        candidate_texts: List[str],
        target_class: int,
        verbose: bool = False,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Evaluate candidate TEXTS using ensemble logits
        
        Strategy:
        1. For each candidate text
        2. Compute ensemble logits (ALL pairs)
        3. Compute loss from ensemble logits
        4. Return all losses
        
        Args:
            messages: User message
            candidate_texts: List of candidate suffix texts
            target_class: Target class index (in union space)
            verbose: Whether to print debug info
            
        Returns:
            losses: [num_candidates] tensor of losses
        """
        losses = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {len(candidate_texts)} Candidates")
            print(f"{'='*60}")
        
        for idx, candidate_text in enumerate(candidate_texts):

            logits_L = self.adapter_target_logits(messages, candidate_text, task_type=task_type)  # (1,L)
            logits_L = logits_L.squeeze(0)

            # ============================== Loss ===============================
            # Compute loss
            # ============================================================
            # target = torch.tensor([target_class], device=self.device)
            loss = self.loss_fn(logits_L, torch.tensor([target_class], device=logits_L.device))
            
            losses.append(loss)
            
            # ============================================================
            # Debug: show first 3 candidates
            # ============================================================
            if verbose and idx < 3:
                pred_class = logits_L.argmax().item()
                target_logit = logits_L[0, target_class].item()
                
                print(f"\nCandidate {idx}:")
                print(f"  Text: '{candidate_text}'")
                print(f"  Loss: {loss.item():.4f}")
                print(f"  Target logit: {target_logit:.4f}")
                print(f"  Predicted class: {pred_class}")
                print(f"  Is target: {pred_class == target_class}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluation Summary:")
            print(f"  Total candidates: {len(losses)}")
            print(f"  Min loss: {min(losses).item():.4f}")
            print(f"  Max loss: {max(losses).item():.4f}")
            print(f"  Mean loss: {sum(losses).item() / len(losses):.4f}")
            print(f"{'='*60}")
        
        return torch.stack(losses)

    def compute_token_gradient_text_based(
        self,
        messages: str,
        suffix_text: str,
        target_class: int,
        verbose: bool = False,
        task_type: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict]:
        """
        Compute gradients using ENSEMBLE logits (correct version)
        
        Key insights:
        1. ALL pairs compute logits ONCE (cached)
        2. ALL pairs have EQUAL weight in ensemble
        3. Only TRAINABLE pairs compute gradients
        """
        gradients_per_pair = []
        tracking_info = {
            'pairs': [],
            'grad_flow': []
        }
        
        # ============================================================
        # STEP 1: Calculate ALL pairs' ensemlogits (trainable + frozen)
        # ============================================================
        base_logits = self.adapter_target_logits(messages, suffix_text, task_type=task_type)  # (1,L)
        base_logits = base_logits.squeeze(0)  # (L,)
        # print("ensemble_logits:", ensemble_logits)
        print("base and union")
        print(len(base_logits))
        print(len(self.target_model_names ))
        assert(len(base_logits)==len(self.target_model_names))
        # ============================================================
        # STEP 3: For each TRAINABLE pair, compute gradients
        # ============================================================
        if verbose:
            print(f"\n{'='*60}")
            print("Step 3: Computing Gradients for Trainable Pairs")
            print(f"{'='*60}")
        
        for idx, pair in enumerate(self.pairs):
            if not pair.trainable:
                continue
            

            print(f"\n  Processing: {pair.name}")
            
            # ============================================================
            # 3.1 Tokenize suffix (using THIS pair's tokenizer)
            # ============================================================
            tokenizer = pair.encoder.get_tokenizer()     
            suffix_ids = tokenizer(
                suffix_text,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].to(self.device)
            
            # ============================================================
            # 3.2 Get embeddings (with gradients)
            # ============================================================
            embed_layer = pair.encoder.get_model().get_input_embeddings()
            suffix_ids = suffix_ids.to(next(embed_layer.parameters()).device)
                    # Ensure embedding supports gradients
            with torch.enable_grad():
                suffix_embeds = embed_layer(suffix_ids).squeeze(0)  # [L, hidden_dim]
                
                # retain_grad only when requires_grad is True
                if suffix_embeds.requires_grad:
                    suffix_embeds.retain_grad()
                else:
                    print(f"    ⚠ suffix_embeds.requires_grad=False, enabling gradients...")
                    suffix_embeds = suffix_embeds.detach().requires_grad_(True)
                    suffix_embeds.retain_grad()
            # suffix_embeds = embed_layer(suffix_ids).squeeze(0)
            suffix_embeds.retain_grad()  
            print(f"    suffix_embeds: shape={suffix_embeds.shape}, requires_grad={suffix_embeds.requires_grad}")
            # ============================================================
            # 3.3 Assemble input
            # ============================================================
            input_embeds, attention_mask = self._assemble_input(
                pair,
                messages,
                suffix_embeds
            )
            print(f"    input_embeds: shape={input_embeds.shape}, requires_grad={input_embeds.requires_grad}")
            print(f"    attention_mask: shape={attention_mask.shape}")
            # ============================================================
            # 3.4 Forward THIS pair (with gradients)
            # ============================================================
            current_logits = pair.forward_embeds(
                embeds=input_embeds,
                attention_mask=attention_mask
            )
            # ============================================================
            # 3. Align to union space (preserving gradients)
            # ============================================================
            current_aligned_logits = self._align_logits_to_union(
                router_logits=current_logits,
                router_name=pair.router.name
            )  # (U,) with gradients
            
            # ============================================================
            # 4. Use helper to collect unified logits (skip current pair)
            # ============================================================
            target_logits = self._collect_unified_logits_with_gradient(
                messages=messages,
                suffix_text=suffix_text,
                cur_pair_idx=idx,
                cur_pair_logits=current_logits,  # gradient-bearing
                task_type=task_type
            )  # (1, U)
            
            # ============================================================
            # 5. Map to target space via adapter
            # ============================================================
            # The adapter is invoked above via `_collect_unified_logits_with_gradient`.
            # Legacy manual-inplace averaging and replacement strategies have been
            # removed in favor of building a router stack and using the adapter's
            # `forward_with_router_stack` to preserve gradients and avoid in-place ops.

            # ============================================================
            # 3.5 Loss + Backward
            # ============================================s================
            target = torch.tensor([target_class], device=self.device)
            loss = self.loss_fn(target_logits, target)
            
            # if verbose:
            print(f"    Loss: {loss.item():.6f}")
            # print(f"combined_logits.requires_grad: {combined_logits.requires_grad}")
            # print(f"loss.requires_grad: {loss.requires_grad}")
            loss.backward()
            
            # ============================================================
            # 3.7 Collect gradients
            # ============================================================
            grad = suffix_embeds.grad
            
            if grad is None:
                print(f"    WARNING: No gradient!")
                continue
            
            gradients_per_pair.append({
                'pair_name': pair.name,
                'tokenizer': tokenizer,
                'tokens': tokenizer.convert_ids_to_tokens(suffix_ids.squeeze(0).tolist()),
                'token_ids': suffix_ids.squeeze(0).tolist(),
                'gradients': grad.detach()
            })
            
            if verbose:
                print("pair_name:",pair.name)
                print("gradients shape:",grad.shape)
                print("grad",grad)
                print(f"    Grad norm: {grad.norm().item():.6f}")
            
            tracking_info['pairs'].append(pair.name)
            tracking_info['grad_flow'].append({
                'pair': pair.name,
                'loss': loss.item(),
                'grad_norm': grad.norm().item()
            })
        
        return gradients_per_pair, tracking_info
    def evaluate_logits(
        self,
        messages: str,
        suffix: str,
        verbose: bool = True
    ) -> Dict:
        """Evaluate logits for the given suffix."""
        
        results = {}
        
        for encoder_idx, encoder in enumerate(self.encoders):
            # Tokenize suffix
            suffix_ids = encoder.get_tokenizer()(
                suffix,
                padding=True,
                return_tensors="pt"
            )["input_ids"].to(encoder.get_model().device)
            
            # Get embeddings
            embed_layer = encoder.get_model().get_input_embeddings()
            suffix_embeds = embed_layer(suffix_ids).squeeze(0)
            
            # Forward
            with torch.no_grad():
                logits= self.routers[encoder_idx].route(
                    messages, suffix
                )
            logits = logits.float()
            probs = F.softmax(logits, dim=0)
            
            results[encoder.name] = {
                'logits': logits.cpu().numpy().tolist(),
                'probs': probs.cpu().numpy().tolist(),
                'predicted_class': logits.argmax().item()
            }
            
            if verbose:
                print(f"\nEncoder: {encoder.name}")
                print(f"  Logits: {logits.cpu().numpy()}")
                print(f"  Probs: {probs.cpu().numpy()}")
                print(f"  Predicted: class {logits.argmax().item()}")
        
        return results
    def run(
        self,
        messages: str,
        target_class: int,
        target_model: str,
        optim_str_init: str = "x x x",
        task_type: Optional[str] = None,
    ) -> GCGResult:
        """Run GCG optimization (TEXT-based with ensemble logits + diagnostics)"""
        config = self.config
        # ============================================================
        # OPTIMIZATION LOOP (TEXT-based)
        # ============================================================
        current_suffix_text = optim_str_init
        
        losses = []
        optim_strings = []
        # Get target class index in union space
        target_class = self.target_model_names.index(target_model)
        for step in tqdm(range(config.num_steps), desc="Optimizing"):
            verbose_this_step = (step == 0) or (step % config.log_interval == 0)
            
            if verbose_this_step and config.verbose:
                print(f"\n{'='*80}")
                print(f"Step {step}: Gradient Computation")
                print(f"{'='*80}")
            
            # ============================================================
            # 1. Compute gradients (TEXT-based, ensemble logits)
            # ============================================================
            gradients_per_pair, tracking_info = self.compute_token_gradient_text_based(
                messages=messages,
                suffix_text=current_suffix_text,
                target_class=target_class,
                verbose=verbose_this_step and config.verbose,
                task_type=task_type
            )
            
            # ============================================================
            # 2. Generate candidate TEXTS
            # ============================================================
            with torch.no_grad():
                candidate_texts = self._generate_candidate_texts(
                    current_text=current_suffix_text,
                    gradients_per_pair=gradients_per_pair,
                    search_width=config.search_width,
                    verbose=verbose_this_step and config.verbose
                )
            
            # ============================================================
            # 3. Evaluate candidates (ensemble logits)
            # ============================================================
            candidate_losses = self._evaluate_candidate_texts(
                messages=messages,
                candidate_texts=candidate_texts,
                target_class=target_class
            )
            
            # ============================================================
            # 4. Select best
            # ============================================================
            best_idx = candidate_losses.argmin()
            current_loss = candidate_losses[best_idx].item()
            current_suffix_text = candidate_texts[best_idx]
            
            losses.append(current_loss)
            optim_strings.append(current_suffix_text)
            
            if current_loss < self.history['best_loss']:
                self.history['best_loss'] = current_loss
                self.history['best_step'] = step
                self.history['best_suffix'] = current_suffix_text
            
            if verbose_this_step and config.verbose:
                print(f"\nStep {step} Summary:")
                print(f"  Current loss: {current_loss:.4f}")
                print(f"  Best loss: {self.history['best_loss']:.4f}")
                print(f"  Current suffix: '{current_suffix_text}'")
            
            torch.cuda.empty_cache()
        
        # ============================================================
        # TEST 5: Final Logits Evaluation
        # ============================================================
        min_loss_idx = losses.index(min(losses))
        best_suffix = optim_strings[min_loss_idx]
        
        # print(f"\n{'='*60}")
        # print("Final Logits Evaluation")
        # print(f"{'='*60}")
        
        # final_results = self.evaluate_logits(
        #     messages=messages,
        #     suffix=best_suffix,
        #     verbose=False
        # )
    

        
        # ============================================================
        # Create result
        # ============================================================
        result = GCGResult(
            best_loss=losses[min_loss_idx],
            best_string=best_suffix,
            best_step=min_loss_idx,
            final_string=optim_strings[-1],
            losses=losses,
            strings=optim_strings
        )
        
        print(f"\n{'='*80}")
        print(f"Optimization Complete!")
        print(f"  Best loss: {result.best_loss:.4f}")
        print(f"  Best suffix: '{result.best_string}'")
        print(f"  Best step: {result.best_step}")
        print(f"{'='*80}\n")
        
        return result

    def ensemble_logits(
    self,
    messages: str,
    suffix_text: str,
    weights: Optional[List[float]] = None,
    verbose: bool = False
) -> torch.Tensor:
        """
        Ensemble logits from all Router-Encoder pairs (mapped to the unified space).

        Args:
            messages: user message text
            suffix_text: optimized suffix text
            weights: per-pair weights (None = automatic)
            verbose: whether to print verbose information

        Returns:
            ensemble_logits: [num_union_classes] combined logits
        """
        # ============================================================
        # 1. Initialize union space
        # ============================================================
        if not hasattr(self, '_model_mappings'):
            Routers_model_list = {pair.router.name: pair.router.get_model_list() for pair in self.pairs}
            self._model_mappings = build_model_index_mapping(self._union_models, Routers_model_list)
            self._union_models = get_model_list_union(Routers_model_list)

            
            if verbose:
                print(f"\n{'='*60}")
                print("Model List Union")
                print(f"{'='*60}")
                print(f"Total union models: {len(self._union_models)}")
                
                # Show coverage for each Router
                for pair in self.pairs:
                    router_models = set(pair.router.get_model_list())
                    coverage = len(router_models) / len(self._union_models) * 100
                    print(f"\n{pair.router.name}:")
                    print(f"  Models: {len(router_models)}")
                    print(f"  Coverage: {coverage:.1f}%")
        
        num_union_classes = len(self._union_models)
        
        # ============================================================
        # 2. Collect logits from all routers (aligned to union space)
        # ============================================================
        aligned_logits_list = []
        
        for pair in self.pairs:
            # Get original logits
            router_logits = pair.router.route(messages, suffix_text)
            
            # Align to union space
            aligned_logits = self._align_logits_to_union(
                router_logits=router_logits,
                router_name=pair.router.name
            )
            
            aligned_logits_list.append(aligned_logits)
            
            if verbose:
                print(f"\n{pair.router.name}:")
                print(f"  Original logits shape: {router_logits.shape}")
                print(f"  Aligned logits shape: {aligned_logits.shape}")
                print(f"  Non-inf count: {(aligned_logits != float('-inf')).sum().item()}")
        
        # ============================================================
        # 3. Weighted average (ignore -inf entries)
        # ============================================================
        if weights is None:
            # Default: trainable pair weight 1.0, frozen pair weight 0.3
            weights = [1.0 for pair in self.pairs]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize ensemble logits
        ensemble_logits = torch.full(
            (num_union_classes,),
            float('-inf'),
            device=aligned_logits_list[0].device,
            dtype=aligned_logits_list[0].dtype
        )
        
        # For each class index: weighted average (only consider non -inf values)
        for class_idx in range(num_union_classes):
            valid_logits = []
            valid_weights = []
            
            for logits, weight in zip(aligned_logits_list, normalized_weights):
                if logits[class_idx] != float('-inf'):
                    valid_logits.append(logits[class_idx])
                    valid_weights.append(weight)
            
            if len(valid_logits) > 0:
                # Normalize valid weights
                total_valid_weight = sum(valid_weights)
                normalized_valid_weights = [w / total_valid_weight for w in valid_weights]
                
                # Weighted average
                # Ensure valid_logits are on the same device
                target_device = aligned_logits_list[0].device
                valid_logits = [logit.to(target_device) for logit in valid_logits]
                ensemble_logits[class_idx] = sum(
                    logit * weight 
                    for logit, weight in zip(valid_logits, normalized_valid_weights)
                )
        
        if verbose:
            print(f"\nEnsemble Logits:")
            print(f"  Shape: {ensemble_logits.shape}")
            print(f"  Non-inf count: {(ensemble_logits != float('-inf')).sum().item()}")
            print(f"  Max logit: {ensemble_logits[ensemble_logits != float('-inf')].max().item():.4f}")
            print(f"  Min logit: {ensemble_logits[ensemble_logits != float('-inf')].min().item():.4f}")
        
        return ensemble_logits


    def _align_logits_to_union(
        self,
        router_logits: torch.Tensor,
        router_name: str
    ) -> torch.Tensor:
        """
        Align a Router's logits into the union space.

        Args:
            router_logits: [num_router_classes] original logits
            router_name: name of the router

        Returns:
            aligned_logits: [num_union_classes] aligned logits (missing entries set to -inf)
        """
        # Get mapping
        if not hasattr(self, '_model_mappings'):
            Routers_model_list = {pair.router.name: pair.router.get_model_list() for pair in self.pairs}
            self._union_models = get_model_list_union(Routers_model_list)
            # print("_union_models",self._union_models)
            # print("len of union models init:", len(self._union_models))
            self._model_mappings = build_model_index_mapping(self._union_models, Routers_model_list)
            # self._union_models = _get_model_list_union()
        # print("len of union models:", len(self._union_models))
        mapping = self._model_mappings[router_name]
        num_union_classes = len(self._union_models)
        
        aligned_logits = torch.full(
            (num_union_classes,),
            float('-inf'),
            device=router_logits.device,
            dtype=router_logits.dtype
        )
        for router_idx, union_idx in mapping.items():
            aligned_logits[union_idx] = router_logits[router_idx]
        
        return aligned_logits


def main():
    # for test
    return


 
if __name__ == "__main__":
    main()