from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Callable, Tuple, Set
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import warnings
import sys
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))

from models.Encoder import EncoderFactory, EncoderSpecBase
from models.Vstar_Builder import VStarBuilder
from transformers import AutoTokenizer, DebertaV2Model,AutoModel




# ============================================================
# Core Interface: Router with Batch Inference Support
# ============================================================

class Router(ABC):
    """
    Unified Router Interface (supports single and batch inference)
    
    Core methods:
    - route(prompt, suffix) -> model_name  # Single routing
    - route_batch(prompts, suffixes) -> List[model_name]  # Batch routing
    - get_logits(prompt, suffix) -> logits  # Single logits
    - get_logits_batch(prompts, suffixes) -> batch_logits  # Batch logits
    - get_model_list() -> List[str]  # Get model list
    """
    
    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self._model_list = []  # Subclass needs to set this
    
    @abstractmethod
    def route(self, prompt: str, suffix: str = "", type: str = "") -> str:
        """
        Routing decision (single)

        Args:
            prompt: User input
            suffix: Adversarial suffix (optional)
            type: Routing type (optional)

        Returns:
            Selected model name
        """
        pass
    
    def route_batch(
        self, 
        prompts: List[str], 
        suffixes: List[str] = None
    ) -> List[str]:
        """
        Batch routing decision
        
        Args:
            prompts: Batch user inputs
            suffixes: Batch adversarial suffixes (optional)
            
        Returns:
            Batch selected model names
        """
        if suffixes is None:
            suffixes = [""] * len(prompts)
        
        # Default implementation: call one by one (subclass can override for optimization)
        return [self.route(p, s) for p, s in zip(prompts, suffixes)]
    
    @abstractmethod
    def get_logits(self, prompt: str, suffix: str = "") -> Optional[torch.Tensor]:
        """
        Get routing logits (single)
        
        Returns:
            logits tensor [num_classes] or None
        """
        pass
    
    def get_logits_batch(
        self,
        prompts: List[str],
        suffixes: List[str] = None
    ) -> Optional[torch.Tensor]:
        """
        Batch get logits (key optimization point)
        
        Args:
            prompts: Batch inputs
            suffixes: Batch suffixes
            
        Returns:
            batch_logits [batch_size, num_classes] or None
        """
        if suffixes is None:
            suffixes = [""] * len(prompts)
        
        # Default implementation: call one by one and stack (subclass should override for optimization)
        logits_list = [self.get_logits(p, s) for p, s in zip(prompts, suffixes)]
        
        if logits_list[0] is None:
            return None
        
        return torch.stack(logits_list)
    
    def get_model_list(self) -> List[str]:
        """
        Get the model list supported by this Router
        
        Returns:
            model_list: List of model names
        """
        return self._model_list.copy()
    
    def set_model_list(self, model_list: List[str]):
        """
        Set model list (for expansion/modification)
        
        Args:
            model_list: New model list
        """
        self._model_list = model_list.copy()
    
    def expand_model_list(self, additional_models: List[str]):
        """
        Expand model list
        
        Args:
            additional_models: Models to be added
        """
        for model in additional_models:
            if model not in self._model_list:
                self._model_list.append(model)
    
    def get_num_classes(self) -> int:
        """Get number of classes (number of models)"""
        return len(self._model_list)
    
    def model_to_index(self, model_name: str) -> int:
        """Model name -> index"""
        try:
            return self._model_list.index(model_name)
        except ValueError:
            raise ValueError(f"Model '{model_name}' not in model_list: {self._model_list}")
    
    def index_to_model(self, index: int) -> str:
        """Index -> model name"""
        if 0 <= index < len(self._model_list):
            return self._model_list[index]
        raise IndexError(f"Index {index} out of range for model_list (size {len(self._model_list)})")
    def supports_embedding_forward(self) -> bool:
        """
        Check if Router supports embedding-level forward
        
        Returns:
            True if supported, False otherwise
        """
        try:
            import inspect
            source = inspect.getsource(self.forward_embeds)
            return "NotImplementedError" not in source
        except:
            return False

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        encoder_name: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Forward pass from embeddings directly (for GCG attack)
        
        Args:
            embeds: [seq_len, hidden_dim] input embeddings
            encoder_name: encoder name (some routers may need this)
            attention_mask: [seq_len] attention mask (optional)
            
        Returns:
            logits: [num_classes] classification logits
            
        Raises:
            NotImplementedError: if Router doesn't support embedding-level operations
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embedding-level forward. "
            "Please implement forward_embeds() method for GCG attack compatibility."
        )

    def forward_embeds_batch(
        self,
        embeds_batch: torch.Tensor,
        encoder_names: Optional[List[str]] = None,
        attention_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Batch embedding-level forward (optimized version)
        
        Args:
            embeds_batch: [batch_size, seq_len, hidden_dim]
            encoder_names: List[str] encoder name list
            attention_masks: [batch_size, seq_len]
            
        Returns:
            logits_batch: [batch_size, num_classes]
        """
        # Default implementation: call one by one (subclass can override for optimization)
        batch_size = embeds_batch.size(0)
        
        if encoder_names is None:
            encoder_names = [None] * batch_size
        
        if attention_masks is None:
            attention_masks = [None] * batch_size
        
        logits_list = []
        for i in range(batch_size):
            logits = self.forward_embeds(
                embeds_batch[i],
                encoder_name=encoder_names[i],
                attention_mask=attention_masks[i] if attention_masks[i] is not None else None
            )
            logits_list.append(logits)
        
        return torch.stack(logits_list)
    def assemble_gcg_input(
    self,
    messages: str,
    suffix_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    prefix_mode: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble input for GCG attack
        
        Different Routers require different input formats:
        - GraphRouter: message + suffix (in Encoder space)
        - P2L Router: message + suffix (in Router's token space)
        
        Args:
            messages: User message
            suffix_embeds: [L, hidden_dim] suffix embeddings
            attention_mask: Optional[torch.Tensor] attention mask for suffix
            
        Returns:
            input_embeds: [total_len, hidden_dim]
            attention_mask: [total_len]
        """
        raise NotImplementedError(
            f"Router '{self.name}' must implement assemble_gcg_input() for GCG attack"
        )


# ============================================================
# Utility Function: Verify model_list Consistency of Router List
# ============================================================

def verify_routers_consistency(routers: List[Router]) -> Dict:
        """
        Verify if model_list of multiple Routers are consistent
        
        Args:
            routers: List of Router objects
            
        Returns:
            result: {
                'consistent': bool,
                'common_models': List[str],
                'differences': Dict[str, List[str]],
                'recommendation': str
            }
        """
        if not routers:
            return {
                'consistent': True,
                'common_models': [],
                'differences': {},
                'recommendation': 'No routers to verify'
            }
        
        # Get all model_lists
        model_lists = {router.name: set(router.get_model_list()) for router in routers}
        
        # Find intersection and union
        common_models = set.intersection(*model_lists.values())
        all_models = set.union(*model_lists.values())
        
        # Check differences
        differences = {}
        for name, models in model_lists.items():
            diff = models - common_models
            if diff:
                differences[name] = list(diff)
        
        # Check if consistent
        consistent = len(differences) == 0
        
        # Generate recommendation
        if consistent:
            recommendation = "✓ All routers have identical model lists"
        else:
            recommendation = (
                f"⚠ Model lists are inconsistent. "
                f"Common models: {len(common_models)}/{len(all_models)}. "
                f"Consider using expand_routers_to_common_models() to align them."
            )
        
        return {
            'consistent': consistent,
            'common_models': sorted(list(common_models)),
            'all_models': sorted(list(all_models)),
            'differences': differences,
            'recommendation': recommendation
        }



import random

class RouterDCModule(nn.Module):
    def __init__(self, backbone, hidden_state_dim=768, node_size=3, similarity_function = "cos"):
        super(RouterDCModule, self).__init__()
        self.backbone = backbone
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = nn.Embedding(node_size, hidden_state_dim)
        std_dev = 0.78
        with torch.no_grad():
            nn.init.normal_(self.embeddings.weight, mean=0, std=std_dev)
        self.similarity_function = similarity_function
            

    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    '''The forward function pass the input to Router and compute the similarity between model output and trainable embedding'''
    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            # If positive model does not well, skip this.
            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # make the last_x ignore the true items
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            # * mask2
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. dataset_id:", dataset_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss


class RouterDCAdapter(Router):
    def __init__(self, backbone = DebertaV2Model.from_pretrained("microsoft/mdeberta-v3-base"), hidden_state_dim=768, node_size=7, similarity_function = "cos", name = "routerDC"):
        super().__init__(name)
        self.RouterDC = RouterDCModule(backbone, hidden_state_dim, node_size, similarity_function)
        
        # Load state_dict
        state_dict = torch.load("/mnt/data2/jiahua/LLM/OpenRouterBench/baselines/RouterDC/logs/router_save/clw_1/slw_0_clw_1_cos_tk_3_lk_3_lr_5e-5_step_1000_t_1_seed_42/best_model.pth")

        self.RouterDC.load_state_dict(state_dict)     


        self.backbone = self.RouterDC.backbone.to(self.device)
        self.hidden_state_dim = hidden_state_dim
        self.node_size = node_size
        self.embeddings = self.RouterDC.embeddings.to(self.device)
        std_dev = 0.78
        self.similarity_function = similarity_function
        self.model = self.backbone.to(self.device)
        self.encoder = self.backbone.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", truncation_side='left', padding=True)

        self.name = name   
        self._model_list = ['mistralai/Mistral-7B-v0.1','meta-math/MetaMath-Mistral-7B', 'itpossible/Chinese-Mistral-7B-v0.1','HuggingFaceH4/zephyr-7b-beta','cognitivecomputations/dolphin-2.6-mistral-7b','meta-llama/Meta-Llama-3-8B','cognitivecomputations/dolphin-2.9-llama3-8b']

    def route(self, prompt: str, suffix: str = "", type: str = "") -> str:
        logits = self.get_logits(prompt, suffix)
        return  logits


    def get_logits(self, prompt: str, suffix: str = "") -> Optional[torch.Tensor]:
        full_text = prompt + suffix
        # tokenized = self.encoder.tokenizer(
        #     full_text,
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        #     return_tensors="pt"
        # ).to(self.device)
        tokenized = self.tokenizer(
            full_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            # embeds = self.encoder.model.get_input_embeddings()(
            #     tokenized['input_ids']
            # )

            outputs = self.encoder(
                input_ids=tokenized['input_ids'],
                attention_mask=tokenized['attention_mask']
            )
            hidden_state = outputs['last_hidden_state'][:, 0, :]  # [batch, hidden_dim]
            logits = self.compute_similarity(hidden_state, self.embeddings.weight)
            logits = logits / 1
            # return logits.squeeze(0), hidden_state.squeeze(0)
        
            # embeds = self.encoder.get_input_embeddings()(
            #     tokenized['input_ids']
            # )
            # hidden_state = embeds['last_hidden_state'][:,0,:]
            # embeds = self.compute_similarity(hidden_state, self.embeddings.weight)

            # t = 1 # temperature
            # logits = embeds / t
            # return logits, hidden_state

        
        # with torch.no_grad():
        #     pooled = embeds.mean(dim=1)  # [1, hidden_dim]
        #     logits, _ = self.forward_embeds(pooled)
        
        return logits.squeeze(0)  # [num_classes]
    
    def compute_similarity(self, input1, input2):
        if self.similarity_function == "cos":
            return (input1 @ input2.T) / (torch.norm(input1,dim=1).unsqueeze(1) * torch.norm(input2,dim=1).unsqueeze(0))
        else:
            return input1 @ input2.T


    '''The forward function pass the input to Router and compute the similarity between model output and trainable embedding'''
    def forward(self, t=1, **input_kwargs):
        x = self.backbone(**input_kwargs)
        # We used the first token as classifier token.
        hidden_state = x['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / t
        return x, hidden_state

    def compute_sample_llm_loss(self, x, index_true, top_k, last_k):
        loss = 0
        top_index_true, top_index = index_true.sort(dim=-1, descending=True)
        last_index_true, negtive_index = index_true.topk(k=last_k, largest=False,dim=-1)

        for i in range(top_k):
            positive_index = top_index[:,i].view(-1,1)

            # If positive model does not well, skip this.
            mask = torch.where(top_index_true[:,i].view(-1,1) > 0, 1, 0)

            top_x = torch.gather(x, 1, positive_index)
            last_x = torch.gather(x, 1, negtive_index)

            # make the last_x ignore the true items
            last_x = torch.where(last_index_true > 0.5, float("-inf"), last_x)

            temp_x = torch.concat([top_x, last_x], dim=-1)

            softmax_x = nn.Softmax(dim=-1)(temp_x)
            log_x = torch.log(softmax_x[:,0])
            log_x = log_x * mask 
            # * mask2
            loss += torch.mean(-log_x)
        return loss
    
    def compute_sample_sample_loss_with_task_tag(self, hidden_state, dataset_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for dataset_id in dataset_ids:
            positive_indexs = torch.nonzero(dataset_ids == dataset_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(dataset_ids != dataset_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. dataset_id:", dataset_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].squeeze()
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss
    
    def compute_cluster_loss(self, hidden_state, cluster_ids, t, H=3):
        similar_score = self.compute_similarity(hidden_state, hidden_state)
        last_k2 = H
        # get the index of corresponding dataset_id
        all_index = []
        for cluster_id in cluster_ids:
            positive_indexs = torch.nonzero(cluster_ids == cluster_id)
            select_positive_index = random.choice(positive_indexs)
            negtive_indexs = torch.nonzero(cluster_ids != cluster_id)
            if len(negtive_indexs) < last_k2:
                print("len of negtive index is smaller than last_k2. cluster_id:", cluster_id)
                continue
            index_of_negtive_indexs = random.sample(range(0, len(negtive_indexs)), last_k2)
            select_negtive_index = negtive_indexs[index_of_negtive_indexs].view(-1)
            select_index = torch.concat([select_positive_index, select_negtive_index])
            all_index.append(select_index)
        all_index = torch.stack(all_index)
        rearrange_similar_score = torch.gather(similar_score, 1, all_index)

        softmax_sample_x = torch.softmax(rearrange_similar_score, dim=-1)
        log_sample_x = torch.log(softmax_sample_x)
        loss = torch.mean(-log_sample_x[:,0])
        return loss

    def get_internal_encoder(self):
        """Get internal encoder instance"""
        class MyDebertaEncoder(EncoderSpecBase):
            def __init__(self, model_name, device="cuda"):
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = DebertaV2Model.from_pretrained(model_name).to(device)
                self.name = model_name
            def get_model(self):
                return self.model
            def get_tokenizer(self):
                return self.tokenizer
            def embedding_dim(self):
                return self.model.config.hidden_size

            # Implement abstract methods
            def embed_matrix(self):
                # Return embedding layer weights if needed
                return self.model.get_input_embeddings().weight

            def token_id(self, token: str):
                # Return single token id
                return self.tokenizer.convert_tokens_to_ids(token)

            def token_ids_seq(self, tokens: list):
                # Return list of token ids
                return self.tokenizer.convert_tokens_to_ids(tokens)
        return MyDebertaEncoder("microsoft/mdeberta-v3-base", device="cuda")


    def assemble_gcg_input(
            self,
            messages: str,
            suffix_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder: Optional[EncoderSpecBase] = None,
            prefix_mode: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            
            tokenizer = self.tokenizer
            target_max_token_len = 512

            # 1. Encode original input (with Padding)
            encoded = tokenizer(
                messages,
                max_length=target_max_token_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Transfer device
            device = encoder.get_model().device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get original Embeddings
            with torch.no_grad():
                input_embeds = encoder.get_model().get_input_embeddings()(input_ids)
            
            input_embeds = input_embeds.squeeze(0)  # [512, hidden_dim]

            # =======================================================
            # Stage 1: Extract valid part (Message) and PAD sample
            # =======================================================
            
            # valid_len is the length of [CLS]...[SEP]
            valid_len = attention_mask.sum().item()
            
            # Extract [CLS] + [Message] (excluding SEP)
            cls_and_message = input_embeds[:valid_len-1]
            
            # Extract [SEP] (to append after Suffix)
            sep_token = input_embeds[valid_len-1].unsqueeze(0)

            # [Key trick] Extract a standard [PAD] Embedding sample
            # As long as valid_len < 512, input_embeds[valid_len] is a PAD
            # If sentence is already full, we may need to get PAD embedding from model again, but usually GCG attack will leave room
            if valid_len < target_max_token_len:
                pad_sample = input_embeds[valid_len].unsqueeze(0) # [1, hidden_dim]
            else:
                # Extreme case: sentence is originally full, cannot get PAD (in this case concatenation will overflow, need to handle truncation)
                # For simplicity, assume there is always room
                pad_sample = torch.zeros_like(sep_token) 

            # =======================================================
            # Stage 2: Assemble active part (Active Part)
            # =======================================================
            
            # Current structure: [CLS] [Message] [Suffix] [SEP]
            if prefix_mode:
                cls = cls_and_message[0].unsqueeze(0)
                message = cls_and_message[1:]
                active_embeds = torch.cat([cls,suffix_embeds, message,sep_token], dim=0)  # [M+L-1, D]
            else:

                active_embeds = torch.cat([
                    cls_and_message,
                    suffix_embeds,
                    sep_token
                ], dim=0)
            
            current_len = active_embeds.size(0)

            # Stage 3: Refill Padding (Refill)

            if current_len > target_max_token_len:
                # A. If too long after concatenation, force truncation (keep beginning, discard end)
                # Note: this may cut off SEP, but this is necessary to maintain tensor shape
                final_embeds = active_embeds[:target_max_token_len]
                final_mask = torch.ones(target_max_token_len, dtype=torch.long, device=device)
                
            else:
                # B. If not full, calculate how many PADs are needed
                pad_needed = target_max_token_len - current_len
                
                # Copy pad_sample to fill the gap
                padding_block = pad_sample.repeat(pad_needed, 1) # [pad_needed, hidden_dim]
                
                # Concatenate final result: [Active] + [Pads]
                final_embeds = torch.cat([active_embeds, padding_block], dim=0)
                
                # Construct corresponding Mask
                # Active part is 1, Padding part is 0
                active_mask = torch.ones(current_len, dtype=torch.long, device=device)
                padding_mask = torch.zeros(pad_needed, dtype=torch.long, device=device)
                final_mask = torch.cat([active_mask, padding_mask], dim=0)

            # Restore batch dimension (if you need [1, 512, 768] later)
            # final_embeds = final_embeds.unsqueeze(0)
            # final_mask = final_mask.unsqueeze(0)

            return final_embeds, final_mask

    def forward_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None

    ) -> torch.Tensor:
        """
        Embedding-level forward (for GCG attack)
        
        
        Args:
            embeds: [seq_len, hidden_dim] input embeddings
            attention_mask: [seq_len] mask
            
        Returns:
            coefficients: [num_models] (requires_grad=True)
        """
        
        # 1. Process input
        input_embeds = embeds
        full_mask = attention_mask
        
        # ✅ Add CLS token embedding (critical!)
        # P2L expects input to end with CLS token
        # cls_token_id = self.tokenizer.cls_token_id
        # cls_embedding = self.model.get_input_embeddings()(
        #     torch.tensor([cls_token_id], device=self.device)
        # )  # [1, hidden_dim]
        
        # # Concatenate CLS
        # input_embeds = torch.cat([input_embeds, cls_embedding], dim=0)  # [seq_len+1, hidden_dim]
        # if full_mask is not None:
        #     cls_mask = torch.ones(1, dtype=torch.long, device=self.device)
        #     full_mask = torch.cat([full_mask, cls_mask], dim=0)
        
        # ============================================================
        # 2. P2L Model Forward
        # ============================================================
        inputs_embeds = input_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
        # print("embeds entrance's input_embeds:", inputs_embeds)
    
        if full_mask is not None:
            full_mask = full_mask.unsqueeze(0)  # [1, seq_len+1]
        # model_dtype = next(self.model.parameters()).dtype
        # inputs_embeds = inputs_embeds.to(dtype=model_dtype)
        # print("Input embeds dtype:", inputs_embeds.dtype)
        # Forward (retain gradients)

        # outputs = self.backbone.forward_embedding(
        #     inputs_embeds=inputs_embeds,
        #     attention_mask=full_mask,
        #     # return_dict=True
        # )

        outputs = self.backbone(
            inputs_embeds=inputs_embeds,   # Your concatenated embedding
            attention_mask=full_mask,
            return_dict=True              # Recommended to return dict format
        )
        hidden_state = outputs.last_hidden_state[:, 0, :]  # [1, hidden_dim]
        # hidden_state = outputs['last_hidden_state'][:,0,:]
        x = self.compute_similarity(hidden_state, self.embeddings.weight)
        x = x / 1
        x = x.squeeze(0)

        return x  # [num_models], requires_grad=True

class RouteLLMAdapter(Router):  
    """RouteLLM Adapter - Direct use of native Router"""  
      
    def __init__(  
        self,  
        name: str,  
        # router_name: str ,  
        type: str = "causal_llm",
        strong_model: str = "gpt-4",  
        weak_model: str = "mixtral-8x7b",  
        device: str = "cuda",  
        config: Optional[dict] = None,
        consider_cost: bool = True

    ):  
        super().__init__(name, device)  
        routellm_root = Path(__file__).parent.parent.parent / "Cur-Routers" / "RouteLLM"
        if str(routellm_root) not in sys.path:
            sys.path.insert(0, str(routellm_root))
        print("Routellm root path:", routellm_root)
        # Directly instantiate Router object  
        from routellm.routers.routers import ROUTER_CLS  
        from routellm.controller import GPT_4_AUGMENTED_CONFIG, ModelPair  
          
        # Use default config or custom config  
        if config is None:  
            config = GPT_4_AUGMENTED_CONFIG  
          
        # Instantiate specified router  
        router_config = config.get(type, {})  
        self.router = ROUTER_CLS[type](**router_config)  
          
        self.type = type  
        self._model_list = [strong_model, weak_model] 
        self.consider_cost = consider_cost

        # Read Threshold
        default_threshold = 0.5
        
        threshold_file = "routellm_thresholds.json"

        # Set config file path (default in the same directory as current script)
        config_path = Path(__file__).parent / threshold_file
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                threshold_data = json.load(f)
                
            # Read corresponding value based on self.type (e.g. "bert", "causal_llm")
            if self.type in threshold_data:
                self.threshold = threshold_data[self.type]
                print(f"Loaded threshold for '{self.type}' from file: {self.threshold}")
            else:
                print(f"Warning: Type '{self.type}' not found in {threshold_file}. Using default.")
                self.threshold = default_threshold
        else:
            print(f"Warning: Threshold file not found at {config_path}. Using default.")
            self.threshold = default_threshold
            print("current type:", self.type)
            print("current threshold:", self.threshold)
        # ------------------ Modification End ------------------

        if(self.type == "causal_llm"):
            self.router_model = self.router.router_model
            self.model = self.router_model.model
            self.tokenizer = self.router_model.tokenizer

        elif(self.type == "bert"):
            self.model = self.router.model
            self.tokenizer = self.router.tokenizer

      
    def route(self, prompt: str, suffix: str = "", type: str = "") -> str:
        """Return routed model name"""
        full_prompt = prompt + suffix  
          
        strong_win_rate = self.router.calculate_strong_win_rate(full_prompt)
        result = torch.tensor([strong_win_rate, 0], dtype=torch.float32, device=self.device)
        if(self.consider_cost):
            strong_win_rate -= self.threshold
            result = torch.tensor([strong_win_rate, 0], dtype=torch.float32, device=self.device)
        return result
    
    def get_logits(self, prompt, suffix = ""):
        full_prompt = prompt + suffix  
          
        full_prompt = prompt + suffix  
        strong_win_rate = self.router.calculate_strong_win_rate(full_prompt)
        result = torch.tensor([strong_win_rate, 0], dtype=torch.float32, device=self.device)
        if(self.consider_cost):
            strong_win_rate -= self.threshold
            result = torch.tensor([strong_win_rate, 0], dtype=torch.float32, device=self.device)
        return result
    def route_batch(  
        self,  
        prompts: List[str],  
        suffixes: List[str] = None  
    ) -> List[str]:  
        """Batch routing"""  
        if suffixes is None:  
            suffixes = [""] * len(prompts)  
          
        # Check if parallel processing is supported  
        if hasattr(self.router, 'NO_PARALLEL') and self.router.NO_PARALLEL:  
            # No parallel support, process sequentially  
            return [self.route(p, s) for p, s in zip(prompts, suffixes)]  
        else:  
            # Parallel support enabled  
            from concurrent.futures import ThreadPoolExecutor  
            with ThreadPoolExecutor(max_workers=10) as executor:  
                results = list(executor.map(  
                    lambda args: self.route(*args),  
                    zip(prompts, suffixes)  
                ))  
            return results
    
    def forward_embeds(self, embeds, encoder_name = None, attention_mask = None,task_type: Optional[str] = None):
        print("running forward_embeds in RouteLLMAdapter")
        if(self.type == "causal_llm"):
            row = {}
            self.orig_vocab_size = self.router_model.orig_vocab_size
            if attention_mask is None:
                attention_mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)
            input_embeds = embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
            attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            print("input_embeds shape:", input_embeds.shape)
            print("attention_mask shape:", attention_mask.shape)
            output_new = self.model(
                inputs_embeds = input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.router_model.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            logits = output_new.logits  # [1, seq_len, vocab_size]
            
            # Get logits of the last token
            last_token_logits = logits[:, -1, :]  # [1, vocab_size]
            # label_token_idx = next(
            #     (i for i, x in enumerate(row["output_ids"]) if x >= self.orig_vocab_size),
            #     None,
            # )
            # if label_token_idx is None:
            #     return None
            score_logits = last_token_logits[0, self.orig_vocab_size:]
            # print("output_new",output_new)
            # print("shape of embeds", input_embeds.shape)
            
            # row["output_ids"] = output_new.logits.squeeze().cpu()
            # print("output_ids",row["output_ids"])
            # output = self.tokenizer.decode(row["output_ids"])
            # print("output_words",output)
            # find the first token within the special tokens range. This is our score prediction.


            # score_logits = np.array(
            #     output_new.scores[label_token_idx][0].to("cpu")[self.orig_vocab_size :]
            # )

            # row["score_logits"] = score_logits
            binary_prob, softmax_scores = self.router_model.compute_routing_prob(score_logits)
            zero_tensor = torch.tensor(0.0, device=binary_prob.device, dtype=binary_prob.dtype)

            if(self.consider_cost):
                binary_prob = binary_prob - self.threshold
                # Create a scalar 0 tensor, matching device and type
                # Use stack to combine, so binary_prob's gradient can propagate back
                result = torch.stack([binary_prob, zero_tensor])
            # Return [strong_prob, weak_prob]
            result = torch.stack([1 - binary_prob, zero_tensor])
            
            print(f"  binary_prob: {binary_prob.item():.4f}")
            print(f"  result requires_grad: {result.requires_grad}")
            return result
            # row["softmax_scores"] = softmax_scores
            row["binary_prob"] = binary_prob

            # row = self.router_model.postprocess(row)
            output = row
            print("output",output)
            if output is None:
            # Route to strong model if output is invalid
                return [1, 0]
            else:
                return torch.tensor([1 - row["binary_prob"], row["binary_prob"]], dtype=torch.float32, device=self.device)
        elif(self.type == "bert"):
            # For bert type, use embedding-level forward
            if attention_mask is None:
                attention_mask = torch.ones(embeds.size(0), dtype=torch.long, device=embeds.device)
            input_embeds = embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
            attention_mask = attention_mask.unsqueeze(0)  # [1, seq_len]
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            print("outputs:",outputs)
            logits = outputs.logits  # [1, seq_len, num_classes]
            probs = torch.softmax(logits, dim=-1)  # [1, 3]
            binary_prob = probs[:, -2:].sum(dim=-1).squeeze(0)  # scalar tensor
            
            print(f"  binary_prob: {binary_prob.item():.4f}")
            print(f"  binary_prob requires_grad: {binary_prob.requires_grad}")
                # Compute prob of label 1 and 2 (tie, tier 2 wins)
            if binary_prob.item() < 0 or binary_prob.item() > 1:
                # Fallback: route to strong model
                return torch.tensor([1.0, 0.0], device=self.device, dtype=torch.float32)
            else:
                zero_tensor = torch.tensor(0.0, device=binary_prob.device, dtype=binary_prob.dtype)
                result = torch.stack([1 - binary_prob, zero_tensor])

                if(self.consider_cost):
                    binary_prob = binary_prob - self.threshold
                    # Create a scalar 0 tensor, matching device and type
                    # Use stack to combine, so binary_prob's gradient can propagate back
                    result = torch.stack([binary_prob, zero_tensor])
            # Return [strong_prob, weak_prob]
                print(f"  result requires_grad: {result.requires_grad}")
                return result  # [2], requires_grad=True
        else:
            raise ValueError(f"Unsupported router type: {self.type}") 

    def assemble_gcg_input(
        self,
        messages: str,
        suffix_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder: Optional[EncoderSpecBase] = None,
        prefix_mode: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_device = suffix_embeds.device
        if(self.type == "causal_llm"):
            input = {}
            input["messages"] = self.router.to_openai_messages([messages])
            row = self.router_model.preprocess(input)
            input_ids = torch.as_tensor(row["input_ids"]).to(self.model.device).reshape(1, -1)
            # print("ids in assemble")
            # Convert each token id to corresponding token
            # flat_input_ids = input_ids.squeeze().tolist()
            # for i in flat_input_ids:
            #     token = self.tokenizer.convert_ids_to_tokens(i)
            #     print(f"id: {i}, token: {token}")

            input_embeds = self.model.get_input_embeddings()(input_ids)   
            # input_embeds = input_embeds.squeeze(0)  # [seq_len, hidden_dim]    
            # full_mask = torch.ones(
            #     input_embeds.size(0),
            #     dtype=torch.long,
            #     device=target_device
            # )
            # return input_embeds, full_mask

            # find the appropriate cut_in pos
            # Get the token ID for start_header_id  
            start_header_id_token = self.tokenizer.convert_tokens_to_ids("Prediction")  
            
            # Find all occurrences of this token in input_ids  
            input_ids_list = input_ids.squeeze().tolist()
            start_header_indices = [i for i, token_id in enumerate(input_ids_list) if token_id == start_header_id_token]
            
            # Verify it's unique and get the index  

            start_header_idx = start_header_indices[-1]  
            print(f"Unique start_header_id found at index: {start_header_idx}")  

            print(f"Found {len(start_header_indices)} occurrences of start_header_id")
            cut_in_pos = start_header_idx - 1  # +1 to include the token itself
            
            msg_embeds_cut_in = input_embeds[:, :cut_in_pos, :]  # [1, cut_in_pos, hidden_dim]
            last_msg_embed = input_embeds[:, cut_in_pos:, :]    # [1,
            msg_embeds_cut_in = msg_embeds_cut_in.squeeze(0)  # [cut_in_pos, hidden_dim]
            last_msg_embed = last_msg_embed.squeeze(0)          # [remaining, hidden_dim
            suffix_embeds = suffix_embeds[1:, :]  # remove the starting bls token embed
            # print("shape of msg_embeds_cut_in:", msg_embeds_cut_in.shape)
            # print("shape of suffix_embeds:", suffix_embeds.shape)
            # print("shape of last_msg_embed:", last_msg_embed.shape)
            if prefix_mode:
                input_embeds = torch.cat([suffix_embeds, msg_embeds_cut_in,last_msg_embed], dim=0)  # [M+L-1, D]
            else:
                input_embeds = torch.cat([
                    msg_embeds_cut_in,
                    suffix_embeds,
                    last_msg_embed
                ], dim=0)
            # print("shape of final input_embeds:", input_embeds.shape)
            # using for test
            input_embeds = input_embeds.to(target_device)
            full_mask = torch.ones(
                input_embeds.size(0),
                dtype=torch.long,
                device=target_device
            )
            return input_embeds, full_mask
        elif(self.type == "bert"):
            inputs = self.tokenizer(
                messages, return_tensors="pt", padding=True, truncation=True
            )
            print("inputs",inputs)

            input_embeds = self.model.get_input_embeddings()(inputs['input_ids'])
            # print("input_embeds shape:", input_embeds.shape)
            # print("suffix_embeds shape:", suffix_embeds.shape)
            msg_embeds_ = input_embeds[:, :-1, :]  # [1, cut_in_pos, hidden_dim]
            suffix_embeds = suffix_embeds[1:,:]  # remove the starting bls token embed
            msg_embed= msg_embeds_.squeeze(0)  # [cut_in_pos, hidden_dim]
            msg_embeds_ = msg_embeds_.squeeze(0)          # [remaining, hidden_dim
            # print("shape of suffix_embeds:", suffix_embeds.shape)
            # print("shape of suffix_embeds:", msg_embed.shape)
            input_embeds = torch.cat([
                msg_embeds_,
                suffix_embeds,
            ], dim=0)
            full_mask = torch.ones(
                input_embeds.size(0),
                dtype=torch.long,
                device=target_device
            )
            return input_embeds, full_mask
        else:
            raise ValueError(f"Unsupported router type: {self.type}")  
    def cost_fn(self, logits: torch.Tensor) -> torch.Tensor:

        return logits


class P2LAdapter(Router):
    """
    P2L (Prompt-to-Leaderboard) Adapter - Local model loading method
    
    Supported model types:
    - lmarena-ai/p2l-135m-bt-01132025 (Qwen2, BT head)
    - lmarena-ai/p2l-1.5b-bt-01132025 (Qwen2, BT head)
    - lmarena-ai/p2l-7b-bt-01132025 (Qwen2, BT head)
    - lmarena-ai/p2l-7b-grk-02222025 (Qwen2, GRK head)
    """
    
    def __init__(
        self,
        name: str = "p2l",
        model_path: str = "lmarena-ai/p2l-7b-bt-01132025",
        model_type: str = "qwen2",  # "qwen2" or "llama"
        head_type: str = "bt",       # "bt", "rk", "grk", "bt_tie"
        loss_type: str = "bt",       # "bt", "rk", etc
        device: str = "cuda",
        max_length: int = 8192,
        local_files_only: bool = False,
        Path_cost_per_model: Path = "/home/haochuntang/Attack-Llm_router/Ensemble_router/data/cost.json",
        consider_cost: bool = False,
    ):
        super().__init__(name, device)
        
        import sys
        from pathlib import Path
        from transformers import AutoTokenizer
        import json
        

        self.model_path = model_path
        self.model_type = model_type
        self.head_type = head_type
        self.loss_type = loss_type
        self.max_length = max_length
        self.local_files_only = local_files_only
        print("consider_cost:", consider_cost)
        self.consider_cost = consider_cost
        # 1. Add P2L code path
        p2l_root = Path(__file__).parent.parent.parent / "Cur-Routers" / "p2l"
        if str(p2l_root) not in sys.path:
            sys.path.insert(0, str(p2l_root))
        
        # Import P2L's model retrieval function (not custom class)
        from p2l.model import get_p2l_model
        
        # 2. Load Tokenizer
        print("\nLoading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device=device
        )
        
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {len(self.tokenizer)}")
        print(f"  CLS token: {self.tokenizer.cls_token} (ID: {self.tokenizer.cls_token_id})")
        
        # 3. Load model_list.json (coefficient to model mapping)
        print("\nLoading model_list...")
        
        try:
            if local_files_only:
                # Load locally
                model_list_path = Path(model_path+"/model_list.json") 
                print("model_list_path:", model_list_path)
                with open(model_list_path, 'r') as f:
                    model_list_data = json.load(f)
            else:
                # Load from HuggingFace Hub
                from huggingface_hub import hf_hub_download
                
                model_list_path = hf_hub_download(
                    repo_id=model_path,
                    filename="model_list.json",
                    local_files_only=local_files_only
                )
                
                with open(model_list_path, 'r') as f:
                    model_list_data = json.load(f)
            
            # ✅ P2L's model_list.json format can be {"models": [...]} or directly [...]
            if isinstance(model_list_data, dict):
                self._model_list = model_list_data.get("models", [])
            else:
                self._model_list = model_list_data
            
        except Exception as e:
            warnings.warn(
                f"Cannot load model_list.json: {e}\n"
                f"Using default model list"
            )
            # Use default Chatbot Arena model list
            self._model_list = [
                "gpt-4-0125-preview",
                "gpt-4-turbo-2024-04-09",
                "claude-3-opus-20240229",
                "gemini-1.5-pro-api-0409-preview",
            ]
        
        self.num_models = len(self._model_list)
        
        
        print(f"✓ model_list loaded")
        print(f"  Number of models: {self.num_models}")
        print(f"  First 5 models: {self._model_list[:5]}")
        
        # ============================================================
        # 4. ✅ Use P2L's get_p2l_model to get correct model class
        # ============================================================
        print("\nLoading model...")
        
        # ✅ This is the correct way (consistent with eval.py)
        model_cls = get_p2l_model(
            model_type=self.model_type,
            loss_type=self.loss_type,
            head_type=self.head_type
        )
        
        print(f"  Model class: {model_cls.__name__}")
        print("is none?", model_cls is None)
        # ✅ Load model (passing CLS_id and num_models)
        self.model = model_cls.from_pretrained(
            model_path,
            CLS_id=self.tokenizer.cls_token_id,  # ← critical parameter
            num_models=self.num_models,          # ← critical parameter
            torch_dtype=torch.bfloat16,
            # device = device,
            device_map= device,
            local_files_only=local_files_only,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"✓ Model loaded")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
        
        # ============================================================
        # 5. Initialize Projection layer (if needed - for non-standard input)
        # ============================================================
        self.projection = None
        
        print(f"\n{'='*80}")
        print(f"✓ P2LAdapter initialization complete")
        print(f"{'='*80}\n")

        # ================================================
        # 6. Load cost configuration (optional)
        # ================================================
        print("\nLoading model cost configuration...")
        with open("/home/haochuntang/Attack-Llm_router/Ensemble_router/data/cost.json", "r", encoding="utf-8") as f:
            model_cost = json.load(f)
        # Convert model_configs list to {model_name: cost} dict
        self.model_cost_dict = {item['model_name']: item['cost'] for item in model_cost['model_configs']}

    
    # Text-level interface
    def cost_fn(self, logits: torch.Tensor, lam: float = 1.0) -> torch.Tensor:
        model_list = self._model_list
        for model, idx in zip(model_list, range(len(model_list))):
            if model in self.model_cost_dict:
                cost = self.model_cost_dict[model]
            else:
                print("missing model in cost configs", model)
                cost = 1.0  # default cost
            logits[idx] = logits[idx] - lam * cost
        return logits

    # Text-level interface
    
    def route(self, prompt: str, suffix: str = "", type: str = "") -> str:
        """
        Routing decision
        
        Args:
            prompt: User input
            suffix: Adversarial suffix
            
        Returns:
            Selected model name
        """
        coefficients = self.get_coefficients(prompt, suffix)
        # print("Coefficients:", coefficients)
        # Select strategy based on head_type
        if self.head_type in ["bt", "bt_tie"]:
            # BT: select model with largest coefficient
            best_idx = coefficients.argmax().item()
        elif self.head_type in ["grk"]:
            # GRK: select model with largest coefficient among those > 0
            valid_mask = coefficients > 0
            if valid_mask.any():
                valid_coefficients = coefficients.clone()
                valid_coefficients[~valid_mask] = float('-inf')
                best_idx = valid_coefficients.argmax().item()
            else:
                best_idx = coefficients.argmax().item()
        else:
            # Default: select largest
            best_idx = coefficients.argmax().item()
        if(self.consider_cost):
            coefficients = self.cost_fn(coefficients, lam=0.001)
        return coefficients
    
    def get_logits(self, prompt: str, suffix: str = "") -> torch.Tensor:
        """
        Get logits (P2L's coefficients can be viewed as logits)
        
        Returns:
            coefficients: [num_models]
        """
        return self.get_coefficients(prompt, suffix)
    
    def get_coefficients(
        self,
        prompt: str,
        suffix: str = "",
        conversation: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Fetch P2L coefficients for a single example
        
        Args:
            prompt: single prompt
            suffix: adversarial suffix
            conversation: optional multi-turn conversation
            
        Returns:
            coefficients: [num_models] BT coefficients or RK coefficients
        """
        # ============================================================
        # 1. Build input (refer to eval.py's P2LPipeline.preprocess)
        # ============================================================
        if conversation is not None:
            # Multi-turn conversation
            messages = [{"role": "user", "content": msg} for msg in conversation[:-1]]
            messages.append({"role": "user", "content": conversation[-1] + suffix})
        else:
            # Single turn
            if isinstance(prompt, list):
                full_text = " ".join(prompt) + suffix
            else:
                full_text = prompt + suffix
            messages = [{"role": "user", "content": full_text}]
        # ✅ Use apply_chat_template (consistent with eval.py)
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        
        # ✅ Add CLS token (critical!)
        formatted = formatted + self.tokenizer.cls_token
        # ============================================================
        # 2. Tokenize
        # ============================================================
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            max_length=self.max_length,
            padding="longest",
            truncation=True,
        ).to(self.model.device)
        # ============================================================
        # 3. Model Forward
        # ============================================================
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                # output_hidden_states=True,
                # return_dict=True,
            )
        embed_layer = self.model.get_input_embeddings()       # nn.Embedding
        embed_matrix = embed_layer.weight                    # [vocab_size, hidden_dim]
        token_embed = embed_layer(inputs['input_ids'])
        
        # ============================================================
        # 4. Extract coefficients
        # ============================================================
        # ✅ P2L model output is P2LOutputs type
        from p2l.model import P2LOutputs
        
        if isinstance(outputs, P2LOutputs):
            coefficients = outputs.coefs.squeeze(0)  # [num_models]
        elif hasattr(outputs, 'coefs'):
            coefficients = outputs.coefs.squeeze(0)
        elif hasattr(outputs, 'logits'):
            # Fallback: use logits of last token
            warnings.warn("P2L model returned logits instead of coefficients")
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            # Need projection
            if self.projection is None:
                vocab_size = logits.size(-1)
                self.projection = nn.Linear(vocab_size, self.num_models).to(self.device)
                nn.init.xavier_normal_(self.projection.weight)
            
            coefficients = self.projection(logits).squeeze(0)
        else:
            raise ValueError(f"Unknown P2L output format: {type(outputs)}")
        
        return coefficients  # [num_models]
    
    # ================================================================
    # Embedding-level interface (forward_embeds) - for GCG attack
    # ================================================================
    
    def forward_embeds(
        self,
        embeds: torch.Tensor,
        encoder_name: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conversation_embeds: Optional[List[torch.Tensor]] = None,
        task_type: Optional[str] = None

    ) -> torch.Tensor:
        """
        Embedding-level forward (used by GCG attacks)

        P2L expects inputs to end with a CLS token.

        Args:
            embeds: [seq_len, hidden_dim] input embeddings
            encoder_name: encoder name
            attention_mask: [seq_len] mask
            conversation_embeds: embeddings for prior turns

        Returns:
            coefficients: [num_models] (requires_grad=True)
        """
        
        # ============================================================
        # 1. Process input
        # ============================================================
        if conversation_embeds is not None:
            # Multi-turn conversation: concatenate all turns
            all_embeds = conversation_embeds + [embeds]
            input_embeds = torch.cat(all_embeds, dim=0)  # [total_len, hidden_dim]
            
            if attention_mask is not None:
                prev_masks = [
                    torch.ones(e.size(0), dtype=torch.long, device=self.device)
                    for e in conversation_embeds
                ]
                full_mask = torch.cat(prev_masks + [attention_mask], dim=0)
            else:
                full_mask = None
        else:
            input_embeds = embeds
            full_mask = attention_mask
        
        # ✅ Add CLS token embedding (critical!)
        
        # ============================================================
        # 2. P2L Model Forward
        # ============================================================
        inputs_embeds = input_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
        # print("embeds entrance's input_embeds:", inputs_embeds[0,:, 0])  # print last 5 tokens embeds
    
        if full_mask is not None:
            full_mask = full_mask.unsqueeze(0)  # [1, seq_len+1]
        # model_dtype = next(self.model.parameters()).dtype
        # inputs_embeds = inputs_embeds.to(dtype=model_dtype)
        # print("Input embeds dtype:", inputs_embeds.dtype)
        # Forward (retain gradients)

        outputs = self.model.forward_embedding(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            # return_dict=True
        )
        
        # ============================================================
        # 3. Extract coefficients
        # ============================================================
        from p2l.model import P2LOutputs
        
        if isinstance(outputs, P2LOutputs):
            coefficients = outputs.coefs.squeeze(0)  # [num_models]
        elif hasattr(outputs, 'coefs'):
            coefficients = outputs.coefs.squeeze(0)
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits[:, -1, :]  # [1, vocab_size]
            
            if self.projection is None:
                vocab_size = logits.size(-1)
                self.projection = nn.Linear(vocab_size, self.num_models).to(self.device)
                nn.init.xavier_normal_(self.projection.weight)
            
            coefficients = self.projection(logits).squeeze(0)
        else:
            raise ValueError(f"Unknown P2L output format: {type(outputs)}")
        # print("Embeds entrance's coefficients:", coefficients)

        if(self.consider_cost):
            coefficients = self.cost_fn(coefficients, lam=0.001)
        
        return coefficients  # [num_models], requires_grad=True
    
    # Batch interface
    
    def get_logits_batch(
        self,
        prompts: List[str],
        suffixes: List[str] = None
    ) -> torch.Tensor:
        """Batch fetch coefficients"""
        if suffixes is None:
            suffixes = [""] * len(prompts)
        
        # Build messages list
        messages_batch = []
        for prompt, suffix in zip(prompts, suffixes):
            full_text = prompt + suffix
            messages = [{"role": "user", "content": full_text}]
            messages_batch.append(messages)
        
        # Batch apply_chat_template
        formatted_batch = [
            self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=False,
                add_special_tokens=False,
            ) + self.tokenizer.cls_token
            for msgs in messages_batch
        ]
        
        # Batch tokenize
        inputs = self.tokenizer(
            formatted_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Batch forward
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract coefficients
        from p2l.model import P2LOutputs
        
        if isinstance(outputs, P2LOutputs):
            return outputs.coefs  # [batch_size, num_models]
        elif hasattr(outputs, 'coefs'):
            return outputs.coefs
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits[:, -1, :]
            
            if self.projection is None:
                vocab_size = logits.size(-1)
                self.projection = nn.Linear(vocab_size, self.num_models).to(self.device)
                nn.init.xavier_normal_(self.projection.weight)
            
            return self.projection(logits)
        else:
            raise ValueError(f"Unknown P2L output format: {type(outputs)}")
    
    def route_batch(
        self,
        prompts: List[str],
        suffixes: List[str] = None
    ) -> List[str]:
        """Batch routing"""
        coefficients_batch = self.get_logits_batch(prompts, suffixes)
        
        # Select strategy based on head_type
        if self.head_type in ["bt", "bt_tie"]:
            best_indices = coefficients_batch.argmax(dim=1).tolist()
        elif self.head_type in ["grk"]:
            valid_mask = coefficients_batch > 0
            coefficients_masked = coefficients_batch.clone()
            coefficients_masked[~valid_mask] = float('-inf')
            best_indices = coefficients_masked.argmax(dim=1).tolist()
        else:
            best_indices = coefficients_batch.argmax(dim=1).tolist()
        
        return [self._model_list[idx] for idx in best_indices]
    
    # ================================================================
    # Helper methods
    # ================================================================
    
    def set_train_mode(self, train: bool = True):
        """Set train/eval mode"""
        if train:
            self.model.train()
        else:
            self.model.eval()
    
    def supports_embedding_forward(self) -> bool:
        """P2L supports embedding-level forward"""
        return True
    def assemble_gcg_input(
        self,
        messages: str,
        suffix_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder: Optional[EncoderSpecBase] = None,
        prefix_mode: bool = False,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assemble P2L router input to keep the CLS token at the end.

        Format: [message_tokens (except the last)] + [suffix] + [last message token]
        """
        # ✅ Use P2L Router's own tokenizer
        tokenizer = self.tokenizer
        target_device = suffix_embeds.device

        messages = [{"role": "user", "content": messages}]
        formated = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        # Tokenize messages
        formated = formated + tokenizer.cls_token
        encoded = tokenizer(
            formated,
            return_tensors="pt",
            max_length=self.max_length,
            padding="longest",
            truncation=True,
        ).to(target_device)
        input_ids = encoded['input_ids']  # [1, seq_len]
        attention_mask = encoded['attention_mask']  # [1, seq_len]
        input_ids = input_ids.to(encoder.get_model().device)
        with torch.no_grad():
            input_embeds = encoder.get_model().get_input_embeddings()(input_ids)       
        # ✅ P2L specific format: keep last token (usually CLS)
        # print("shape of input_embeds before assembling:", input_embeds.shape)
        # input_embeds's batch dim is fixed to 1
        
        input_embeds = input_embeds.squeeze(0)  # [seq_len, hidden_dim]
        # print("shape of input_embeds after unsqueeze:", input_embeds.shape)
        last_msg_embed = input_embeds[-1].unsqueeze(0)      # [1, hidden_dim]
        before_embed = input_embeds[:3]   # [3, hidden_dim] - embeddings of first 3 tokens
        # msg_embeds_except_last = input_embeds[:-1]         # [M-1, hidden_dim]
        # Remove from 4th to second-to-last
        if input_embeds.size(0) > 4:
            msg_embeds_except_last = input_embeds[3:-1]
        else:
            msg_embeds_except_last = input_embeds
        # print("shape of msg_embeds_except_last:", msg_embeds_except_last.shape)
        # print("shape of suffix_embeds:", suffix_embeds.shape)
        # print("shape of last_msg_embed:", last_msg_embed.shape)
        # ✅ P2L specific format: keep last token (usually CLS)
        # Concatenate: [msg_except_last] + [suffix] + [last_token]
        if prefix_mode:
            input_embeds = torch.cat([before_embed,suffix_embeds, msg_embeds_except_last,last_msg_embed], dim=0)  # [M+L-1, D]
        else:

            input_embeds = torch.cat([
                before_embed,
                msg_embeds_except_last,
                suffix_embeds,
                last_msg_embed
            ], dim=0)
        # input_embeds = input_embeds.unsqueeze(0)  # [seq_len + suffix_len, hidden_dim]
        # print("shape of input_embeds after assembling:", input_embeds.shape)
        # attention mask all 1s, same length as input_embeds
        full_mask = torch.ones(
            input_embeds.size(0),
            dtype=torch.long,
            device=target_device
        )
        
        return input_embeds, full_mask

class GraphRouterAdapter(Router):
    """
    Adapter for GraphRouter (GNN-based router)
    """
    
    def __init__(
        self,
        name: str = "graph-router",
        config_path: str = None,
        model_path: str = None,
        encoder_name: str = "all-MiniLM-L6-v2",
        device: str = "cuda"
    ):
        super().__init__(name, device)
        
        import sys
        from pathlib import Path
        import yaml
        import pickle
        import json
        import pandas as pd
        import re

        # ============================================================
        # 1. Determine GraphRouter root directory
        # ============================================================
        graph_router_root = Path(__file__).parent.parent.parent / "Cur-Routers" / "GraphRouter"
        self. graph_router_root = graph_router_root
        if str(graph_router_root) not in sys.path:
            sys.path.insert(0, str(graph_router_root))
        
        from model.graph_nn import EncoderDecoderNet,form_data
        # from model.multi_task_graph_router import form_data
        # ============================================================
        # 2. Load config file (using absolute path)
        # ============================================================
        print("\nLoading config file...")
        print("config_path before processing:", config_path)
        if config_path is None:
            config_path = graph_router_root / "configs" / "config.yaml"
            print("Using default config file:", config_path)
        else:
            config_path = Path(config_path)
            # If relative path, convert to absolute path
            if not config_path.is_absolute():
                config_path = graph_router_root / "configs" / "config.yaml"
    
        print(f"\n{'='*60}")
        print("GraphRouterAdapter initializing")
        print(f"{'='*60}")
        print(f"GraphRouter root: {graph_router_root}")
        print(f"Config file: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # ============================================================
        # 3. Load LLM descriptions and embeddings (✅ convert relative to absolute path)
        # ============================================================
        
        # LLM description (JSON)
        llm_description_path = self.config['llm_description_path']
        llm_description_path = Path(llm_description_path)
        
        # ✅ If relative path, based on GraphRouter root directory
        if not llm_description_path.is_absolute():
            llm_description_path = graph_router_root / llm_description_path
        
        print(f"LLM description file: {llm_description_path}")
        
        with open(llm_description_path, 'r', encoding='utf-8') as f:
            llm_description = json.load(f)
        
        self._model_list = list(llm_description.keys())
        self.num_llms = len(self._model_list)
        print(f"LLM list: {self._model_list}")
        print(f"LLM count: {self.num_llms}")
        
        # LLM embeddings (PKL)
        llm_embedding_path = self.config['llm_embedding_path']
        llm_embedding_path = Path(llm_embedding_path)
        
        # ✅ If relative path, based on GraphRouter root directory
        if not llm_embedding_path.is_absolute():
            llm_embedding_path = graph_router_root / llm_embedding_path
        
        print(f"LLM embeddings file: {llm_embedding_path}")
        
        with open(llm_embedding_path, 'rb') as f:
            self.llm_embeddings = pickle.load(f)  # [num_llms, llm_dim]
        
        print(f"LLM embeddings shape: {self.llm_embeddings.shape}")
                # LLM description (JSON)
        task_description_path = self.config['task_description_path']
        task_description_path = Path(task_description_path)
        
        # ✅ If relative path, based on GraphRouter root directory
        if not task_description_path.is_absolute():
            task_description_path = graph_router_root / task_description_path
        
        print(f"Task description file: {task_description_path}")
        
        with open(task_description_path, 'r', encoding='utf-8') as f:
            self.task_description = json.load(f)
        print("task_description:", self.task_description)
        self._task_list = list(self.task_description.keys())
        self.num_tasks = len(self._task_list)
        print(f"Task list: {self._task_list}")
        print(f"Task count: {self.num_tasks}")
        # ============================================================
        # 4. Initialize GNN model
        # ============================================================
        query_dim = self.config.get('query_dim', 384)  # SentenceTransformer default dimension
        llm_dim = self.llm_embeddings.shape[1]
        hidden_dim = self.config['embedding_dim']
        edge_dim = self.config['edge_dim']
        
        self.model = EncoderDecoderNet(
            query_feature_dim=query_dim,
            llm_feature_dim=llm_dim,
            hidden_features=hidden_dim,
            in_edges=edge_dim
        ).to(device)
        # self.model = EncoderDecoderNet(
        #     query_feature_dim=query_feature_dim, 
        #     llm_feature_dim=llm_feature_dim,
        #      hidden_features=hidden_features_size,
        #      in_edges=in_edges_size).to(device)
        print("\nGNN model structure:")
        print(f"  Query dim: {query_dim}")
        print(f"  LLM dim: {llm_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Edge dim: {edge_dim}")
        # self.check_embedding_match()

        # ============================================================
        # 5. Load trained model weights
        # ============================================================
        model_path = Path(model_path) if model_path is not None else None
        if model_path is None:
            model_path = self.config.get('model_path', 'model_path/best_model.pth')
        else:
            if not model_path.is_absolute():
                model_path = graph_router_root / model_path
        model_path = graph_router_root / model_path

        print(f"Load model weights from: {model_path}")
        
        model_path = Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"\n✓ Loaded model weights: {model_path}")
        else:
            warnings.warn(f"⚠ Model weights not found: {model_path}, using random initialization")
        # Print model weights (for debug)
        # print("\nGNN model weight parameters:")
        # for name, param in self.model.named_parameters():
        #     print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
        # GCG attack requires train mode (retain gradients)
        # Use eval mode for normal inference
        # self.model.eval()  # default eval, switch to train for GCG
        
        # ============================================================
        # 6. Initialize Sentence Encoder (for text -> embedding)
        # ============================================================
        from sentence_transformers import SentenceTransformer
        
        # Use same encoder as during training
        encoder_name = self.config.get('sentence_encoder', 'all-MiniLM-L6-v2')
        self.sentence_encoder = SentenceTransformer(encoder_name).to(device)
        self.sentence_encoder.eval()
        self.encoder = self.sentence_encoder
        # self.encoder = self.sentence_encoder  # compatible with old interface
        
        print(f"\nSentence Encoder: {encoder_name}")
        
        # ============================================================
        # 7. Preprocess LLM features (convert to tensor, no gradients needed)
        # ============================================================
        self.llm_features_tensor = torch.tensor(
            self.llm_embeddings,
            dtype=torch.float,
            device=device,
            requires_grad=False  # LLM features fixed, not optimized
        )
        

        # ============================================================
        # 9. Optional: preload historical context nodes (for predict_new_query)
        # ============================================================
        self.form_data_helper = form_data(self.device)  

        self._context_ready = False 
        self._load_context_from_disk(graph_router_root)

        print(f"\n{'='*60}")
        print("✓ GraphRouterAdapter initialization complete")
        print(f"{'='*60}\n")
    
    # Text-level interface (route, get_logits)
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling compatible with BaseModelOutput and tuple outputs.

        Args:
            model_output: BaseModelOutput or tuple
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled: [batch_size, hidden_dim]
        """
        # ✅ Fix: compatible with both formats
        if hasattr(model_output, 'last_hidden_state'):
            # BaseModelOutput format
            token_embeddings = model_output.last_hidden_state
        elif isinstance(model_output, tuple):
            # Tuple format
            token_embeddings = model_output[0]
        else:
            raise TypeError(f"Unsupported model_output type: {type(model_output)}")
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask

    def route(
        self,
        prompt: str,
        suffix: str = "",
        type: str = "",
        # task_desc: str = "The Alpaca dataset is designed for instruction-following tasks, where the model is required to generate coherent and contextually appropriate responses to given instructions or prompts. It focuses on understanding diverse user requests and providing informative and accurate outputs based on those instructions."
    ) -> str:
        """
        Text-level routing decision.

        Args:
            prompt: user query
            suffix: adversarial suffix (optional)
            task_desc: task description

        Returns:
            chosen LLM name
        """
        # print("task_desc:", self.task_description)
        if(type is None):
            warnings.warn(f"Unknown task type: {type}, using 'general'")
            type = "general"
            task_desc = self.task_description.get("mmlu")['feature']
        else:
            # print(f"Using task type: {type}")
            task_desc = self.task_description.get(type)['feature']
        logits = self.get_logits(prompt, suffix, task_desc)
        # best_llm_idx = logits.argmax().item()
        # return self._model_list[best_llm_idx]
        return logits    
    def get_logits(
        self,
        prompt: str,
        suffix: str = "",
        task_desc: str = "general"
    ) -> torch.Tensor:
        """
        Get routing logits at text level.

        Returns:
            logits: [num_llms] score for each LLM
        """
        # Concatenate prompt + suffix
        full_text = prompt + " " + suffix if suffix else prompt
        
        # Encode using SentenceTransformer
        with torch.no_grad():
            query_emb = self.sentence_encoder.encode(
                [full_text],
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False  # ← add this line

            )  # [1, query_dim]
            # print("full_text:", full_text)
            # print("shape of query_emb:", query_emb.shape)
            # print('using sentence encoder to get query_emb first 5:', query_emb[0][:5])
            
            task_emb = self.sentence_encoder.encode(
                [task_desc],
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False  # ← add this line
            )  # [1, task_dim]
            # print("task_desc:", task_desc)
            # print('using sentence encoder to get task_emb first 5:', task_emb[0][:5])
        query_emb = query_emb.clone()
        task_emb = task_emb.clone()
        # GNN forward
        # logits = self._gnn_forward(
        #     query_features=query_emb,
        #     task_id=task_emb,
        #     llm_features=self.llm_features_tensor
        # )
        logits = self.predict_new_query(
            new_query_emb=query_emb.squeeze(0),
            new_task_emb=task_emb.squeeze(0)    

        )
        
        return logits  # [num_llms]
    
    # ================================================================
    # Embedding-level interface (forward_embeds) - for GCG attack
    # ================================================================
    
    def forward_embeds(
        self,
        embeds: torch.Tensor,
        encoder_name: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conversation_embeds: Optional[List[torch.Tensor]] = None,
        encoder: Optional[EncoderSpecBase] = None,
        track_gradients: bool = True,
        task_type: str = "general",
    ) -> torch.Tensor:
        """
        Embedding-level forward for GCG attacks.

        Gradient flow overview:
            suffix_embeds [L, hidden_dim] (requires_grad=True)
                ↓ concat with message
            input_embeds [M+L, hidden_dim]
                ↓ encoder forward
            hidden_states [M+L, hidden_dim']
                ↓ mean pooling (2D -> 1D)
            query_emb [hidden_dim']
                ↓ (optional) projection
            query_emb_aligned [query_dim]
                ↓ GNN forward
            logits [num_llms]
                ↓ backward

        Args:
            embeds: [seq_len, hidden_dim] encoder outputs
            encoder_name: encoder name (for logging)
            attention_mask: [seq_len] attention mask (optional)
            task_desc: task description

        Returns:
            logits: [num_llms] routing scores (with gradients)
        """
        if(encoder is None):
            encoder = self.encoder
        else:
            encoder = encoder
        grad_info = {} if track_gradients else None
        attention_mask = torch.ones(
            embeds.size(0),
            dtype=torch.long,
            device=self.device
        )
        # Step 1: Encoder forward
        if track_gradients:
            embeds.retain_grad()
        # print("dtype of input_embeds:", input_embeds.dtype)
        # print("dtype of attention_mask:", attention_mask.dtype if attention_mask is not None else None)
        transformer_model = encoder[0].auto_model  # get underlying Transformer

        encoder_outputs = transformer_model(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0) if attention_mask is not None else None
        )
        # print("encoder_outputs",encoder_outputs)
        # pooled = mean_pooling(model_output,encoded_input['attention_mask'])
        # ============================================================
        # ✅ Step 2: Mean pooling
        # ============================================================

        if attention_mask is not None:
            mask_for_pooling = attention_mask.unsqueeze(0)  # [1, seq_len]
        else:
            mask_for_pooling = torch.ones(
                1, embeds.size(0), 
                dtype=torch.long, 
                device=embeds.device
            )
        pooled = self.mean_pooling(encoder_outputs, mask_for_pooling)

        # pooled = mean_pooling(encoder_outputs, mask_for_pooling)  # [hidden_dim]

        # ✅ Step 3: Normalize (consistent with SentenceTransformer)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        node_features = pooled.squeeze(0)  # [hidden_dim]
        # print("nodef",node_features.shape)
        if track_gradients:
            node_features.retain_grad()
            grad_info['input_embeds'] = embeds
            grad_info['node_features'] = node_features

        query_emb = node_features
        expected_dim = self.get_query_dim()
        # print("expected_dim:", expected_dim)
        # print('before projection, query_embshape:', query_emb.shape[0])
        # Get actual dimension of query_emb
        assert query_emb.shape[0] == expected_dim

            # Need projection
            # if self.projection is None or self.projection.in_features != query_emb.size(0):
            #     # Dynamically create projection layer
            #     self.projection = nn.Linear(
            #         query_emb.size(0),
            #         expected_dim
            #     ).to(self.device)
                
            #     # Xavier initialization
            #     nn.init.xavier_normal_(self.projection.weight)
            #     if self.projection.bias is not None:
            #         nn.init.zeros_(self.projection.bias)
                
            # print(f"  Create projection layer: {query_emb.size(0)} -> {expected_dim}")
            
            # Project
            # query_emb = self.projection(query_emb)  # [expected_dim]
        
        # Now query_emb is [query_dim], as expected by GNN
        query_features = query_emb.unsqueeze(0)  # [1, query_dim]
        # print('after projection, query_features:', query_features)
        # ============================================================
        # Step 3: Task Embedding (no gradients)
        # ============================================================
        if task_type is None:
            warnings.warn(f"Unknown task type: {task_type}, using 'general'")
            task_type = "general"
            task_desc = self.task_description.get("mmlu")['feature']
        else:
            task_desc = self.task_description.get(task_type)['feature']
        with torch.no_grad():
            task_emb = self.sentence_encoder.encode(
                [task_desc],
                convert_to_tensor=True,
                device=self.device
            )  # [1, task_dim]
        
        task_id = task_emb.clone().detach().requires_grad_(False)
        
        # ============================================================
        # Step 4: GNN Forward (retain gradients)
        # ============================================================
        # logits = self._gnn_forward(
        #     query_features=query_features,
        #     task_id=task_id,
        #     llm_features=self.llm_features_tensor,
        #     requires_grad=True  # ✅ Critical: GNN needs gradients
        # )
        logits = self.predict_new_query(
            new_query_emb=query_features.squeeze(0),      # [query_dim]
            new_task_emb=task_id.squeeze(0)  # [task_dim]
        )
        # print(f"  logits shape: {logits.shape}")
        # print(f"  logits requires_grad: {logits.requires_grad}") 
        return logits  # [num_llms], requires_grad=True
    
    # Internal utility methods
    def _gnn_forward(
            self,
            query_features: torch.Tensor,
            task_id: torch.Tensor,
            llm_features: torch.Tensor,
            requires_grad: bool = False
        ) -> torch.Tensor:
            
            # 1. Determine total number of nodes
            # FeatureAlign concatenates features: [Query(1), LLMs(N)]
            # So Query index is 0
            # LLM indices are 1, 2, ..., N
            num_llms = self.num_llms
            
            # ==========================================
            # Fix 1: index offset (LLM starts from 1)
            # ==========================================
            edge_org = [0] * num_llms
            edge_des = [i + 1 for i in range(num_llms)]  # ✅ Fix: start from 1
            
            edge_index = torch.tensor(
                [edge_org, edge_des],
                dtype=torch.long,
                device=self.device
            )
            
            # ==========================================
            # Fix 2: placeholder edge_weight
            # ==========================================
            # Must provide a Tensor to satisfy network input requirements, even if we don't use it
            # Dimension must be self.config['edge_dim'] (usually 2 or 3)
            edge_weight = torch.zeros(
                num_llms,
                self.config['edge_dim'], 
                dtype=torch.float,
                device=self.device
            )
            
            # ==========================================
            # Fix 3: correct Mask logic
            # ==========================================
            # edge_mask: True here means "this is our prediction target"
            # We need to predict all Query->LLM edges
            edge_mask = torch.ones(num_llms, dtype=torch.bool, device=self.device)
            
            # edge_can_see: True here means "convolution layer can see these edges as background knowledge"
            # Because we have no Context, only edges to predict, to prevent data leakage,
            # and because we filled fake features (all zeros), we must hide them.
            edge_can_see = torch.zeros(num_llms, dtype=torch.bool, device=self.device)
            
            # Mode switching
            if requires_grad:
                self.model.train()
            else:
                self.model.eval()
                
            # Forward propagation
            logits = self.model(
                task_id=task_id,
                query_features=query_features,
                llm_features=llm_features,
                edge_index=edge_index,
                edge_mask=edge_mask,     # predict these
                edge_can_see=edge_can_see, # don't look at anything during convolution
                edge_weight=edge_weight  # although passed all zeros, masked by can_see
            )
            
            return logits
    
    # Utility methods
    
    def set_train_mode(self, train: bool = True):
        """Toggle train/eval mode"""
        if train:
            self.model.train()
        else:
            self.model.eval()
    
    def get_query_dim(self) -> int:
        """Get expected query embedding dimension"""
        return self.config.get('query_dim', 384)
    def assemble_gcg_input(
            self,
            messages: str,
            suffix_embeds: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder: Optional[EncoderSpecBase] = None,
            prefix_mode: bool = False,

        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            GraphRouter input assembly
            
            Format: [message_tokens] + [suffix]
            
            GraphRouter operates in Encoder space, so use Encoder's tokenizer
            """
            # GraphRouter needs to know which Encoder to use
            # Assuming GraphRouter has a primary_encoder attribute
            if(encoder is None):
                encoder = self.encoder
            else:
                encoder = encoder
            tokenizer = encoder.get_tokenizer()

            # Issue: here should fix the tokenizer to be used
            # Tokenize messages
            tokens = tokenizer(
                messages,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Use Encoder's embedding layer
            embed_layer = encoder.get_model().get_input_embeddings().to(self.device)
            msg_embeds = embed_layer(tokens['input_ids']).squeeze(0)  # [M, hidden_dim]
            last_msg_embed = msg_embeds[-1].unsqueeze(0)  # [1, hidden_dim]
            msg_embeds_except_last = msg_embeds[:-1]      # [M-1, hidden_dim]
            suffix_embeds = suffix_embeds.to(msg_embeds_except_last.device)
            if prefix_mode:
                input_embeds = torch.cat([suffix_embeds, msg_embeds_except_last,last_msg_embed], dim=0)  # [M+L-1, D]
            else:

                input_embeds = torch.cat([msg_embeds_except_last, suffix_embeds, last_msg_embed], dim=0)
            
            # Create attention mask
            msg_mask = tokens['attention_mask'].squeeze(0)
            
            if attention_mask is not None:
                suffix_mask = attention_mask
            else:
                suffix_mask = torch.ones(
                    suffix_embeds.size(0),
                    dtype=torch.long,
                    device=self.device
                )
            
            full_mask = torch.cat([msg_mask, suffix_mask], dim=0)

            return input_embeds, full_mask
    # Context loading and new query prediction (does not affect original route / forward_embeds)
    def _safe_json_list(self, text: str):
        text = re.sub(r'\s+', ', ', text.strip())
        try:
            return json.loads(text)
        except Exception:
            text = text.replace("[[,", "[[")
            return json.loads(text)

    def _load_context_from_disk(self, graph_router_root: Path):
        """Load historical queries/tasks/edge info, skip if missing, does not affect normal routing."""
        router_data_path = self.config.get("saved_router_data_path")
        if router_data_path is None:
            print("No saved_router_data_path provided, skipping context loading.")
            return
        router_data_path = Path(router_data_path)
        if not router_data_path.is_absolute():
            router_data_path = graph_router_root / router_data_path
        if not router_data_path.exists():
            warnings.warn(f"Cannot find historical data file: {router_data_path}, skipping context loading.")
            return

        try:
            df = pd.read_csv(router_data_path)
            # df = df[df['task_id'] == 'gsm8k']

        except Exception as e:
            warnings.warn(f"Failed to read historical data: {e}")
            return
        
        # Parse embedding column
        target_context_size = 240 
        
        # Calculate total number of complete Queries
        total_queries_in_csv = len(df) // self.num_llms
        
        if total_queries_in_csv > target_context_size:
            import random
            # 1. Generate index list of all Queries [0, 1, 2, ..., Total-1]
            all_query_indices = list(range(total_queries_in_csv))
            
            # 2. Randomly sample N Query indices
            # Use random.sample for sampling without replacement
            selected_query_indices = random.sample(all_query_indices, target_context_size)
            
            # 3. Find all row numbers corresponding to these Queries
            # For example: if Query 5 is selected, keep rows [5*N, 5*N+1, ..., 5*N+(N-1)]
            selected_rows = []
            for q_idx in selected_query_indices:
                start_row = q_idx * self.num_llms
                end_row = start_row + self.num_llms
                selected_rows.extend(range(start_row, end_row))
            
            # 4. Slice DataFrame by row numbers
            df = df.iloc[selected_rows].reset_index(drop=True)
            print(f"[Context] Sampled: randomly selected {target_context_size} Queries from {total_queries_in_csv} as context.")
        else:
            print(f"[Context] Loaded all: total {total_queries_in_csv} Queries.")
        query_raw = df["query_embedding"].tolist()
        task_raw = df["task_description_embedding"].tolist()
        effect_list = df["effect"].tolist()
        cost_list = df["cost"].tolist() if "cost" in df.columns else [0.0] * len(effect_list)

        num_queries = len(query_raw) // self.num_llms
        unique_idx = list(range(0, len(query_raw), self.num_llms))

        query_embs = []
        task_embs = []
        for i in unique_idx:
            q_str = query_raw[i]
            t_str = task_raw[i]
            
            # Parse string to list
            q_vec = self._safe_json_list(q_str) 
            t_vec = self._safe_json_list(t_str)
            
            # Fix: append the entire vector directly, not the 0th element
            query_embs.append(q_vec[0]) 
            task_embs.append(t_vec[0])

        # for q in query_raw:
        #     parsed = self._safe_json_list(q)
        #     query_embs.append(parsed[0])
        # for t in task_raw:
        #     parsed = self._safe_json_list(t)
        #     task_embs.append(parsed[0])

        # query_embs = torch.tensor(
        #     [query_embs[i] for i in unique_idx],
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        # task_embs = torch.tensor(
        #     [task_embs[i] for i in unique_idx],
        #     dtype=torch.float32,
        #     device=self.device,
        # )

        # Edges: historical query -> each LLM
        edge_org = [i for i in range(num_queries) for _ in range(self.num_llms)]  
        edge_des = list(range(self.num_llms)) * num_queries  #

        # Combine edge_weight, support edge_dim>=2: [effect, cost, ...]
        edge_dim = self.config.get('edge_dim', 2)
        edge_weight_tensor = torch.zeros(len(edge_org), edge_dim, device=self.device)        
        edge_weight_tensor[:, 0] = torch.tensor(cost_list, device=self.device)   # Col 0: Cost
        edge_weight_tensor[:, 1] = torch.tensor(effect_list, device=self.device) # Col 1: Effect
        self._ctx_edge_weight = edge_weight_tensor
        self._ctx_query = torch.tensor(query_embs, dtype=torch.float32, device=self.device)
        self._ctx_task = torch.tensor(task_embs, dtype=torch.float32, device=self.device)
        # self._ctx_edge_org = torch.tensor(edge_org, device=self.device, dtype=torch.long)
        # self._ctx_edge_des = torch.tensor(edge_des, device=self.device, dtype=torch.long)
        # self._ctx_edge_weight = edge_weight    # [E, edge_dim]
        self._ctx_edge_org = edge_org  
        self._ctx_edge_des = edge_des  
        self._ctx_effect_list = effect_list  
        effect_by_llm = {}  
        cost_by_llm = {}
        for i, llm_name in enumerate(self._model_list):  
            llm_effects = [self._ctx_effect_list[j] for j in range(i, len(self._ctx_effect_list), self.num_llms)]  
            effect_by_llm[llm_name] = np.mean(llm_effects)  
            llm_cost = [cost_list[j] for j in range(i, len(cost_list), self.num_llms)]  
            cost_by_llm[llm_name] = np.mean(llm_cost)
        print("Average effect by LLM:", effect_by_llm)
        print("Average cost by LLM:", cost_by_llm)
        self._ctx_cost_list = cost_list 
        self._context_ready = True
        print(f"✓ Historical context loaded: {num_queries} queries, {len(edge_org)} edges.")

    def predict_new_query(
        self,
        new_query_emb: torch.Tensor,
        new_task_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Use historical context with a new query to compute logits for the new query.
        - new_query_emb: [query_dim] (must match training dimension)
        - new_task_emb:  [task_dim]
        """
        if not self._context_ready:
            warnings.warn("Context not loaded; falling back to single-query inference.")
            return self._gnn_forward(
                query_features=new_query_emb.unsqueeze(0),
                task_id=new_task_emb.unsqueeze(0),
                llm_features=self.llm_features_tensor,
            )
        # 1. Merge historical data and new query  
        # ------------------------------------------------------------------
        # 1. Concatenate node features on GPU (Zero Copy to CPU)
        # ------------------------------------------------------------------
        # _ctx_query is already on GPU
        batch_q = torch.cat([self._ctx_query, new_query_emb.unsqueeze(0)], dim=0)
        batch_t = torch.cat([self._ctx_task, new_task_emb.unsqueeze(0)], dim=0)
        
        # ------------------------------------------------------------------
        # 2. Build graph index (Query -> LLM)
        # ------------------------------------------------------------------
        # Our goal is to replicate form_data logic: Query Index: 0..N, LLM Index: N+1..N+M
        
        total_queries = batch_q.shape[0]
        num_llms = self.num_llms
        
        # Construct source nodes (Query)
        # src: [0,0,0, ..., Q,Q,Q]
        # Note: _ctx_edge_org is list, converting to tensor is slow, recommend converting to GPU Tensor in init
        # Here assume not converted in init, directly generate new (because we know topology is simple)
        
        src = torch.arange(total_queries, device=self.device).repeat_interleave(num_llms)
        
        # Construct target nodes (LLM)
        # LLM node indices start from total_queries
        llm_start_idx = total_queries
        dst = torch.arange(llm_start_idx, llm_start_idx + num_llms, device=self.device).repeat(total_queries)
        
        edge_index = torch.stack([src, dst], dim=0)
        
        # ------------------------------------------------------------------
        # 3. Build Edge Attributes (Correct Order: Cost, Effect)
        # ------------------------------------------------------------------
        # _ctx_edge_weight should be stored as GPU Tensor in init
        # Assume _ctx_effect_list and _ctx_cost_list are lists
        
        # In init should store like this:
        # ctx_costs = torch.tensor(cost_list, device=self.device).reshape(-1, 1)
        # ctx_effects = torch.tensor(effect_list, device=self.device).reshape(-1, 1)
        # self.ctx_combined = torch.cat([ctx_costs, ctx_effects], dim=1) # [Cost, Effect]
        
        # Here for compatibility with your code:
        # Construct new edge attributes [Cost=0, Effect=0]
        new_edge_attr = torch.zeros(num_llms,self.config['edge_dim'], device=self.device)
        
        # Concatenate (assume self._ctx_edge_weight is already [Cost, Effect] order GPU Tensor)
        # If not, please fix order in _load_context_from_disk!
        masked_ctx_weight = self._ctx_edge_weight.clone()
    
    # 2. Set Cost column (column 0) to all zeros
        masked_ctx_weight[:, 0] = self._ctx_edge_weight[:, 0] * 0.01        
        edge_weight = torch.cat([masked_ctx_weight, new_edge_attr], dim=0)

        # edge_weight = torch.cat([self._ctx_edge_weight, new_edge_attr], dim=0)
        
        # ------------------------------------------------------------------
        # 4. Build Masks
        # ------------------------------------------------------------------
        num_ctx_edges = self._ctx_edge_weight.shape[0]
        num_new_edges = num_llms
        
        # edge_mask: predict last N edges
        edge_mask = torch.cat([
            torch.zeros(num_ctx_edges, dtype=torch.bool, device=self.device),
            torch.ones(num_new_edges, dtype=torch.bool, device=self.device)
        ])
        
        # edge_can_see: can only see Context
        edge_can_see = torch.cat([
            torch.ones(num_ctx_edges, dtype=torch.bool, device=self.device),
            torch.zeros(num_new_edges, dtype=torch.bool, device=self.device)
        ])    
        logits = self.model(
            task_id=batch_t,
            query_features=batch_q,
            llm_features=self.llm_features_tensor,
            edge_index=edge_index,
            edge_mask=edge_mask,
            edge_can_see=edge_can_see,
            edge_weight=edge_weight
        )
        # self.model.eval()
        # with torch.no_grad():
        #     logits_all = self.model(
        #         task_id=t_all,
        #         query_features=q_all,
        #         llm_features=self.llm_features_tensor,
        #         edge_index=torch.stack([edge_org, edge_des], dim=0),
        #         edge_mask=edge_mask,
        #         edge_can_see=edge_can_see,
        #         edge_weight=edge_weight,
        #     )
        # print("logits_all shape:", logits_all.shape)
        # Get num_llms logits for new query
        return logits  # [num_llms]
    def safe_parse(self, text):
        text = re.sub(r'\s+', ', ', text.strip())
        try:
            return json.loads(text)[0]
        except:
            return json.loads(text.replace("[[,", "[["))[0]

    def check_embedding_match(self):

        csv_path = self.config.get("saved_router_data_path")
        csv_path = self.graph_router_root / csv_path
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path).head(1)
        encoder_name = self.config.get('sentence_encoder', 'all-MiniLM-L6-v2')

        # Get text and Embedding from CSV
        csv_text = df['query'].iloc[0] # assume column name is query
        csv_emb = np.array(self.safe_parse(df['query_embedding'].iloc[0]))
        
        print(f"\nSample Text: '{csv_text}'")
        print(f"CSV Embedding Shape: {csv_emb.shape}")
        
        print(f"\nInitializing Encoder: {encoder_name}...")
        model = SentenceTransformer(encoder_name)
        
        # Generate new Embedding
        my_emb = model.encode([csv_text])[0]
        print(f"My Embedding Shape: {my_emb.shape}")
        
        # Dimension check
        if csv_emb.shape != my_emb.shape:
            print("\n❌ Fatal error: embedding dimensions do not match!")
            print(f"Training embedding shape: {csv_emb.shape}")
            print(f"Inference embedding shape: {my_emb.shape}")
            return

        # Similarity check
        csv_emb_tensor = torch.tensor(csv_emb, dtype=torch.float32)
        my_emb_tensor = torch.tensor(my_emb, dtype=torch.float32)
        sim = torch.cosine_similarity(csv_emb_tensor.unsqueeze(0), my_emb_tensor.unsqueeze(0))[0].item()
        print(f"\n🔍 Cosine Similarity: {sim:.4f}")
        
        if sim > 0.9:
            print("✅ Embeddings match; encoder choice is correct.")
        else:
            print("❌ Embeddings do not match!")
            print("Reason: training embeddings used a different encoder than the current one.")
            print("Impact: GNN receives misaligned features and produces static rankings.")
    # =========================================================
class LowRankRouter(Router):
    """
    Low-rank semantic router extracted from LowRankSemanticAdapter
    
    Structure:
    - Input: Text (str)
    - Process 1: HFTextEmbedder (Mini BERT) → (D,) embedding
    - Process 2: semantic_encoder (Low-Rank MLP: D -> rank -> L) → (L,) logits
    - Output: Target Space Logits (L,)
    """
    
    def __init__(
        self,
        semantic_encoder: nn.Module,       # ✅ Low-rank MLP (D -> rank -> L)
        target_model_names: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",      
        name: str = "semantic_router",
        device: str = "cuda",
        local_files_only: bool = False,
        tokenizer :Optional[AutoTokenizer]=None,
        encoder :Optional[AutoModel]=None,
    ):
        super().__init__(name, device)
        if(tokenizer is None):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)
        else:
            self.tokenizer = tokenizer
        if(encoder is None):
            self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            self.encoder = encoder
        self.encoder.train().to(device)
        self.model = self.encoder
        # for pair gennerate
        # semantic_router.encoder.eval()

        # ✅ Core parameters: embedder and semantic_encoder
        self.semantic_encoder = semantic_encoder.to(device)
        
        self._model_list = target_model_names  # ✅ Output space
        
        
        # Get dimension info (for debug)
        try:
            embedding_dim = self.semantic_encoder[0].in_features
            rank = self.semantic_encoder[0].out_features
            num_target = self.semantic_encoder[-1].out_features
            print(f"[SemanticRouter] Initialized")
            print(f"  - Embedder: Text → {embedding_dim}-dim")
            print(f"  - Semantic Encoder: {embedding_dim} → {rank} → {num_target}")
            print(f"  - Output: {len(target_model_names)} target models")
        except:
            print(f"[SemanticRouter] Initialized (dim info unavailable)")
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        toks = {k: v.to(self.device) for k, v in toks.items()}
        out = self.encoder(**toks)
        # Mean Pooling
        last = out.last_hidden_state
        attn = toks['attention_mask']
        mask = attn.unsqueeze(-1).expand(last.size()).float()
        emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        pooled = F.normalize(emb, p=2, dim=-1)
        pooled_cpu = pooled.detach().cpu()

        # print(f"[SemanticRouter.encode_texts] pooled[:10]: {pooled_cpu[:, :10]}")

        return pooled
    
    def route(self, prompt: str, suffix: str = "", type: str = "") -> torch.Tensor:
        """
        Single routing pass: text → embedding → logits
        
        Args:
            prompt: input text
            suffix: optional suffix
            
        Returns:
            logits: (L,) logits in target space
        """
        return self.get_logits(prompt, suffix, type)
    
    def route_batch(
        self,
        prompts: List[str],
        suffixes: List[str] = None
    ) -> torch.Tensor:
        """
        Batch routing
        
        Args:
            prompts: list of texts
            suffixes: optional list of suffixes
            
        Returns:
            logits: (B, L) logits
        """
        if suffixes is None:
            suffixes = [""] * len(prompts)
        
        full_texts = [p + s for p, s in zip(prompts, suffixes)]
        # Truncate text to prevent excessive length
        for i in range(len(full_texts)):
            full_texts[i] = full_texts[i][:512]
        # Encode all texts
        with torch.no_grad():
            text_embeddings = self.encode_texts(full_texts)  # (B, D)
            logits = self.semantic_encoder(text_embeddings)   # (B, L)
        return logits
    
    def get_logits(
        self,
        prompt: str,
        suffix: str = "",
        type: str = ""
    ) -> Optional[torch.Tensor]:
        """Get logits for a single query"""
        text_embeddings = self.encode_texts([prompt + suffix])  # (1, D)
        logits = self.semantic_encoder(text_embeddings)   # (1, L)
        return logits.squeeze(0)
    
    def get_logits_batch(
        self,
        prompts: List[str],
        suffixes: List[str] = None
    ) -> Optional[torch.Tensor]:
        """Get logits for a batch"""
        return self.route_batch(prompts, suffixes)
    
    def get_model_list(self) -> List[str]:
        return self._model_list
    
    def forward_embeds(
        self,
        embeds: torch.Tensor,
        encoder_name: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        task_type: Optional[str] = None
    ) -> torch.Tensor:
        """
        Embedding-level forward (used for GCG attacks)

        Process mirrors encode_texts:
        - embeds: (seq_len, D) encoder embeddings
        - mean pool to (D,)
        - semantic_encoder produces (L,) logits

        Args:
            embeds: (seq_len, D) token embeddings
            attention_mask: (seq_len,) attention mask

        Returns:
            logits: (L,) target-space logits (with gradient)
        """
        # ✅ If embeds is 1D, expand to 2D (batch)
        # if embeds.dim() == 1:
        #     embeds = embeds.unsqueeze(0)  # (1, D)
        
        # ============================================================
        # 1. Mean Pooling (refer to encode_texts logic)
        # ============================================================
        if attention_mask is None:
            # If no mask, assume all tokens are valid
            attention_mask = torch.ones(
                embeds.size(0),
                dtype=torch.float,
                device=embeds.device
            )
        else:
            # Ensure mask is float type
            attention_mask = attention_mask.float()
        seq_len = embeds.size(0)
        position_ids = torch.arange(seq_len, device=embeds.device).unsqueeze(0) 
        encoder_out = self.encoder(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            position_ids=position_ids,  # ✅ Critical: manually pass position IDs
            return_dict=True
        )

        # print("[DEBUG forward_embeds] input_embeds[:,:1]:", embeds[:, :1].detach().cpu().numpy())
        # print("[DEBUG forward_embeds] attention_mask[:1]:", attention_mask[:1].detach().cpu().numpy())
        # Expand mask to match embeds dimensions (seq_len, D)
        token_hidden = encoder_out.last_hidden_state  # [1, seq_len, dim]
        # print("[DEBUG forward_embeds] token_hidden[0,:5,:10]:", token_hidden[0, :, 0].detach().cpu().numpy())

        mask_expanded = (
            attention_mask.unsqueeze(0).unsqueeze(-1).expand_as(token_hidden).float()
        )        
        # Mean pooling: (seq_len, D) -> (D,)
        pooled = (token_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp_min(1e-8)
        pooled = F.normalize(pooled, p=2, dim=-1)
        
        # L2 normalize (refer to encode_texts F.normalize)
        # pooled = F.normalize(pooled.unsqueeze(0), p=2, dim=-1)  # (D,)
        pooled_cpu = pooled.detach().cpu()
        # print(f"[SemanticRouter.forward_embeds] pooled[:10]: {pooled_cpu[:, :10]}")
        # ============================================================
        # 2. Get logits through semantic_encoder
        # ============================================================
        # ✅ Retain gradients (don't use no_grad)
        logits = self.semantic_encoder(pooled).squeeze(0)  # (1, L)
        
        return logits # (L,), requires_grad=True
    def assemble_gcg_input(
        self,
        messages: str,
        suffix_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder: Optional[EncoderSpecBase] = None,
        prefix_mode: bool = False,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        token_ids = tokens['input_ids']  # [1, M]
        token_mask = tokens['attention_mask']  # [1, M]
        # print("[DEBUG assemble_gcg_input] message input_ids:", token_ids[0, :50].tolist())
        # try:
        #     print("[DEBUG assemble_gcg_input] message tokens[:50]:", [self.tokenizer.convert_ids_to_tokens(int(i)) for i in token_ids[0, :50].tolist()])
        # except Exception:
        #     pass

        # Use encoder's embedding layer to get full sequence embedding
        with torch.no_grad():
            embed_layer = self.encoder.get_input_embeddings()
            msg_embeds = embed_layer(token_ids)  # [1, M, D]
        first_msg_embed = msg_embeds[0, 0, :].unsqueeze(0)  # [1, D]
        last_msg_embed = msg_embeds[0, -1, :].unsqueeze(0)  # [1, D]
        msg_embeds_except_last = msg_embeds[0, 1:-1, :]      #
        # msg_embeds = msg_embeds.squeeze(0)  # [M, D]
        # print("[DEBUG assemble_gcg_input] msg_embeds[:5,:10]:", msg_embeds[:5, :10].detach().cpu().numpy())
        # print("[DEBUG assemble_gcg_input] suffix_embeds[:5,:10]:", suffix_embeds[:5, :10].detach().cpu().numpy())

        msg_mask = token_mask.squeeze(0)  # [M]
        if prefix_mode:
            input_embeds = torch.cat([first_msg_embed,suffix_embeds, msg_embeds_except_last,last_msg_embed], dim=0)  # [M+L-1, D]
        else:
            input_embeds = torch.cat([first_msg_embed,msg_embeds_except_last, suffix_embeds,last_msg_embed], dim=0)  # [M+L-1, D]

        if attention_mask is not None:
            suffix_mask = attention_mask
        else:
            suffix_mask = torch.ones(suffix_embeds.size(0), dtype=torch.long, device=self.device)
        full_mask = torch.cat([msg_mask, suffix_mask], dim=0)  
        # print("[DEBUG assemble_gcg_input] assembled input_embeds[:5,:10]:", input_embeds[:5, :10].detach().cpu().numpy())
        # print("[DEBUG assemble_gcg_input] full_mask[:50]:", full_mask[:50].detach().cpu().numpy())
        return input_embeds, full_mask

def create_router(router_type: str, **kwargs) -> Router:
    
    adapters = {
        "routellm": RouteLLMAdapter,
        "routerdc": RouterDCAdapter,
        "p2l": P2LAdapter,
        "graphrouter": GraphRouterAdapter,
        "graph": GraphRouterAdapter,
        "LowRank": LowRankRouter,
    }
    if router_type not in adapters:
        raise ValueError(
            f"Unknown router type: {router_type}\n"
            f"Available types: {list(adapters.keys())}"
        )
    
    return adapters[router_type](**kwargs)

class RouterFactory:
    """Router factory class (backward compatible)"""

    @staticmethod
    def create(router_type: str, **kwargs) -> Router:
        """Create a router (delegates to create_router)"""
        return create_router(router_type, **kwargs)


def test_graph_router():
    """Test GraphRouter with GCG attack"""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # 1. Load encoders
    print("1. Loading encoders...")
    encoders = EncoderFactory.create_all_from_yaml(device=device)
    test_encoders = encoders[:2]
    print(f"   Using {len(test_encoders)} encoders\n")
    
    # 2. Create GraphRouter
    print("2. Creating GraphRouter...")
    router = RouterFactory.create(
        router_type="graph",
        name="graph-router",
        config_path="configs/config.yaml",
        model_path="model_path/best_model.pth",
        device=device
    )
    print(f"   ✓ Router created: {router.name}")
    print(f"   ✓ Models: {router.get_model_list()}\n")
    
    # 3. Test text-level routing
    print("3. Testing text-level routing...")
    test_queries = [
        "What is the capital of France?",
        "Explain quantum computing",
        "Write a Python function to sort a list"
    ]
    
    for query in test_queries:
        model = router.route(query)
        print(f"   Query: '{query}'")
        print(f"   Routed to: {model}\n")
    
    # 4. Test with GCG attack
    print("4. Testing GCG attack...")
    
    vstar_builder = VStarBuilder(
        encoders=test_encoders,
        k=64,
        device=device
    )

# Usage example
# from models.RouterGCG import RouterGCG, GCGTrainerConfig


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    # Example usage:
    # test_graph_router()
    # # Create multiple Routers
    # router1 = create_router(
    #     "routerdc",
    #     name="routerdc-1",
    #     checkpoint_path="model1.pth",
    #     candidates=["gpt-3.5", "gpt-4"]
    # )
    
    # router2 = create_router(
    #     "routerdc",
    #     name="routerdc-2",
    #     checkpoint_path="model2.pth",
    #     candidates=["gpt-3.5", "gpt-4", "claude"]  # one more model
    # )
    
    # routers = [router1, router2]
    
    # # Verify consistency
    # print("\n1. Verify model_list consistency:")
    # result = verify_routers_consistency(routers)
    # print(f"  Consistent: {result['consistent']}")
    # print(f"  Common models: {result['common_models']}")
    # print(f"  Differences: {result['differences']}")
    # print(f"  {result['recommendation']}")
    
    # # Expand to unified list
    # print("\n2. Expand to unified model_list (union):")
    # unified = expand_routers_to_common_models(routers, strategy="union")
    
    # # Verify again
    # print("\n3. Verify after expansion:")
    # result = verify_routers_consistency(routers)
    # print(f"  {result['recommendation']}")
    
    # print("\n" + "=" * 60)