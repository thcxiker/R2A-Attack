from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn import functional as F
import torch
import json


def load_json_config(config_path: str) -> dict:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to JSON file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
def load_model_list(model_list_path: str) -> list:
    """
    Load model list from text file.
    
    Args:
        model_list_path: Path to model list file (one model per line)
    
    Returns:
        List of model names or paths
    """
    with open(model_list_path, "r", encoding="utf-8") as f:
        model_list = [line.strip() for line in f if line.strip()]
    return model_list
def load_router_data(data_path: str) -> list:
    """
    Load router dataset.
    
    Args:
        data_path: Path to dataset file (JSON or CSV format)
    
    Returns:
        List of data records
    """
    if data_path.endswith(".json"):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif data_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(data_path)
        data = df.to_dict(orient="records")
    else:
        raise ValueError("Only JSON and CSV formats are supported")
    return data
def get_model_list_union(Routers_model_list: Dict[str, List[str]]) -> List[str]:
    """
    Get the union of all router model lists.
    
    Returns:
        Union of all router model lists (sorted alphabetically)
    """
    all_models = set()
    for _, router_models in Routers_model_list.items():
        all_models.update(router_models)
    # Sort alphabetically for consistency
    union_models = sorted(list(all_models))
    return union_models


def build_model_index_mapping(Target_list: List[str], Routers_model_list: Dict[str, List[str]]) -> Dict[str, Dict[str, int]]:
    """
    Build mapping from each router's models to target model indices.
    
    Args:
        Target_list: List of target model names
        Routers_model_list: Dictionary mapping router names to their model lists
    
    Returns:
        Dictionary mapping router names to their model index mappings
    """
    # Create target_model → index mapping
    target_model_to_idx = {model: idx for idx, model in enumerate(Target_list)}

    mappings = {}
    for router_name, model_list in Routers_model_list.items():
        router_mapping = {}
        for router_idx, model in enumerate(model_list):
            union_idx = target_model_to_idx.get(model, -1)
            router_mapping[router_idx] = union_idx
        mappings[router_name] = router_mapping

    return mappings

def build_model_alignment(target_models: List[str], 
                          router_models: List[str]) -> Tuple[List[int], List[str]]:
    """
    Build mapping from router models to target models.
    
    Args:
        target_models: Union space model list
        router_models: Router's model list
    
    Returns:
        (mapping indices, missing models)
        - mapping[i] = index where router_models[i] maps to target_models
        - mapping[i] = -1 if model doesn't exist in target
    """
    target_index = {m: idx for idx, m in enumerate(target_models)}
    mapping: List[int] = []
    missing: List[str] = []
    
    for model in router_models:
        idx = target_index.get(model, -1)
        mapping.append(idx)
        if idx < 0:
            missing.append(model)
    
    return mapping, missing


def align_logits_to_target(logits: torch.Tensor, 
                           mapping: List[int], 
                           num_target_models: int,
                           fill_value: float = 0.0) -> torch.Tensor:
    """
    Align router's logits to target model space.
    
    Args:
        logits: Router's original logits (num_router_models,)
        mapping: Index mapping list
        num_target_models: Total number of target models
        fill_value: Fill value for unmapped positions
    
    Returns:
        aligned_logits: Aligned logits (num_target_models,)
    """
    aligned = torch.full((num_target_models,), fill_value, dtype=torch.float32)
    limit = min(len(mapping), logits.numel())
    
    for src_idx in range(limit):
        tgt_idx = mapping[src_idx]
        if tgt_idx >= 0:
            aligned[tgt_idx] = logits[src_idx]
    
    return aligned


def normalize_logits(logits: torch.Tensor, method: str = "softmax") -> torch.Tensor:
    """
    Normalize logits.
    
    Args:
        logits: Input logits
        method: Normalization method
            - "softmax": Convert to probability distribution [0, 1], sum to 1
            - "minmax": Scale to [0, 1]
            - "zscore": Standardize to mean 0, std 1
            - "layernorm": Layer normalization
            - "none": No normalization
    
    Returns:
        Normalized logits
    """
    if method == "softmax":
        return F.softmax(logits, dim=-1)
    elif method == "minmax":
        min_val = logits.min()
        max_val = logits.max()
        if max_val - min_val < 1e-8:
            return torch.zeros_like(logits)
        return (logits - min_val) / (max_val - min_val)
    elif method == "zscore":
        mean = logits.mean()
        std = logits.std()
        if std < 1e-8:
            return torch.zeros_like(logits)
        return (logits - mean) / std
    elif method == "layernorm":
        return F.layer_norm(logits, normalized_shape=(logits.size(-1),))
    elif method == "none":
        return logits
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_logits_from_entry(entry: Dict[str, Any]) -> Optional[torch.Tensor]:
    """
    Extract logits from data entry.
    
    Args:
        entry: Data entry dictionary with possible keys:
            - "logits": logits data
            - "coefs": coefficient data (equivalent to logits)
            - "coefficients": coefficient data
            - "scores": score data
    
    Returns:
        Logits tensor (C,) or None
    """
    # Try multiple possible key names
    for key in ("logits", "coefs", "coefficients", "scores"):
        logits = entry.get(key)
        if logits is not None:
            break
    
    if logits is None:
        return None
    
    # Handle case where it's already a Tensor
    if isinstance(logits, torch.Tensor):
        vec = logits
    else:
        # Handle list type
        if not isinstance(logits, list) or len(logits) == 0:
            return None
        
        # Handle nested list [[...]] or [...]
        first = logits[0] if isinstance(logits[0], list) else logits
        if len(first) == 0:
            return None
        
        vec = torch.tensor(first, dtype=torch.float32)
    
    # Ensure it's a 1D tensor
    if vec.dim() == 2:
        vec = vec.squeeze(0)
    elif vec.dim() > 2:
        raise ValueError(f"logits dims too high: {vec.dim()}")
    
    return vec.to(torch.float32)


def extract_prompt_from_entry(entry: Dict[str, Any]) -> str:
    """
    Extract prompt text from data entry.
    
    Args:
        entry: Data entry dictionary with possible keys:
            - "question": Question text
            - "prompt": Prompt text
            - "content": Content text
            - "text": Text
            - "messages": Messages text
    
    Returns:
        Prompt string
    """
    for key in ("question", "prompt", "content", "text", "messages"):
        if key in entry and isinstance(entry[key], str):
            return entry[key]
    
    # Return empty string if not found
    return ""





def extract_question_id_from_entry(entry: Dict[str, Any]) -> Optional[Any]:
    """
    Extract question_id from data entry.
    
    Args:
        entry: Data entry dictionary
    
    Returns:
        question_id or None
    """
    for key in ("question_id", "id", "qid", "index"):
        qid = entry.get(key)
        if qid is not None:
            return qid
    return None