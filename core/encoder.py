# encoder_robust_attack.py
# Refactored with Abstract Base Classes for better modularity
# Now supports loading from YAML configuration
# -------------------------------------------------------------

import math
import random
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModel

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    torch = None
    nn = None
    F = None

# ============================================================
# 0. YAML CONFIGURATION LOADER
# ============================================================

class EncoderConfig:
    """
    Configuration manager for encoders loaded from YAML
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Args:
            config_path: Path to encoders.yaml file
                        If None, searches in default locations
        """
        if config_path is None:
            config_path = self._find_config_file()
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find encoders.yaml in default locations"""
        search_paths = [
            "config/encoders.yaml",
            "../config/encoders.yaml",
            "../../config/encoders.yaml",
            "/home/haochuntang/Attack-Llm_router/Ensemble_router/config/encoders.yaml"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            "Could not find encoders.yaml. Please specify config_path explicitly."
        )
    
    def _load_config(self) -> Dict:
        """Load YAML configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_encoder_configs(self, role: Optional[str] = None) -> List[Dict]:
        """
        Get encoder configurations
        
        Args:
            role: Filter by role (e.g., 'train_eot', 'eval_only', 'router_backbone')
                 If None, returns all EOT encoders
        
        Returns:
            List of encoder config dicts
        """
        encoders = self.config.get('encoders4EOT', [])
        
        if role:
            encoders = [enc for enc in encoders if enc.get('role') == role]
        
        return encoders
    
    def get_api_encoders(self) -> List[Dict]:
        """Get API-only encoder configurations"""
        return self.config.get('API_ONLY_encoders', [])
    
    def get_defaults(self) -> Dict:
        """Get default configuration values"""
        return self.config.get('defaults', {})
    
    def get_encoder_by_name(self, name: str) -> Optional[Dict]:
        """
        Get encoder configuration by name
        
        Args:
            name: Encoder name
            
        Returns:
            Encoder config dict or None if not found
        """
        all_encoders = self.get_encoder_configs() + self.get_api_encoders()
        
        for enc in all_encoders:
            if enc.get('name') == name:
                return enc
        
        return None


# ============================================================
# 1. ABSTRACT ENCODER SPECIFICATION
# ============================================================

class EncoderSpecBase(ABC):
    """
    Abstract base class for encoder specifications
    
    Defines core interfaces that all encoders must implement:
    - Embedding matrix access
    - Text to token ID conversion
    - Device management
    """
    
    def __init__(self, name: str, device: Optional[str] = None):
        """
        Args:
            name: Unique identifier for the encoder
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
        """
        self.name = name
        self.device = device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._cache_token_ids: Dict[str, int] = {}
        
    @abstractmethod
    def embed_matrix(self) -> torch.Tensor:
        """
        Get the encoder's embedding matrix
        
        Returns:
            torch.Tensor: Embedding matrix of shape [|V_e|, d_e]
                         where |V_e| is vocab size and d_e is embedding dimension
        """
        pass
    
    @abstractmethod
    def token_id(self, surface: str) -> Optional[int]:
        """
        Convert a string to a single token ID
        
        Args:
            surface: Input string
            
        Returns:
            int: Token ID, or None if the string cannot be encoded as a single token
        """
        pass
    
    @abstractmethod
    def token_ids_seq(self, surface: str) -> List[int]:
        """
        Convert a string to a sequence of token IDs
        
        Args:
            surface: Input string
            
        Returns:
            List[int]: List of token IDs
        """
        pass
    
    @abstractmethod
    def get_model(self) -> Any:
        """
        Get the underlying model object
        
        Returns:
            Model object (typically a transformers model)
        """
        pass
    
    @abstractmethod
    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer object
        
        Returns:
            Tokenizer object
        """
        pass
    
    def clear_cache(self):
        """Clear the token ID cache"""
        self._cache_token_ids.clear()
    
    def vocab_size(self) -> int:
        """
        Get vocabulary size
        
        Returns:
            int: Number of tokens in vocabulary
        """
        return self.embed_matrix().size(0)
    
    def embedding_dim(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            int: Dimension of embedding vectors
        """
        return self.embed_matrix().size(1)


# ============================================================
# 2. CONCRETE IMPLEMENTATIONS
# ============================================================

@dataclass
class TransformerEncoderSpec(EncoderSpecBase):
    """
    Encoder implementation based on Hugging Face Transformers
    
    Supports all transformer architectures (BERT, GPT, RoBERTa, etc.)
    """
    
    tokenizer: PreTrainedTokenizer = field(default=None)
    model: PreTrainedModel = field(default=None)
    pooling: str = field(default="mean")  # mean, cls, max, last
    normalize: Optional[str] = field(default=None)  # l2, None
    
    def __init__(self, 
                 name: str,
                 model_name_or_path: Optional[str] = None,
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 model: Optional[PreTrainedModel] = None,
                 device: Optional[str] = None,
                 use_causal_lm: bool = False,
                 pooling: str = "mean",
                 normalize: Optional[str] = None,
                 dtype: str = "float16",
                 trust_remote_code: bool = False):
        """
        Args:
            name: Unique encoder identifier
            model_name_or_path: Hugging Face model name or path
            tokenizer: Pre-loaded tokenizer (optional)
            model: Pre-loaded model (optional)
            device: Device to run on
            use_causal_lm: Whether to use CausalLM model (True) or base model (False)
            pooling: Pooling method for embeddings
            normalize: Normalization method (l2 or None)
            dtype: Model dtype (float16, float32, bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        super().__init__(name=name, device=device)
        
        self.pooling = pooling
        self.normalize = normalize
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Auto-load if model_name_or_path is provided
        if model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                use_fast=False  

            )
            
            if use_causal_lm:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code
                )
            else:
                self.model = AutoModel.from_pretrained(
                    model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code
                )
        else:
            # Otherwise use provided tokenizer and model
            assert tokenizer is not None and model is not None, \
                "Must provide either model_name_or_path or both tokenizer and model"
            self.tokenizer = tokenizer
            self.model = model
        
        # Move to specified device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Freeze parameters to prevent accidental updates
        for param in self.model.parameters():
            param.requires_grad = False
    
    def embed_matrix(self) -> torch.Tensor:
        """Return the model's input embedding matrix"""
        return self.model.get_input_embeddings().weight
    
    def token_id(self, surface: str) -> Optional[int]:
        """
        Convert string to single token ID (with caching)
        
        Returns:
            int: Token ID, or None if not a single token
        """
        # Check cache
        if surface in self._cache_token_ids:
            return self._cache_token_ids[surface]
        
        # Encode and check if it's a single token
        ids = self.tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            self._cache_token_ids[surface] = ids[0]
            return ids[0]
        return None
    
    def token_ids_seq(self, surface: str) -> List[int]:
        """Convert string to token ID sequence"""
        return self.tokenizer.encode(surface, add_special_tokens=False)
    
    def get_model(self) -> PreTrainedModel:
        """Return the underlying transformer model"""
        return self.model
    
    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the tokenizer"""
        return self.tokenizer
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token ID sequence to text
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            str: Decoded text
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


@dataclass
class CustomEncoderSpec(EncoderSpecBase):
    """
    Custom encoder specification
    
    Allows users to provide custom embedding matrices and encoding functions
    Suitable for non-transformer architectures or special requirements
    """
    
    embedding_matrix: torch.Tensor = field(default=None)
    encode_fn: callable = field(default=None)
    decode_fn: callable = field(default=None)
    custom_model: Any = field(default=None)
    
    def __init__(self,
                 name: str,
                 embedding_matrix: torch.Tensor,
                 encode_fn: callable,
                 decode_fn: Optional[callable] = None,
                 custom_model: Any = None,
                 device: Optional[str] = None):
        """
        Args:
            name: Unique encoder identifier
            embedding_matrix: Embedding matrix [vocab_size, embed_dim]
            encode_fn: Encoding function with signature fn(text: str) -> List[int]
            decode_fn: Decoding function (optional) with signature fn(ids: List[int]) -> str
            custom_model: Custom model object (optional)
            device: Device to run on
        """
        super().__init__(name=name, device=device)
        self.embedding_matrix = embedding_matrix.to(self.device)
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.custom_model = custom_model
        
        if custom_model is not None:
            try:
                custom_model.to(self.device)
                custom_model.eval()
            except:
                pass
    
    def embed_matrix(self) -> torch.Tensor:
        """Return the custom embedding matrix"""
        return self.embedding_matrix
    
    def token_id(self, surface: str) -> Optional[int]:
        """Convert string to single token ID using custom encoding function"""
        if surface in self._cache_token_ids:
            return self._cache_token_ids[surface]
        
        ids = self.encode_fn(surface)
        if len(ids) == 1:
            self._cache_token_ids[surface] = ids[0]
            return ids[0]
        return None
    
    def token_ids_seq(self, surface: str) -> List[int]:
        """Convert to token ID sequence using custom encoding function"""
        return self.encode_fn(surface)
    
    def get_model(self) -> Any:
        """Return the custom model (if any)"""
        return self.custom_model
    
    def get_tokenizer(self) -> Any:
        """Custom encoders may not have a tokenizer"""
        return None
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode using custom decoding function
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            str: Decoded text, or string representation of IDs if no decoder available
        """
        if self.decode_fn:
            return self.decode_fn(token_ids)
        return str(token_ids)


# ============================================================
# 3. ENCODER FACTORY
# ============================================================






class EncoderFactory:
    """
    Encoder factory class
    
    Provides convenient methods to create various types of encoders
    """
    
    @staticmethod
    def create_transformer_encoder(
        name: str,
        model_name_or_path: str,
        device: Optional[str] = None,
        use_causal_lm: bool = True,
        **kwargs
    ) -> TransformerEncoderSpec:
        """
        Create a Transformer encoder
        
        Args:
            name: Encoder name
            model_name_or_path: HuggingFace model path
            device: Device to run on
            use_causal_lm: Whether to use CausalLM
            **kwargs: Additional arguments (pooling, normalize, dtype, etc.)
            
        Returns:
            TransformerEncoderSpec instance
        """
        return TransformerEncoderSpec(
            name=name,
            model_name_or_path=model_name_or_path,
            device=device,
            use_causal_lm=use_causal_lm,
            **kwargs
        )
    
    @staticmethod
    def create_from_yaml(
        encoder_name: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        override_model_path: Optional[str] = None
    ) -> TransformerEncoderSpec:
        """
        Create encoder from YAML configuration
        
        Args:
            encoder_name: Name of encoder in YAML config
            config_path: Path to encoders.yaml (optional)
            device: Device to run on (overrides config)
            override_model_path: Override model path from config
            
        Returns:
            TransformerEncoderSpec instance
        """
        # Load configuration
        config_manager = EncoderConfig(config_path)
        encoder_config = config_manager.get_encoder_by_name(encoder_name)
        
        if encoder_config is None:
            raise ValueError(f"Encoder '{encoder_name}' not found in configuration")
        
        # Check if it's an API-only encoder
        if encoder_config.get('provider') in ['openai_api', 'api_only']:
            raise ValueError(
                f"Encoder '{encoder_name}' is API-only and cannot be instantiated locally. "
                f"Use API-based implementation instead."
            )
        
        # Get defaults
        defaults = config_manager.get_defaults()
        
        # Merge configuration
        model_path = override_model_path or encoder_config.get('model')
        dtype = encoder_config.get('dtype', defaults.get('dtype', 'bfloat16'))
        print("dtype of encoder",dtype)
        pooling = encoder_config.get('pooling', 'mean')
        normalize = encoder_config.get('normalize')
        trust_remote_code = encoder_config.get('trust_remote_code', defaults.get('trust_remote_code', False))
        
        # Determine if it's a causal LM (check model type from config notes or provider)
        provider = encoder_config.get('provider', 'hf_local')
        # For sentence transformers and embedding models, use base model
        use_causal_lm = encoder_config.get('use_causal_lm', False)  # 默认False
        return TransformerEncoderSpec(
            name=encoder_name,
            model_name_or_path=model_path,
            device=device,
            use_causal_lm=use_causal_lm,
            pooling=pooling,
            normalize=normalize,
            dtype=dtype,
            trust_remote_code=trust_remote_code
        )
    
    @staticmethod
    def create_all_from_yaml(
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        role: Optional[str] = None
    ) -> List[TransformerEncoderSpec]:
        """
        Create all encoders from YAML configuration
        
        Args:
            config_path: Path to encoders.yaml (optional)
            device: Device to run on
            role: Filter by role (train_eot, eval_only, router_backbone)
            
        Returns:
            List of TransformerEncoderSpec instances
        """
        config_manager = EncoderConfig(config_path)
        encoder_configs = config_manager.get_encoder_configs(role=role)
        
        encoders = []
        for enc_config in encoder_configs:
            encoder_name = enc_config.get('name')
            try:
                encoder = EncoderFactory.create_from_yaml(
                    encoder_name=encoder_name,
                    config_path=config_path,
                    device=device
                )
                encoders.append(encoder)
                print(f"Loaded encoder: {encoder_name}")
            except Exception as e:
                print(f"Failed to load encoder '{encoder_name}': {e}")
        
        return encoders
    
    @staticmethod
    def create_custom_encoder(
        name: str,
        embedding_matrix: torch.Tensor,
        encode_fn: callable,
        decode_fn: Optional[callable] = None,
        device: Optional[str] = None
    ) -> CustomEncoderSpec:
        """
        Create a custom encoder
        
        Args:
            name: Encoder name
            embedding_matrix: Embedding matrix
            encode_fn: Encoding function
            decode_fn: Decoding function (optional)
            device: Device to run on
            
        Returns:
            CustomEncoderSpec instance
        """
        return CustomEncoderSpec(
            name=name,
            embedding_matrix=embedding_matrix,
            encode_fn=encode_fn,
            decode_fn=decode_fn,
            device=device
        )
    
    @staticmethod
    def create_from_existing(
        name: str,
        tokenizer: Any,
        model: Any,
        device: Optional[str] = None
    ) -> TransformerEncoderSpec:
        """
        Create encoder from existing tokenizer and model
        
        Args:
            name: Encoder name
            tokenizer: Pre-loaded tokenizer
            model: Pre-loaded model
            device: Device to run on
            
        Returns:
            TransformerEncoderSpec instance
        """
        return TransformerEncoderSpec(
            name=name,
            tokenizer=tokenizer,
            model=model,
            device=device
        )


# ============================================================
# 4. UTILITY FUNCTIONS FOR ENCODER SPECS
# ============================================================

def validate_encoder_spec(encoder: EncoderSpecBase) -> bool:
    """
    Validate that an encoder specification is properly implemented
    
    Args:
        encoder: Encoder to validate
        
    Returns:
        bool: Whether the encoder is valid
    """
    try:
        # Check embedding matrix
        emb = encoder.embed_matrix()
        assert isinstance(emb, torch.Tensor), "embed_matrix must return torch.Tensor"
        assert emb.dim() == 2, "embed_matrix must be 2D"
        
        # Check basic encoding functionality
        test_str = "test"
        ids = encoder.token_ids_seq(test_str)
        assert isinstance(ids, list), "token_ids_seq must return list"
        
        return True
    except Exception as e:
        print(f"Encoder validation failed: {e}")
        return False


def compare_encoders(enc1: EncoderSpecBase, enc2: EncoderSpecBase, test_strings: List[str]) -> Dict:
    """
    Compare two encoders on the same inputs
    
    Args:
        enc1: First encoder
        enc2: Second encoder
        test_strings: List of test strings
        
    Returns:
        Dict: Dictionary containing comparison results
    """
    results = {
        "encoder1": enc1.name,
        "encoder2": enc2.name,
        "vocab_size_1": enc1.vocab_size(),
        "vocab_size_2": enc2.vocab_size(),
        "embed_dim_1": enc1.embedding_dim(),
        "embed_dim_2": enc2.embedding_dim(),
        "common_single_tokens": 0,
        "total_tests": len(test_strings),
        "differences": []
    }
    
    for s in test_strings:
        id1 = enc1.token_id(s)
        id2 = enc2.token_id(s)
        
        if id1 is not None and id2 is not None:
            results["common_single_tokens"] += 1
        elif id1 != id2:
            results["differences"].append({
                "string": s,
                "enc1_id": id1,
                "enc2_id": id2,
                "enc1_seq": enc1.token_ids_seq(s),
                "enc2_seq": enc2.token_ids_seq(s)
            })
    
    return results




if __name__ == "__main__":
    print("=" * 60)
    print("Testing EncoderSpec with YAML Configuration")
    print("=" * 60)
    
    # Test 1: Load single encoder from YAML
    print("\n1. Loading encoder from YAML...")
    try:
        encoder1 = EncoderFactory.create_from_yaml(
            encoder_name="minilm_sent",
            device="cpu"
        )
        print(f"   Name: {encoder1.name}")
        print(f"   Vocab Size: {encoder1.vocab_size()}")
        print(f"   Embedding Dim: {encoder1.embedding_dim()}")
        print(f"   Device: {encoder1.device}")
        print("load success!")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Load all encoders from YAML
    print("\n2. Loading all encoders from YAML...")
    try:
        encoders = EncoderFactory.create_all_from_yaml(device="cpu")
        print(f"   Loaded {len(encoders)} encoders")
        for enc in encoders:
            print(f"   - {enc.name}: {enc.embedding_dim()}d")
    except Exception as e:
        print(f"   Error: {e}")
    

    
    # Test encoding
    for enc in encoders:

        test_word = "hello"
        token_id = enc.token_id(test_word)
        token_seq = enc.token_ids_seq(test_word)
        print(f"   '{test_word}' -> single token: {token_id}")
        print(f"   '{test_word}' -> sequence: {token_seq}")
        
    print("\n" + "=" * 60)
    print("EncoderSpec testing complete!")
    print("=" * 60)