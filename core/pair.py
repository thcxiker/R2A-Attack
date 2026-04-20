import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from reroute.core.router import Router
from reroute.core.encoder import EncoderSpecBase,TransformerEncoderSpec


class RouterEncoderPair:
    """
    Router-Encoder binding helper.

    Ensures Router and Encoder consistency (useful for joint training).

    Examples:
        # Option 1: Router has an internal encoder
        pair = RouterEncoderPair(
            router=p2l_router,
            encoder=None,  # use router's internal encoder automatically
            trainable=True
        )

        # Option 2: Provide an explicit encoder
        pair = RouterEncoderPair(
            router=graph_router,
            encoder=minilm_encoder,
            trainable=True
        )
    """
    
    def __init__(
        self,
        router: Router,
        encoder: Optional[EncoderSpecBase] = None,
        trainable: bool = True,
        name: Optional[str] = None
    ):
        """
        Args:
            router: Router instance
            encoder: Encoder instance (None = use router's internal encoder)
            trainable: whether the pair participates in gradient updates
            name: optional custom name
        """
        self.router = router
        self.trainable = trainable
        self._name = name or f"{router.name}_pair"
        
        # ============================================================
        # Bind or extract an Encoder
        # ============================================================
        if encoder is not None:
            # user provided an encoder
            self.encoder = encoder
            self.encoder_source = "external"
            print(f"  RouterEncoderPair: using external encoder '{encoder.name}'")
        
        elif hasattr(router, 'get_internal_encoder'):
            # Router provides an internal encoder
            self.encoder = router.get_internal_encoder()
            self.encoder_source = "internal"
            print(f"  RouterEncoderPair: extracted internal encoder from router")
        
        elif hasattr(router, 'model'):
            # Fallback: wrap router.model as an encoder
            self.encoder = self._wrap_router_model(router)
            self.encoder_source = "wrapped"
            print(f"  RouterEncoderPair: wrapped router.model as encoder")
        
        else:
            raise ValueError(
                f"Router '{router.name}' has no encoder! "
                "Please provide an encoder or implement get_internal_encoder()."
            )
        
        # ============================================================
        # Validate compatibility
        # ============================================================
        self._validate_compatibility()
    
    @property
    def name(self) -> str:
        return self._name
    
    def _wrap_router_model(self, router: Router):
    # """Wrap a Router's model to behave like an Encoder."""
    
        class RouterModelWrapper:
            """Wrapper: Router's internal model as EncoderSpec"""
            
            def __init__(self, router: Router):
                self.router = router
                self._name = f"{router.name}_encoder"
                
                if not hasattr(router, 'model'):
                    raise AttributeError(f"Router {router.name} has no 'model'")
                
                self._model = router.model
                
                if hasattr(router, 'tokenizer'):
                    self._tokenizer = router.tokenizer
                else:
                    from transformers import AutoTokenizer
                    model_path = getattr(router, 'model_path', None)
                    if model_path:
                        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
                    else:
                        raise AttributeError(
                            f"Router {router.name} has no tokenizer"
                        )
            
            @property
            def name(self) -> str:
                return self._name
            
            def get_model(self):
                return self._model
            
            def get_tokenizer(self):
                return self._tokenizer
            
            def embedding_dim(self) -> int:
                """Return the embedding dimension."""
                if hasattr(self._model, 'config'):
                    if hasattr(self._model.config, 'hidden_size'):
                        return self._model.config.hidden_size

                embed_layer = self._model.get_input_embeddings()
                return embed_layer.embedding_dim
            
            # ============================================================
            # expose embedding matrix for builders/utilities
            # ============================================================
            def embed_matrix(self) -> torch.Tensor:
                """Return the model's embedding matrix [vocab_size, embed_dim]."""
                embed_layer = self._model.get_input_embeddings()
                embed_matrix = embed_layer.weight.data
                return embed_matrix
            
            def encode(self, text: str) -> torch.Tensor:
                """Encode text into a vector representation."""
                if hasattr(self.router, 'encode'):
                    return self.router.encode(text)
                
                inputs = self._tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self._model.device)
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    hidden_states = outputs.last_hidden_state
                    embeddings = hidden_states.mean(dim=1).squeeze(0)
                
                return embeddings
    
        return RouterModelWrapper(router)
    

    
    def _validate_compatibility(self):
        """Validate Router and Encoder compatibility."""

        # Check 1: Encoder implements required methods
        required_methods = ['get_model', 'get_tokenizer', 'embedding_dim']
        for method in required_methods:
            if not hasattr(self.encoder, method):
                raise AttributeError(
                    f"Encoder '{self.encoder.name}' missing method: {method}"
                )

        # Check 2: Router supports embedding-level forward
        if not hasattr(self.router, 'forward_embeds'):
            import warnings
            warnings.warn(
                f"Router '{self.router.name}' does not implement "
                f"forward_embeds(). GCG attacks may not work."
            )
    
    def forward_embeds(
        self,
        embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Embedding-level forward.

        Args:
            embeds: [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
            attention_mask: optional attention mask tensor

        Returns:
            logits: model output logits
        """
        return self.router.forward_embeds(
            embeds=embeds,
            attention_mask=attention_mask
        )
    

    
    def get_model_list(self):
        """Return the router's available model list."""
        return self.router.get_model_list()
    
    def to_float32(self):
        """Convert the encoder's model parameters to float32 in-place."""
        model = self.encoder.get_model()
        first_param = next(model.parameters())
        original_dtype = first_param.dtype

        if original_dtype != torch.float32:
            model.float()
            print(f"  {self.name}: {original_dtype} → float32 ✓")
        else:
            print(f"  {self.name}: float32 ✓")
    
    def __repr__(self):
        status = "trainable" if self.trainable else "frozen"
        return (
            f"RouterEncoderPair(\n"
            f"  router={self.router.name},\n"
            f"  encoder={self.encoder.name} ({self.encoder_source}),\n"
            f"  embedding_dim={self.encoder.embedding_dim()},\n"
            f"  status={status}\n"
            f")"
        )