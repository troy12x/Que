"""
Configuration classes for InfinityFormer
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, List


from transformers import PretrainedConfig

@dataclass
class InfinityFormerConfig(PretrainedConfig):
    """
    Configuration class for InfinityFormer model.
    """
    model_type = "infinityformer"

    # Model dimensions
    vocab_size: int = 50257  # Default GPT-2 vocab size
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    use_return_dict: bool = True
    use_cache: bool = True  # Default for generation
    output_attentions: bool = False
    output_hidden_states: bool = False
    
    # InfinityFormer specific parameters
    use_rotary_embeddings: bool = True
    rotary_embedding_base: int = 10000
    use_multi_scale_memory: bool = True
    num_memory_scales: int = 3
    memory_compression_ratio: float = 0.5
    memory_compression_frequency: int = 100
    kernel_type: str = 'elu'  # Options: 'elu', 'relu', 'learnable'
    kernel_epsilon: float = 0.1
    
    # Gating mechanism
    use_gating: bool = True
    gate_init_bias: float = -2.0  # Initialize gates to be closed
    
    # Training parameters
    use_gradient_checkpointing: bool = False
    gradient_checkpointing_use_reentrant: bool = True # Pytorch default for checkpoint
    gradient_checkpointing_frequency: int = 1
    
    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be a multiple of `num_attention_heads` "
                f"({self.num_attention_heads})"
            )
            
        if self.kernel_type not in ['elu', 'relu', 'learnable']:
            raise ValueError(f"`kernel_type` must be one of 'elu', 'relu', or 'learnable', got {self.kernel_type}")
