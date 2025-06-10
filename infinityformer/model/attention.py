import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from ..config import InfinityFormerConfig

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for relative position information.
    """
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def _get_rotary_embeddings(self, x: torch.Tensor, seq_dim: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.size(seq_dim)
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self.rotate_half(x) * sin)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, num_heads, head_dim]
        cos, sin = self._get_rotary_embeddings(x, seq_dim=1)
        return self.apply_rotary_pos_emb(x, cos.unsqueeze(0).unsqueeze(2), sin.unsqueeze(0).unsqueeze(2))


class KernelFunction(nn.Module):
    """
    Kernel function for linear attention.
    """
    def __init__(self, config: InfinityFormerConfig):
        super().__init__()
        self.kernel_type = config.kernel_type
        self.epsilon = config.kernel_epsilon
        
        if self.kernel_type == 'learnable':
            self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_type == 'elu':
            return F.elu(x) + 1.0 + self.epsilon
        elif self.kernel_type == 'relu':
            return F.relu(x) + self.epsilon
        elif self.kernel_type == 'learnable':
            return F.elu(x * self.temperature) + 1.0 + self.epsilon
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")


class MultiScaleMemory(nn.Module):
    """
    Multi-scale memory module with adaptive decay and gating.
    """
    def __init__(self, config: InfinityFormerConfig, layer_idx: int = 0):
        super().__init__()
        self.num_scales = config.num_memory_scales
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        
        # Memory decay parameters (one per scale)
        self.decay_factors = nn.Parameter(torch.ones(self.num_scales) * 0.9) # Example: Initialize with 0.9
        
        # Gating parameters
        self.gate_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales) # Initialize to equal weighting
        self.gate_bias = nn.Parameter(torch.zeros(1)) # Not currently used in forward, but kept for potential future use
        
        # Memory compression
        self.compression_ratio = config.memory_compression_ratio
        self.compression_frequency = config.memory_compression_frequency
        self.step_counter = 0 # To track steps for periodic compression
        
        # Initialize memory states
        # persistent=False means this buffer will not be saved in the model's state_dict
        self.register_buffer("memory", None, persistent=False)

        # Debug prints for initialization

        
    def init_memory(self, batch_size: int, device: torch.device) -> None:
        """Initialize memory states."""
        # Memory shape: [batch_size, num_scales, num_heads, head_dim_key, head_dim_value]
        # Assuming head_dim_key == head_dim_value == self.head_dim for simplicity here
        self.memory = torch.zeros(
            batch_size, self.num_scales, self.num_heads, self.head_dim, self.head_dim,
            device=device
        )

        
    def compress_memory(self) -> None:
        """Compress the memory using SVD-based compression."""
        if self.memory is None or self.compression_ratio >= 1.0:
            return # No memory to compress or no compression needed
            
        batch_size = self.memory.size(0)
        compressed_memories = [] # List to store compressed (U, S, Vh) tuples
        
        for b in range(batch_size):
            scale_memories = []
            for s in range(self.num_scales):
                head_memories = []
                for h in range(self.num_heads):
                    mem = self.memory[b, s, h] # [head_dim, head_dim]
                    
                    # Apply SVD
                    U, S_diag, Vh = torch.linalg.svd(mem, full_matrices=False)
                    
                    # Truncate based on compression ratio
                    k = max(1, int(self.head_dim * self.compression_ratio))
                    U_trunc = U[:, :k]
                    S_trunc = S_diag[:k] # S is a vector of singular values
                    Vh_trunc = Vh[:k, :]
                    
                    compressed = (U_trunc, S_trunc, Vh_trunc)
                    head_memories.append(compressed)
                scale_memories.append(head_memories)
            compressed_memories.append(scale_memories)
            
        self.compressed_memory = compressed_memories # Store the list of tuples
        self.memory = None # Free the original memory tensor
        
    def decompress_memory(self, device: torch.device) -> None:
        """Decompress the memory when needed."""
        if not hasattr(self, 'compressed_memory') or self.compressed_memory is None:
            return # No compressed memory to decompress
            
        # Assuming compressed_memory is a list of lists of lists of (U,S,Vh) tuples
        batch_size = len(self.compressed_memory)
        # Re-initialize self.memory tensor to store decompressed values
        memory_reconstructed = torch.zeros(
            batch_size, self.num_scales, self.num_heads, self.head_dim, self.head_dim,
            device=device
        )
        
        for b in range(batch_size):
            for s in range(self.num_scales):
                for h in range(self.num_heads):
                    U_trunc, S_trunc, Vh_trunc = self.compressed_memory[b][s][h]
                    # Reconstruct the matrix: M = U_trunc @ diag(S_trunc) @ Vh_trunc
                    mem_reconstructed_single = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
                    memory_reconstructed[b, s, h] = mem_reconstructed_single.to(device)
        
        self.memory = memory_reconstructed
        delattr(self, 'compressed_memory') # Remove the compressed version
    
    def update_memory(self, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Update the memory with new key-value pairs.
        k, v: [batch_size, seq_len, num_heads, head_dim]
        mask: Optional, [batch_size, seq_len]
        """
        batch_size, seq_len, num_heads, head_dim_k = k.shape
        _, _, _, head_dim_v = v.shape

        if self.memory is None and not hasattr(self, 'compressed_memory'):
            self.init_memory(batch_size, k.device)
        elif hasattr(self, 'compressed_memory') and self.memory is None:
            self.decompress_memory(k.device)
        
        # Apply mask if provided (mask is [B, S])
        if mask is not None:
            # Expand mask to match k, v dimensions: [B, S, 1, 1]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1) 
            k = k * mask_expanded
            v = v * mask_expanded
        
        # Compute outer product for new information: k_i^T v_i
        # kv_new: [batch_size, seq_len, num_heads, head_dim_k, head_dim_v]
        kv_new = torch.einsum('bsnk,bsnv->bsnkv', k, v) 
        
        # Sum new information over the sequence length for this update step.
        # This represents the "chunk" of new information from the current input sequence.
        # new_info_chunk: [batch_size, num_heads, head_dim_k, head_dim_v]
        new_info_chunk = kv_new.sum(dim=1) 
        
        # Reshape for broadcasting with memory scales: [batch_size, 1, num_heads, head_dim_k, head_dim_v]
        new_info_chunk_expanded = new_info_chunk.unsqueeze(1)

        # Get decay factors, apply sigmoid to keep them in (0,1)
        # decay_factors_sig: [num_scales] -> [1, num_scales, 1, 1, 1] for broadcasting
        decay_factors_sig = torch.sigmoid(self.decay_factors).view(1, self.num_scales, 1, 1, 1)
        
        # Update memory: M_t = decay * M_{t-1} + (1 - decay) * new_info_chunk
        # self.memory: [batch_size, num_scales, num_heads, head_dim_k, head_dim_v]
        self.memory = self.memory.detach() * decay_factors_sig + (1.0 - decay_factors_sig) * new_info_chunk_expanded
        


        self.step_counter += 1
        if self.training and self.compression_frequency > 0 and self.step_counter % self.compression_frequency == 0:
            self.compress_memory()
    
    def forward(self, q: torch.Tensor, output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Attend to the memory using the query.
        
        Args:
            q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            output_attentions: Whether to return attention weights (gate weights for memory scales).
            
        Returns:
            Tuple of (Attention output, Optional Attention weights)
            Attention output shape: [batch_size, seq_len, num_heads, head_dim_value]
            Attention weights shape: [num_scales] if output_attentions is True, else None
        """
        if self.memory is None and not hasattr(self, 'compressed_memory'):
            # If no memory, output is zeros. If output_attentions is True, return None for weights.
            # q has shape [B, S, N, Dk], output should be [B, S, N, Dv] (assuming Dk=Dv)
            return torch.zeros_like(q), None 
        
        if hasattr(self, 'compressed_memory') and self.memory is None:
            self.decompress_memory(q.device) # Ensure memory is available

        # FIX: Check and correct memory state batch size if it differs from the query's batch size.
        # This is critical when switching from a larger training batch to a smaller evaluation batch.
        if self.memory.shape[0] != q.shape[0]:
            self.memory = self.memory[:q.shape[0], ...]
        
        # Einsum for attention computation with memory
        # q (query):           [batch_size (b), seq_len (s), num_heads (n), head_dim_key (k)]
        # self.memory (memory): [batch_size (b), num_memory_scales (m), num_heads (n), head_dim_key (k), head_dim_value (v)]
        # output_scaled:       [batch_size (b), num_memory_scales (m), seq_len (s), num_heads (n), head_dim_value (v)]
        # Note: The squeeze(2) was removed as self.memory is now directly [B, M, N, Dk, Dv]
        output_scaled = torch.einsum('bsnk,bmnkv->bmsnv', q, self.memory)
        
        # Apply gating to combine outputs from different scales
        gate_weights_softmaxed = F.softmax(self.gate_weights, dim=0)  # [num_scales]
        # Reshape for broadcasting: [1, num_scales, 1, 1, 1]
        gate_weights_view = gate_weights_softmaxed.view(1, self.num_scales, 1, 1, 1)  

        # Weighted sum of outputs from different scales
        # output_scaled: [b, m, s, n, v] * gate_weights_view: [1, m, 1, 1, 1] -> sum over m
        output = (output_scaled * gate_weights_view).sum(dim=1)  # [batch_size, seq_len, num_heads, head_dim_value]

        attn_weights_to_return = None
        if output_attentions:
            attn_weights_to_return = gate_weights_softmaxed.detach() # Return the softmaxed gate weights
        
        return output, attn_weights_to_return



class LinearAttention(nn.Module):
    """
    Linear attention mechanism with multi-scale memory.
    """
    def __init__(self, config: InfinityFormerConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.layer_idx = layer_idx
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
        # Kernel function
        self.kernel = KernelFunction(config)
        
        # Multi-scale memory
        self.use_memory = config.use_multi_scale_memory
        if self.use_memory:
            self.memory = MultiScaleMemory(config, layer_idx)
        
        # Gating mechanism
        self.use_gating = config.use_gating
        if self.use_gating:
            self.gate = nn.Linear(self.hidden_size, 1)
            nn.init.constant_(self.gate.bias, config.gate_init_bias)
        
        # Rotary position embeddings
        self.use_rotary = config.use_rotary_embeddings
        if self.use_rotary:
            self.rotary_emb = RotaryPositionEmbedding(self.head_dim, config.rotary_embedding_base)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys, and values
        q = self.q_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        v = self.v_proj(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Reshape to [batch_size, seq_len, num_heads, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings if enabled
        if self.use_rotary:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        
        # Apply kernel function to queries and keys
        q = self.kernel(q) * self.scaling
        k = self.kernel(k)
        
        # Compute output from current context (standard linear attention)
        # k_t: [batch_size, seq_len, num_heads, head_dim]
        # v: [batch_size, seq_len, num_heads, head_dim]
        # attn_weights: [batch_size, num_heads, seq_len, seq_len]
        # output: [batch_size, seq_len, num_heads, head_dim]
        attn_weights = torch.einsum('bqhd,bkhd->bhqk', q, k)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attn_weights = attn_weights + attention_mask
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context_output = torch.einsum('bhqk,bkhd->bqhd', attn_weights, v)
        
        # Update memory with current key-value pairs if memory is enabled
        if self.use_memory:
            self.memory.update_memory(k, v, attention_mask.squeeze(1).squeeze(1) if attention_mask is not None else None)

            # Get output from memory and unpack the tuple
            memory_output, _ = self.memory(q)



            # Apply gating to combine context and memory outputs
            if self.use_gating:
                # Compute gate values
                gate_input = hidden_states
                if attention_mask is not None and attention_mask.dim() == 4:
                    gate_input = gate_input * attention_mask.squeeze(1).transpose(-1, -2).float()
                gate = torch.sigmoid(self.gate(gate_input))  # [batch_size, seq_len, 1]
                gate = gate.unsqueeze(-1)  # [batch_size, seq_len, 1, 1]

                # Combine outputs using the gate
                output = gate * context_output + (1 - gate) * memory_output
            else:
                # Combine by simple addition
                output = context_output + memory_output
        else:
            output = context_output
        
        # Reshape and project back to hidden size
        output = output.reshape(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(output)
        
        return output, attn_weights
