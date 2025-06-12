import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Helper classes and functions adapted for Qwen3
class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

# This is a placeholder for your unique kernel function
class KernelFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Using ReLU as a simple, effective non-linear kernel for demonstration
        self.kernel_fn = nn.ReLU()

    def forward(self, x):
        return self.kernel_fn(x)

# This is a placeholder for your recurrent memory
class RecurrentMemory(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        # Simple memory state initialized with zeros
        self.state = None

    def update_memory(self, k, v):
        # A simple form of memory: just the last key/value pair
        # A real implementation would be more complex (e.g., EMA, reservoirs)
        self.state = (k, v)

    def forward(self, q):
        if self.state is None:
            # Return zeros if memory is empty
            return torch.zeros_like(q)
        
        # Simple attention over the single memory state
        mem_k, mem_v = self.state
        # In a real scenario, you'd want to ensure shapes are compatible
        # This is a simplified example
        return F.scaled_dot_product_attention(q.transpose(1,2), mem_k.transpose(1,2), mem_v.transpose(1,2)).transpose(1,2)

class InfinityFormerAttention(nn.Module):
    """
    A hybrid attention module that combines Qwen3's GQA structure with
    InfinityFormer's concepts of linear attention (via a kernel)
    and recurrent memory.
    """
    def __init__(self, config: "Qwen3Config", layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        # Qwen3-style GQA projections (names must match for surgery)
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=getattr(config, 'attention_bias', False))
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=getattr(config, 'attention_bias', False))

        # Qwen3-style RMSNorm for Q/K (names must match for surgery)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # --- Your Unique Features ---
        self.kernel = KernelFunction(config)
        self.memory = RecurrentMemory(config, layer_idx)
        self.gate = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Standard Qwen3 projections and normalization
        query_states = self.q_norm(self.q_proj(hidden_states).view(bsz, q_len, self.num_attention_heads, self.head_dim))
        key_states = self.k_norm(self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)
        
        past_key_value = (key_states, value_states) if self.config.use_cache else None

        # GQA: repeat k/v heads
        key_states_gqa = repeat_kv(key_states, self.num_key_value_groups)
        value_states_gqa = repeat_kv(value_states, self.num_key_value_groups)

        # --- InfinityFormer's Core Logic --- #
        # 1. Apply kernel to queries and keys for linear attention
        q_kernel = self.kernel(query_states)
        k_kernel = self.kernel(key_states_gqa)

        # 2. Get output from recurrent memory
        memory_output = self.memory(q_kernel)

        # 3. Compute standard attention output
        # Transpose for scaled_dot_product_attention: [b, s, h, d] -> [b, h, s, d]
        q_for_sdpa = q_kernel.transpose(1, 2)
        k_for_sdpa = k_kernel.transpose(1, 2)
        v_for_sdpa = value_states_gqa.transpose(1, 2)
        
        context_output_sdpa = F.scaled_dot_product_attention(
            q_for_sdpa, k_for_sdpa, v_for_sdpa, attn_mask=attention_mask
        )
        context_output = context_output_sdpa.transpose(1, 2)

        # 4. Combine context and memory with a gate
        gate_values = torch.sigmoid(self.gate(hidden_states)).view(bsz, q_len, self.num_attention_heads, self.head_dim)
        combined_output = gate_values * context_output + (1 - gate_values) * memory_output

        # 5. Update memory with the new key-value state (using the non-repeated keys)
        self.memory.update_memory(self.kernel(key_states), value_states)
        # --- End of Core Logic ---

        # Reshape and project back to hidden size
        attn_output = combined_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # For compatibility, we don't return attention weights
        attn_weights = None 

        return attn_output, attn_weights, past_key_value

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

    def rotate_half(self, x):
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)
