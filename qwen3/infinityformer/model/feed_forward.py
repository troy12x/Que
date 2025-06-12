import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedFeedForward(nn.Module):
    """
    Gated feed-forward network with GELU activation and gating mechanism.
    """
    def __init__(self, config: 'InfinityFormerConfig'):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Main feed-forward layers
        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size * 2)  # Double for gating
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Dropout
        self.activation_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Apply gated activation (GLU variant)
        hidden_states = self.fc1(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=-1)
        hidden_states = hidden_states * torch.sigmoid(gate)  # GLU gating
        hidden_states = F.gelu(hidden_states)  # GELU activation
        hidden_states = self.activation_dropout(hidden_states)
        
        # Project back to hidden size
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states
