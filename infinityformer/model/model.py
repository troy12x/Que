import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Dict, Any
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.generation import GenerationMixin

from ..config import InfinityFormerConfig

from .attention import LinearAttention, MultiScaleMemory # Added MultiScaleMemory
from .feed_forward import GatedFeedForward
import torch.nn.functional as F # Added for F.dropout

class InfinityFormerLayer(nn.Module):
    """
    A single layer of the InfinityFormer model.
    """
    def __init__(self, config: InfinityFormerConfig, layer_idx: int):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = LinearAttention(config, layer_idx)
        
        # Memory Attention components
        self.mem_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # Using a separate norm for mem QKV inputs
        self.mem_q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mem_k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mem_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.mem_attn = MultiScaleMemory(config, layer_idx=layer_idx) # Initialize MultiScaleMemory

        self.dropout_prob = config.hidden_dropout_prob # Renamed from self.dropout to self.dropout_prob for clarity
        
        # Feed-forward network
        self.ffn = GatedFeedForward(config)
        # self.final_layer_norm is defined but not explicitly used if GFF handles its own norm/residual structure.
        # Keeping it defined as per original code for now.
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        
        # Gradient checkpointing
        self.gradient_checkpointing = config.use_gradient_checkpointing
    
    def _split_heads(self, tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
        """Splits hidden_size dim into attn_num_heads and head_dim."""
        new_shape = tensor.size()[:-1] + (num_heads, head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combines attn_num_heads and head_dim into hidden_size."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.embed_dim,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states: torch.Tensor, # [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None, # [batch_size, seq_len] or [batch_size, 1, from_seq_len, to_seq_len]
        past_key_value: Optional[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]] = None, 
        # past_key_value = (self_attn_kv_cache, mem_attn_state_cache)
        # self_attn_kv_cache = (prev_k, prev_v) from LinearAttention
        # mem_attn_state_cache = self.mem_attn.memory tensor
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]: 
        
        residual = hidden_states
        batch_size, seq_len, _ = hidden_states.shape

        self_attn_past_kv = None
        mem_attn_past_state = None
        if past_key_value is not None:
            self_attn_past_kv = past_key_value[0]
            if len(past_key_value) > 1:
                 mem_attn_past_state = past_key_value[1]
        
        # --- Self-Attention Block --- 
        hidden_states_ln1 = self.self_attn_layer_norm(hidden_states)
        
        self_attn_outputs = self.self_attn(
            hidden_states=hidden_states_ln1,
            attention_mask=attention_mask,
            past_key_value=self_attn_past_kv,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        self_attn_output = self_attn_outputs[0] # [batch, seq_len, embed_dim]
        new_self_attn_kv_cache = self_attn_outputs[1] if use_cache else None
        
        self_attn_weights = None
        if output_attentions:
            if use_cache and len(self_attn_outputs) > 2:
                self_attn_weights = self_attn_outputs[2]
            elif not use_cache and len(self_attn_outputs) > 1:
                self_attn_weights = self_attn_outputs[1]
            # else: self_attn_weights remains None, which is already initialized

        self_attn_output = F.dropout(self_attn_output, p=self.dropout_prob, training=self.training)

        # --- Memory Attention Block --- 
        # Project hidden_states_ln1 (or hidden_states) to Q, K, V for memory attention
        # Using hidden_states_ln1 as it's already computed and normed.
        q_for_mem_flat = self.mem_q_proj(hidden_states_ln1) # [B, S, H]
        k_for_mem_flat = self.mem_k_proj(hidden_states_ln1) # [B, S, H]
        v_for_mem_flat = self.mem_v_proj(hidden_states_ln1) # [B, S, H]

        # Split heads for memory QKV
        q_for_mem = self._split_heads(q_for_mem_flat, self.num_heads, self.head_dim) # [B, N, S, Dk]
        k_for_mem = self._split_heads(k_for_mem_flat, self.num_heads, self.head_dim) # [B, N, S, Dk]
        v_for_mem = self._split_heads(v_for_mem_flat, self.num_heads, self.head_dim) # [B, N, S, Dv]
        
        # Transpose QKV to be [B, S, N, Dk/Dv] for MultiScaleMemory
        q_for_mem = q_for_mem.permute(0, 2, 1, 3) # [B, S, N, Dk]
        k_for_mem = k_for_mem.permute(0, 2, 1, 3) # [B, S, N, Dk]
        v_for_mem = v_for_mem.permute(0, 2, 1, 3) # [B, S, N, Dv]

        if use_cache and mem_attn_past_state is not None:
            if self.mem_attn.memory is None or self.mem_attn.memory.shape != mem_attn_past_state.shape:
                self.mem_attn.memory = mem_attn_past_state.clone() 
            else:
                self.mem_attn.memory.copy_(mem_attn_past_state)
        
        # TODO: Determine appropriate mask for memory update if attention_mask is causal.
        # For now, passing None. If attention_mask is a padding mask [B,S], it could be used.
        self.mem_attn.update_memory(k_for_mem, v_for_mem, mask=None) 
        
        mem_output_raw, mem_gate_weights = self.mem_attn(q_for_mem, output_attentions=output_attentions)
        # mem_output_raw is [B, S, N, Dv]
        mem_output = mem_output_raw.contiguous().view(batch_size, seq_len, self.embed_dim) # Reshape to [B, S, H]
        mem_output = F.dropout(mem_output, p=self.dropout_prob, training=self.training)

        # --- Combine Attention Outputs --- 
        combined_attn_output = self_attn_output + mem_output # Simple addition
        
        # First residual connection
        hidden_states = residual + combined_attn_output
        
        # --- Feed-Forward Network Block --- 
        # Assuming GatedFeedForward (self.ffn) handles its own LayerNorm and residual connection (Pre-LN style for FFN block)
        hidden_states = self.ffn(hidden_states)
        
        # --- Output Construction --- 
        outputs = (hidden_states,)
        
        present_key_value = None
        if use_cache:
            new_mem_attn_state = self.mem_attn.memory.detach().clone() if self.mem_attn.memory is not None else None
            present_key_value = (new_self_attn_kv_cache, new_mem_attn_state)
            outputs += (present_key_value,)
        
        if output_attentions:
            all_attentions = (self_attn_weights, mem_gate_weights)
            outputs += (all_attentions,)
            
        return outputs


class InfinityFormerEmbeddings(nn.Module):
    """
    Token and position embeddings for InfinityFormer.
    """
    def __init__(self, config: InfinityFormerConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Position IDs for position embeddings
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights with normal distribution."""
        nn.init.normal_(self.word_embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """
    
        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
            inputs_embeds: [batch_size, seq_len, hidden_size]
            
        Returns:
            [batch_size, seq_len, hidden_size]
        """
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        
        seq_length = input_shape[1]
        max_pos_embed = self.position_embeddings.num_embeddings

        # Truncate inputs if they are longer than max_position_embeddings
        if seq_length > max_pos_embed:
            if input_ids is not None:
                input_ids = input_ids[:, :max_pos_embed]
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, :max_pos_embed, :]
            seq_length = max_pos_embed

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        else:
            # Even if position_ids are provided, they cannot be longer than the model's max length
            if position_ids.shape[1] > seq_length:
                position_ids = position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        position_embeddings = self.position_embeddings(position_ids)
        
        # Sum token and position embeddings
        embeddings = inputs_embeds + position_embeddings
        
        # Apply layer norm and dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class InfinityFormerModel(nn.Module):
    """
    The full InfinityFormer model without a specific head.
    """
    def __init__(self, config: InfinityFormerConfig):
        super().__init__()
        self.config = config
        self.dtype = getattr(config, 'torch_dtype', torch.float32)
        
        # Embeddings
        self.embeddings = InfinityFormerEmbeddings(config)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            InfinityFormerLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embeddings.word_embeddings = value
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        if input_ids is not None:
            seq_length = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]
        else:
            # When using cache, input_ids and inputs_embeds can be None, and seq_length is 1
            seq_length = 0

        max_pos_embed = self.config.max_position_embeddings
        if seq_length > max_pos_embed:
            if attention_mask is not None:
                if attention_mask.dim() == 4:
                    attention_mask = attention_mask[:, :, :, :max_pos_embed]
                elif attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, :max_pos_embed]

        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            position_ids: [batch_size, seq_len]
            inputs_embeds: [batch_size, seq_len, hidden_size]
            return_dict: Whether to return a dict or a tuple
            
        Returns:
            Last hidden states and optional attention weights
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if hasattr(self.config, 'use_cache') else False)
        output_attentions = self.config.output_attentions # Get from config
        output_hidden_states = self.config.output_hidden_states # Get from config

        if past_key_values is None and use_cache:
            # Each layer's past_key_value is now a tuple (self_attn_kv_cache, mem_attn_state_cache)
            past_key_values = [(None, None)] * self.config.num_hidden_layers 
            
        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            # input_ids = input_ids.view(-1, input_shape[-1]) # Not needed if already [batch, seq_len]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Prepare attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=self.dtype)  # Ensure correct dtype
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = hidden_states.to(self.dtype)

        # Initialize accumulators for outputs
        all_hidden_states_list = [] if output_hidden_states else None
        all_attentions_list = [] if output_attentions else None
        all_new_past_key_values_list = [] if use_cache else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states_list.append(hidden_states)

            layer_input_past_key_value = past_key_values[i] if past_key_values is not None else (None, None)
            
            if self.training and self.config.use_gradient_checkpointing and hasattr(torch.utils.checkpoint, 'checkpoint') and self.config.is_gradient_checkpointing:
                # Define a custom forward function for checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # inputs are: hidden_states, attention_mask, layer_past_kv, use_cache_flag, output_attentions_flag
                        # The past_key_value (inputs[2]) is the specific layer's past_key_value tuple
                        return module(inputs[0], attention_mask=inputs[1], past_key_value=inputs[2], use_cache=inputs[3], output_attentions=inputs[4])
                    return custom_forward

                # Pass non-tensor arguments that are fixed across the batch or are flags
                # Convert boolean flags to tensors for checkpoint compatibility if necessary
                current_use_cache_tensor = torch.tensor(use_cache, device=hidden_states.device) 
                current_output_attentions_tensor = torch.tensor(output_attentions, device=hidden_states.device)

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask, 
                    layer_input_past_key_value, 
                    current_use_cache_tensor, 
                    current_output_attentions_tensor,
                    preserve_rng_state=True,
                    use_reentrant=self.config.gradient_checkpointing_use_reentrant # Use config for reentrant
                )
            else:
                # Pop arguments that are explicitly passed to layer_module to avoid conflict with **kwargs
                kwargs_for_layer = kwargs.copy()
                kwargs_for_layer.pop('output_attentions', None)
                kwargs_for_layer.pop('output_hidden_states', None)
                # Note: use_cache is also explicitly passed, but usually not an issue if also in kwargs unless layer_module itself has **kwargs

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=layer_input_past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs_for_layer
                )
        
            hidden_states = layer_outputs[0]

            if use_cache:
                # layer_outputs[1] is (new_self_attn_kv_cache, new_mem_attn_state)
                all_new_past_key_values_list.append(layer_outputs[1])
        
            if output_attentions:
                # layer_outputs[2] (if use_cache) or layer_outputs[1] (if not use_cache) is (self_attn_weights, mem_gate_weights)
                current_attentions = layer_outputs[2] if use_cache and len(layer_outputs) > 2 else (layer_outputs[1] if not use_cache and len(layer_outputs) > 1 else None)
                if current_attentions is not None:
                     all_attentions_list.append(current_attentions)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states_list.append(hidden_states)

        # Define accumulator tuples from lists populated during the layer loop
        all_new_past_key_values_accumulator = tuple(all_new_past_key_values_list) if use_cache and all_new_past_key_values_list is not None else None
        all_hidden_states_accumulator = tuple(all_hidden_states_list) if output_hidden_states and all_hidden_states_list is not None else None
        all_attentions_accumulator = tuple(all_attentions_list) if output_attentions and all_attentions_list is not None else None

        # Apply final layer norm to the last hidden state
        hidden_states = self.final_layer_norm(hidden_states)

        if not return_dict:
            _output_tuple = (hidden_states,)
            if use_cache:
                _output_tuple = _output_tuple + (all_new_past_key_values_accumulator,)
            if output_hidden_states:
                _output_tuple = _output_tuple + (all_hidden_states_accumulator,)
            if output_attentions:
                _output_tuple = _output_tuple + (all_attentions_accumulator,)
            return _output_tuple
        else:  # return_dict is True
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=all_new_past_key_values_accumulator,  # Already None if not use_cache
                hidden_states=all_hidden_states_accumulator,  # Already None if not output_hidden_states
                attentions=all_attentions_accumulator  # Already None if not output_attentions
            )
        
        if use_cache:
            # layer_outputs = (hidden_state, new_past_key_value, optional_attentions)
            all_new_past_key_values_accumulator = all_new_past_key_values_accumulator + (layer_outputs[1],)
        
        if self.config.output_attentions:
            # Attentions are at index 2 if use_cache and returned, else at index 1 if returned without cache
            attention_idx = 2 if use_cache and len(layer_outputs) > 2 else (1 if not use_cache and len(layer_outputs) > 1 else -1)
            if attention_idx != -1 and layer_outputs[attention_idx] is not None:
                 all_attentions_accumulator.append(layer_outputs[attention_idx])

        if self.config.output_hidden_states:
            all_hidden_states_accumulator = all_hidden_states_accumulator + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)
        
        if not return_dict:
            output_elements = [hidden_states]
            if use_cache and all_new_past_key_values_accumulator is not None:
                output_elements.append(all_new_past_key_values_accumulator)
            # Hugging Face expects hidden_states and attentions as tuples even if not return_dict
            if self.config.output_hidden_states and all_hidden_states_accumulator is not None:
                 output_elements.append(all_hidden_states_accumulator)
            if self.config.output_attentions and all_attentions_accumulator:
                output_elements.append(tuple(all_attentions_accumulator))
            return tuple(output_elements)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=all_new_past_key_values_accumulator if use_cache else None,
            hidden_states=all_hidden_states_accumulator if self.config.output_hidden_states else None,
            attentions=tuple(all_attentions_accumulator) if self.config.output_attentions and all_attentions_accumulator else None,
        )


from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from ..config import InfinityFormerConfig # Ensure this import is correct relative to your project structure

class InfinityFormerForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = InfinityFormerConfig
    base_model_prefix = "infinity_former"

    """
    InfinityFormer with a language modeling head on top.
    """
    def __init__(self, config: InfinityFormerConfig):
        super().__init__(config)  # Pass config to PreTrainedModel's __init__
        
        # Main model - this is specific to InfinityFormerForCausalLM
        self.infinity_former = InfinityFormerModel(config)
        
        # Language modeling head - this is specific to InfinityFormerForCausalLM
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if needed
        if self.config.tie_word_embeddings: # Use self.config as it's set by super().__init__(config)
            self.lm_head.weight = self.infinity_former.get_input_embeddings().weight

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, position_ids=None, **kwargs):
        # Determine past_length, robustly handling different past_key_values structures or absence
        if past_key_values is not None:
            try:
                # past_key_values is a list of tuples: [(self_attn_kv, mem_state), ...]
                # self_attn_kv is a tuple: (key_cache, value_cache)
                # key_cache (and value_cache) has shape [batch_size, num_heads, sequence_length, head_dim]
                # So, past_key_values[0][0][0] is the key_cache of the first layer.
                # We need its sequence_length dimension.
                if past_key_values[0] is not None and past_key_values[0][0] is not None and past_key_values[0][0][0] is not None:
                    past_length = past_key_values[0][0][0].shape[2]
                else: # Handle cases where the cache might be partially None (e.g. first step of generation)
                    past_length = 0
            except (TypeError, IndexError, AttributeError):
                # Fallback if past_key_values is not structured as expected or is malformed
                past_length = 0
        else:
            past_length = 0

        # Create position_ids if not provided
        if position_ids is None:
            current_length = input_ids.shape[1]
            # Position IDs should correspond to the absolute positions in the sequence
            position_ids = torch.arange(past_length, past_length + current_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) # Expand to batch size [B, S]

        # If past_key_values are used (indicating a KV cache or recurrent state from previous steps),
        # typically only the last token of input_ids is needed for the next step.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            # If input_ids is sliced, position_ids should also correspond to this single token.
            # The position_ids created above would be for the full current_length.
            # If current_length was > 1, we need the last position_id.
            if position_ids.shape[1] > 1: # If position_ids were for more than one token
                 position_ids = position_ids[:, -1:]
        
        # Ensure attention_mask is present, defaulting to all ones if not.
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape) # Mask for the current input_ids

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values, # Passed through; model.forward needs to handle it
        }
        
        # Pass use_cache to the model's forward method.
        # The model.forward (and its sub-modules) must be able to accept and use/ignore this.
        model_inputs["use_cache"] = kwargs.get("use_cache", True)
        
        return model_inputs

        # Weight initialization is handled by PreTrainedModel's __init__ (via post_init(),
        # which calls _init_weights if defined on this class). No explicit call needed here.

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def get_input_embeddings(self) -> nn.Module:
        return self.infinity_former.get_input_embeddings()
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        self.infinity_former.set_input_embeddings(value)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = value.weight
    
    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # Changed from Tuple[Tuple[torch.Tensor]] for InfinityFormer's memory
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or [batch_size, 1, 1, seq_len]
            position_ids: [batch_size, seq_len]
            inputs_embeds: [batch_size, seq_len, hidden_size]
            labels: [batch_size, seq_len] with indices in [-100, 0, ..., vocab_size-1]
            return_dict: Whether to return a dict or a tuple
            
        Returns:
            If labels are provided, returns (loss, logits), otherwise returns logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if hasattr(self.config, 'use_cache') else False)
        
        # Get model outputs
        outputs = self.infinity_former(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # Get sequence output
        sequence_output = outputs[0] if not return_dict else outputs.last_hidden_state
        # If use_cache is True, outputs will be a model output object from InfinityFormerModel
        # that should contain past_key_values. Otherwise, it might be a tuple.
        current_past_key_values = outputs.past_key_values if use_cache and hasattr(outputs, 'past_key_values') else None
        
        # Get logits
        lm_logits = self.lm_head(sequence_output)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (lm_logits,) + (outputs[1:] if isinstance(outputs, tuple) else (outputs.hidden_states, outputs.attentions))
            if use_cache:
                output = output + (current_past_key_values,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=current_past_key_values, # This will be the updated memory state from InfinityFormerModel
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )