"""
Training utilities for InfinityFormer.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from tqdm import tqdm
import numpy as np
import wandb # Added for Weights & Biases integration

from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..config import InfinityFormerConfig


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    no_decay: Optional[list] = None
) -> Optimizer:
    """
    Create an AdamW optimizer with weight decay fix.
    
    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay
        adam_epsilon: Epsilon for Adam optimizer
        no_decay: List of parameter names to exclude from weight decay
        
    Returns:
        Optimizer instance
    """
    if no_decay is None:
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
    # Separate parameters into those with and without weight decay
    params_with_wd = []
    params_without_wd = []
    
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
            
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    
    # Group parameters
    optimizer_grouped_parameters = [
        {
            "params": params_with_wd,
            "weight_decay": weight_decay,
        },
        {
            "params": params_without_wd,
            "weight_decay": 0.0,
        },
    ]
    
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)


def get_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: str = "linear",
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        scheduler_type: Type of scheduler ('linear', 'cosine', 'reduce_on_plateau')
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "linear":
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)
    
    elif scheduler_type == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: torch.device = None,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    progress_bar: bool = True,
    # W&B related arguments
    current_epoch: int = 0, # For logging
    global_step: int = 0, # For logging and tracking
    logging_steps: int = 100, # How often to log to W&B
    # For calculating fractional epoch for W&B
    steps_per_epoch: Optional[int] = None 
) -> Tuple[Dict[str, float], int]: # Return metrics and updated global_step
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Number of steps to accumulate gradients
        use_amp: Whether to use automatic mixed precision
        scaler: Gradient scaler for AMP
        progress_bar: Whether to show progress bar
        
    Returns:
        Dictionary with training metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    total_loss = 0.0
    total_steps = 0
    
    # Initialize progress bar if needed
    if progress_bar:
        train_loader = tqdm(train_loader, desc="Training")
    
    # Initialize gradient accumulation
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision if enabled
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            outputs = model(**batch)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights if we've accumulated enough gradients
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip gradients
            if max_grad_norm > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # Update learning rate
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Update metrics for epoch summary
            # Note: loss here is already scaled by gradient_accumulation_steps
            # So, loss.item() is the average loss over accumulation steps for this update.
            actual_batch_loss = loss.item() * gradient_accumulation_steps
            total_loss += actual_batch_loss
            total_steps += 1 # This counts optimizer steps
            global_step += 1

            # Log to W&B at specified intervals
            if logging_steps > 0 and global_step % logging_steps == 0:
                # Determine sequence length for token calculation
                seq_len_for_tokens = wandb.config.get('max_seq_length')
                if seq_len_for_tokens is None:
                    seq_len_for_tokens = wandb.config.get('block_size')
                # Fallback if both are None, though unlikely if data is prepared
                if seq_len_for_tokens is None and hasattr(model, 'config') and hasattr(model.config, 'max_position_embeddings'):
                     seq_len_for_tokens = model.config.max_position_embeddings
                if seq_len_for_tokens is None: # Ultimate fallback
                    seq_len_for_tokens = 0 # Avoid error, but indicates missing config

                # Ensure config values are present, provide defaults if not (though they should be from args)
                per_device_bs = wandb.config.get('per_device_train_batch_size', 1)
                grad_accum_steps = wandb.config.get('gradient_accumulation_steps', 1)
                
                # Note: This calculation assumes single-process training or that global_step is consistent across processes.
                # If using DDP, typically only rank 0 logs. The per_device_train_batch_size * n_gpus would be total batch size.
                # For simplicity here, we use per_device_train_batch_size. If n_gpu is in wandb.config, could use it.
                tokens_processed_cumulative = global_step * per_device_bs * grad_accum_steps * seq_len_for_tokens

                log_metrics = {
                    "train/batch_loss": actual_batch_loss,
                    "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                    "trainer/global_step": global_step,
                    "train/epoch_fractional": current_epoch + ( (step + 1) / steps_per_epoch if steps_per_epoch and steps_per_epoch > 0 else (step + 1) / len(train_loader) ),
                    "trainer/tokens_processed_cumulative": tokens_processed_cumulative
                }
                # Add other potential metrics like gradient norm if calculated
                # log_metrics["train/grad_norm"] = grad_norm # if you calculate it
                wandb.log(log_metrics)

    # Calculate average loss for the epoch
    # total_steps here is the number of optimizer steps in this epoch
    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0 
    
    return {"loss": avg_loss, "learning_rate": optimizer.param_groups[0]["lr"]}, global_step


import math # Added for perplexity calculation

def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device = None,
    progress_bar: bool = True,
) -> Dict[str, float]:
    """
    Evaluate the model on the given dataset.
    
    Args:
        model: The model to evaluate
        eval_loader: DataLoader for evaluation data
        device: Device to evaluate on
        progress_bar: Whether to show progress bar
        
    Returns:
        Dictionary with evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    # Initialize progress bar if needed
    if progress_bar:
        eval_loader = tqdm(eval_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs[0].item()
            
            # Update metrics
            total_loss += loss
            total_steps += 1
    
    # Calculate average loss
    avg_loss = total_loss / len(eval_loader) if len(eval_loader) > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss is not None and avg_loss != float('inf') else float('inf')
    
    return {"loss": avg_loss, "perplexity": perplexity}


def generate_text(
    model: nn.Module,
    tokenizer, # Needed for EOS token ID
    input_ids_prompt: torch.Tensor, # Changed from prompt: str
    max_new_tokens: int = 20,      # Changed from max_length, and default adjusted
    temperature: float = 1.0,
    top_k: int = 50,
    # top_p and repetition_penalty are not used in this simplified loop, 
    # but could be added back if more advanced sampling is needed.
    device: torch.device = None,
    use_cache: bool = False, # Defaulting to False as it's simpler for this manual loop
) -> torch.Tensor: # Returns a Tensor of IDs
    """
    Generate a sequence of token IDs starting from input_ids_prompt.

    Args:
        model: The model to use for generation.
        tokenizer: Tokenizer, used for EOS token ID.
        input_ids_prompt: Tensor of input token IDs (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. 0 means greedy.
        top_k: Top-k sampling parameter. 0 means no top-k filtering.
        device: Device to use for generation.
        use_cache: Whether the model should use its KV cache (if implemented).

    Returns:
        Tensor of generated token IDs, including the prompt (batch_size, prompt_len + new_tokens_len).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device) # Ensure model is on the correct device

    # Ensure input_ids_prompt is on the correct device
    input_ids_prompt = input_ids_prompt.to(device)
    generated_ids = input_ids_prompt.clone()

    if max_new_tokens <= 0:
        # logger.warning("max_new_tokens is <= 0, returning original input_ids_prompt.")
        return generated_ids # Return original if no new tokens requested

    model_max_len = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else float('inf')
    # logger.debug(f"generate_text: model_max_len = {model_max_len}, use_cache = {use_cache}, initial_prompt_len = {input_ids_prompt.shape[1]}")

    with torch.no_grad():
        for i in range(max_new_tokens):
            # Check if the *current* length of generated_ids (which will be input to the model)
            # already meets or exceeds model_max_len. If so, we can't proceed if use_cache=False,
            # as the model call itself would fail or subsequent appends would be problematic.
            if not use_cache and generated_ids.shape[1] >= model_max_len:
                # logger.warning(
                #     f"Stopping generation: current input length ({generated_ids.shape[1]}) for model call "
                #     f"meets or exceeds model_max_len ({model_max_len}) with use_cache=False. "
                #     f"Generated {i} new tokens out of requested {max_new_tokens}."
                # )
                break
            
            current_input_ids = generated_ids
            attention_mask = torch.ones_like(current_input_ids).to(device)

            outputs = model(input_ids=current_input_ids, attention_mask=attention_mask, return_dict=True, use_cache=use_cache)
            next_token_logits = outputs.logits[:, -1, :]

            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                top_k_values, _ = torch.topk(next_token_logits, top_k, dim=-1)
                kth_value = top_k_values[:, -1].unsqueeze(-1)
                indices_to_remove = next_token_logits < kth_value
                next_token_logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            
            if temperature == 0:
                 next_token_id = torch.argmax(probs, dim=-1).unsqueeze(-1)
            else:
                 next_token_id = torch.multinomial(probs, num_samples=1)

            # If we are already at model_max_len and use_cache is False, we shouldn't append further
            # as the next iteration's check generated_ids.shape[1] >= model_max_len would pass, but the model
            # would have been fed something of model_max_len. Appending makes it model_max_len + 1.
            # The check at the start of the loop handles this: if generated_ids.shape[1] == model_max_len, it breaks.
            # So, it's safe to append here.
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                # logger.info(f"EOS token ({tokenizer.eos_token_id}) generated. Stopping.")
                break

    return generated_ids # Return the full tensor of IDs
