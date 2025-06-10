#!/usr/bin/env python3
"""
Example script for training and evaluating the InfinityFormer model.
"""

import os
import argparse
import logging
import torch
import wandb # Added for Weights & Biases integration
from tqdm import tqdm

import sys
import os

# Add the project root directory (c:\no-backdrop) to sys.path
# This allows Python to find the 'infinityformer' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from datasets import load_dataset as hf_load_dataset

from infinityformer import (
    InfinityFormerConfig,
    InfinityFormerForCausalLM,
)
from infinityformer.utils import (
    load_dataset,
    get_dataloader,
    DataCollatorForLanguageModeling,
    get_optimizer,
    get_scheduler,
    train_epoch,
    evaluate,
    generate_text,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate InfinityFormer model")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from huggingface.co/models"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name"
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=1024, help="The maximum total sequence length after tokenization."
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    
    # Training arguments
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="Max gradient norm."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help=(
            "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html"
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run an evaluation every X steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help=(
            "Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. "
            "Default is unlimited checkpoints"
        ),
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
        "which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of evaluation examples to this "
        "value if set.",
    )
    
    # InfinityFormer specific arguments
    parser.add_argument(
        "--use_rotary_embeddings",
        action="store_true",
        help="Whether to use rotary position embeddings.",
    )
    parser.add_argument(
        "--use_multi_scale_memory",
        action="store_true",
        help="Whether to use multi-scale memory.",
    )
    parser.add_argument(
        "--num_memory_scales",
        type=int,
        default=3,
        help="Number of memory scales to use.",
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="elu",
        choices=["elu", "relu", "learnable"],
        help="Type of kernel function to use for linear attention.",
    )
    
    # Weights & Biases arguments
    parser.add_argument(
        "--wandb_project", type=str, default="infinityformer-pretraining", help="Weights & Biases project name."
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="Weights & Biases entity (username or team). Optional, W&B will use default if not set."
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Optional custom name for the W&B run."
    )
    parser.add_argument(
        "--wandb_run_id", type=str, default=None, help="Optional W&B run ID to resume a specific run."
    )

    args = parser.parse_args()
    
    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    
    return args

def main():
    args = parse_args()
    
    # Initialize W&B as early as possible
    # It's good practice to allow project and entity to be set via args or environment variables
    # For now, using a default project name and letting W&B pick up entity from env/login.
    # We will add CLI arguments for these later if needed.
    wandb.init(
        project=args.wandb_project if hasattr(args, 'wandb_project') and args.wandb_project else "infinityformer-pretraining",
        entity=args.wandb_entity if hasattr(args, 'wandb_entity') and args.wandb_entity else None, # Optional: your W&B entity (username or team)
        name=args.wandb_run_name if hasattr(args, 'wandb_run_name') and args.wandb_run_name else None, # Optional: a custom name for the run
        config=args,  # Log all hyperparameters from argparse
        resume="allow", # Allows resuming a run if id is passed or auto-detected
        id=args.wandb_run_id if hasattr(args, 'wandb_run_id') and args.wandb_run_id else None # For resuming specific runs
    )
    
    # Set seed before initializing model.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Consider adding numpy and random seeds here if they are used elsewhere
        # import numpy as np
        # import random
        # np.random.seed(args.seed)
        # random.seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info(f"device: {device}, n_gpu: {n_gpu}, 16-bits training: {args.fp16}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_kwargs = {
        "use_fast": not args.use_slow_tokenizer,
        # trust_remote_code is not a standard arg for AutoTokenizer.from_pretrained
        # It's usually passed to AutoModel.from_pretrained. If needed for a specific tokenizer,
        # it might require conditional logic or ensuring the arg is always present.
        # For now, removing it to avoid potential errors with standard tokenizers.
        # "trust_remote_code": getattr(args, 'trust_remote_code', False) 
    }
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, **tokenizer_kwargs)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    else:
        logger.info("No tokenizer specified, defaulting to 'Qwen/Qwen3-0.6B' tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", **tokenizer_kwargs)

    # Set pad_token_id if not set for models like GPT-2
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model config (vocab_size will be updated after tokenizer)
    logger.info("Initializing model config...")
    # Ensure these args are defined in parse_args or have defaults in InfinityFormerConfig
    config = InfinityFormerConfig(
        hidden_size=getattr(args, 'hidden_size', 256),
        num_hidden_layers=getattr(args, 'num_hidden_layers', 6),
        num_attention_heads=getattr(args, 'num_attention_heads', 8),
        intermediate_size=getattr(args, 'intermediate_size', 1024),
        max_position_embeddings=args.max_seq_length,
        use_rotary_embeddings=args.use_rotary_embeddings,
        use_multi_scale_memory=args.use_multi_scale_memory,
        num_memory_scales=args.num_memory_scales,
        kernel_type=args.kernel_type,
        tie_word_embeddings=getattr(args, 'tie_word_embeddings', True) # From previous fix
    )
    config.vocab_size = len(tokenizer) # Update config vocab size based on tokenizer

    # Load and process dataset
    logger.info("Loading and processing dataset...")
    train_dataset = None
    eval_dataset = None

    if args.dataset_name:
        raw_datasets = hf_load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys() and args.do_eval and "train" in raw_datasets.keys():
            logger.warning("Validation set not found in dataset. Splitting train set for validation.")
            train_split = raw_datasets["train"].train_test_split(test_size=0.1, seed=getattr(args, 'seed', 42))
            raw_datasets["train"] = train_split["train"]
            raw_datasets["validation"] = train_split["test"]

        column_names = raw_datasets["train"].column_names if "train" in raw_datasets else raw_datasets[list(raw_datasets.keys())[0]].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], truncation=True, max_length=args.max_seq_length, padding=False)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        if "train" in tokenized_datasets: train_dataset = tokenized_datasets["train"]
        if args.do_eval and "validation" in tokenized_datasets: eval_dataset = tokenized_datasets["validation"]

    elif args.train_file:
        from infinityformer.utils.data_utils import TextDataset # Ensure import
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=args.train_file,
            block_size=args.max_seq_length,
            overwrite_cache=args.overwrite_cache
        )
        if args.validation_file and args.do_eval:
            eval_dataset = TextDataset(
                tokenizer=tokenizer,
                file_path=args.validation_file,
                block_size=args.max_seq_length,
                overwrite_cache=args.overwrite_cache
            )
    else:
        logger.info("No dataset specified. Creating a small dummy in-memory dataset with the loaded tokenizer.")
        dummy_texts = [
            "This is a sample sentence for the InfinityFormer model.",
            "Language modeling is an interesting task.",
            "InfinityFormer aims for long context understanding."
        ] * 10 
        
        processed_dummy_data = []
        for text in dummy_texts:
            encoded = tokenizer(text, truncation=True, max_length=args.max_seq_length) # No padding here
            processed_dummy_data.append({'input_ids': encoded['input_ids']})
        
        class ListDataset(torch.utils.data.Dataset):
            def __init__(self, data_list):
                self.data_list = data_list
            def __len__(self):
                return len(self.data_list)
            def __getitem__(self, idx):
                return self.data_list[idx]

        train_dataset = ListDataset(processed_dummy_data)
    
    if train_dataset is None and args.do_train:
        raise ValueError("Failed to load or create a training dataset, but --do_train was specified.")
    if eval_dataset is None and args.do_eval:
        logger.warning("Evaluation dataset not loaded or created, but --do_eval was specified. Skipping evaluation.")
        args.do_eval = False

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, # Causal LM
        pad_to_multiple_of=8 if args.fp16 else None 
    )

    # Create dataloaders
    train_dataloader = None
    if args.do_train and train_dataset:
        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=getattr(args, 'dataloader_num_workers', 0) # Add num_workers if available in args
        )
    
    eval_dataloader = None
    if args.do_eval and eval_dataset:
        eval_dataloader = get_dataloader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=getattr(args, 'dataloader_num_workers', 0)
        )

    # Initialize model
    logger.info("Initializing model...")
    model = InfinityFormerForCausalLM(config)
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
    
    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
        scheduler_type=args.lr_scheduler_type,
    )
    
    # Enable gradient checkpointing if specified
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Enable mixed precision training if specified
    if args.fp16:
        scaler = torch.amp.GradScaler(enabled=args.fp16)
    else:
        scaler = None
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    epochs_trained = 0
    
    # Check if continuing training from a checkpoint
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        global_step = checkpoint['global_step']
        epochs_trained = checkpoint['epoch']
        
        logger.info(f"  Continuing training from checkpoint, resuming from step {global_step}")
    
    # Training loop
    for epoch in range(epochs_trained, int(args.num_train_epochs)):
        model.train()
        
        # Training epoch
        # Calculate steps_per_epoch for accurate fractional epoch logging in W&B
        steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
        if len(train_dataloader) % args.gradient_accumulation_steps != 0:
            steps_per_epoch +=1 # account for the last partial accumulation batch if any

        train_metrics, global_step = train_epoch(
            model=model,
            train_loader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_amp=args.fp16,
            scaler=scaler,
            current_epoch=epoch,
            global_step=global_step,
            logging_steps=args.logging_steps,
            steps_per_epoch=steps_per_epoch
        )
        # global_step is now updated by train_epoch
        
        # Log epoch summary metrics (console)
        logger.info(f"Epoch {epoch + 1} Summary - Avg Training Loss: {train_metrics['loss']:.4f}, Final LR: {train_metrics['learning_rate']:.2e}")
        
        # Log epoch summary to W&B (optional, as detailed logs are within train_epoch)
        # We can log the average epoch loss here, and validation metrics if any.
        epoch_summary_log = {
            "epoch/train_loss_avg": train_metrics['loss'],
            "epoch/learning_rate_final": train_metrics['learning_rate'],
            "epoch/epoch_num": epoch + 1,
            "epoch/global_step_end_of_epoch": global_step 
        }
        wandb.log(epoch_summary_log)

        # Evaluation
        if args.do_eval:
            eval_metrics = evaluate(
                model=model,
                eval_loader=eval_dataloader,
                device=device,
                use_amp=args.fp16,
                scaler=scaler # Pass scaler if use_amp is True
            )
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {eval_metrics['loss']:.4f}, Validation Perplexity: {eval_metrics['perplexity']:.4f}")
            
            wandb.log({
                "eval/loss": eval_metrics['loss'],
                "eval/perplexity": eval_metrics['perplexity'],
                "epoch/epoch_num_eval": epoch + 1,
                "trainer/global_step_eval": global_step
            })

            # Potentially update ReduceLROnPlateau scheduler
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(eval_metrics['loss'])
        
        # Save checkpoint (based on args.checkpointing_steps or end of epoch)
        # The step-based checkpointing is inside train_epoch if args.checkpointing_steps is an int.
        # This section can be for end-of-epoch checkpointing if desired, or removed if step-based is sufficient.
        # For now, let's assume the original intent was more frequent checkpointing controlled by args.checkpointing_steps (str like 'epoch' or 'step')
        # The current code has checkpointing_steps inside main loop, which is fine for 'epoch' type.
        if args.checkpointing_steps == "epoch" or (args.checkpointing_steps and str(args.checkpointing_steps).isdigit() and int(args.checkpointing_steps) == 0): # Save at end of every epoch or if steps is 0
            if args.output_dir is not None:
                checkpoint_output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}-step-{global_step}")
                os.makedirs(checkpoint_output_dir, exist_ok=True)
                
                # Save model, optimizer, scheduler, and args
                torch.save({
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'args': args,
                    'config': model.config.to_dict() if hasattr(model, 'config') else None
                }, os.path.join(checkpoint_output_dir, "training_state.pt"))
                
                # Save tokenizer
                tokenizer.save_pretrained(checkpoint_output_dir)
                logger.info(f"Saved epoch checkpoint to {checkpoint_output_dir}")
                
                # Optionally, log checkpoint as W&B artifact
                # artifact_name = f'model-epoch-{epoch+1}-step-{global_step}'
                # artifact = wandb.Artifact(artifact_name, type='model')
                # artifact.add_dir(checkpoint_output_dir)
                # wandb.log_artifact(artifact)

        # Check if max_train_steps is reached (if set)
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            logger.info(f"Reached max_train_steps ({args.max_train_steps}). Stopping training.")
            break # Exit epoch loop
    
    # Save final model
    if args.output_dir is not None:
        output_dir = os.path.join(args.output_dir, "final")
        os.makedirs(output_dir, exist_ok=True)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
        }, os.path.join(output_dir, "pytorch_model.bin"))
        
        logger.info(f"Saved final model to {output_dir}")
    
    # Example generation
    logger.info("Generating example text...")
    
    prompt_text = "Adrian Wallace was born on" # Using the same prompt text
    
    # Tokenize the prompt text
    # Ensure config is available here, or pass args.max_seq_length
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True, max_length=config.max_position_embeddings if 'config' in locals() else args.max_seq_length)
    input_ids_prompt = inputs.input_ids.to(device)

    # Generate text
    generated_ids_tensor = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_ids_prompt=input_ids_prompt,
        max_new_tokens=100,  # Changed from max_length, kept original value
        temperature=0.9,
        top_k=50,
        # top_p=0.95, # top_p is not used in the current generate_text function
        device=device,
        use_cache=getattr(args, 'use_cache_for_generation', False) # Added use_cache, defaults to False
    )
    
    # Decode the generated IDs
    generated_sequence_ids = generated_ids_tensor[0] 
    generated_text_output = tokenizer.decode(generated_sequence_ids, skip_special_tokens=True)
    logger.info(f"Prompt: {prompt_text}")
    logger.info(f"Generated: {generated_text_output}")

if __name__ == "__main__":
    main()
