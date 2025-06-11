import os
import sys
import argparse
import logging
import torch

# Add project root to sys.path to allow for local package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler
from datasets import load_dataset as hf_load_dataset

# Local imports
from infinityformer.model import InfinityFormerConfig, InfinityFormerForCausalLM
from huggingface_hub import create_repo
from evaluate import run_mmlu_evaluation, run_piqa_evaluation
from infinityformer.utils import (
    DataCollatorForLanguageModeling,
    get_optimizer
)


# --- Distributed Training Setup ---
def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process(single_gpu_mode=False):
    """Checks if the current process is the main process."""
    if single_gpu_mode:
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Checkpoint Saving ---
def save_checkpoint(model, tokenizer, args, global_step):
    """Saves model, tokenizer, and arguments to a checkpoint directory."""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Saving model checkpoint to {checkpoint_dir}")

    # Unwrap the model if using DDP
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save model and tokenizer using Hugging Face's `save_pretrained`
    # safe_serialization=False is required for models with tied weights like this one.
    model_to_save.save_pretrained(checkpoint_dir, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training arguments for easy resuming
    torch.save(args, os.path.join(checkpoint_dir, "training_args.bin"))
    logger.info(f"Checkpoint saved successfully to {checkpoint_dir}")

def setup_logging(single_gpu_mode=False):
    """Sets up logging, restricting verbose logs to the main process."""
    log_level = logging.INFO if is_main_process(single_gpu_mode) else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale pre-training script for InfinityFormer")
    
    # Model & Tokenizer
    parser.add_argument("--model_config_name", type=str, help="Path to a model config json file.")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3-0.6B", help="Tokenizer name or path.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use slow tokenizer.")
    
    # Data
    parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config name.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--preprocessing_num_workers", type=int, default=8, help="Num workers for preprocessing.")
    
    # Training
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--mmlu_eval_steps", type=int, default=1000, help="Run MMLU evaluation every N steps. Disabled if 0.")
    parser.add_argument("--mmlu_limit_subjects", type=int, default=-1, help="Limit MMLU to the first N subjects for quick testing.")
    parser.add_argument("--piqa_eval_steps", type=int, default=1000, help="Run PIQA evaluation every N steps. Disabled if 0.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Override num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--num_warmup_steps", type=int, default=1000, help="Number of warmup steps.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for H100).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    
    # Checkpointing & Logging
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps.")
    parser.add_argument("--wandb_project", type=str, default="infinityformer-pretraining", help="Weights & Biases project name.")
    parser.add_argument("--single_gpu", action="store_true", help="Run on a single GPU without distributed training for testing.")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push checkpoints to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The model ID (repository name) on the Hugging Face Hub.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Create a private repository on the Hub.")

    args = parser.parse_args()
    return args

# --- Main Training Logic ---
def main():
    args = parse_args()
    
    # --- Setup based on mode (single GPU vs. distributed) ---
    if args.single_gpu:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
        setup_logging(single_gpu_mode=True)
    else:
        local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        setup_logging()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    if is_main_process(args.single_gpu):
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub and is_main_process(args.single_gpu):
            if args.hub_model_id is None:
                raise ValueError("Must specify --hub_model_id when pushing to the Hub.")
            print(f"Pushing checkpoints to repository: {args.hub_model_id}")
            # Use HUGGING_FACE_HUB_TOKEN environment variable or `huggingface-cli login`
            create_repo(args.hub_model_id, private=args.hub_private_repo, exist_ok=True)

    if is_main_process(args.single_gpu):
        wandb.init(project=args.wandb_project, config=args)

    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading and preprocessing dataset...")
    raw_datasets = hf_load_dataset(args.dataset_name, args.dataset_config_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=args.max_seq_length)

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
    )
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get("test")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_sampler = RandomSampler(train_dataset) if args.single_gpu else DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=data_collator, num_workers=args.preprocessing_num_workers)

    eval_dataloader = None
    if eval_dataset:
        eval_sampler = SequentialSampler(eval_dataset) if args.single_gpu else DistributedSampler(eval_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size, sampler=eval_sampler, collate_fn=data_collator, num_workers=args.preprocessing_num_workers)

    logger.info("Initializing model...")
    if args.model_config_name:
        config = InfinityFormerConfig.from_pretrained(args.model_config_name)
    elif args.single_gpu:
        logger.info("Using a small model configuration for single-GPU testing.")
        config = InfinityFormerConfig(
            vocab_size=len(tokenizer),
            hidden_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,  # 4 * hidden_size
            max_position_embeddings=args.max_seq_length,
        )
    else:
        logger.info("Using default ~500M parameter model configuration.")
        config = InfinityFormerConfig(
            vocab_size=len(tokenizer),
            hidden_size=768,
            num_hidden_layers=54,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=args.max_seq_length,
        )
    
    if args.gradient_checkpointing:
        config.use_cache = False

    model = InfinityFormerForCausalLM(config).to(device)
    if not args.single_gpu:
        # For debugging purposes, we are forcing find_unused_parameters=True.
        # This will help identify which parameters are not receiving gradients.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # This will print the names of parameters that are not used in the forward pass.

    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    global_step = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model_to_load = model.module if not args.single_gpu else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']

    logger.info("***** Starting Training *****")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    for epoch in range(start_epoch, args.num_train_epochs):
        model.train()
        if not args.single_gpu:
            train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(range(len(train_dataloader)), disable=not is_main_process(args.single_gpu), desc=f"Epoch {epoch+1}/{args.num_train_epochs}")

        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(args.gradient_accumulation_steps)

                if is_main_process(args.single_gpu) and global_step % args.logging_steps == 0:
                    train_loss = loss.item() * args.gradient_accumulation_steps
                    lr = lr_scheduler.get_last_lr()[0]
                    wandb.log({
                        "train/loss": train_loss,
                        "train/learning_rate": lr,
                        "trainer/global_step": global_step,
                        "epoch": epoch
                    })
                    tqdm.write(f"Step: {global_step} | Loss: {train_loss:.4f} | LR: {lr:.2e}")

                if global_step > 0 and global_step % args.checkpointing_steps == 0:
                    if is_main_process(args.single_gpu):
                        # Save locally first
                        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        model.save_pretrained(checkpoint_dir, safe_serialization=True)
                        tokenizer.save_pretrained(checkpoint_dir)
                        tqdm.write(f"Saved checkpoint to {checkpoint_dir}")

                        # Push to Hub if enabled
                        if args.push_to_hub:
                            try:
                                tqdm.write(f"Pushing checkpoint-{global_step} to the Hub...")
                                # The push_to_hub function will use the logged-in user or env token
                                model.push_to_hub(
                                    repo_id=args.hub_model_id,
                                    commit_message=f"Training checkpoint {global_step}",
                                    private=args.hub_private_repo,
                                    safe_serialization=True
                                )
                                tokenizer.push_to_hub(
                                    repo_id=args.hub_model_id,
                                    commit_message=f"Training checkpoint {global_step}",
                                    private=args.hub_private_repo
                                )
                                tqdm.write(f"Successfully pushed to {args.hub_model_id}")
                            except Exception as e:
                                tqdm.write(f"Failed to push to Hub: {e}")

                if args.mmlu_eval_steps > 0 and global_step > 0 and global_step % args.mmlu_eval_steps == 0:
                    if is_main_process(args.single_gpu):
                        tqdm.write(f"\n--- Running MMLU Evaluation at step {global_step} ---")
                        mmlu_accuracy = run_mmlu_evaluation(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            limit_subjects=args.mmlu_limit_subjects
                        )
                        tqdm.write(f"--- MMLU Eval complete. Accuracy: {mmlu_accuracy:.4f} ---\n")
                        wandb.log({
                            "eval/mmlu_accuracy": mmlu_accuracy,
                            "trainer/global_step": global_step
                        })

                if args.piqa_eval_steps > 0 and global_step > 0 and global_step % args.piqa_eval_steps == 0:
                    if is_main_process(args.single_gpu):
                        piqa_accuracy = run_piqa_evaluation(
                            model=model,
                            tokenizer=tokenizer,
                            device=device
                        )
                        wandb.log({
                            "eval/piqa_accuracy": piqa_accuracy,
                            "trainer/global_step": global_step
                        })

                if global_step >= args.max_train_steps:
                    break
        
        if eval_dataloader:
            model.eval()
            eval_losses = []
            for batch in tqdm(eval_dataloader, disable=not is_main_process(args.single_gpu), desc="Evaluating"):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                        outputs = model(**batch)
                        loss = outputs.loss
                eval_losses.append(loss.unsqueeze(0))
            
            if not args.single_gpu:
                all_eval_losses = [torch.zeros_like(eval_losses[0]) for _ in range(world_size)]
                dist.all_gather(all_eval_losses, torch.cat(eval_losses, dim=0).mean().unsqueeze(0))
                avg_eval_loss = torch.cat(all_eval_losses).mean().item()
            else:
                avg_eval_loss = torch.cat(eval_losses).mean().item()
            
            if is_main_process(args.single_gpu):
                perplexity = torch.exp(torch.tensor(avg_eval_loss)).item()
                logger.info(f"Epoch {epoch+1} Eval Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.4f}")
                wandb.log({"eval/loss": avg_eval_loss, "eval/perplexity": perplexity, "epoch": epoch + 1})

        if global_step >= args.max_train_steps:
            break

    if is_main_process(args.single_gpu):
        logger.info("Training complete. Saving final model.")
        final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
        os.makedirs(final_checkpoint_dir, exist_ok=True)
        model_to_save = model.module if not args.single_gpu else model
        torch.save(model_to_save.state_dict(), os.path.join(final_checkpoint_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(final_checkpoint_dir)
        model_to_save.config.to_json_file(os.path.join(final_checkpoint_dir, "config.json"))

    if not args.single_gpu:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
