"""
Script to evaluate the long-context capabilities of InfinityFormer
using a 'needle in a haystack' / passkey retrieval task.
"""

import argparse
import logging
import random
import time
import json
import os
import sys

import torch
from transformers import AutoTokenizer

# Add the project root directory (c:\no-backdrop) to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from infinityformer import InfinityFormerConfig, InfinityFormerForCausalLM
from infinityformer.utils.training import generate_text # We'll use our manual one

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_long_input(target_token_length, tokenizer, device, model_max_len):
    """Creates a long input sequence of approximately target_token_length."""
    filler_sentence = "This is a filler sentence to create a long context for architectural stress testing. The quick brown fox jumps over the lazy dog. "
    # Ensure it's not empty if target_token_length is very small
    if target_token_length <= 0: 
        target_token_length = 10 
        logger.warning("Target token length was <=0, setting to 10.")

    input_text = ""
    # Build up text by repeating the filler sentence.
    # This is a rough way to control length; precise token count is achieved by truncation later.
    # Estimate repeats needed. Add a bit more to ensure we likely exceed target_token_length before tokenizing.
    # Average token length of filler_sentence can be pre-calculated if needed for better estimation.
    estimated_tokens_per_filler = len(tokenizer.encode(filler_sentence, add_special_tokens=False))
    if estimated_tokens_per_filler == 0: estimated_tokens_per_filler = 1 # Avoid division by zero
    num_repeats = (target_token_length // estimated_tokens_per_filler) + 5 # Add some buffer
    
    input_text = filler_sentence * num_repeats
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    
    # Truncate to the exact target_token_length (or as close as possible if source is shorter)
    if len(tokens) > target_token_length:
        tokens = tokens[:target_token_length]
    elif len(tokens) < target_token_length:
        logger.warning(f"Generated filler text has {len(tokens)} tokens, which is less than target {target_token_length}. Using all generated tokens.")
    
    if not tokens: # Ensure tokens is not empty
        logger.warning(f"Generated token list is empty for target_token_length: {target_token_length}. Using a default short sequence.")
        tokens = tokenizer.encode("default sequence for testing", add_special_tokens=False)

    # Number of special tokens that will be added (e.g., BOS and EOS)
    # Current implementation always adds two positions for BOS and EOS conceptually.
    num_added_special_tokens = 0
    if tokenizer.bos_token_id is not None:
        num_added_special_tokens +=1
    if tokenizer.eos_token_id is not None:
        # If bos and eos are same, we only count one actual slot if they are added as [BOS_EOS] + tokens
        # But our code does [BOS] + tokens + [EOS], so it's always two slots if both are defined.
        # For simplicity, assuming two distinct slots are used if both .bos_token_id and .eos_token_id are present.
        # The current construction is `[tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]`
        # This means 2 tokens are added if both are defined.
        # If one is None, then 1 token is added. If both None, 0. (Handled by final_token_list construction)
        pass # Logic below will handle this by checking actual additions.
    
    # Tentatively, let's assume 2 special tokens will be added for calculating allowed_content_len.
    # A more precise calculation happens when constructing final_token_list.
    # This is to ensure 'tokens' (the content) is sized appropriately first.
    # Max length for the 'tokens' (content part) to fit within model_max_len when special tokens are added.
    # model_max_len is config.max_position_embeddings for the model.
    
    # Calculate how many special tokens will actually be prepended/appended
    _temp_special_tokens_count = 0
    if tokenizer.bos_token_id is not None: _temp_special_tokens_count +=1
    if tokenizer.eos_token_id is not None: _temp_special_tokens_count +=1

    allowed_content_len = model_max_len - _temp_special_tokens_count

    if allowed_content_len < 0:
        logger.error(f"model_max_len ({model_max_len}) is too small to fit {_temp_special_tokens_count} special tokens. Content will be empty.")
        allowed_content_len = 0

    if len(tokens) > allowed_content_len:
        logger.info(
            f"Content tokens (length {len(tokens)}, target {target_token_length}) "
            f"truncated to {allowed_content_len} to fit within model_max_len ({model_max_len}) "
            f"with {_temp_special_tokens_count} special tokens."
        )
        tokens = tokens[:allowed_content_len]

    if not tokens and target_token_length > 0 and allowed_content_len <= 0:
         logger.warning(
             f"Content tokens became empty. Original target: {target_token_length}, model_max_len: {model_max_len}, "
             f"allowed_content_len: {allowed_content_len}. Input will only consist of special tokens if any."
         )

    # Construct final list of tokens carefully
    final_token_list = []
    if tokenizer.bos_token_id is not None:
        final_token_list.append(tokenizer.bos_token_id)
    final_token_list.extend(tokens) # Add the (potentially truncated) content tokens
    if tokenizer.eos_token_id is not None:
        final_token_list.append(tokenizer.eos_token_id)
    
    input_ids = torch.tensor(final_token_list, dtype=torch.long).unsqueeze(0).to(device)

    # Final safeguard: verify that the constructed input_ids does not exceed model_max_len
    if input_ids.shape[1] > model_max_len:
        logger.critical(
            f"CRITICAL LOGIC ERROR in create_long_input: Final input_ids length {input_ids.shape[1]} "
            f"exceeds model_max_len {model_max_len}. Forcibly truncating input_ids. This should be fixed."
        )
        input_ids = input_ids[:, :model_max_len] # Hard truncate as a last resort
    
    return input_ids


def run_evaluation(args):
    logger.info(f"Starting long context architectural stress test with args: {args}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        # Attempt to set a common BOS token if missing, or use EOS as fallback
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            pass # Already has one
        elif hasattr(tokenizer, 'cls_token_id') and tokenizer.cls_token_id is not None: # e.g. BERT
            tokenizer.bos_token = tokenizer.cls_token
            tokenizer.bos_token_id = tokenizer.cls_token_id
        else: # Fallback to EOS
            tokenizer.bos_token = tokenizer.eos_token
            tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Set tokenizer.bos_token to: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Create InfinityFormerConfig from command-line arguments
    logger.info("Creating InfinityFormerConfig from command-line arguments.")
    config = InfinityFormerConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.model_capacity_seq_len,
        use_rotary_embeddings=args.use_rotary_embeddings,
        use_multi_scale_memory=args.use_multi_scale_memory,
        num_memory_scales=args.num_memory_scales,
        kernel_type=args.kernel_type,
        tie_word_embeddings=args.tie_word_embeddings
    )
    logger.info(f"Initialized Config: {config}")

    logger.info("Initializing a new model from the generated configuration (untrained). This script tests processing capability.")
    model = InfinityFormerForCausalLM(config)

    model.to(device)
    model.eval()

    model_max_true_seq_len = config.max_position_embeddings
    logger.info(f"Model configured for max_position_embeddings: {model_max_true_seq_len}")

    results = []
    context_lengths_to_test = list(range(args.min_test_seq_len, args.max_test_seq_len + 1, args.test_seq_len_step))

    for ctx_len_tokens in context_lengths_to_test:
        if ctx_len_tokens <= 0:
            logger.warning(f"Skipping invalid test sequence length: {ctx_len_tokens}")
            continue
        if ctx_len_tokens > model_max_true_seq_len:
            logger.warning(f"Requested test sequence length {ctx_len_tokens} exceeds model's configured capacity {model_max_true_seq_len}. Skipping.")
            continue
        
        num_successful_processing = 0
        for i in range(args.num_tests_per_length):
            logger.info(f"Test {i+1}/{args.num_tests_per_length} for target processing length {ctx_len_tokens} tokens.")

            input_ids = create_long_input(
                ctx_len_tokens, 
                tokenizer, 
                device,
                model_max_true_seq_len # Model's absolute capacity
            )
            
            actual_input_length = input_ids.shape[1]
            if actual_input_length > model_max_true_seq_len + 2 : # +2 for BOS/EOS, though create_long_input should handle this
                 logger.warning(f"Generated input_ids length {actual_input_length} still exceeds model max capacity {model_max_true_seq_len}. This indicates an issue in token budgeting. Skipping test.")
                 continue
            if actual_input_length == 0:
                logger.warning("Generated input_ids is empty. Skipping test.")
                continue

            processing_successful_this_run = False
            generated_answer_text = "ERROR: Generation failed or not attempted."
            generation_time = -1.0
            try:
                start_time = time.time()
                generated_output_ids = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids_prompt=input_ids, # Changed 'input_ids' to 'input_ids_prompt'
                    max_new_tokens=args.generate_num_tokens,
                    temperature=0.7, # Can be higher for untrained model, doesn't matter much
                    top_k=50,
                    device=device,
                    use_cache=False 
                )
                end_time = time.time()
                generation_time = end_time - start_time
                
                generated_part_ids = generated_output_ids[0, actual_input_length:]
                generated_answer_text = tokenizer.decode(generated_part_ids, skip_special_tokens=True).strip()
                processing_successful_this_run = True
                num_successful_processing += 1
            except Exception as e:
                logger.error(f"Error during model processing or generation for context length {ctx_len_tokens}: {e}", exc_info=True)
                generated_answer_text = f"ERROR: {e}"
            
            logger.info(f"  Input length fed to model: {actual_input_length}")
            logger.info(f"  Generated sample ({args.generate_num_tokens} tokens): {generated_answer_text}")
            logger.info(f"  Processing Successful: {processing_successful_this_run} (Time: {generation_time:.2f}s)")
            
            results.append({
                "target_context_length_tokens": ctx_len_tokens,
                "actual_input_tokens_fed_to_model": actual_input_length,
                "test_iteration": i + 1,
                "generated_output_sample": generated_answer_text,
                "processing_successful": processing_successful_this_run,
                "processing_time_s": generation_time,
            })

        logger.info(f"Successfully processed {num_successful_processing}/{args.num_tests_per_length} tests for target context length {ctx_len_tokens}.")

    if args.output_file:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {args.output_file}")

    logger.info("Architectural stress test finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test InfinityFormer architectural capability to process long sequences.")
    
    # Model Architecture Arguments
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=6)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--use_rotary_embeddings", action="store_true", help="Use rotary position embeddings.")
    parser.add_argument("--no_rotary_embeddings", action="store_false", dest="use_rotary_embeddings", help="Do not use rotary position embeddings.")
    parser.set_defaults(use_rotary_embeddings=True)
    
    parser.add_argument("--use_multi_scale_memory", action="store_true", help="Use multi-scale memory.")
    parser.add_argument("--no_multi_scale_memory", action="store_false", dest="use_multi_scale_memory", help="Do not use multi-scale memory.")
    parser.set_defaults(use_multi_scale_memory=True)

    parser.add_argument("--num_memory_scales", type=int, default=3)
    parser.add_argument("--kernel_type", type=str, default="elu", choices=["elu", "relu", "learnable"])
    parser.add_argument("--tie_word_embeddings", action="store_true", help="Tie word embeddings.")
    parser.add_argument("--no_tie_word_embeddings", action="store_false", dest="tie_word_embeddings", help="Do not tie word embeddings.")
    parser.set_defaults(tie_word_embeddings=True)

    # Sequence Length Arguments for Model and Testing
    parser.add_argument("--model_capacity_seq_len", type=int, default=2048, help="Maximum sequence length the model architecture will be configured for (max_position_embeddings).")
    parser.add_argument("--min_test_seq_len", type=int, default=512, help="Minimum sequence length (tokens) to test processing for.")
    parser.add_argument("--max_test_seq_len", type=int, default=2048, help="Maximum sequence length (tokens) to test processing for (must be <= model_capacity_seq_len).")
    parser.add_argument("--test_seq_len_step", type=int, default=512, help="Step size for sequence lengths to test.")
    
    # Other Necessary Arguments
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path or name of the tokenizer.")
    parser.add_argument("--num_tests_per_length", type=int, default=1, help="Number of processing tests per sequence length.")
    parser.add_argument("--generate_num_tokens", type=int, default=10, help="Number of tokens to attempt to generate after processing the long sequence.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional JSON file to save detailed results.")

    args = parser.parse_args()

    # Validate sequence length arguments
    if args.max_test_seq_len > args.model_capacity_seq_len:
        raise ValueError("max_test_seq_len cannot be greater than model_capacity_seq_len.")
    if args.min_test_seq_len <= 0:
        raise ValueError("min_test_seq_len must be positive.")

    run_evaluation(args)

# Example usage:
# python examples/evaluate_long_context.py --tokenizer_name_or_path gpt2 --model_capacity_seq_len 2048 --min_test_seq_len 512 --max_test_seq_len 2048 --test_seq_len_step 256 --hidden_size 256 --num_hidden_layers 4 --num_attention_heads 4 --intermediate_size 512 --use_rotary_embeddings --use_multi_scale_memory --num_memory_scales 2 --kernel_type elu --generate_num_tokens 5 --output_file arch_stress_test_results.json
