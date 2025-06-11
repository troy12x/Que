import torch
import time
import sys
import os

# Add the project root directory (c:\no-backdrop) to sys.path
# This allows Python to find the 'infinityformer' package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infinityformer.model.model import InfinityFormerModel, InfinityFormerConfig

def benchmark_throughput(model, device, sequence_lengths, batch_size=1, warmup_steps=5, benchmark_steps=10):
    """
    Benchmarks the throughput of a model in tokens per second.

    Args:
        model (torch.nn.Module): The model to benchmark.
        device (torch.device): The device to run the benchmark on.
        sequence_lengths (list): A list of sequence lengths to test.
        batch_size (int): The batch size for the input tensors.
        warmup_steps (int): Number of warmup iterations before timing.
        benchmark_steps (int): Number of iterations to average for timing.
    """
    model.to(device)
    model.train()  # Ensure model is in training mode for backward pass

    print(f"\n{'='*40}")
    print(f"  Starting Throughput Benchmark")
    print(f"{'='*40}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size}")
    print(f"Warmup Steps: {warmup_steps}")
    print(f"Benchmark Steps: {benchmark_steps}")
    print(f"\n{'Sequence Length':<20} | {'Throughput (tokens/s)':<25}")
    print(f"{'--------------------':<20} | {'-------------------------':<25}")

    for seq_len in sequence_lengths:
        input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
        total_tokens = batch_size * seq_len

        # Warmup phase
        for _ in range(warmup_steps):
            _ = model(input_ids)
            # In a real scenario, you'd have loss.backward(), but for pure throughput,
            # we can simulate a backward pass on a dummy loss from the output.
            dummy_loss = model(input_ids).last_hidden_state.sum()
            dummy_loss.backward()
            model.zero_grad()

        # Benchmark phase
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(benchmark_steps):
            outputs = model(input_ids)
            dummy_loss = outputs.last_hidden_state.sum()
            dummy_loss.backward()
            model.zero_grad()
        
        torch.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_step = total_time / benchmark_steps
        throughput = total_tokens / avg_time_per_step

        print(f"{seq_len:<20} | {throughput:>25.2f}")

    print(f"\n{'='*40}")
    print("  Benchmark Complete")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    # --- Configuration ---
    # This should match the configuration of the model you want to test.
    config = InfinityFormerConfig(
        vocab_size=50257,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=131072,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        use_gradient_checkpointing=True,
    )

    # --- Model Initialization ---
    model = InfinityFormerModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Benchmark will run on CPU, which will be very slow.")

    # --- Benchmark Execution ---
    # Sequence lengths to test, similar to the paper's graph
    sequence_lengths_to_test = [4096, 8192, 16384, 32768]

    benchmark_throughput(
        model=model,
        device=device,
        sequence_lengths=sequence_lengths_to_test,
        batch_size=1 # Using batch size 1 to measure pure sequence length scaling
    )
