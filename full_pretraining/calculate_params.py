import argparse
import math

def calculate_params(vocab_size, hidden_size, num_hidden_layers, intermediate_size, tied_embeddings=True):
    """Calculates the approximate number of parameters in a standard transformer model."""

    # Embedding parameters (shared input/output if tied)
    embedding_params = vocab_size * hidden_size
    
    # Attention block parameters (QKV projections + output projection)
    # Each of the 4 matrices is H x H
    attention_params = num_hidden_layers * (4 * hidden_size * hidden_size)

    # MLP block parameters (up-projection + down-projection)
    # Two matrices: H x I and I x H
    mlp_params = num_hidden_layers * (2 * hidden_size * intermediate_size)

    # Total parameters (ignoring smaller params like layer norms for approximation)
    total_params = embedding_params + attention_params + mlp_params
    
    # If embeddings are not tied, the final LM head adds more parameters
    if not tied_embeddings:
        total_params += vocab_size * hidden_size

    return {
        "embedding_params": embedding_params,
        "attention_params": attention_params,
        "mlp_params": mlp_params,
        "total_params": total_params
    }

def find_config_for_target_params(target_params, vocab_size, intermediate_ratio=4):
    """Suggests a model configuration for a target number of parameters."""
    # This is a search problem. We iterate over common hidden sizes to find the best fit.
    common_hidden_sizes = [768, 1024, 1280, 1536, 2048]
    
    best_config = None
    min_diff = float('inf')

    for h in common_hidden_sizes:
        # Simplified formula: Total ≈ vocab*H + L * H^2 * (4 + 2*ratio)
        # We solve for L (num_hidden_layers)
        embedding_params = vocab_size * h
        params_in_layers = target_params - embedding_params
        
        # Params per layer from attention (4*H^2) and MLP (2*H*I = 2*H*(ratio*H))
        params_per_layer = h * h * (4 + 2 * intermediate_ratio)
        
        if params_per_layer <= 0 or params_in_layers <= 0:
            continue

        num_layers = round(params_in_layers / params_per_layer)
        
        if num_layers <= 0:
            continue
            
        # Calculate actual params with this suggested config
        calculated_params = calculate_params(
            vocab_size=vocab_size,
            hidden_size=h,
            num_hidden_layers=int(num_layers),
            intermediate_size=h * intermediate_ratio
        )['total_params']
        
        diff = abs(calculated_params - target_params)
        
        if diff < min_diff:
            min_diff = diff
            best_config = {
                'hidden_size': h,
                'num_hidden_layers': int(num_layers),
                'intermediate_size': h * intermediate_ratio,
                'num_attention_heads': h // 64, # A common heuristic for head dimension of 64
                'calculated_params': calculated_params
            }
            
    return best_config

def main():
    parser = argparse.ArgumentParser(description="Calculate model parameters for InfinityFormer.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Subparser for calculating params from a given config ---
    calc_parser = subparsers.add_parser("calculate", help="Calculate parameters for a specific configuration.")
    calc_parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size (e.g., Qwen2 uses ~152k).")
    calc_parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size (dimension of embeddings)." )
    calc_parser.add_argument("--num_hidden_layers", type=int, required=True, help="Number of transformer layers.")
    calc_parser.add_argument("--intermediate_size", type=int, required=True, help="Size of the MLP intermediate layer.")
    
    # --- Subparser for finding a config for a target size ---
    find_parser = subparsers.add_parser("find", help="Find a configuration for a target parameter size.")
    find_parser.add_argument("--target_params", type=float, required=True, help="Target number of parameters (e.g., 500e6 for 500M)." )
    find_parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size (e.g., Qwen2 uses ~152k).")

    args = parser.parse_args()

    if args.command == "calculate":
        params = calculate_params(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            intermediate_size=args.intermediate_size
        )
        print("--- Parameter Calculation ---")
        print(f"  Embedding Layer: {params['embedding_params']:,} parameters")
        print(f"  Attention Blocks:  {params['attention_params']:,} parameters")
        print(f"  MLP Blocks:        {params['mlp_params']:,} parameters")
        print("-----------------------------")
        print(f"  Total Parameters:  {params['total_params']:,} ( ~{params['total_params']/1e6:.2f}M )")

    elif args.command == "find":
        config = find_config_for_target_params(args.target_params, args.vocab_size)
        print(f"--- Suggested Config for ~{args.target_params/1e6:.0f}M Parameters ---")
        if config:
            print(f"  hidden_size:         {config['hidden_size']}")
            print(f"  num_hidden_layers:   {config['num_hidden_layers']}")
            print(f"  intermediate_size:   {config['intermediate_size']}")
            print(f"  num_attention_heads: {config['num_attention_heads']}")
            print("-------------------------------------------------")
            print(f"  Calculated Params:   {config['calculated_params']:,} ( ~{config['calculated_params']/1e6:.2f}M )")
            print("\nNote: This is an approximation. The actual number may vary slightly based on layer norms, biases, etc.")
        else:
            print("Could not find a suitable configuration with the given constraints.")

if __name__ == "__main__":
    main()
