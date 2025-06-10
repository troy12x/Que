# InfinityFormer: Linear Complexity Transformer with Unlimited Context

InfinityFormer is a transformer architecture designed for linear time/space complexity and unlimited context length. It achieves this through a combination of kernel-based linear attention, multi-scale recurrent memory, and adaptive gating mechanisms.

## Key Features

- **Linear Complexity**: O(N) time and space complexity with respect to sequence length
- **Unlimited Context**: Handles arbitrarily long sequences without fixed window size
- **Multi-Scale Memory**: Captures patterns at different timescales with adaptive decay
- **Kernel-Based Attention**: Efficient attention approximation using feature maps
- **Rotary Positional Encodings**: Relative position information with RoPE
- **Adaptive Gating**: Learns to balance between memory and current context

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/infinityformer.git
cd infinityformer
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the InfinityFormer model on your dataset:

```bash
python examples/train_infinityformer.py \
    --train_file /path/to/train.txt \
    --validation_file /path/to/validation.txt \
    --output_dir /path/to/save/model \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 1024 \
    --use_rotary_embeddings \
    --use_multi_scale_memory \
    --num_memory_scales 3 \
    --kernel_type elu
```

### Inference

To generate text using a trained model:

```python
from infinityformer import InfinityFormerForCausalLM
from transformers import AutoTokenizer
import torch

# Load model and tokenizer
model = InfinityFormerForCausalLM.from_pretrained("/path/to/save/model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # or your custom tokenizer

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

## Architecture Overview

InfinityFormer consists of the following key components:

1. **Kernel-Based Linear Attention**: Approximates full attention using feature maps for O(N) complexity
2. **Multi-Scale Memory**: Maintains multiple recurrent memories with different timescales
3. **Rotary Position Embeddings (RoPE)**: Provides relative position information
4. **Adaptive Gating**: Learns to balance between memory and current context
5. **Memory Compression**: Periodically compresses memory to maintain efficiency

## Implementation Details

- **Kernel Functions**: Supports ELU, ReLU, and learnable kernel functions
- **Memory Management**: Implements efficient memory compression and decompression
- **Gradient Checkpointing**: Reduces memory usage during training
- **Mixed Precision Training**: Supports FP16 training for faster training

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.0+
- tqdm
- numpy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use InfinityFormer in your research, please cite:

```bibtex
@misc{infinityformer2023,
  author = {Your Name},
  title = {InfinityFormer: Linear Complexity Transformer with Unlimited Context},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/infinityformer}}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
