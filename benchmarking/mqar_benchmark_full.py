import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from infinityformer.model.model import InfinityFormerForCausalLM
from infinityformer.config import InfinityFormerConfig
import math

class MQARDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=256, vocab_size=10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.samples = []
        
        # Generate samples
        for _ in range(num_samples):
            # Create pairs of key-value pairs
            pairs = np.random.randint(1, vocab_size, size=(seq_len // 4, 2))
            # Create sequence: key1, value1, key2, value2, ...
            sequence = np.zeros(seq_len, dtype=np.int64)
            sequence[::4] = pairs[:, 0]  # keys
            sequence[1::4] = pairs[:, 1]  # values
            # Create targets: same as sequence but shifted by 1
            targets = np.roll(sequence, -1)
            targets[-1] = 0  # Last token target is padding
            
            self.samples.append((sequence, targets))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence, targets = self.samples[idx]
        return {
            'input_ids': torch.tensor(sequence),
            'labels': torch.tensor(targets)
        }

def compute_accuracy(logits, targets, value_positions):
    """Compute accuracy only on value positions."""
    # Select only value positions
    logits_at_values = logits[:, value_positions, :]
    targets_at_values = targets[:, value_positions]
    
    predictions = logits_at_values.argmax(dim=-1)
    correct = (predictions == targets_at_values).float().mean().item()
    return correct

def main():
    # Configuration
    config = InfinityFormerConfig(
        vocab_size=10000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=1,
        intermediate_size=1024,
        max_position_embeddings=256,
        memory_compression_ratio=0.5,
        memory_compression_frequency=10,
        kernel_type='elu',
        kernel_epsilon=1e-6
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model
    model = InfinityFormerForCausalLM(config)
    model.to(device)

    # Dataset
    train_dataset = MQARDataset(num_samples=10000, seq_len=256)
    
    # Dataloader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 19
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_ids'].to(device)
            targets = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(inputs, labels=targets)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Compute epoch metrics
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save best model
        if epoch % 10 == 0:
            save_path = f"mqar_model_epoch_{epoch}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
