# Transformer Layer Implementation

A clean, production-ready implementation of a Transformer layer in PyTorch.

## Features

This implementation includes:

- **Multi-Head Self-Attention**: Implements the scaled dot-product attention mechanism with multiple attention heads
- **Feed-Forward Network**: Position-wise fully connected feed-forward network
- **Layer Normalization**: Applied after each sub-layer
- **Residual Connections**: Skip connections around each sub-layer
- **Dropout**: Configurable dropout for regularization

## Architecture

The transformer layer follows the architecture from "Attention is All You Need" (Vaswani et al., 2017):

```
Input
  |
  +--> Multi-Head Attention --> Dropout --> Add & Norm
  |                                            |
  +--------------------------------------------+
  |
  +--> Feed-Forward Network --> Dropout --> Add & Norm
  |                                           |
  +-------------------------------------------+
  |
Output
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
import torch
from transformer_layer import TransformerLayer

# Initialize transformer layer
d_model = 512      # Model dimension
num_heads = 8      # Number of attention heads
d_ff = 2048        # Feed-forward dimension
dropout = 0.1      # Dropout rate

transformer = TransformerLayer(d_model, num_heads, d_ff, dropout)

# Create input tensor (batch_size, seq_len, d_model)
x = torch.randn(32, 100, 512)

# Forward pass
output = transformer(x)  # Shape: (32, 100, 512)
```

### With Attention Mask

```python
# Create attention mask (optional)
mask = torch.ones(32, 1, 100, 100)  # Allow all positions to attend
mask[:, :, :, 50:] = 0  # Mask out last 50 positions

# Forward pass with mask
output = transformer(x, mask=mask)
```

## Components

### MultiHeadAttention
- Implements multi-head scaled dot-product attention
- Splits input into multiple heads for parallel attention
- Includes linear projections for Q, K, V and output

### FeedForward
- Two-layer fully connected network
- Uses ReLU activation
- Typically uses d_ff = 4 * d_model

### TransformerLayer
- Complete transformer encoder block
- Combines attention and feed-forward with residual connections
- Includes layer normalization and dropout

## Testing

Run the included test:

```bash
python transformer_layer.py
```

Expected output:
```
Transformer Layer Test
==================================================
Input shape: torch.Size([2, 10, 512])
Output shape: torch.Size([2, 10, 512])
Number of parameters: 7,087,616
Test passed successfully!
```

## Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| d_model | Model dimension | 512 |
| num_heads | Number of attention heads | 8 |
| d_ff | Feed-forward dimension | 2048 (4 * d_model) |
| dropout | Dropout rate | 0.1 |

## License

MIT
