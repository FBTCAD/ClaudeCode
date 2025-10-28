"""
Transformer Layer Implementation

This module implements a standard transformer layer with multi-head self-attention
and feed-forward network, following the architecture from "Attention is All You Need".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            Q: Query tensor (batch_size, num_heads, seq_len, d_k)
            K: Key tensor (batch_size, num_heads, seq_len, d_k)
            V: Value tensor (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask tensor

        Returns:
            attention_output: Weighted values
            attention_weights: Attention weights
        """
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            output: Multi-head attention output (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear projection
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(attention_output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Args:
        d_model: Dimension of the model
        d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for feed-forward network.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)

        Returns:
            output: Feed-forward output (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerLayer(nn.Module):
    """
    A single Transformer layer (encoder block).

    This layer consists of:
    1. Multi-head self-attention mechanism
    2. Add & Norm (residual connection + layer normalization)
    3. Position-wise feed-forward network
    4. Add & Norm (residual connection + layer normalization)

    Args:
        d_model: Dimension of the model (embedding dimension)
        num_heads: Number of attention heads
        d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the transformer layer.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            output: Transformer layer output (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attention_output))

        # Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Hyperparameters
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # Create transformer layer
    transformer_layer = TransformerLayer(d_model, num_heads, d_ff, dropout)

    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass
    output = transformer_layer(x)

    print("Transformer Layer Test")
    print("=" * 50)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in transformer_layer.parameters())}")
    print("\nTest passed successfully!")
