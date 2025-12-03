import torch
import torch.nn as nn

from typing import Optional, Tuple
from torch.nn import functional as F


def apply_rotary_positional_embedding(x: torch.Tensor, theta_is: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor x.

    Args:
        x : Input tensor of shape (B, T, D).
        theta_is : Precomputed complex frequencies of shape (T, D//2).

    Returns:
        torch.Tensor: Tensor with RoPE applied, same shape as input x.
    """

    # Reshape x to separate real and imaginary parts (pairs of features)
    # Shape: (B, T, D)    -> (B, T, D//2, 2)
    x_combined = x.float().reshape(*x.shape[:-1], -1, 2)

    # Convert to complex numbers
    # Shape: (B, T, D//2)
    x_complex = torch.view_as_complex(x_combined)

    # Reshape theta_is for broadcasting
    # Shape: (T, D//2) -> (1, T, D//2)
    # Add batch dimension (dim=0)
    theta_is = theta_is.unsqueeze(0)

    # Apply rotation by multiplying complex numbers
    # Broadcasting works:
    # (B, T, D//2) * (1, T, D//2) -> (B, T, D//2)
    x_rotated = x_complex * theta_is.to(x_complex.device)

    # Convert back to real numbers
    # Shape: (B, T, D//2, 2)
    x_out = torch.view_as_real(x_rotated)

    # Reshape back to original input shape
    # Shape: (B, T, D)
    # Flatten the last two dimensions (D//2, 2) -> D
    x_out = x_out.flatten(start_dim=-2)
    return x_out.type_as(x)


class RotaryPositionalEmbedding:
    def __init__(
        self,
        embedding_size: int,
        block_size: int,
        base_frequency: int = 10000,
        device: Optional[torch.device] = None
    ) -> None:
        """
        Initializes the Rotary Positional Embedding.

        Args:
            embedding_size : Dimension of the head embeddings. Must be even.
            block_size : Maximum sequence length.
            base_frequency : The base_frequency value for frequency calculation (theta_i).
            device : Device to store the precomputed frequencies.
        """
        super().__init__()
        if embedding_size % 2 != 0:
            raise ValueError("Dimension must be even for RoPE.")

        self.embedding_size = embedding_size
        self.block_size = block_size
        self.base_frequency = base_frequency
        self.device = device

        # Precompute theta values for RoPE
        # For each i in [0, embedding_size/ 2 ):
        #   theta_i = 1 / (base_frequency^(2 * i / embedding_size))
        half_dimension = self.embedding_size // 2
        frequency_range = torch.arange(half_dimension, dtype=torch.float32)
        exponent = 2 * frequency_range / self.embedding_size
        denominator = torch.pow(self.base_frequency, exponent)
        # Shape: (embedding_size // 2,)
        theta_is = 1.0 / denominator

        # Calculate frequencies for each position: m * theta
        # Shape: (block_size, embedding_size / 2)
        position_indices = torch.arange(self.block_size, dtype=torch.float32)
        frequencies = torch.outer(position_indices, theta_is)

        # Calculate complex numbers in polar form: cos(m*theta) + i*sin(m*theta)
        # Shape: (block_size, embedding_size / 2)
        self.theta_is = torch.polar(torch.ones_like(frequencies), frequencies)

        if self.device:
            self.theta_is = self.theta_is.to(self.device)

    def get_theta_is(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """
        Returns the precomputed complex frequencies for a given sequence length.

        Args:
            sequence_length : The sequence length (T).
            device : Target device.

        Returns:
            Complex frequencies of shape (sequence_length, embedding_size / 2).
        """
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum precomputed length {self.block_size}")

        # Return the slice for the current sequence length T
        self.theta_is = self.theta_is.to(device)
        return self.theta_is[:sequence_length]


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        block_size: int,
        dropout: float,
        rotary_embeddings: RotaryPositionalEmbedding
    ) -> None:
        """
        Initialize a single attention head.
        Args:
            embedding_size : Embedding dimension.
            head_size : Size of the head (embedding_size / num_heads).
            block_size : Maximum sequence length.
            dropout : Dropout rate.
            rotary_embeddings : Instance of RotaryPositionalEmbedding for RoPE.
        """

        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rotary_embeddings = rotary_embeddings
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: (B, T, C) -> (Batch, Sequence length, Embedding Dimension)
        _, T, _ = x.shape
        k = self.key(x)   # (B, T, head size)
        q = self.query(x)  # (B, T, head size)

        # --- Apply RoPE ---
        # Get rotary embeddings for the current sequence length T
        # Shape: (T, head size // 2)
        theta_is = self.rotary_embeddings.get_theta_is(T, x.device)

        # Apply RoPE to q and k
        q = apply_rotary_positional_embedding(q, theta_is)
        k = apply_rotary_positional_embedding(k, theta_is)
        # --- End RoPE ---

        # Compute attention scores ("affinities")
        # (B, T, head size) @ (B, head size, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # Apply causal mask dynamically
        tril = torch.tril(torch.ones(T, T, device=x.device))
        weights = weights.masked_fill(tril == 0, float('-inf'))

        # Softmax and dropout
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of values
        # Shape: (B, T, head size)
        v = self.value(x)
        # (B, T, T) @ (B, T, head size) -> (B, T, head size)
        return weights @ v


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        dropout: float,
        rotary_embeddings: RotaryPositionalEmbedding
    ) -> None:
        """
        Initialize multiple attention heads.
        Args:
            embedding_size : Embedding dimension.
            num_heads : Number of attention heads.
            head_size : Size of each head (embedding_size / num_heads).
            block_size : Maximum sequence length.
            dropout : Dropout rate.
            rotary_embeddings : Instance of RotaryPositionalEmbedding for RoPE.
        """

        super().__init__()
        self.heads = nn.ModuleList([
            Head(
                embedding_size=embedding_size,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout,
                rotary_embeddings=rotary_embeddings
            )
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concatenate results from all heads
        # (B, T, num_heads * head_size) = (B, T, embedding_size)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.projection(out))


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
        self,
        embedding_size: int,
        dropout: float
    ) -> None:
        """
        Initialize the feed-forward layer.
        Args:
            embedding_size : Embedding dimension.
            dropout : Dropout rate.
        """

        super().__init__()
        self.expanding_factor = 4
        self.network = nn.Sequential(
            nn.Linear(embedding_size, self.expanding_factor * embedding_size),
            nn.ReLU(),
            nn.Linear(self.expanding_factor * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        embedding_size: int,
        number_of_heads: int,
        block_size: int,
        dropout: float,
        rotary_embeddings: RotaryPositionalEmbedding
    ) -> None:
        """
        Initialize a transformer block.
        Args:
            embedding_size : Embedding dimension.
            number_of_heads : Number of attention heads.
            block_size : Maximum sequence length.
            dropout : Dropout rate.
            rotary_embeddings : Instance of RotaryPositionalEmbedding for RoPE.
        """

        super().__init__()
        head_size = embedding_size // number_of_heads
        error_message = f"embedding_size {embedding_size} must be divisible by number_of_heads {number_of_heads}"
        assert head_size * number_of_heads == embedding_size, error_message
        self.self_attention = MultiHeadAttention(
            embedding_size=embedding_size,
            num_heads=number_of_heads,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
            rotary_embeddings=rotary_embeddings
        )
        self.feed_forward = FeedForward(embedding_size, dropout)
        self.layer_norm_1 = nn.LayerNorm(embedding_size)
        self.layer_norm_2 = nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply self-attention with RoPE (handled inside self_attention) + residual connection
        x = x + self.self_attention(self.layer_norm_1(x))
        return x + self.feed_forward(self.layer_norm_2(x))


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_size: int,
        number_of_heads: int,
        block_size: int,
        number_of_blocks: int,
        dropout: float,
        device: str,
        ignore_index: int = -100,
        rope_base_frequency: int = 10000
    ) -> None:
        """
        Initialize the GPT language model.
        Args:
            vocab_size : Size of the vocabulary.
            embedding_size : Embedding dimension.
            number_of_heads : Number of attention heads.
            block_size : Maximum sequence length.
            number_of_blocks : Number of transformer blocks.
            dropout : Dropout rate.
            device : Device to run the model on (e.g., 'cuda' or 'cpu').
            ignore_index : Index to ignore in loss calculation (default: -100).
            rope_base_frequency : Base frequency for RoPE (default: 10000).
        """

        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device
        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads

        if embedding_size % number_of_heads != 0:
            raise ValueError(
                "embedding_size must be divisible by number_of_heads")

        head_size = embedding_size // number_of_heads
        self.rotary_embeddings = RotaryPositionalEmbedding(
            embedding_size=head_size,
            block_size=block_size,
            base_frequency=rope_base_frequency,
            device=device
        )

        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.Sequential(*[
            Block(
                embedding_size=embedding_size,
                number_of_heads=number_of_heads,
                block_size=block_size,
                dropout=dropout,
                rotary_embeddings=self.rotary_embeddings
            )
            for _ in range(number_of_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(embedding_size)
        self.final_linear_layer = nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)
        # Move all modules and buffers (theta_is) to the device
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights for the model.
        Args:
            module : The module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape
        # Shape: (B, T, embedding_size)
        x = self.token_embedding_table(input_tokens)

        # Shape: (B, T, embedding_size)
        x = self.blocks(x)

        # Final normalization and linear layer
        # Shape: (B, T, embedding_size)
        x = self.final_layer_norm(x)
        # Shape: (B, T, vocab_size)
        logits = self.final_linear_layer(x)

        loss = None
        if targets is not None:
            # Shape: (B*T, C) where C = vocab_size
            B_logits, T_logits, C_logits = logits.shape
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            # Shape: (B*T)
            targets = targets.view(B * T)
            loss = F.cross_entropy(
                input=logits_for_loss,
                target=targets,
                ignore_index=self.ignore_index
            )

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens from the model.
        Args:
            input_tokens : Input tensor of shape (B, T).
            max_new_tokens : Number of new tokens to generate.
        Returns:
            Generated tokens of shape (B, T + max_new_tokens).
        """

        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)

            # Focus only on the last time step
            # Shape: (B, T, C) -> (B, C), where C = vocab_size
            logits = logits[:, -1, :]
            probabilities = F.softmax(logits, dim=-1)
            # Sample from the distribution
            # Shape: (B, C) -> (B, 1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
            # Append sampled index to the running sequence
            # Shape: (B, T) + (B, 1) -> (B, T+1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)

        return input_tokens

    def advanced_generation(
        self,
        input_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generates new tokens from the model with advanced sampling techniques.
        Args:
            input_tokens : Input tensor of shape (B, T).
            max_new_tokens : Number of new tokens to generate.
            temperature : Temperature for scaling logits.
            top_k : Number of top tokens to consider for sampling.
            top_p : Cumulative probability threshold for nucleus sampling.
        Returns:
            Generated tokens of shape (B, T + max_new_tokens).
        """

        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)

            # Pluck the logits at the final step and scale by desired temperature
            # Shape: (B, T, C) -> (B, C), where C = vocab_size
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                vocab_size = logits.size(-1)
                # Shape: (B, C) -> (B, k)
                top_k_values, _ = torch.topk(
                    input=logits,
                    k=min(top_k, vocab_size)
                )
                # Set other values to -infinity
                logits[logits < top_k_values[:, [-1]]] = -float('Inf')

            probabilities = F.softmax(logits, dim=-1)

            # Optionally apply nucleus sampling (top-p)
            if top_p is not None:
                sorted_probabilities, sorted_indices = torch.sort(
                    input=probabilities,
                    descending=True
                )
                cumulative_probabilities = torch.cumsum(
                    input=sorted_probabilities,
                    dim=-1
                )

                # Remove tokens with cumulative probability above the threshold (nucleus)
                sorted_indices_to_remove = cumulative_probabilities > top_p
                # Shift everything right to keep the first token above the threshold
                sorted_indices_to_remove[..., 1:] = \
                    sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = torch.zeros_like(
                    input=logits,
                    dtype=torch.bool
                )
                indices_to_remove.scatter_(
                    dim=1,
                    index=sorted_indices,
                    src=sorted_indices_to_remove
                )

                # Zero out probabilities for removed tokens
                probabilities[indices_to_remove] = 0.0
                # Renormalize probabilities
                probabilities /= probabilities.sum(dim=-1, keepdim=True)

            # Sample from the final distribution
            # Shape: (B, C) -> (B, 1)
            idx_next = torch.multinomial(input=probabilities, num_samples=1)

            # Append sampled index to the running sequence and continue
            # Shape: (B, T) + (B, 1) -> (B, T+1)
            input_tokens = torch.cat(tensors=(input_tokens, idx_next), dim=1)

        return input_tokens


if __name__ == "__main__":
    # Example usage
    vocab_size = 16394
    embedding_size = 512  # Must be divisible by number_of_heads
    number_of_heads = 1   # Example: 8 heads -> head_size = 32
    block_size = 1024
    number_of_blocks = 1
    dropout = 0.2
    head_size = embedding_size // number_of_heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    # print(f"Using device: {device}")

    error_message = f"head_size ({head_size}) must be even for RoPE to work as expected."
    assert head_size % 2 == 0, error_message

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        number_of_heads=number_of_heads,
        block_size=block_size,
        number_of_blocks=number_of_blocks,
        dropout=dropout,
        device=device
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    print(
        f"Model created with {embedding_size=}, {number_of_heads=}, head_size={embedding_size//number_of_heads}")

    # Create dummy input
    input_tokens = torch.randint(0, vocab_size, (2, 50), device=device)

    # Test forward pass
    # Use input as target for testing shape
    logits, loss = model(input_tokens, targets=input_tokens)
    if loss is not None:
        print("Loss:", loss.item())

    # Test generation
    print("Generating...")
    # Start generation from first 10 tokens
    generated_tokens = model.generate(input_tokens[:, :10], max_new_tokens=20)
    print("Generated tokens shape:", generated_tokens.shape)
    print("Generated sequence example (first batch):\n",
          generated_tokens[0].tolist())

    # Test advanced generation
    print("\nAdvanced Generating (top_k=5, temp=0.8)...")
    generated_tokens_adv = model.advanced_generation(
        input_tokens[:, :10],
        max_new_tokens=20,
        temperature=0.8,
        top_k=10
    )
    print("Generated tokens shape (adv):", generated_tokens_adv.shape)
    print("Generated sequence example (adv, first batch):\n",
          generated_tokens_adv[0].tolist())
