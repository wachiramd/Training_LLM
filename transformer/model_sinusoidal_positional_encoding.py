import torch
import torch.nn as nn
import math

from typing import Optional, Tuple
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd: int, num_heads: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        head_size = n_embd // n_head
        error_message = f"n_embd {n_embd} must be divisible by n_head {n_head}"
        assert head_size * n_head == n_embd, error_message
        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        device: str,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device
        self.n_embd = n_embd

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.register_buffer(
            'positional_encoding',
            self._create_sinusoidal_encoding(block_size, n_embd)
        )

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.to(device)

    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Creates sinusoidal positional encoding matrix."""
        positional_encodings = torch.zeros(max_len, d_model)

        # div_term = exp(arange(0, d_model, 2) * -(log(10000.0) / d_model))
        # This produces: div_term[i] = 1 / (10000 ** (2i/d_model))
        exponent = torch.arange(0, d_model, 2).float()
        scale = -math.log(10000.0) / d_model
        div_term = torch.exp(exponent * scale)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        positional_encodings[:, 0::2] = torch.sin(positions * div_term)
        positional_encodings[:, 1::2] = torch.cos(positions * div_term)

        # Shape: (max_len, d_model) -> (1, max_len, d_model)
        positional_encodings = positional_encodings.unsqueeze(0)
        return positional_encodings

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        # Shape: (B, T, n_embd)
        token_embedding = self.token_embedding_table(input_tokens)
        # Shape: (1, T, n_embd)
        positional_encoding = self.positional_encoding[:, :T, :].to(
            self.device)

        x = token_embedding + positional_encoding
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear_layer(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(
                logits, targets, ignore_index=self.ignore_index)

        return logits, loss

    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            # Crop input_tokens to the last block_size tokens
            # Sinusoidal encoding is applied inside the forward pass based on sequence length
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)  # (B, T+1)
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
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness (higher = more random).
            top_k: Limits generation to the top-k most likely tokens.
            top_p: Limits generation to tokens with cumulative probability <= top_p.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            # Crop input_tokens to the last block_size tokens
            cropped_input = input_tokens[:, -self.block_size:]
            # Forward pass to get logits
            logits, _ = self(cropped_input)
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature  # becomes (B, C)

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # Optionally apply nucleus sampling (top-p)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift everything right to keep the first token above the threshold
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # Scatter sorted tensors to original indexing
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                # Zero out probabilities for removed tokens
                probs[indices_to_remove] = 0.0
                # Renormalize probabilities
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence and continue
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)  # (B, T+1)

        return input_tokens


if __name__ == "__main__":
    # Example usage
    vocab_size = 16394
    embedding_size = 512
    number_of_heads = 8
    block_size = 1024
    number_of_blocks = 1
    dropout = 0.2
    head_size = embedding_size // number_of_heads
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_heads,
        block_size=block_size,
        n_layer=number_of_blocks,
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
