import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention with RPE bias"""

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', tril)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, head_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, T, C)
            head_bias: Relative position bias for this specific head (T, T)
        """
        _, T, _ = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # head_bias shape is (T, T). Unsqueeze to add batch dim for broadcasting -> (1, T, T).
        # (B, T, T) + (1, T, T) -> (B, T, T)
        weights = weights + head_bias.unsqueeze(0)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel, using Head """

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        head_size: int,
        block_size: int,
        dropout: float,
        max_relative_distance: int = 16
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_relative_distance = max_relative_distance
        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.num_buckets = 2 * self.max_relative_distance + 1
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets,
            self.num_heads
        )

    def _compute_relative_position_bias(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        """ Computes the relative position bias tensor for ALL heads. """
        query_positions = torch.arange(sequence_length, device=device)
        key_positions = torch.arange(sequence_length, device=device)

        # Shape (T, T)
        relative_position = key_positions[None, :] - query_positions[:, None]
        # Shift range to positive values [0, 2 * max_relative_distance]
        relative_indices = relative_position + self.max_relative_distance
        # Clamp to ensure indices are within the range [0, num_buckets - 1]
        relative_indices = torch.clamp(
            input=relative_indices,
            min=0,
            max=self.num_buckets - 1
        )
        # Lookup biases for all heads: (T, T) -> (T, T, num_heads)
        bias = self.relative_attention_bias(relative_indices)
        # (T, T, num_heads) -> (num_heads, T, T)
        bias = bias.permute(2, 0, 1)
        return bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        relative_bias = self._compute_relative_position_bias(T, x.device)

        head_outputs = []
        for i, head_module in enumerate(self.heads):
            # Slice shape: (T, T)
            head_bias_slice = relative_bias[i]
            head_output = head_module(x=x, head_bias=head_bias_slice)
            head_outputs.append(head_output)

        # Shape (B, T, num_heads * head_size) -> (B, T, n_embd)
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class Block(nn.Module):
    """ Transformer block using the RPE Attention """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float,
        max_relative_distance: int
    ) -> None:
        super().__init__()
        head_size = n_embd // n_head
        error_message = f"n_embd {n_embd} must be divisible by n_head {n_head}"
        assert head_size * n_head == n_embd, error_message

        self.self_attention = MultiHeadAttention(
            n_embd=n_embd,
            num_heads=n_head,
            head_size=head_size,
            block_size=block_size,
            dropout=dropout,
            max_relative_distance=max_relative_distance
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
        ignore_index: int = -100,
        max_relative_distance: int = 16
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout,
                max_relative_distance=max_relative_distance
            ) for _ in range(n_layer)
        ])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Avoid re-initializing RPE bias if desired (e.g., keep default zero init)
            # This check is basic; better ways might involve setting an attribute
            is_rpe_bias = (hasattr(module, 'weight') and
                           module.weight.shape == (2 * self.blocks[0].self_attention.max_relative_distance + 1,
                                                   self.blocks[0].self_attention.num_heads))
            if not is_rpe_bias:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        x = self.token_embedding_table(input_tokens)
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
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens

    def advanced_generation(
        self, input_tokens: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: Optional[int] = None, top_p: Optional[float] = None
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens


if __name__ == "__main__":
    vocab_size = 10000
    embedding_size = 128  # n_embd
    number_of_heads = 4  # n_head
    block_size = 64      # Max context length
    number_of_blocks = 2  # n_layer
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_relative_distance = 8

    print(f"Using device: {device}")
    print(f"Initializing GPTLanguageModel with {max_relative_distance=}")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_heads,
        block_size=block_size,
        n_layer=number_of_blocks,
        dropout=dropout,
        device=device,
        max_relative_distance=max_relative_distance
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    B = 2  # Batch size
    T = 30  # Sequence length (<= block_size)
    input_tokens = torch.randint(0, vocab_size, (B, T), device=device)

    print(f"\nTesting forward pass with input shape: {input_tokens.shape}")
    logits, loss = model(input_tokens, targets=input_tokens)
    if loss is not None:
        print(f"Loss: {loss.item():.4f}")
    else:
        print("Forward pass completed, no loss calculated.")
    print(f"Logits shape: {logits.shape}")

    print("\nTesting generation...")
    gen_input = input_tokens[:, :5]
    print(f"Generation input shape: {gen_input.shape}")
    generated_tokens = model.generate(gen_input, max_new_tokens=10)
    print(f"Generated tokens shape: {generated_tokens.shape}")
    print("Generated sequence example (first batch):\n",
          generated_tokens[0].tolist())
