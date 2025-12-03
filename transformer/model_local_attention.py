import torch
import torch.nn as nn

from typing import Optional, Tuple
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float, window_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.window_size = window_size

        if self.window_size > block_size:
            print(
                f"Warning: window_size ({self.window_size}) > block_size ({block_size}). Clamping window_size to block_size.")
            self.window_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch, Time (sequence length), Channels (embedding dimension)
        B, T, C = x.shape

        # Project input to Key, Query vectors
        # k: (B, T, head_size)
        k = self.key(x)
        # q: (B, T, head_size)
        q = self.query(x)

        # Compute attention scores (affinities)
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # Scale by 1/sqrt(dimension of key)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        # Create row and column indices
        # row_indices: (T, 1), e.g., [[0], [1], ..., [T-1]]
        row_indices = torch.arange(T, device=x.device).unsqueeze(1)
        # col_indices: (1, T), e.g., [[0, 1, ..., T-1]]
        col_indices = torch.arange(T, device=x.device).unsqueeze(0)

        # 1. Causal mask: Prevents attention to future tokens.
        # True where query index (row) >= key index (column).
        # causal_mask: (T, T)
        # Example for T=4:
        # [[T, F, F, F],
        #  [T, T, F, F],
        #  [T, T, T, F],
        #  [T, T, T, T]]
        causal_mask = row_indices >= col_indices

        # 2. Local window mask: Restricts attention to a local window around the current token.
        # True if key index (column) is within `window_size` of query index (row).
        # (q_idx - window_size + 1 <= k_idx)
        # local_mask: (T, T)
        # Example: T=4, window_size=2
        # local_mask (T=4, window_size=2) is:
        # [[T, T, T, T],
        #  [T, T, T, T],
        #  [F, T, T, T],
        #  [F, F, T, T]]
        local_mask = col_indices >= row_indices - self.window_size + 1

        # 3. Combine masks: Attention is allowed only if both causal and local conditions are met.
        # final_mask: (T, T)
        # Example for T=4, window_size=2:
        # [[T, F, F, F],
        #  [T, T, F, F],
        #  [F, T, T, F],
        #  [F, F, T, T]]
        final_mask = causal_mask & local_mask

        # Apply the combined mask to the attention scores.
        # Set scores to -infinity where final_mask is False (0).
        # weights: (B, T, T)
        weights = weights.masked_fill(final_mask == 0, float('-inf'))
        # Apply softmax to get attention probabilities.
        weights = F.softmax(weights, dim=-1)
        # Apply dropout to attention weights.
        weights = self.dropout(weights)

        # (B, T, head_size)
        v = self.value(x)

        # Perform weighted aggregation of Values.
        # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd: int, num_heads: int, head_size: int, block_size: int, dropout: float, window_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                block_size=block_size,
                dropout=dropout,
                window_size=window_size
            )
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, num_heads * head_size)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # (B, T, n_embd)
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

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, window_size: int) -> None:
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
            window_size=window_size
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class GPTLanguageModel(nn.Module):
    """ GPT-style Language Model """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        device: str,
        window_size: int,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device
        self.window_size = window_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, window_size)
            for _ in range(n_layer)
        ])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        """ Initializes weights for linear and embedding layers. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        token_embedding = self.token_embedding_table(input_tokens)
        positional_embedding = self.position_embedding_table(
            torch.arange(T, device=self.device))

        x = token_embedding + positional_embedding
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
        Generates new tokens sequentially using basic multinomial sampling.

        Args:
            input_tokens: The initial context tokens (B, T_initial).
            max_new_tokens: The maximum number of new tokens to generate.

        Returns:
            The input tokens concatenated with the generated tokens (B, T_initial + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat(
                (input_tokens, idx_next), dim=1)
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
        Generates new tokens using advanced sampling techniques (temperature, top-k, top-p).

        Args:
            input_tokens: The initial context tokens (B, T_initial).
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: Controls randomness (higher = more random, lower = more deterministic).
            top_k: Limits sampling to the k most likely tokens.
            top_p: Limits sampling to tokens whose cumulative probability exceeds p (nucleus sampling).

        Returns:
            The input tokens concatenated with the generated tokens (B, T_initial + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop input context to the maximum block size
            cropped_input = input_tokens[:, -self.block_size:]
            # Get model predictions (logits)
            logits, _ = self(cropped_input)
            # Focus on the logit for the last token and apply temperature scaling
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # --- Top-k Sampling ---
            if top_k is not None:
                # Get the values of the top-k logits
                v, _ = torch.topk(logits, min(
                    top_k, logits.size(-1)))  # v shape (B, k)
                # Set logits lower than the k-th highest logit to negative infinity
                # This effectively removes them from consideration in softmax
                # Use [-1] to get the k-th value
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)

            # --- Top-p (Nucleus) Sampling ---
            if top_p is not None:
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True)  # Both (B, vocab_size)
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(
                    sorted_probs, dim=-1)  # (B, vocab_size)
                # Find indices where cumulative probability exceeds top_p
                # We want to *keep* indices *up to* the point where cumulative sum exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the removal mask right by one: keep the first token that pushes sum over top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                # Always keep the most probable token
                sorted_indices_to_remove[..., 0] = 0

                # Create a mask to remove tokens based on their original positions
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                # Set probabilities of tokens to remove to 0
                probs[indices_to_remove] = 0.0
                # Renormalize the probabilities so they sum to 1 again
                probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample the next token index based on the (potentially modified) probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append the sampled token index to the running sequence
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
    dropout = 0.1
    window_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print(f"Local Attention Window Size: {window_size}")

    head_size = embedding_size // number_of_heads
    if embedding_size % number_of_heads != 0:
        raise ValueError("embedding_size must be divisible by number_of_heads")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_heads,
        block_size=block_size,
        n_layer=number_of_blocks,
        dropout=dropout,
        device=device,
        window_size=window_size
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    print(
        f"Model created with {embedding_size=}, {number_of_heads=}, head_size={head_size}, {window_size=}")

    # Create dummy input (Batch size = 2, Sequence length = 50)
    input_tokens = torch.randint(0, vocab_size, (2, 50), device=device)

    print("\nTesting forward pass...")
    try:
        logits, loss = model(input_tokens, targets=input_tokens)
        if loss is not None:
            print("Forward pass successful. Loss:", loss.item())
        else:
            print("Forward pass successful. No loss calculated (targets=None).")
        # Expected: (B*T, C) -> (2*50, 16394)
        print("Logits shape:", logits.shape)
    except Exception as e:
        print(f"Error during forward pass: {e}")

    print("\nTesting generation...")
    try:
        # Start generation from first 10 tokens
        context = input_tokens[:, :10]
        generated_tokens = model.generate(context, max_new_tokens=20)
        print("Generation successful.")
        # Expected: (B, 10 + 20) -> (2, 30)
        print("Generated tokens shape:", generated_tokens.shape)
        print("Generated sequence example (first batch):\n",
              generated_tokens[0].tolist())
    except Exception as e:
        print(f"Error during generation: {e}")

    print("\nTesting advanced generation (top_k=10, temp=0.8)...")
    try:
        context = input_tokens[:, :10]
        generated_tokens_adv = model.advanced_generation(
            context,
            max_new_tokens=20,
            temperature=0.8,
            top_k=10
        )
        print("Advanced generation successful.")
        # Expected: (2, 30)
        print("Generated tokens shape (adv):", generated_tokens_adv.shape)
        print("Generated sequence example (adv, first batch):\n",
              generated_tokens_adv[0].tolist())
    except Exception as e:
        print(f"Error during advanced generation: {e}")
