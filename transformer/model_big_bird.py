import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Head(nn.Module):
    """ One head of Big Bird-inspired self-attention """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        dropout: float,
        window_size: int,
        num_global_tokens: int,
        num_random_tokens: int
    ) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        assert window_size >= 1, "window_size must be at least 1"
        assert num_global_tokens >= 0, "num_global_tokens cannot be negative"
        assert num_random_tokens >= 0, "num_random_tokens cannot be negative"

        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.num_random_tokens = num_random_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch, Time (sequence length), Channels (embedding dimension)
        B, T, C = x.shape

        # k and q are (B, T, head_size)
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores ("affinities")
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5

        device = x.device
        # Shape (T, 1)
        row_indices = torch.arange(T, device=device).unsqueeze(1)
        # Shape (1, T)
        col_indices = torch.arange(T, device=device).unsqueeze(0)

        # 1. Base Causal Mask: q_idx >= k_idx (token i can only attend to token j if j <= i)
        causal_mask = row_indices >= col_indices

        # 2. Local Window Mask Component:
        # Token i attends to token j if j is in [i - window_size + 1, i]
        condition_1 = col_indices >= row_indices - self.window_size + 1
        condition_2 = col_indices <= row_indices
        local_attention_mask = condition_1 & condition_2

        # 3. Global Mask Component:
        # Connection is allowed if query is global OR key is global.
        query_is_global = row_indices < self.num_global_tokens
        key_is_global = col_indices < self.num_global_tokens
        global_attention_mask_component = query_is_global | key_is_global

        # 4. Random Attention Mask Component:
        random_attention_mask = torch.zeros(
            size=(T, T),
            dtype=torch.bool,
            device=device
        )
        self.num_random_tokens = min(self.num_random_tokens, T)
        # Generate random column indices for each row (query token)
        # Shape: (T, num_random_tokens)
        random_cols = torch.randint(
            low=0,
            high=T,
            size=(T, self.num_random_tokens),
            device=device
        )

        # Shape: (T, num_random_tokens)
        row_selector = torch.arange(T, device=device)
        row_selector = row_selector.repeat_interleave(self.num_random_tokens)
        row_selector = row_selector.view(T, self.num_random_tokens)
        random_attention_mask[row_selector, random_cols] = True

        # Combine local and global and random components
        combined_mask = local_attention_mask | global_attention_mask_component | random_attention_mask

        final_mask = combined_mask & causal_mask
        weights = weights.masked_fill(final_mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # (B, T, head_size)
        v = self.value(x)
        out = weights @ v
        return out


class BigBirdInspiredAttention(nn.Module):
    """ Multiple heads of Big Bird-inspired self-attention in parallel """

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        dropout: float,
        window_size: int,
        num_global_tokens: int,
        num_random_tokens: int
    ) -> None:
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError("n_embd must be divisible by num_heads")

        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([
            Head(
                n_embd=n_embd,
                head_size=head_size,
                dropout=dropout,
                window_size=window_size,
                num_global_tokens=num_global_tokens,
                num_random_tokens=num_random_tokens
            )
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, n_embd)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

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

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        window_size: int,
        num_global_tokens: int,
        num_random_tokens: int
    ) -> None:
        super().__init__()
        self.self_attention = BigBirdInspiredAttention(
            n_embd=n_embd,
            num_heads=n_head,
            dropout=dropout,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            num_random_tokens=num_random_tokens
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
        num_global_tokens: int,
        num_random_tokens: int,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        assert num_global_tokens <= block_size, f"num_global_tokens ({num_global_tokens}) cannot exceed block_size ({block_size})"
        assert window_size <= block_size, f"window_size ({window_size}) cannot exceed block_size ({block_size})"

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                window_size=window_size,
                num_global_tokens=num_global_tokens,
                num_random_tokens=num_random_tokens
            )
            for _ in range(n_layer)
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
        Generates new tokens from the model.

        Args:
            input_tokens: The initial input tokens.
            max_new_tokens: The maximum number of tokens to generate.

        Returns:
            The generated tokens.
        """
        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size:]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
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
    # Example usage
    vocab_size = 16394
    embedding_size = 512
    number_of_heads = 8
    block_size = 1024
    number_of_layers = 1
    dropout_rate = 0.1

    window_s = 128
    num_global_tokens = 4
    num_random_tokens = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print("Initializing model with BigBird-Inspired Attention (Local + Global)")
    print(
        f"Local Window Size: {window_s}, Number of Global Tokens: {num_global_tokens}")

    head_size = embedding_size // number_of_heads
    if embedding_size % number_of_heads != 0:
        raise ValueError("embedding_size must be divisible by number_of_heads")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_heads,
        block_size=block_size,
        n_layer=number_of_layers,
        dropout=dropout_rate,
        device=device,
        window_size=window_s,
        num_global_tokens=num_global_tokens,
        num_random_tokens=num_random_tokens,
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")
    print(
        f"Model created with {embedding_size=}, {number_of_heads=}, head_size={head_size}")
    print(
        f"BigBird params: window_size={window_s}, num_global_tokens={num_global_tokens}")

    # Create dummy input (Batch size = 2, Sequence length = 50)
    test_seq_len = 64
    if num_global_tokens >= test_seq_len:
        print(
            f"Warning: num_global_tokens ({num_global_tokens}) >= test_seq_len ({test_seq_len}). Adjust for meaningful test.")

    input_tokens = torch.randint(
        0, vocab_size, (2, test_seq_len), device=device)

    print("\nTesting forward pass...")
    try:
        logits, loss = model(input_tokens, targets=input_tokens)
        if loss is not None:
            print("Forward pass successful. Loss:", loss.item())
        else:
            print("Forward pass successful. No loss calculated (targets=None).")
        print(
            f"Logits shape: ({input_tokens.shape[0]}, {input_tokens.shape[1]}, {vocab_size})")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting generation...")
    try:
        context_len = min(10, test_seq_len)
        context = input_tokens[:, :context_len]
        generated_tokens = model.generate(context, max_new_tokens=20)
        print("Generation successful.")
        print("Generated tokens shape:", generated_tokens.shape)
        print("Generated sequence example (first batch):\n",
              generated_tokens[0].tolist())
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting advanced generation (top_k=10, temp=0.8)...")
    try:
        context_len = min(10, test_seq_len)
        context = input_tokens[:, :context_len]
        generated_tokens_adv = model.advanced_generation(
            context,
            max_new_tokens=20,
            temperature=0.8,
            top_k=10
        )
        print("Advanced generation successful.")
        print("Generated tokens shape (adv):", generated_tokens_adv.shape)
        print("Generated sequence example (adv, first batch):\n",
              generated_tokens_adv[0].tolist())
    except Exception as e:
        print(f"Error during advanced generation: {e}")
        import traceback
        traceback.print_exc()
