import torch
import torch.nn as nn

from typing import Optional, Tuple
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    """ Grouped-Query Attention (GQA) module """

    def __init__(self, n_embd: int, num_heads: int, block_size: int, dropout: float, num_kv_heads: Optional[int] = None) -> None:
        super().__init__()
        # If num_kv_heads is not specified, default to standard MHA (num_kv_heads = num_heads)
        # If specified as 1, it's MQA. Otherwise, it's GQA.
        if num_kv_heads is None:
            self.num_kv_heads = num_heads
        else:
            self.num_kv_heads = num_kv_heads

        if num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads})")

        self.num_query_heads = num_heads
        self.head_size = n_embd // num_heads
        self.num_queries_per_kv = self.num_query_heads // self.num_kv_heads

        assert self.head_size * num_heads == n_embd, "n_embd must equal num_heads * head_size"

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(
            n_embd,
            self.num_kv_heads * self.head_size,
            bias=False
        )
        self.value = nn.Linear(
            n_embd,
            self.num_kv_heads * self.head_size,
            bias=False
        )

        self.projection = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        tril = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer('tril', tril.view(1, 1, block_size, block_size))

    def _repeat_kv(self, x: torch.Tensor, repetition_times: int) -> torch.Tensor:
        """
        Repeats the heads of Key/Value tensors to match the number of query heads.
        Input: (B, num_kv_heads, T, head_size)
        Output: (B, num_query_heads, T, head_size)
        """
        if repetition_times == 1:
            return x

        x = x.repeat_interleave(repetition_times, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch size, Sequence length, Embedding dimensionality
        B, T, C = x.shape

        # (B, T, n_embd)
        q = self.query(x)
        # (B, num_query_heads, T, head_size)
        q = q.view(B, T, self.num_query_heads, self.head_size).transpose(1, 2)

        # (B, T, num_kv_heads * head_size)
        k = self.key(x)
        # (B, num_kv_heads, T, head_size)
        k = k.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        # (B, T, num_kv_heads * head_size)
        v = self.value(x)
        # (B, num_kv_heads, T, head_size)
        v = v.view(B, T, self.num_kv_heads, self.head_size).transpose(1, 2)

        # (B, num_query_heads, T, head_size)
        k = self._repeat_kv(k, self.num_queries_per_kv)
        v = self._repeat_kv(v, self.num_queries_per_kv)

        # Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
        # att_weights: (B, num_query_heads, T, head_size) @ (B, num_query_heads, head_size, T) -> (B, num_query_heads, T, T)
        att_weights = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)

        # Apply causal mask
        mask = self.tril[:, :, :T, :T] == 0
        att_weights = att_weights.masked_fill(mask, float('-inf'))
        att_weights = F.softmax(att_weights, dim=-1)
        att_weights = self.attn_dropout(att_weights)

        # (B, num_query_heads, T, head_size)
        y = att_weights @ v

        # Concatenate head outputs back together
        # (B, T, C=num_query_heads*head_size)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.projection(y))
        return y


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

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float, num_kv_heads: Optional[int] = None) -> None:
        super().__init__()
        self.self_attention = GroupedQueryAttention(
            n_embd=n_embd,
            num_heads=n_head,
            block_size=block_size,
            dropout=dropout,
            num_kv_heads=num_kv_heads
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
        num_kv_heads: Optional[int] = None,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else n_head

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout, num_kv_heads)
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
        """ Defines the forward pass of the model. """
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
    number_of_query_heads = 8
    number_of_kv_heads = 4
    block_size = 1024
    number_of_blocks = 1
    dropout = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print("Initializing model with Grouped-Query Attention (GQA)")
    print(
        f"Query Heads: {number_of_query_heads}, KV Heads: {number_of_kv_heads}")

    head_size = embedding_size // number_of_query_heads
    if embedding_size % number_of_query_heads != 0:
        raise ValueError(
            "embedding_size must be divisible by number_of_query_heads")
    if number_of_query_heads % number_of_kv_heads != 0:
        raise ValueError(
            f"number_of_query_heads ({number_of_query_heads}) must be divisible by number_of_kv_heads ({number_of_kv_heads})")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_query_heads,
        block_size=block_size,
        n_layer=number_of_blocks,
        dropout=dropout,
        device=device,
        num_kv_heads=number_of_kv_heads
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    print(
        f"Model created with {embedding_size=}, Query Heads={number_of_query_heads}, KV Heads={number_of_kv_heads}, head_size={head_size}")

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
