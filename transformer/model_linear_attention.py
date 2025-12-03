import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LinearAttention(nn.Module):
    """ Causal Linear Attention module """

    def __init__(self, n_embd: int, num_heads: int, dropout: float, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        assert self.head_size * num_heads == n_embd, "n_embd must be divisible by num_heads"

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.epsilon = epsilon  # For numerical stability in division

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch size, Sequence length, Embedding dimensionality
        B, T, C = x.shape

        # 1. Project to Q, K, V
        # Q, K, V are all (B, T, C)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 2. Reshape for multi-head attention
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        q_h = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k_h = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v_h = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # 3. Apply feature map to Q and K
        phi_q_h = self._feature_map(q_h)
        phi_k_h = self._feature_map(k_h)

        # 4. Causal Linear Attention calculation
        # Based on Katharopoulos et al. (2020) "Transformers are RNNs"
        # Attention_i = (phi(Q_i)^T * sum_{j=1 to i} (phi(K_j) * V_j^T) ) / (phi(Q_i)^T * sum_{j=1 to i} phi(K_j) + epsilon)
        # Let S_i_prime = sum_{j=1 to i} (phi(K_j) * V_j^T) (matrix of size head_size x head_size)
        # Let Z_i_prime = sum_{j=1 to i} phi(K_j) (vector of size head_size)

        # Calculate S_cumulative (cumulative sum of outer products phi_k * v^T)
        kv_outer_products = torch.matmul(
            phi_k_h.unsqueeze(-1),  # (B, num_heads, T, head_size, 1)
            v_h.unsqueeze(-2)  # (B, num_heads, T, 1, head_size)
        )
        # (B, num_heads, T, head_size, head_size)
        S_cumulative = torch.cumsum(kv_outer_products, dim=2)

        # Calculate Z_cumulative (cumulative sum of phi_k)
        # (B, num_heads, T, head_size)
        Z_cumulative = torch.cumsum(phi_k_h, dim=2)

        # Numerator: (phi(Q_i)^T * S_i_prime) for each i
        # (B, num_heads, T, head_size)
        numerator = torch.matmul(
            phi_q_h.unsqueeze(-2),  # (B, num_heads, T, 1, head_size)
            S_cumulative  # (B, num_heads, T, head_size, head_size)
        ).squeeze(-2)

        # Denominator: (phi(Q_i)^T * Z_i_prime + epsilon) for each i
        # (B, num_heads, T, 1)
        denominator = torch.sum(
            phi_q_h * Z_cumulative,  # (B, num_heads, T, head_size)
            dim=-1,
            keepdim=True
        ) + self.epsilon

        # (B, num_heads, T, head_size)
        y_h = numerator / denominator

        # 5. Concatenate heads and project
        # (B, num_heads, T, head_size) -> (B, T, num_heads, head_size) -> (B, T, C)
        y = y_h.transpose(1, 2).contiguous().view(B, T, C)
        y = self.projection(y)
        y = self.dropout(y)

        return y


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

    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = LinearAttention(
            n_embd=n_embd,
            num_heads=n_head,
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
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, dropout)
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
            B_logits, T_logits, C_logits = logits.shape
            logits = logits.view(B_logits * T_logits, C_logits)
            targets = targets.view(B_logits * T_logits)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")
    print("Initializing model with Linear Attention")

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
        device=device
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")
    print(
        f"Model created with {embedding_size=}, {number_of_heads=}, head_size={head_size}")

    # Create dummy input (Batch size = 2, Sequence length = 50)
    input_tokens = torch.randint(0, vocab_size, (2, 50), device=device)

    print("\nTesting forward pass...")
    try:
        logits, loss = model(input_tokens, targets=input_tokens)
        if loss is not None:
            print("Forward pass successful. Loss:", loss.item())
        else:
            print("Forward pass successful. No loss calculated (targets=None).")
        print(
            f"Logits shape (before reshape for loss): ({input_tokens.shape[0]}, {input_tokens.shape[1]}, {vocab_size})")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()

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
        import traceback
        traceback.print_exc()

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
        import traceback
        traceback.print_exc()
