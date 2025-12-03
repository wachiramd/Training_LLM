from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_positional_embedding(
    x: torch.Tensor, theta_is: torch.Tensor
) -> torch.Tensor:
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
        n_embd: int,
        block_size: int,
        base_frequency: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initializes the Rotary Positional Embedding.

        Args:
            n_embd : Dimension of the head embeddings. Must be even.
            block_size : Maximum sequence length.
            base_frequency : The base_frequency value for frequency calculation (theta_i).
            device : Device to store the precomputed frequencies.
        """
        super().__init__()
        if n_embd % 2 != 0:
            raise ValueError("Dimension must be even for RoPE.")

        self.n_embd = n_embd
        self.block_size = block_size
        self.base_frequency = base_frequency
        self.device = device

        # Precompute theta values for RoPE
        # For each i in [0, n_embd/ 2 ):
        #   theta_i = 1 / (base_frequency^(2 * i / n_embd))
        half_dimension = self.n_embd // 2
        frequency_range = torch.arange(half_dimension, dtype=torch.float32)
        exponent = 2 * frequency_range / self.n_embd
        denominator = torch.pow(self.base_frequency, exponent)
        # Shape: (n_embd // 2,)
        theta_is = 1.0 / denominator

        # Calculate frequencies for each position: m * theta
        # Shape: (block_size, n_embd / 2)
        position_indices = torch.arange(self.block_size, dtype=torch.float32)
        frequencies = torch.outer(position_indices, theta_is)

        # Calculate complex numbers in polar form: cos(m*theta) + i*sin(m*theta)
        # Shape: (block_size, n_embd / 2)
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
            Complex frequencies of shape (sequence_length, n_embd / 2).
        """
        if sequence_length > self.block_size:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds maximum precomputed length {self.block_size}"
            )

        # Return the slice for the current sequence length T
        self.theta_is = self.theta_is.to(device)
        return self.theta_is[:sequence_length]


class DeepSeekMLAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2, simplified without RoPE.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        q_compression_dim: int,
        kv_compression_dim: int,
        dropout: float,
        rotary_embeddings: RotaryPositionalEmbedding,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.n_embd = n_embd
        self.rotary_embeddings = rotary_embeddings

        # --- Query Compression Path ---
        # Down-projection for query
        self.W_dq = nn.Linear(n_embd, q_compression_dim, bias=False)
        # LayerNorm after query down-projection
        self.q_layer_norm = nn.LayerNorm(q_compression_dim)
        # Up-projection for query
        self.W_uq = nn.Linear(q_compression_dim, n_embd, bias=False)

        # --- Key-Value Joint Compression Path ---
        # Down-projection for key-value
        self.W_dkv = nn.Linear(n_embd, kv_compression_dim, bias=False)
        # LayerNorm after key-value down-projection
        self.kv_layer_norm = nn.LayerNorm(kv_compression_dim)
        # Up-projection for key (from compressed KV)
        self.W_uk = nn.Linear(kv_compression_dim, n_embd, bias=False)
        # Up-projection for value (from compressed KV)
        self.W_uv = nn.Linear(kv_compression_dim, n_embd, bias=False)

        # Output projection
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)

        # Dropout for attention probabilities (used in F.scaled_dot_product_attention)
        self.attn_dropout = dropout
        # Dropout for the final output of the attention module
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch size, Sequence length, Embedding dimensionality
        B, T, C = x.shape

        # 1. Query Path
        # (B, T, C) -> (B, T, q_compression_dim)
        compressed_q_latent = self.W_dq(x)
        compressed_q_latent_norm = self.q_layer_norm(compressed_q_latent)
        # (B, T, q_compression_dim) -> (B, T, C)
        q_final = self.W_uq(compressed_q_latent_norm)

        # 2. Key-Value Path
        # (B, T, C) -> (B, T, kv_compression_dim)
        compressed_kv_latent = self.W_dkv(x)
        compressed_kv_latent_norm = self.kv_layer_norm(compressed_kv_latent)
        # (B, T, kv_compression_dim) -> (B, T, C)
        k_final = self.W_uk(compressed_kv_latent_norm)
        # (B, T, kv_compression_dim) -> (B, T, C)
        v_final = self.W_uv(compressed_kv_latent_norm)

        # 3. Apply Rotary Positional Embedding to Q, K
        theta_is = self.rotary_embeddings.get_theta_is(T, x.device)
        q_rope = apply_rotary_positional_embedding(q_final, theta_is)
        k_rope = apply_rotary_positional_embedding(k_final, theta_is)

        # 4. Reshape Q, K, V for multi-head attention
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        q_heads = q_rope.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k_heads = k_rope.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v_heads = v_final.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # 5. Scaled Dot-Product Attention
        # F.scaled_dot_product_attention handles softmax, scaling, and causal masking.
        y_heads = F.scaled_dot_product_attention(
            query=q_heads,
            key=k_heads,
            value=v_heads,
            attn_mask=None,  # Not needed when is_causal=True for causal language modeling
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        # 6. Concatenate heads and apply final projection
        # (B, n_head, T, head_size) -> (B, T, n_head, head_size) -> (B, T, C)
        y_concat = y_heads.transpose(1, 2).contiguous().view(B, T, C)

        # (B, T, C)
        output = self.W_o(y_concat)
        output = self.output_dropout(output)
        return output


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

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
    """Transformer block using DeepSeekMLAttention"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        q_compression_dim: int,
        kv_compression_dim: int,
        dropout: float,
        rotary_embeddings: RotaryPositionalEmbedding,
    ) -> None:
        super().__init__()
        self.self_attention = DeepSeekMLAttention(
            n_embd=n_embd,
            n_head=n_head,
            q_compression_dim=q_compression_dim,
            kv_compression_dim=kv_compression_dim,
            dropout=dropout,
            rotary_embeddings=rotary_embeddings,
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attention(x)
        x = x + attn_out
        x_normed_after_attn = self.layer_norm_1(x)

        ff_out = self.feed_forward(x_normed_after_attn)
        x = x_normed_after_attn + ff_out
        output_of_block = self.layer_norm_2(x)
        return output_of_block


class GPTLanguageModel(nn.Module):
    """GPT-style Language Model using DeepSeekMLAttention"""

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        n_head: int,
        block_size: int,
        n_layer: int,
        dropout: float,
        device: str,
        q_compression_dim: int,
        kv_compression_dim: int,
        ignore_index: int = -100,
        rope_base_frequency: int = 10000,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.rotary_embeddings = RotaryPositionalEmbedding(
            n_embd=n_embd,
            block_size=block_size,
            base_frequency=rope_base_frequency,
            device=device,
        )

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd=n_embd,
                    n_head=n_head,
                    q_compression_dim=q_compression_dim,
                    kv_compression_dim=kv_compression_dim,
                    dropout=dropout,
                    rotary_embeddings=self.rotary_embeddings,
                )
                for _ in range(n_layer)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.final_linear_layer = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes weights for linear and embedding layers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_tokens: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_tokens.shape

        x = self.token_embedding_table(input_tokens)
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear_layer(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index)

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
            cropped_input = input_tokens[:, -self.block_size :]
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
        top_p: Optional[float] = None,
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
            cropped_input = input_tokens[:, -self.block_size :]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)

            if top_p is not None:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    1, sorted_indices, sorted_indices_to_remove
                )
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens


if __name__ == "__main__":
    vocab_size = 16394
    n_embd = 512
    n_head = 8
    block_size = 1024
    n_layer = 1
    dropout_rate = 0.1

    # --- MLA Parameters ---
    head_dim = n_embd // n_head
    kv_compression_dim = 4 * head_dim
    q_compression_dim = n_embd // 2
    # ----------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print("Initializing model with DeepSeek MLA")
    print(
        f"Query Compression Dim: {q_compression_dim}, KV Compression Dim: {kv_compression_dim}"
    )

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        block_size=block_size,
        n_layer=n_layer,
        dropout=dropout_rate,
        device=device,
        q_compression_dim=q_compression_dim,
        kv_compression_dim=kv_compression_dim,
    )

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")
    print(f"Model created with {n_embd=}, {n_head=}, head_size={head_dim}")
    print(
        f"MLA params: q_compression_dim={q_compression_dim}, kv_compression_dim={kv_compression_dim}"
    )

    test_seq_len = 64
    input_tokens = torch.randint(0, vocab_size, (2, test_seq_len), device=device)

    print("\nTesting forward pass...")
    try:
        logits, loss = model(input_tokens, targets=input_tokens)
        if loss is not None:
            print("Forward pass successful. Loss:", loss.item())
        else:
            print("Forward pass successful. No loss calculated.")
        print(
            f"Logits shape: ({input_tokens.shape[0]}, {input_tokens.shape[1]}, {vocab_size})"
        )
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
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback

        traceback.print_exc()
