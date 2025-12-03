import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization base on the following implementation:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L34-L77
    """

    def __init__(self, dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        # The gain parameter, initialized to 1
        self.gain = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        output = self._norm(x.float()).to(original_dtype)
        return output * self.gain


class DeepSeekMLAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2, simplified without RoPE.
    """

    def __init__(
        self,
        n_embd: int,
        num_heads: int,
        q_compression_dim: int,
        kv_compression_dim: int,
        dropout: float,
        epsilon: float = 1e-6
    ) -> None:
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.n_embd = n_embd

        # --- Query Compression Path ---
        self.W_dq = nn.Linear(n_embd, q_compression_dim, bias=False)
        self.q_norm = RMSNorm(q_compression_dim, epsilon=epsilon)
        self.W_uq = nn.Linear(q_compression_dim, n_embd, bias=False)

        # --- Key-Value Joint Compression Path ---
        self.W_dkv = nn.Linear(n_embd, kv_compression_dim, bias=False)
        self.kv_norm = RMSNorm(kv_compression_dim, epsilon=epsilon)
        self.W_uk = nn.Linear(kv_compression_dim, n_embd, bias=False)
        self.W_uv = nn.Linear(kv_compression_dim, n_embd, bias=False)

        self.W_o = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout_p = dropout
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # 1. Query Path
        compressed_q_latent = self.W_dq(x)
        compressed_q_latent_norm = self.q_norm(compressed_q_latent)
        q_final = self.W_uq(compressed_q_latent_norm)

        # 2. Key-Value Path
        compressed_kv_latent = self.W_dkv(x)
        compressed_kv_latent_norm = self.kv_norm(compressed_kv_latent)
        k_final = self.W_uk(compressed_kv_latent_norm)
        v_final = self.W_uv(compressed_kv_latent_norm)

        # 3. Reshape Q, K, V
        q_heads = q_final.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k_heads = k_final.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v_heads = v_final.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # 4. Scaled Dot-Product Attention
        y_heads = F.scaled_dot_product_attention(
            query=q_heads, key=k_heads, value=v_heads,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True
        )

        # 5. Concatenate heads and apply final projection
        y_concat = y_heads.transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(y_concat)
        output = self.output_dropout(output)
        return output


class FeedForward(nn.Module):
    """
    A Feed-Forward Network module using SwiGLU activation.
    This structure is inspired by models like Llama.
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L307-L348
    """

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()

        # Calculate the intermediate hidden dimension for SwiGLU.
        # The original FFN expands n_embd to 4 * n_embd.
        # To maintain a similar parameter count with SwiGLU's 3 linear layers,
        # the intermediate hidden dimension is set to (2/3) of that expansion.
        ffn_expansion_factor = 4
        hidden_dim_swiglu = int((ffn_expansion_factor * n_embd) * (2/3))

        self.w_gate = nn.Linear(n_embd, hidden_dim_swiglu, bias=False)
        self.w_up = nn.Linear(n_embd, hidden_dim_swiglu, bias=False)
        self.w_down = nn.Linear(hidden_dim_swiglu, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, hidden_dim_swiglu)
        gate_value = self.w_gate(x)
        up_value = self.w_up(x)
        activated_value = F.silu(gate_value) * up_value

        # (B, T, n_embd)
        output = self.w_down(activated_value)
        output = self.dropout(output)
        return output


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        q_compression_dim: int,
        kv_compression_dim: int,
        dropout: float,
        epsilon: float = 1e-6
    ) -> None:
        super().__init__()
        self.self_attention = DeepSeekMLAttention(
            n_embd=n_embd,
            num_heads=n_head,
            q_compression_dim=q_compression_dim,
            kv_compression_dim=kv_compression_dim,
            dropout=dropout,
            epsilon=epsilon
        )
        self.feed_forward = FeedForward(n_embd, dropout)
        self.norm1 = RMSNorm(n_embd, epsilon=epsilon)
        self.norm2 = RMSNorm(n_embd, epsilon=epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Normalization
        x_norm1 = self.norm1(x)
        attn_out = self.self_attention(x_norm1)
        x = x + attn_out

        x_norm2 = self.norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + ff_out
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
        q_compression_dim: int,
        kv_compression_dim: int,
        epsilon: float = 1e-6,
        ignore_index: int = -100
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.block_size = block_size
        self.device = device

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(
                n_embd=n_embd,
                n_head=n_head,
                q_compression_dim=q_compression_dim,
                kv_compression_dim=kv_compression_dim,
                dropout=dropout,
                epsilon=epsilon
            )
            for _ in range(n_layer)
        ])

        self.final_norm = RMSNorm(n_embd, epsilon=epsilon)
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
        positional_indices = torch.arange(T, device=self.device)
        positional_embedding = self.position_embedding_table(positional_indices)
        x = token_embedding + positional_embedding
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.final_linear_layer(x)

        loss = None
        if targets is not None:
            B_logits, T_logits, C_logits = logits.shape
            logits_for_loss = logits.view(B_logits * T_logits, C_logits)
            targets_for_loss = targets.view(B_logits * T_logits)
            loss = F.cross_entropy(
                logits_for_loss, targets_for_loss, ignore_index=self.ignore_index)
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
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0.0
                probs = probs / probs.sum(dim=-1, keepdim=True)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens


if __name__ == "__main__":
    vocab_size = 16394
    embedding_size = 512
    number_of_heads = 8
    block_s = 1024
    number_of_layers = 1
    dropout_rate = 0.1
    head_dim = embedding_size // number_of_heads
    kv_comp_dim = 4 * head_dim
    q_comp_dim = embedding_size // 2
    eps_norm_val = 1e-5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Initializing model with DeepSeekMLAttention (internal RMSNorm), SwiGLU FFN, and block-level RMSNorm (Pure RMSNorm model)")
    print(f"Query Compression Dim: {q_comp_dim}, KV Compression Dim: {kv_comp_dim}, RMSNorm Eps: {eps_norm_val}")

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=embedding_size,
        n_head=number_of_heads,
        block_size=block_s,
        n_layer=number_of_layers,
        dropout=dropout_rate,
        device=device,
        q_compression_dim=q_comp_dim,
        kv_compression_dim=kv_comp_dim,
        epsilon=eps_norm_val
    )
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size / 1e6:.2f}M parameters")

    test_seq_len = 64
    input_tokens = torch.randint(0, vocab_size, (2, test_seq_len), device=device)
    print("\nTesting forward pass...")
    try:
        logits, loss = model(input_tokens, targets=input_tokens)
        if loss is not None:
            print("Forward pass successful. Loss:", loss.item())
        else:
            print("Forward pass successful. No loss calculated.")
        print(f"Logits shape: ({input_tokens.shape[0]}, {input_tokens.shape[1]}, {vocab_size})")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback
        traceback.print_exc()
