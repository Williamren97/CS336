from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.utils.nn import softmax
    
def scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(K.size(-1))
    if mask is not None:
        # Convention for this assignment/tests: mask == True means "keep",
        # mask == False means "mask out".
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    if pdrop is not None and pdrop > 0:
        attn = nn.functional.dropout(attn, p=pdrop, training=True)
    return torch.matmul(attn, V)


class GELU(nn.Module):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return 0.5 * x * (1 + torch.erf(x / np.sqrt(2)))


def silu(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x)


def apply_rope(
    *,
    x: torch.Tensor,
    theta: float,
    token_positions: torch.Tensor,
    d_model: int,
) -> torch.Tensor:
    """
    Apply RoPE to the last dimension of `x` (size d_model).

    `token_positions` is broadcastable to x.shape[:-1] (typically (..., seq_len)).
    """

    if d_model % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")
    if x.shape[-1] != d_model:
        raise ValueError(f"Expected x.shape[-1] == d_model ({d_model}), got {x.shape[-1]}")

    # Make token_positions shape broadcastable to x[..., 0]
    # Expected: (..., seq_len)
    while token_positions.ndim < x.ndim - 1:
        token_positions = token_positions.unsqueeze(0)

    device = x.device
    dtype = x.dtype

    half = d_model // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    # angles: (..., seq_len, half)
    angles = token_positions.to(torch.float32).unsqueeze(-1) * inv_freq
    angles = angles.to(dtype)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        rms_x = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)+self.eps)
        x = (x / rms_x) * self.weight
        return x

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward network:
        out = W2( SiLU(W1(x)) * W3(x) )
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_pdrop: float,
        *,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_head = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.FloatTensor, *, token_positions: torch.Tensor | None = None) -> torch.FloatTensor:
        B, T, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, heads, T, d_head)
        q = q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        if self.rope_theta is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
            # Apply RoPE on head dimension (d_head)
            q = apply_rope(x=q, theta=self.rope_theta, token_positions=token_positions.unsqueeze(1), d_model=self.d_head)
            k = apply_rope(x=k, theta=self.rope_theta, token_positions=token_positions.unsqueeze(1), d_model=self.d_head)

        # Causal mask: True means "keep".
        causal = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=0)
        causal = causal.unsqueeze(0).unsqueeze(0)  # (1,1,T,T) broadcast

        out = scaled_dot_product_attention(K=k, Q=q, V=v, mask=causal, pdrop=self.attn_pdrop)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.output_proj(out)
        
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        resid_pdrop: float,
        *,
        max_seq_len: int | None = None,
        rope_theta: float | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            attn_pdrop=attn_pdrop,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.drop1 = nn.Dropout(resid_pdrop)

        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.drop2 = nn.Dropout(resid_pdrop)
    
    def forward(self, x: torch.FloatTensor, *, token_positions: torch.Tensor | None = None) -> torch.FloatTensor:
        x = x + self.drop1(self.attn(self.ln1(x), token_positions=token_positions))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attn_pdrop: float,
        resid_pdrop: float,
        *,
        rope_theta: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.context_length = context_length

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                )
                for _ in range(num_layers)
            ]
        )
        self.drop = nn.Dropout(resid_pdrop)
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        B, T = x.shape
        tok = self.token_embeddings(x)
        tok = self.drop(tok)

        token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = tok
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)
        h = self.ln_final(h)
        return self.lm_head(h)

class TransformerBlockAblation(nn.Module):
    def __init__(self, d_model: int,
                 num_heads: int,
                 d_ff: int,
                 attn_pdrop: float,
                 resid_pdrop: float,
                 no_rmsnorm: bool=False,
                 parallel_layers: bool=False,
                 post_norm: bool=False):
        super(TransformerBlockAblation, self).__init__()
        if not no_rmsnorm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.drop1 = nn.Dropout(resid_pdrop)
        
        self.ffn = FFN(d_model, d_ff)
        self.drop2 = nn.Dropout(resid_pdrop)

        self.no_rmsnorm = no_rmsnorm
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if self.no_rmsnorm:
            x = x + self.drop1(self.attn(x))
            x = x + self.drop2(self.ffn(x))
        elif self.parallel_layers:
            x1 = x + self.drop1(self.attn(self.ln1(x)))
            x2 = x + self.drop2(self.ffn(self.ln2(x)))
            x = x1 + x2
        elif self.post_norm:
            x = self.ln1(x + self.drop1(self.attn(x)))
            x = self.ln2(x + self.drop2(self.ffn(x)))
        else:
            x = x + self.drop1(self.attn(self.ln1(x)))
            x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class TransformerLMAblation(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int,
                 d_model: int, num_heads: int, d_ff: int, attn_pdrop: float,
                 resid_pdrop: float,
                 no_rmsnorm: bool=False, parallel_layers: bool=False, post_norm: bool=False,
                 **kwargs):

        super(TransformerLMAblation, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([
            TransformerBlockAblation(d_model, num_heads, d_ff, attn_pdrop, resid_pdrop,
                                  no_rmsnorm, parallel_layers, post_norm
                                 ) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(resid_pdrop)
        if not no_rmsnorm:
            self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.no_rmsnorm = no_rmsnorm
        self.parallel_layers = parallel_layers
        self.post_norm = post_norm

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        B, T = x.size()
        positions = torch.arange(T, device=x.device).expand(B, T)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = self.drop(x)
        for layer in self.layers:
            x = layer(x)
        if not self.no_rmsnorm:
            x = self.ln_final(x)
        x = self.lm_head(x)
        return x