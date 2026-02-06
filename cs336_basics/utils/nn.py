from __future__ import annotations

import torch


def softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """
    Numerically-stable softmax.
    """

    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exps = torch.exp(shifted)
    return exps / torch.sum(exps, dim=dim, keepdim=True)


def cross_entropy(inputs: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
    """
    Cross-entropy over a batch of logits.
    """

    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))
    return (-log_probs).mean()


def gradient_clipping(parameters, max_norm: float) -> None:
    """
    Clip gradients in-place to have a maximum combined L2 norm of `max_norm`.
    Matches the semantics of `torch.nn.utils.clip_grad_norm_`
    """

    grads = [p.grad for p in parameters if getattr(p, "grad", None) is not None]
    if not grads:
        return

    # Compute global norm
    total_norm_sq = torch.zeros((), device=grads[0].device)
    for g in grads:
        total_norm_sq = total_norm_sq + g.detach().pow(2).sum()
    total_norm = total_norm_sq.sqrt()

    if total_norm <= max_norm:
        return

    scale = max_norm / (total_norm + 1e-12)
    for g in grads:
        g.detach().mul_(scale)




# import torch
# import torch.nn as nn
# import math

# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int):
#         super().__init__()
#         assert d_model % num_heads == 0, "Hidden size must be divisible by num_heads"
        
#         self.d_model = d_model      # 比如 512
#         self.num_heads = num_heads  # 比如 8
#         self.head_dim = d_model // num_heads # 每一头的大小，比如 64
        
#         # 定义投影层：将输入映射到 Q, K, V
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

#     def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
#         # x 形状: [Batch, Seq, Hidden] (B, S, H)
#         batch, seq, _ = x.shape
        
#         # 1. 线性投影并切分多头
#         # 转换路径: (B, S, H) -> (B, S, nH, hD) -> (B, nH, S, hD)
#         q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         k = self.k_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
#         v = self.v_proj(x).view(batch, seq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

#         # 2. 计算缩放点积注意力 (Scaled Dot-Product Attention)
#         # q: (B, nH, S, hD), k.T: (B, nH, hD, S) -> scores: (B, nH, S, S)
#         # 这里用 matmul (@) 自动处理最后两个维度的矩阵乘法
#         attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
#         # 3. 应用掩码 (Causal Mask / Padding Mask)
#         if mask is not None:
#             # mask 是 True 的地方填充负无穷，让 Softmax 结果为 0
#             attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
            
#         attn_weights = torch.softmax(attn_logits, dim=-1) # (B, nH, S, S)
        
#         # 4. 注意力加权投影
#         # (B, nH, S, S) @ (B, nH, S, hD) -> (B, nH, S, hD)
#         out = attn_weights @ v
        
#         # 5. 合并多头 (维度的逆向魔术)
#         # 转换路径: (B, nH, S, hD) -> (B, S, nH, hD) -> (B, S, H)
#         # 注意: permute 之后必须接 contiguous()，否则下一步 view 会报错！
#         out = out.permute(0, 2, 1, 3).contiguous().view(batch, seq, self.d_model)
        
#         # 最后经过输出投影层
#         return self.out_proj(out)