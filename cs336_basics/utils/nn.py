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

