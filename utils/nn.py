import torch

def softmax(in_features: torch.FloatTensor, dim: int) -> torch.FloatTensor:
    """
    Compute the softmax of a tensor along a specified dimension.
    Args:
        in_features: A tensor of any shape.
        dim: The dimension along which to compute the softmax.
    Returns:
        A tensor of the same shape as `in_features`, with the softmax along the specified dimension.
    """
    exps = torch.exp(in_features - torch.max(in_features, dim=dim, keepdim=True).values)
    sum_exps = torch.sum(exps, dim=dim, keepdim=True)
    softmax_output = exps / sum_exps
    return softmax_output

def cross_entropy(inputs, targets):
    """
    Args:
        inputs: torch.FloatTensor
            FloatTensor of shape (batch_size, num_classes). inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets: torch.LongTensor
            LongTensor of shape (batch_size, ) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Tensor of shape () with the average cross-entropy loss across examples.
    """
    # Compute the log of the softmax of the inputs.
    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)

    # Gather the log probabilities of the target classes.

    log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))

    # Compute the negative log likelihood.
    loss = -log_probs.mean()

    return loss

def gradient_clipping(parameters, max_norm):
    """
    Clip gradients to have a maximum norm of `max_norm`.
    Args:
        parameters: Iterable of torch.Tensor
            The parameters of the model.
        max_norm: float
            Maximum L2 norm for the gradients.
    """
    # Compute the L2 norm of the gradients.
    total_norm_2 = sum([torch.sum(p.grad ** 2) for p in parameters])
    total_norm = total_norm_2 ** 0.5

    # If the total norm is larger than `max_norm`, scale all gradients to have a norm of `max_norm`.
    if total_norm > max_norm:
        for p in parameters:
            p.grad.detach().mul_(max_norm / total_norm)