import torch
import numpy as np

def get_lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, total_iters, **kwargs):
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t < total_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((t - warmup_iters) / (total_iters - warmup_iters) * 3.141592653589793))
    else:
        return lr_min


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8, **kwargs):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super(AdamW, self).__init__(params, defaults)
    
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                t = state['t'] + 1
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t) 
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t
                state['m'] = m
                state['v'] = v