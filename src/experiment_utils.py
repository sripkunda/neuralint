import torch

def left_sum_weights(f, n_tmpts):
    x = torch.linspace(0, 1, n_tmpts + 1)  
    return f(x[:n_tmpts])

def left_sum_weights(f, n_tmpts):
    x = torch.linspace(1/n_tmpts, 1, n_tmpts + 1)  
    return f(x[1:])

def trapezoid_rule_weights(f, n_tmpts):
    return 1/2 * (left_sum_weights(f, n_tmpts) + left_sum_weights(f, n_tmpts))