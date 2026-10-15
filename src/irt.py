"""Minimal IRT fitting (joint MAP) with a fixed guessing floor, in PyTorch.

Model: P(correct_mi) = c + (1-c) * sigmoid(a_i * (theta_m - b_i))
  rasch=True  -> a_i = 1 (Rasch with guessing floor; theta scale free, in logits)
  rasch=False -> 2PL with theta ~ N(0,1) (standardised), log a_i ~ N(0, 0.5^2)
Priors: b_i ~ N(0, 3^2); theta ~ N(0, 2^2) for Rasch.
"""
import numpy as np
import torch


def fit_irt(X, c=0.0, rasch=True, iters=600, device="cpu", seed=0):
    """X: (M models, N items) array with 0/1 (NaN = missing). Returns dict(theta, b, a)."""
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    mask = ~torch.isnan(X)
    Y = torch.nan_to_num(X, 0.0)
    M, N = X.shape
    # init from classical statistics
    p_item = (Y.sum(0) / mask.sum(0)).clamp(0.02, 0.98)
    p_true = ((p_item - c) / (1 - c)).clamp(0.02, 0.98)
    b = torch.nn.Parameter(-torch.log(p_true / (1 - p_true)))
    p_m = (Y.sum(1) / mask.sum(1)).clamp(0.02, 0.98)
    pt_m = ((p_m - c) / (1 - c)).clamp(0.02, 0.98)
    theta = torch.nn.Parameter(torch.log(pt_m / (1 - pt_m)) - torch.log(pt_m / (1 - pt_m)).mean())
    loga = torch.nn.Parameter(torch.zeros(N, device=device))
    params = [b, theta] + ([] if rasch else [loga])
    opt = torch.optim.LBFGS(params, lr=0.5, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        a = torch.ones_like(loga) if rasch else loga.exp()
        z = a[None] * (theta[:, None] - b[None])
        P = c + (1 - c) * torch.sigmoid(z)
        P = P.clamp(1e-6, 1 - 1e-6)
        ll = (Y * torch.log(P) + (1 - Y) * torch.log(1 - P))[mask].sum()
        prior = -(b ** 2).sum() / (2 * 9)
        if rasch:
            prior = prior - (theta ** 2).sum() / (2 * 4)
        else:
            prior = prior - (theta ** 2).sum() / 2 - (loga ** 2).sum() / (2 * 0.25)
        loss = -(ll + prior)
        loss.backward()
        return loss

    opt.step(closure)
    a = np.ones(N) if rasch else loga.detach().exp().cpu().numpy()
    return dict(theta=theta.detach().cpu().numpy(), b=b.detach().cpu().numpy(), a=a, c=c)


GUESS = {"arc": 0.25, "hellaswag": 0.25, "winogrande": 0.5, "truthfulqa": 0.0, "gsm8k": 0.0}


def guess_floor(task):
    return 0.25 if task.startswith("mmlu_") else GUESS[task]
