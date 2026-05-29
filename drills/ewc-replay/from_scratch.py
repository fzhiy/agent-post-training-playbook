"""EWC (Elastic Weight Consolidation) + Experience Replay, from scratch.

Continual learning: train Task 1, then adapt to Task 2 without forgetting.
Two anti-forgetting strategies implemented from first principles:

  (a) EWC  -- Kirkpatrick et al. 2017, arXiv:1612.00796
      After Task 1, estimate the Fisher information diagonal F_i for each
      weight theta_i.  Add a quadratic penalty to Task-2 loss:
        L_EWC = L_task2 + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
      where theta* are the Task-1 optimal weights.  Weights that were
      important for Task 1 (high F_i) get penalised more for drifting.

  (b) Replay -- Robins 1995 (rehearsal)
      Keep a small ring-buffer of (x, y) pairs from Task 1.  When training
      on Task 2, interleave replay batches so gradients from both tasks
      are mixed.  No extra parameter overhead.

No external datasets. No network calls. Pure torch + stdlib.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tiny model: two linear layers, enough to exhibit catastrophic forgetting.
# ---------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """Two-layer MLP: in_dim -> hidden -> out_dim."""

    def __init__(self, in_dim: int = 4, hidden: int = 16, out_dim: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Fisher diagonal estimation (EWC step after Task 1)
# ---------------------------------------------------------------------------

def estimate_fisher(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 200,
) -> dict[str, torch.Tensor]:
    """Diagonal Fisher information estimate (Monte-Carlo, per-parameter).

    For a regression model we use the squared-error log-likelihood:
        log p(y | x, theta) = -0.5 * (y_hat - y)^2  (Gaussian, unit variance)
    so the gradient of log p is just the negative MSE gradient.

    We accumulate E[grad^2] over the dataset as the diagonal Fisher:
        F_i = (1/N) * sum_n (d log p / d theta_i)^2

    Args:
        model:    model already trained on Task 1 (parameters = theta*).
        x, y:     Task-1 data (used to score the likelihood).
        n_samples: how many data points to average over.

    Returns:
        dict mapping param name -> F_i tensor (same shape as param).
    """
    model.eval()
    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(p)
        for name, p in model.named_parameters()
    }

    idx = torch.randperm(x.size(0))[:n_samples]
    x_sub, y_sub = x[idx], y[idx]

    for xi, yi in zip(x_sub, y_sub):
        model.zero_grad()
        pred = model(xi.unsqueeze(0))
        # log-likelihood gradient (MSE => Gaussian likelihood)
        loss = F.mse_loss(pred.squeeze(), yi)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2

    for name in fisher:
        fisher[name] /= n_samples

    return fisher


# ---------------------------------------------------------------------------
# EWC penalty
# ---------------------------------------------------------------------------

def ewc_penalty(
    model: nn.Module,
    fisher: dict[str, torch.Tensor],
    theta_star: dict[str, torch.Tensor],
) -> torch.Tensor:
    """EWC quadratic penalty: sum_i F_i * (theta_i - theta*_i)^2.

    Each weight is penalised proportional to how important it was for
    Task 1 (its Fisher value) times how far it has drifted from the
    Task-1 solution theta*.
    """
    penalty = torch.tensor(0.0)
    for name, p in model.named_parameters():
        fi = fisher[name]
        diff = p - theta_star[name].detach()
        penalty = penalty + (fi * diff ** 2).sum()
    return penalty


# ---------------------------------------------------------------------------
# Experience Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity FIFO ring buffer storing (x, y) tensors row-by-row.

    Usage:
        buf = ReplayBuffer(capacity=100)
        buf.add(x_task1, y_task1)          # store Task-1 data
        x_r, y_r = buf.sample(batch_size)  # sample for Task-2 training
    """

    def __init__(self, capacity: int = 200):
        self.capacity = capacity
        self._xs: Deque[torch.Tensor] = deque()
        self._ys: Deque[torch.Tensor] = deque()

    def add(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch; evict oldest rows if capacity is exceeded."""
        for xi, yi in zip(x, y):
            self._xs.append(xi.detach().clone())
            self._ys.append(yi.detach().clone())
        while len(self._xs) > self.capacity:
            self._xs.popleft()
            self._ys.popleft()

    def sample(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n rows uniformly (with replacement if n > len)."""
        indices = random.choices(range(len(self._xs)), k=n)
        xs = torch.stack([self._xs[i] for i in indices])
        ys = torch.stack([self._ys[i] for i in indices])
        return xs, ys

    def __len__(self) -> int:
        return len(self._xs)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_task(
    n: int,
    in_dim: int,
    weight: torch.Tensor,
    noise: float = 0.05,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a deterministic linear regression task.

    y = x @ weight + noise
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    x = torch.randn(n, in_dim, generator=rng)
    y = (x @ weight).squeeze(-1) + noise * torch.randn(n, generator=rng)
    return x, y


def train_one_epoch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    extra_loss: torch.Tensor | None = None,
) -> float:
    """One full-batch gradient step; returns scalar loss."""
    model.train()
    optimizer.zero_grad()
    pred = model(x).squeeze(-1)
    loss = F.mse_loss(pred, y)
    total = loss if extra_loss is None else loss + extra_loss
    total.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    pred = model(x).squeeze(-1)
    return F.mse_loss(pred, y).item()


# ---------------------------------------------------------------------------
# Three training regimes
# ---------------------------------------------------------------------------

def train_naive(
    model: nn.Module,
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    epochs_t1: int = 300,
    epochs_t2: int = 300,
    lr: float = 1e-2,
) -> float:
    """Naive sequential fine-tuning: train T1, then overwrite with T2.

    Returns Task-1 loss AFTER Task-2 training (the catastrophic forgetting
    baseline — expected to be high).
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs_t1):
        train_one_epoch(model, x1, y1, opt)
    for _ in range(epochs_t2):
        train_one_epoch(model, x2, y2, opt)
    return eval_loss(model, x1, y1)


def train_ewc(
    model: nn.Module,
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    epochs_t1: int = 300,
    epochs_t2: int = 300,
    lr: float = 1e-2,
    ewc_lambda: float = 50.0,
) -> float:
    """EWC: estimate Fisher after T1, add quadratic penalty during T2.

    Returns Task-1 loss AFTER Task-2 training (should be markedly lower
    than the naive baseline).
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # -- Phase 1: train on Task 1 --
    for _ in range(epochs_t1):
        train_one_epoch(model, x1, y1, opt)

    # -- Anchor: record theta* and Fisher --
    theta_star = {name: p.detach().clone() for name, p in model.named_parameters()}
    fisher = estimate_fisher(model, x1, y1)

    # -- Phase 2: train on Task 2 with EWC penalty --
    for _ in range(epochs_t2):
        penalty = ewc_lambda * ewc_penalty(model, fisher, theta_star)
        opt.zero_grad()
        pred = model(x2).squeeze(-1)
        loss = F.mse_loss(pred, y2) + penalty
        loss.backward()
        opt.step()

    return eval_loss(model, x1, y1)


def train_replay(
    model: nn.Module,
    x1: torch.Tensor,
    y1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    epochs_t1: int = 300,
    epochs_t2: int = 300,
    lr: float = 1e-2,
    replay_ratio: float = 0.5,
    buffer_capacity: int = 200,
) -> float:
    """Experience Replay: interleave Task-1 samples during Task-2 training.

    At each Task-2 step the loss is:
        L = (1 - replay_ratio) * L_task2 + replay_ratio * L_replay

    Returns Task-1 loss AFTER Task-2 training (should be lower than naive).
    """
    buf = ReplayBuffer(capacity=buffer_capacity)
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    # -- Phase 1: train on Task 1 and store data in buffer --
    for _ in range(epochs_t1):
        train_one_epoch(model, x1, y1, opt)
    buf.add(x1, y1)

    # -- Phase 2: train on Task 2 with interleaved replay --
    n_replay = max(1, int(replay_ratio * x2.size(0)))
    for _ in range(epochs_t2):
        opt.zero_grad()
        pred2 = model(x2).squeeze(-1)
        loss2 = F.mse_loss(pred2, y2)

        xr, yr = buf.sample(n_replay)
        pred_r = model(xr).squeeze(-1)
        loss_r = F.mse_loss(pred_r, yr)

        loss = (1 - replay_ratio) * loss2 + replay_ratio * loss_r
        loss.backward()
        opt.step()

    return eval_loss(model, x1, y1)
