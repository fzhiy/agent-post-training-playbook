"""Correctness tests: EWC and Replay both significantly reduce forgetting.

    python test_ewc_replay.py          # plain run
    python -m pytest test_ewc_replay.py

All three experiments use fixed seeds -> fully deterministic.
"""
import torch

from from_scratch import (
    SmallMLP,
    make_task,
    train_naive,
    train_ewc,
    train_replay,
)

# ---------------------------------------------------------------------------
# Shared data: two unrelated linear regression tasks, same input dimension.
# Task 1: y = x @ w1;  Task 2: y = x @ w2  (w1, w2 nearly orthogonal)
# ---------------------------------------------------------------------------
IN_DIM = 4
N = 256

torch.manual_seed(42)
W1 = torch.randn(IN_DIM, 1)
W2 = torch.randn(IN_DIM, 1)

X1, Y1 = make_task(N, IN_DIM, W1, seed=1)
X2, Y2 = make_task(N, IN_DIM, W2, seed=2)

EPOCHS = 300
LR = 1e-2
# How much better EWC/Replay must be vs naive (absolute MSE units).
FORGETTING_REDUCTION_THRESHOLD = 0.5


def fresh_model(seed: int = 0) -> SmallMLP:
    torch.manual_seed(seed)
    return SmallMLP(in_dim=IN_DIM, hidden=16, out_dim=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_naive_forgets():
    """Baseline: naive fine-tuning leads to high Task-1 loss.

    This confirms the toy setup actually exhibits catastrophic forgetting,
    making the anti-forgetting results meaningful.
    """
    loss_t1_after = train_naive(
        fresh_model(0), X1, Y1, X2, Y2,
        epochs_t1=EPOCHS, epochs_t2=EPOCHS, lr=LR,
    )
    # After naive Task-2 training the model should have drifted far from T1.
    assert loss_t1_after > 0.3, (
        f"Expected significant forgetting (loss > 0.3), got {loss_t1_after:.4f}. "
        "Check task design or training length."
    )


def test_ewc_reduces_forgetting():
    """EWC penalty markedly reduces catastrophic forgetting vs naive."""
    naive_loss = train_naive(
        fresh_model(0), X1, Y1, X2, Y2,
        epochs_t1=EPOCHS, epochs_t2=EPOCHS, lr=LR,
    )
    ewc_loss = train_ewc(
        fresh_model(0), X1, Y1, X2, Y2,
        epochs_t1=EPOCHS, epochs_t2=EPOCHS, lr=LR,
        ewc_lambda=50.0,
    )
    reduction = naive_loss - ewc_loss
    assert reduction >= FORGETTING_REDUCTION_THRESHOLD, (
        f"EWC should reduce T1 loss by >= {FORGETTING_REDUCTION_THRESHOLD:.2f}, "
        f"got naive={naive_loss:.4f}, ewc={ewc_loss:.4f}, reduction={reduction:.4f}"
    )


def test_replay_reduces_forgetting():
    """Experience replay markedly reduces catastrophic forgetting vs naive."""
    naive_loss = train_naive(
        fresh_model(0), X1, Y1, X2, Y2,
        epochs_t1=EPOCHS, epochs_t2=EPOCHS, lr=LR,
    )
    replay_loss = train_replay(
        fresh_model(0), X1, Y1, X2, Y2,
        epochs_t1=EPOCHS, epochs_t2=EPOCHS, lr=LR,
        replay_ratio=0.5,
    )
    reduction = naive_loss - replay_loss
    assert reduction >= FORGETTING_REDUCTION_THRESHOLD, (
        f"Replay should reduce T1 loss by >= {FORGETTING_REDUCTION_THRESHOLD:.2f}, "
        f"got naive={naive_loss:.4f}, replay={replay_loss:.4f}, reduction={reduction:.4f}"
    )


def test_ewc_task2_still_learned():
    """EWC must not prevent Task-2 from being learned (anti-forgetting is not free
    if it completely blocks new learning)."""
    model = fresh_model(0)
    torch.manual_seed(0)
    # Sanity: record random-init task-2 loss
    from from_scratch import eval_loss
    loss_before = eval_loss(model, X2, Y2)

    ewc_loss_t2_after = None

    # Run EWC training, capture Task-2 loss at end
    import torch.nn.functional as F
    from from_scratch import (
        estimate_fisher, ewc_penalty, make_task
    )
    import torch.optim as optim

    model2 = fresh_model(0)
    opt = optim.SGD(model2.parameters(), lr=LR)

    # Phase 1
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model2(X1).squeeze(-1)
        F.mse_loss(pred, Y1).backward()
        opt.step()

    theta_star = {n: p.detach().clone() for n, p in model2.named_parameters()}
    fisher = estimate_fisher(model2, X1, Y1)

    # Phase 2
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model2(X2).squeeze(-1)
        loss = F.mse_loss(pred, Y2) + 50.0 * ewc_penalty(model2, fisher, theta_star)
        loss.backward()
        opt.step()

    ewc_loss_t2_after = eval_loss(model2, X2, Y2)
    # Task-2 loss should drop substantially from random-init level
    assert ewc_loss_t2_after < loss_before * 0.5, (
        f"EWC model failed to learn Task 2: before={loss_before:.4f}, "
        f"after={ewc_loss_t2_after:.4f}"
    )


def test_fisher_all_nonnegative():
    """Fisher diagonal values must all be >= 0 (they are squared gradients)."""
    from from_scratch import estimate_fisher
    model = fresh_model(0)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    import torch.nn.functional as F
    for _ in range(EPOCHS):
        opt.zero_grad()
        F.mse_loss(model(X1).squeeze(-1), Y1).backward()
        opt.step()
    fisher = estimate_fisher(model, X1, Y1)
    for name, f in fisher.items():
        assert (f >= 0).all(), f"Fisher[{name}] has negative values"


def test_replay_buffer_capacity():
    """ReplayBuffer evicts oldest entries and never exceeds capacity."""
    from from_scratch import ReplayBuffer
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(torch.randn(4, 4), torch.randn(4))
    assert len(buf) == 10, f"Expected 10 entries, got {len(buf)}"

    # Overflow: adding 8 more should keep only the 10 most recent
    buf.add(torch.randn(8, 4), torch.randn(8))
    assert len(buf) == 10, f"Buffer exceeded capacity: {len(buf)}"

    # Sample must return the right shape
    xs, ys = buf.sample(3)
    assert xs.shape == (3, 4)
    assert ys.shape == (3,)


if __name__ == "__main__":
    test_naive_forgets()
    print("  naive forgetting confirmed")
    test_ewc_reduces_forgetting()
    print("  EWC reduces forgetting")
    test_replay_reduces_forgetting()
    print("  Replay reduces forgetting")
    test_ewc_task2_still_learned()
    print("  EWC still learns Task 2")
    test_fisher_all_nonnegative()
    print("  Fisher values non-negative")
    test_replay_buffer_capacity()
    print("  ReplayBuffer capacity correct")
    print("\nall ewc-replay drills passed")
