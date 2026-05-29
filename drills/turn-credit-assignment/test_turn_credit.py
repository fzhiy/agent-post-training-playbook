"""Correctness tests for turn-credit-assignment primitives.

    python test_turn_credit.py            # plain run
    python -m pytest test_turn_credit.py  # or via pytest

All tests are assertion-based, deterministic (fixed seeds / hand-constructed
inputs), and require only torch + stdlib.
"""
import torch

from from_scratch import (
    discounted_return,
    group_relative_advantages,
    masked_pg_loss,
)


# ---------------------------------------------------------------------------
# discounted_return
# ---------------------------------------------------------------------------

def test_discounted_return_single_step():
    """Single reward: G_0 = r_0."""
    rewards = torch.tensor([5.0])
    G = discounted_return(rewards, gamma=0.9)
    assert G.shape == (1,)
    assert torch.isclose(G[0], torch.tensor(5.0)), G


def test_discounted_return_two_steps():
    """Two steps: G_0 = r_0 + gamma*r_1; G_1 = r_1."""
    rewards = torch.tensor([1.0, 2.0])
    gamma = 0.5
    G = discounted_return(rewards, gamma=gamma)
    assert torch.isclose(G[0], torch.tensor(1.0 + 0.5 * 2.0))
    assert torch.isclose(G[1], torch.tensor(2.0))


def test_discounted_return_geometric_series():
    """Constant reward r: G_0 = r * (1 - gamma^T) / (1 - gamma)."""
    T, r, gamma = 10, 1.0, 0.9
    rewards = torch.ones(T) * r
    G = discounted_return(rewards, gamma=gamma)
    expected_G0 = r * (1 - gamma ** T) / (1 - gamma)
    assert torch.isclose(G[0], torch.tensor(expected_G0), atol=1e-5), G[0]


def test_discounted_return_gamma_zero():
    """gamma=0: each return equals its immediate reward."""
    rewards = torch.tensor([3.0, 1.0, 4.0])
    G = discounted_return(rewards, gamma=0.0)
    assert torch.allclose(G, rewards)


def test_discounted_return_shape():
    rewards = torch.randn(20)
    G = discounted_return(rewards, gamma=0.99)
    assert G.shape == rewards.shape


# ---------------------------------------------------------------------------
# group_relative_advantages
# ---------------------------------------------------------------------------

def test_group_advantages_zero_mean_per_group():
    """Core GRPO property: within each group, advantages must sum to zero."""
    torch.manual_seed(42)
    returns = torch.tensor([1.0, 3.0, 5.0,   # group 0
                            2.0, 4.0,         # group 1
                            7.0, 7.0, 7.0])   # group 2 (all equal)
    group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    adv = group_relative_advantages(returns, group_ids)

    for gid in [0, 1, 2]:
        mask = group_ids == gid
        group_sum = adv[mask].sum()
        assert torch.isclose(group_sum, torch.tensor(0.0), atol=1e-5), \
            f"group {gid} advantages not zero-mean: sum={group_sum}"


def test_group_advantages_identical_returns_zero():
    """If all returns in a group are equal, all advantages are zero."""
    returns = torch.tensor([3.0, 3.0, 3.0])
    group_ids = torch.tensor([0, 0, 0])
    adv = group_relative_advantages(returns, group_ids)
    assert torch.allclose(adv, torch.zeros(3), atol=1e-6), adv


def test_group_advantages_independent_groups():
    """Advantages in one group are unaffected by returns in another group."""
    returns = torch.tensor([0.0, 1.0,     # group 0: small spread
                            0.0, 100.0])  # group 1: large spread
    group_ids = torch.tensor([0, 0, 1, 1])
    adv = group_relative_advantages(returns, group_ids)

    # Both groups should be zero-mean regardless of their absolute scale
    assert torch.isclose(adv[0:2].sum(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(adv[2:4].sum(), torch.tensor(0.0), atol=1e-5)


def test_group_advantages_shape():
    torch.manual_seed(7)
    N = 16
    returns = torch.randn(N)
    group_ids = torch.randint(0, 4, (N,))
    adv = group_relative_advantages(returns, group_ids)
    assert adv.shape == (N,)


def test_group_advantages_ordering():
    """Higher return within a group => positive advantage; lower => negative."""
    returns = torch.tensor([1.0, 5.0])    # 5 > mean=3 > 1
    group_ids = torch.tensor([0, 0])
    adv = group_relative_advantages(returns, group_ids)
    assert adv[0] < 0 and adv[1] > 0, adv


# ---------------------------------------------------------------------------
# masked_pg_loss
# ---------------------------------------------------------------------------

def test_masked_loss_ignores_masked_tokens():
    """Flipping log_probs at masked positions must not change the loss."""
    torch.manual_seed(0)
    B, T = 2, 6
    log_probs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    # mask: only positions 1, 3 are agent tokens
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 1] = True
    mask[:, 3] = True

    loss_before = masked_pg_loss(log_probs, advantages, mask)

    # Overwrite the masked-out positions with large garbage values
    log_probs_perturbed = log_probs.clone()
    log_probs_perturbed[:, 0] = 999.0
    log_probs_perturbed[:, 2] = -999.0
    log_probs_perturbed[:, 4] = 0.0
    log_probs_perturbed[:, 5] = 123.4

    loss_after = masked_pg_loss(log_probs_perturbed, advantages, mask)
    assert torch.isclose(loss_before, loss_after), \
        f"loss changed when perturbing masked tokens: {loss_before} vs {loss_after}"


def test_masked_loss_all_masked_out_is_zero():
    """If no tokens are active, loss should be zero (no active tokens to average)."""
    B, T = 3, 5
    log_probs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    mask = torch.zeros(B, T, dtype=torch.bool)   # everything masked
    loss = masked_pg_loss(log_probs, advantages, mask)
    assert torch.isclose(loss, torch.tensor(0.0)), loss


def test_masked_loss_formula():
    """Manual ground-truth check on a tiny example."""
    # B=1, T=3; only token 1 is active
    log_probs  = torch.tensor([[0.5, -1.0, 0.2]])
    advantages = torch.tensor([[2.0,  3.0, 1.0]])
    mask       = torch.tensor([[False, True, False]])

    loss = masked_pg_loss(log_probs, advantages, mask)
    # Expected: -( 3.0 * (-1.0) ) / 1 = 3.0
    expected = torch.tensor(3.0)
    assert torch.isclose(loss, expected, atol=1e-6), loss


def test_masked_loss_per_sequence_advantage():
    """Accept (B,) per-sequence advantage (broadcast form)."""
    torch.manual_seed(1)
    B, T = 4, 8
    log_probs  = torch.randn(B, T)
    advantages = torch.randn(B)           # one scalar per sequence
    mask       = torch.ones(B, T, dtype=torch.bool)
    loss = masked_pg_loss(log_probs, advantages, mask)
    assert loss.shape == ()               # scalar
    assert torch.isfinite(loss)


def test_masked_loss_negative_sign():
    """Positive advantage + high log_prob should reduce loss (more negative direction)."""
    B, T = 1, 2
    log_probs_high = torch.tensor([[0.0, 0.0]])        # high log-prob
    log_probs_low  = torch.tensor([[-10.0, -10.0]])   # low log-prob
    advantages = torch.tensor([[1.0, 1.0]])
    mask = torch.ones(B, T, dtype=torch.bool)
    loss_high = masked_pg_loss(log_probs_high, advantages, mask)
    loss_low  = masked_pg_loss(log_probs_low,  advantages, mask)
    # Gradient ascent on return => loss is lower (more negative) when log-prob is higher
    assert loss_high < loss_low, f"high log_prob should give lower loss: {loss_high} vs {loss_low}"


def test_masked_loss_shape():
    torch.manual_seed(3)
    B, T = 4, 12
    log_probs  = torch.randn(B, T)
    advantages = torch.randn(B, T)
    mask       = torch.randint(0, 2, (B, T)).bool()
    loss = masked_pg_loss(log_probs, advantages, mask)
    assert loss.shape == ()   # scalar


# ---------------------------------------------------------------------------
# End-to-end: full mini-trajectory
# ---------------------------------------------------------------------------

def test_end_to_end_pipeline():
    """Smoke-test the full pipeline: returns -> GRPO advantages -> masked PG loss."""
    torch.manual_seed(99)

    # 4 trajectories in 2 groups (2 rollouts each)
    B, T = 4, 10
    gamma = 0.95

    # Per-step rewards for each trajectory (each row is one trajectory's rewards)
    all_rewards = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    group_ids = torch.tensor([0, 0, 1, 1])

    # Step 1: discounted return for each trajectory (use G_0 as trajectory return)
    traj_returns = torch.stack([
        discounted_return(all_rewards[i], gamma=gamma)[0]
        for i in range(B)
    ])
    assert traj_returns.shape == (B,)
    assert (traj_returns[[0, 2]] > traj_returns[[1, 3]]).all(), \
        "trajectories with terminal reward should have higher return"

    # Step 2: GRPO advantages
    adv = group_relative_advantages(traj_returns, group_ids)
    assert adv.shape == (B,)
    for gid in [0, 1]:
        mask = group_ids == gid
        assert torch.isclose(adv[mask].sum(), torch.tensor(0.0), atol=1e-5)

    # Step 3: masked PG loss
    log_probs  = torch.randn(B, T)
    # odd positions = agent tokens; even = tool output tokens
    action_mask = torch.zeros(B, T, dtype=torch.bool)
    action_mask[:, 1::2] = True

    loss = masked_pg_loss(log_probs, adv, action_mask)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_discounted_return_single_step()
    test_discounted_return_two_steps()
    test_discounted_return_geometric_series()
    test_discounted_return_gamma_zero()
    test_discounted_return_shape()

    test_group_advantages_zero_mean_per_group()
    test_group_advantages_identical_returns_zero()
    test_group_advantages_independent_groups()
    test_group_advantages_shape()
    test_group_advantages_ordering()

    test_masked_loss_ignores_masked_tokens()
    test_masked_loss_all_masked_out_is_zero()
    test_masked_loss_formula()
    test_masked_loss_per_sequence_advantage()
    test_masked_loss_negative_sign()
    test_masked_loss_shape()

    test_end_to_end_pipeline()

    print("all turn-credit-assignment drills passed ✓")
