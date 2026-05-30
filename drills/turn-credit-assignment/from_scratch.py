"""Agentic-RL turn credit assignment, from scratch.

No high-level RL libraries (e.g. trl.PPOTrainer, trl.GRPOTrainer,
torch.distributions convenience wrappers, or any framework that provides
discounted-return / advantage / policy-gradient loss out of the box) —
every primitive must be implemented from first principles.

Covers three primitives needed to train an LLM agent with RL:

  1. discounted_return(rewards, gamma)          -- scalar per step
  2. group_relative_advantages(returns,         -- GRPO-style critic-free baseline
                                group_ids)
  3. masked_pg_loss(log_probs, advantages,      -- policy-gradient loss that trains
                   action_mask)                 -- ONLY on agent action tokens

In multi-turn agentic settings the trajectory interleaves agent tokens (actions,
reasoning) with environment tokens (tool outputs, observations).  Only agent
tokens should receive gradient; environment tokens are masked out.

GRPO reference: DeepSeekMath (Shao et al., 2024), arXiv:2402.03300.

Requires: torch (stdlib math only otherwise).
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# 1. Discounted return
# ---------------------------------------------------------------------------

def discounted_return(rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute discounted return G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}.

    Args:
        rewards: 1-D tensor of shape (T,), one scalar reward per time step.
        gamma:   discount factor in [0, 1].

    Returns:
        returns: 1-D tensor of shape (T,), G_t for each step t.

    The loop runs backwards from T-1 to 0; each step accumulates
    G_t = r_t + gamma * G_{t+1}.  This is O(T) and avoids building the
    full discounted-sum matrix.
    """
    T = rewards.size(0)
    returns = torch.zeros_like(rewards)
    running = torch.tensor(0.0, dtype=rewards.dtype)
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


# ---------------------------------------------------------------------------
# 2. GRPO-style group-relative advantages
# ---------------------------------------------------------------------------

def group_relative_advantages(
    returns: torch.Tensor,   # (N,) one return per trajectory/turn
    group_ids: torch.Tensor, # (N,) integer group label per trajectory/turn
) -> torch.Tensor:
    """Critic-free baseline: subtract per-group mean, divide by per-group std.

    GRPO (arXiv:2402.03300) samples G responses for the same prompt and
    normalises within that group instead of learning a value function.
    This removes the need for a critic network and keeps the baseline
    unbiased -- the group mean is a simple Monte-Carlo estimate of V(s).

    For each sample i in group g:
        advantage_i = (return_i - mean_g) / (std_g + eps)

    Properties guaranteed by this normalisation:
      - Each group's advantages are zero-mean  (by construction).
      - Variance within each group is ~1       (unless all returns are identical).
      - Groups with a single sample get advantage 0 (no comparison possible).

    Args:
        returns:   1-D tensor, one scalar return per sample.
        group_ids: 1-D integer tensor, same length; identifies which prompt/group
                   each sample belongs to.

    Returns:
        advantages: 1-D tensor, same shape as returns.
    """
    eps = 1e-8
    advantages = torch.zeros_like(returns)
    for gid in group_ids.unique():
        mask = group_ids == gid
        g_returns = returns[mask]
        g_mean = g_returns.mean()
        g_std = g_returns.std(unbiased=False)   # population std; unbiased needs N>=2
        advantages[mask] = (g_returns - g_mean) / (g_std + eps)
    return advantages


# ---------------------------------------------------------------------------
# 3. Masked policy-gradient loss
# ---------------------------------------------------------------------------

def masked_pg_loss(
    log_probs:   torch.Tensor,  # (B, T) log-probabilities of the chosen tokens
    advantages:  torch.Tensor,  # (B, T) or (B,)  per-token or per-sequence advantage
    action_mask: torch.Tensor,  # (B, T) bool, True = agent action token (train here)
) -> torch.Tensor:
    """REINFORCE-style policy-gradient loss, masked to agent action tokens only.

    In an agentic trajectory the token sequence looks like:

        [system] [obs_1] [action_1] [tool_out_1] [action_2] [tool_out_2] ...

    Observation / tool-output tokens must NOT receive gradient -- their
    log-probabilities are outside the agent's control.  action_mask marks
    which positions are agent action tokens (True) and which are environment
    tokens (False).

    Loss per active token: -advantage * log_prob
    (negative because we *minimise* the loss but *maximise* expected return).

    The loss is averaged over active tokens so that sequences of different
    lengths contribute equally.

    Args:
        log_probs:   (B, T) log π_θ(a_t | s_t) for each token.
        advantages:  (B, T) A_t broadcast-compatible with log_probs, OR (B,)
                     per-sequence advantage that will be expanded to (B, T).
        action_mask: (B, T) bool tensor; True where gradient should flow.

    Returns:
        scalar loss (mean over active tokens across the batch).
    """
    if advantages.dim() == 1:
        # per-sequence advantage -> broadcast to every token in that sequence
        advantages = advantages.unsqueeze(1).expand_as(log_probs)

    # Element-wise policy-gradient objective: -A * log π
    pg = -advantages * log_probs               # (B, T)

    # Zero out environment / observation tokens
    pg = pg * action_mask.float()

    # Average only over active tokens (not the full B*T budget)
    n_active = action_mask.float().sum().clamp(min=1.0)
    return pg.sum() / n_active
