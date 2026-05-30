"""Self-Refine loop, from scratch — toy continuous objective.

No pre-built self-refinement frameworks (e.g. guidance, dspy.Predict with
self-refine, or any library that automates the generate→score→reflect→edit
loop) — the whole point is to wire every iteration step by hand.

Concept (Madaan et al., 2023, arXiv:2303.17651):
    generate -> score -> reflect -> edit -> repeat, keeping the best so far.

No LLM needed here. The "model" is a small MLP that outputs a candidate
vector; the scorer is a known quadratic; the "reflector" nudges the
candidate using the gradient of the score. This isolates the loop logic
from any language-model machinery.

Loop invariant: best_score never decreases across iterations.

Requires: torch (stdlib only, no external datasets or network calls).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Scorer (oracle) — the objective we want to maximise
# ---------------------------------------------------------------------------

class QuadraticScorer:
    """Score a candidate vector x: score = -||x - target||^2.

    Score is in (-inf, 0]; 0 is perfect.  We maximise, so higher = better.
    The target is fixed and known only to the scorer, not to the generator.
    """

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target.detach().clone()

    def __call__(self, x: torch.Tensor) -> float:
        """Return scalar float score for candidate x (1-D tensor)."""
        return -((x - self.target) ** 2).sum().item()


# ---------------------------------------------------------------------------
# Generator — produces the initial candidate
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """Tiny MLP: () -> d-dimensional candidate vector.

    Takes no input; the parameters *are* the state that the loop improves.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.head = nn.Linear(1, d)  # maps a constant 1 -> d dims

    def forward(self) -> torch.Tensor:
        # constant input — the MLP is just a learnable affine map
        dummy = torch.ones(1, 1)
        return self.head(dummy).squeeze(0)  # shape (d,)


# ---------------------------------------------------------------------------
# Reflector — proposes an edit given the current candidate and its score
# ---------------------------------------------------------------------------

def reflect_and_edit(
    candidate: torch.Tensor,
    scorer: QuadraticScorer,
    step_size: float = 0.1,
) -> torch.Tensor:
    """Reflect on why the candidate is suboptimal; propose an edit.

    Mechanism: one gradient-ascent step on the score w.r.t. the candidate.
    In an LLM Self-Refine system, this step is replaced by a language model
    that reads the candidate + score and rewrites the candidate; the
    structural loop is identical.

    Returns a *new* candidate tensor (no in-place mutation).
    """
    x = candidate.detach().requires_grad_(True)
    score = -((x - scorer.target) ** 2).sum()  # maximise -> ascend
    score.backward()
    with torch.no_grad():
        edited = x + step_size * x.grad  # gradient-ascent step
    return edited.detach()


# ---------------------------------------------------------------------------
# Self-Refine loop
# ---------------------------------------------------------------------------

def self_refine(
    generator: Generator,
    scorer: QuadraticScorer,
    *,
    n_iterations: int = 20,
    step_size: float = 0.1,
) -> dict:
    """Run the Self-Refine loop and return a results dict.

    Steps per iteration
    -------------------
    1. Generate (or keep current) candidate.
    2. Score the candidate.
    3. If score > best_score, accept as new best.
    4. Reflect: propose an edited candidate for the next round.

    Returns
    -------
    {
        "best_candidate": torch.Tensor,
        "best_score":     float,
        "score_history":  list[float],   # score at each iteration
        "initial_score":  float,
    }
    """
    with torch.no_grad():
        candidate = generator()

    initial_score = scorer(candidate)
    best_candidate = candidate.clone()
    best_score = initial_score
    score_history: list[float] = [initial_score]

    for _ in range(n_iterations):
        # --- reflect & edit ---
        candidate = reflect_and_edit(candidate, scorer, step_size=step_size)

        # --- score ---
        score = scorer(candidate)
        score_history.append(score)

        # --- keep best (best-of-N invariant) ---
        if score > best_score:
            best_score = score
            best_candidate = candidate.clone()

    return {
        "best_candidate": best_candidate,
        "best_score": best_score,
        "score_history": score_history,
        "initial_score": initial_score,
    }


# ---------------------------------------------------------------------------
# Quick demo (not part of the tests)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    d = 8
    target = torch.ones(d) * 3.0
    scorer = QuadraticScorer(target)

    gen = Generator(d)
    nn.init.zeros_(gen.head.weight)
    nn.init.zeros_(gen.head.bias)   # start far from target

    results = self_refine(gen, scorer, n_iterations=30, step_size=0.15)

    print(f"initial score : {results['initial_score']:.4f}")
    print(f"final best    : {results['best_score']:.4f}")
    print(f"score history : {[f'{s:.3f}' for s in results['score_history']]}")
    improvement = results["best_score"] - results["initial_score"]
    print(f"improvement   : {improvement:.4f}  (>= 0: {improvement >= 0})")
