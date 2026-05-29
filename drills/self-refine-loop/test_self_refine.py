"""Correctness tests for the Self-Refine loop.

    python test_self_refine.py            # plain run
    python -m pytest test_self_refine.py  # or via pytest

All tests are deterministic (fixed manual_seed). No network, no datasets.
"""
import torch
import torch.nn as nn

from from_scratch import Generator, QuadraticScorer, self_refine, reflect_and_edit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_setup(d: int = 8, seed: int = 0) -> tuple:
    """Return (generator, scorer, target) with zero-initialised generator."""
    torch.manual_seed(seed)
    target = torch.randn(d) * 2.0
    scorer = QuadraticScorer(target)
    gen = Generator(d)
    # start from zero so distance from target is non-trivial
    nn.init.zeros_(gen.head.weight)
    nn.init.zeros_(gen.head.bias)
    return gen, scorer, target


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_final_score_not_worse_than_initial():
    """The loop must never return a best_score worse than initial."""
    gen, scorer, _ = _make_setup()
    results = self_refine(gen, scorer, n_iterations=20, step_size=0.1)
    assert results["best_score"] >= results["initial_score"] - 1e-6, (
        f"best={results['best_score']:.6f} < initial={results['initial_score']:.6f}"
    )


def test_score_monotonically_non_decreasing_best():
    """The running-best vector of scores must be non-decreasing."""
    gen, scorer, _ = _make_setup(seed=1)
    results = self_refine(gen, scorer, n_iterations=30, step_size=0.1)

    running_best = results["initial_score"]
    for i, s in enumerate(results["score_history"]):
        running_best = max(running_best, s)
    # the final best_score must equal the max of the history
    assert abs(results["best_score"] - running_best) < 1e-6, (
        "best_score is not the maximum of the score history"
    )


def test_loop_actually_improves():
    """With a sensible step size the loop must strictly improve over 50 steps."""
    gen, scorer, _ = _make_setup(seed=2, d=16)
    results = self_refine(gen, scorer, n_iterations=50, step_size=0.2)
    # gradient ascent on a quadratic is guaranteed to improve
    assert results["best_score"] > results["initial_score"] + 1e-3, (
        "Expected strict improvement; check step_size or n_iterations"
    )


def test_reflect_edit_moves_closer_to_target():
    """A single reflect_and_edit step must strictly reduce distance to target."""
    torch.manual_seed(3)
    d = 4
    target = torch.tensor([1.0, -2.0, 0.5, 3.0])
    scorer = QuadraticScorer(target)
    x = torch.zeros(d)
    dist_before = ((x - target) ** 2).sum().item()
    x_new = reflect_and_edit(x, scorer, step_size=0.1)
    dist_after = ((x_new - target) ** 2).sum().item()
    assert dist_after < dist_before, (
        f"Edit did not reduce distance: {dist_before:.4f} -> {dist_after:.4f}"
    )


def test_best_candidate_scores_match_best_score():
    """best_candidate when re-scored must reproduce best_score exactly."""
    gen, scorer, _ = _make_setup(seed=4)
    results = self_refine(gen, scorer, n_iterations=20, step_size=0.1)
    recomputed = scorer(results["best_candidate"])
    assert abs(recomputed - results["best_score"]) < 1e-6, (
        f"Re-scored best candidate {recomputed:.6f} != best_score {results['best_score']:.6f}"
    )


def test_score_history_length():
    """score_history must contain exactly n_iterations + 1 entries (incl. initial)."""
    gen, scorer, _ = _make_setup(seed=5)
    n = 15
    results = self_refine(gen, scorer, n_iterations=n, step_size=0.1)
    assert len(results["score_history"]) == n + 1, (
        f"Expected {n + 1} entries, got {len(results['score_history'])}"
    )


def test_zero_iterations_returns_initial():
    """With n_iterations=0 the loop returns the generator's initial output."""
    gen, scorer, _ = _make_setup(seed=6)
    results = self_refine(gen, scorer, n_iterations=0, step_size=0.1)
    assert results["best_score"] == results["initial_score"]
    assert len(results["score_history"]) == 1


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_final_score_not_worse_than_initial()
    test_score_monotonically_non_decreasing_best()
    test_loop_actually_improves()
    test_reflect_edit_moves_closer_to_target()
    test_best_candidate_scores_match_best_score()
    test_score_history_length()
    test_zero_iterations_returns_initial()
    print("all self-refine drills passed ✓")
