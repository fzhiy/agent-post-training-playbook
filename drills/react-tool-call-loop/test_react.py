"""Correctness tests for the from-scratch ReAct loop + SFT label masking.

    python test_react.py            # plain run
    python -m pytest test_react.py  # or via pytest

Two kinds of checks:
  * deterministic string/structure assertions on a SCRIPTED mock LLM — the
    loop is exact, so we can pin the tool-call turn and the n-th observation
    string verbatim;
  * a numeric allclose: our from-scratch masked_cross_entropy must equal
    F.cross_entropy(ignore_index=-100), proving the mask drops exactly the
    environment tokens and nothing else.
"""
import torch
import torch.nn.functional as F

from from_scratch import (
    IGNORE_INDEX,
    build_sft_labels,
    masked_cross_entropy,
    parse_react_step,
    run_react_loop,
)


def _scripted_llm(completions):
    """Turn a list of canned completions into an llm(prompt)->str callable."""
    it = iter(completions)
    return lambda _prompt: next(it)


# --- a fixed 2-step episode reused across tests --------------------------
QUESTION = "What is 2 + 3, then double it?"
COMPLETIONS = [
    "Thought: I should add first.\nAction: calc\nAction Input: 2 + 3\n",
    "Thought: 5 doubled is 10.\nFinal Answer: 10\n",
]
TOOLS = {"calc": lambda arg: str(eval(arg, {"__builtins__": {}}))}  # toy, sandboxed


# ---------------------------------------------------------------------------
# 1. Parser
# ---------------------------------------------------------------------------

def test_parse_action_step():
    step = parse_react_step(COMPLETIONS[0])
    assert not step.is_final
    assert step.action == "calc"
    assert step.action_input == "2 + 3"
    assert step.thought == "I should add first."


def test_parse_final_step():
    step = parse_react_step(COMPLETIONS[1])
    assert step.is_final
    assert step.final_answer == "10"
    assert step.action is None


def test_final_answer_wins_over_stray_action():
    # A completion that both names an Action and a Final Answer must terminate:
    # a model that decided to stop is not dragged into another tool call.
    text = "Thought: done.\nAction: calc\nAction Input: 9\nFinal Answer: 7\n"
    step = parse_react_step(text)
    assert step.is_final and step.final_answer == "7"


# ---------------------------------------------------------------------------
# 2. The loop
# ---------------------------------------------------------------------------

def test_tool_called_at_right_turn():
    traj = run_react_loop(_scripted_llm(COMPLETIONS), TOOLS, QUESTION)
    # exactly one tool call, on turn 1, with the parsed input and real result
    assert traj.calls == [("calc", "2 + 3", "5")]
    assert len(traj.steps) == 2


def test_final_answer_terminates():
    traj = run_react_loop(_scripted_llm(COMPLETIONS), TOOLS, QUESTION)
    assert traj.stop_reason == "final_answer"
    assert traj.final_answer == "10"


def test_nth_observation_is_exact():
    # Deterministic mock => the observation injected after turn 1 is pinned
    # verbatim. This is the analytic ground truth for the loop's plumbing.
    traj = run_react_loop(_scripted_llm(COMPLETIONS), TOOLS, QUESTION)
    obs_segments = [text for text, role in traj.segments if role == "observation"]
    assert obs_segments == ["Observation: 5\n"]
    assert "Observation: 5\n" in traj.transcript
    # and the observation lands AFTER the action, BEFORE the final answer
    assert traj.transcript.index("Action: calc") \
        < traj.transcript.index("Observation: 5") \
        < traj.transcript.index("Final Answer: 10")


def test_max_steps_termination():
    # A model that never emits Final Answer must stop at the budget, not hang.
    loop_forever = lambda _p: "Thought: keep going.\nAction: calc\nAction Input: 1\n"
    traj = run_react_loop(loop_forever, TOOLS, QUESTION, max_steps=3)
    assert traj.stop_reason == "max_steps"
    assert traj.final_answer is None
    assert len(traj.calls) == 3
    assert all(c[0] == "calc" for c in traj.calls)  # routed, not error-pathed


def test_unknown_tool_is_reported_not_crashed():
    bad = ["Thought: try x.\nAction: nope\nAction Input: foo\n",
           "Thought: ok.\nFinal Answer: done\n"]
    traj = run_react_loop(_scripted_llm(bad), TOOLS, QUESTION)
    assert traj.calls[0][2] == "Error: unknown tool 'nope'"
    assert traj.stop_reason == "final_answer"


def test_malformed_parse_is_reported_not_crashed():
    # A completion with neither Action nor Final Answer must not crash: the
    # loop reports a parse error as the observation and keeps going.
    bad = ["Thought: I'm confused and emit no action.\n",
           "Thought: ok now.\nFinal Answer: recovered\n"]
    traj = run_react_loop(_scripted_llm(bad), TOOLS, QUESTION)
    assert traj.calls[0] == (None, None, "Error: could not parse an Action or Final Answer")
    assert traj.stop_reason == "final_answer" and traj.final_answer == "recovered"


# ---------------------------------------------------------------------------
# 3. SFT label masking
# ---------------------------------------------------------------------------

def test_labels_mask_prompt_and_observation():
    seg = [([0, 1], "prompt"), ([2, 3], "agent"),
           ([4], "observation"), ([5, 6], "agent")]
    ids, labels = build_sft_labels(seg)
    assert ids == [0, 1, 2, 3, 4, 5, 6]
    assert labels == [IGNORE_INDEX, IGNORE_INDEX, 2, 3, IGNORE_INDEX, 5, 6]


def test_masked_ce_matches_torch_reference():
    # from-scratch masked CE must equal F.cross_entropy(ignore_index=-100)
    torch.manual_seed(0)
    N, V = 7, 11
    logits = torch.randn(N, V)
    labels = torch.tensor([-100, -100, 2, 3, -100, 5, 6])
    ours = masked_cross_entropy(logits, labels)
    ref = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
    assert torch.allclose(ours, ref, atol=1e-6), (ours - ref).abs().item()


def test_masked_ce_ignores_observation_tokens():
    # Corrupting a MASKED (observation) position must NOT change the loss;
    # corrupting an AGENT position must. This proves the mask is load-bearing.
    torch.manual_seed(1)
    N, V = 7, 11
    logits = torch.randn(N, V)
    labels = torch.tensor([-100, -100, 2, 3, -100, 5, 6])
    base = masked_cross_entropy(logits, labels)

    # perturb a single (non-target) class so the softmax actually moves —
    # adding a constant to a whole row is shift-invariant and would prove nothing.
    masked_pos = logits.clone()
    masked_pos[4, 0] += 100.0  # an ignored position -> loss unchanged
    assert torch.allclose(masked_cross_entropy(masked_pos, labels), base, atol=1e-6)

    agent_pos = logits.clone()
    agent_pos[3, 0] += 100.0   # a trained position (target id 3) -> loss changes
    assert not torch.allclose(masked_cross_entropy(agent_pos, labels), base, atol=1e-6)


def test_all_masked_returns_zero():
    logits = torch.randn(3, 5)
    labels = torch.full((3,), IGNORE_INDEX)
    assert masked_cross_entropy(logits, labels).item() == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ✓")
    print("all react-tool-call-loop drills passed ✓")
