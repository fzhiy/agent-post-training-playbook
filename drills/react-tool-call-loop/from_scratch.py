"""Minimal ReAct tool-calling loop + SFT label masking, from scratch.

No agent/LLM frameworks (no langchain, llama-index, transformers `Agent`,
smolagents, autogen, openai tool-runner, ...) — the whole point is to derive
and defend, line by line:

  1. parse_react_step  -- turn a model completion into (thought, action, input)
                          or a Final Answer (text format, à la ReAct).
  2. run_react_loop    -- think -> act -> observe, looping until Final Answer
                          or a step budget; tools are plain Python callables.
  3. build_sft_labels  -- the loss mask that trains ONLY agent-emitted tokens;
                          question / tool-output tokens are set to ignore_index.
  4. masked_cross_entropy -- from-scratch CE that honours ignore_index, so the
                          masking is verifiable numerically (vs F.cross_entropy).

The "LLM" is injected as a plain callable `llm(prompt) -> str`, so the loop is
fully deterministic and testable with a scripted mock. See README.md for the
math, the format, and the stratified follow-up questions.

Requires: torch (only for `masked_cross_entropy`); the loop and parser are
pure-Python stdlib.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100  # the value F.cross_entropy skips; standard HF convention.


# ---------------------------------------------------------------------------
# 1. Parse a single ReAct step
# ---------------------------------------------------------------------------

@dataclass
class ReActStep:
    """One model turn: either an Action (tool call) or a Final Answer."""
    thought: str
    action: str | None        # tool name; None on the final step
    action_input: str | None  # raw tool argument; None on the final step
    final_answer: str | None   # set only on the final step
    is_final: bool


def parse_react_step(text: str) -> ReActStep:
    """Parse a completion in the ReAct text format.

    Non-final step:
        Thought: <reasoning>
        Action: <tool_name>
        Action Input: <argument>

    Final step:
        Thought: <reasoning>
        Final Answer: <answer>

    Final Answer is checked FIRST: a completion that contains it terminates the
    episode even if a stray "Action:" line is also present, so a model that
    decides to stop cannot be dragged into another tool call by malformed text.
    """
    thought_m = re.search(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer):|\Z)",
                          text, re.DOTALL)
    thought = thought_m.group(1).strip() if thought_m else ""

    # No re.DOTALL: the answer is the rest of ITS line, not everything after
    # (trailing lines a model might emit must not leak into final_answer).
    final_m = re.search(r"Final Answer:\s*(.*)", text)
    if final_m:
        return ReActStep(thought=thought, action=None, action_input=None,
                         final_answer=final_m.group(1).strip(), is_final=True)

    action_m = re.search(r"Action:\s*(.*?)\s*\n", text)
    input_m = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
    action = action_m.group(1).strip() if action_m else None
    action_input = input_m.group(1).strip() if input_m else None
    return ReActStep(thought=thought, action=action, action_input=action_input,
                     final_answer=None, is_final=False)


# ---------------------------------------------------------------------------
# 2. The ReAct loop
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    transcript: str                       # full think/act/observe text stream
    steps: list[ReActStep]                # one per model turn
    calls: list[tuple[str | None, str | None, str]]  # (tool, input, observation); tool=None on a parse failure
    final_answer: str | None
    stop_reason: str                      # "final_answer" | "max_steps"
    segments: list[tuple[str, str]] = field(default_factory=list)  # (text, role)


def run_react_loop(
    llm,                       # callable: prompt (str) -> completion (str)
    tools: dict,               # name -> callable(arg: str) -> observation (str)
    question: str,
    *,
    max_steps: int = 8,
    observation_prefix: str = "Observation: ",
) -> Trajectory:
    """Run think -> act -> observe until a Final Answer or the step budget.

    Each iteration:
      1. call the LLM on the running transcript -> a completion (agent tokens);
      2. parse it; if it is a Final Answer, stop;
      3. otherwise route Action to the matching tool, run it, and INJECT the
         observation back into the transcript (these are environment tokens).

    `segments` records every chunk with its role so build_sft_labels can mask
    exactly the environment-injected text. The model is NEVER asked to predict
    its own observations — that is the whole point of the mask.
    """
    transcript = f"Question: {question}\n"
    segments: list[tuple[str, str]] = [(transcript, "prompt")]
    steps: list[ReActStep] = []
    calls: list[tuple[str | None, str | None, str]] = []

    for _ in range(max_steps):
        completion = llm(transcript)
        if not completion.endswith("\n"):
            completion += "\n"
        step = parse_react_step(completion)
        steps.append(step)
        transcript += completion
        segments.append((completion, "agent"))  # model-generated -> trainable

        if step.is_final:
            return Trajectory(transcript, steps, calls, step.final_answer,
                              "final_answer", segments)

        if step.action is None:
            # Neither an Action nor a Final Answer parsed — tell the model so,
            # and keep looping (it gets a chance to recover until max_steps).
            observation = "Error: could not parse an Action or Final Answer"
        elif step.action not in tools:
            observation = f"Error: unknown tool '{step.action}'"
        else:
            observation = tools[step.action](step.action_input)
        calls.append((step.action, step.action_input, observation))

        obs_text = f"{observation_prefix}{observation}\n"
        transcript += obs_text
        segments.append((obs_text, "observation"))  # injected -> masked out

    return Trajectory(transcript, steps, calls, None, "max_steps", segments)


# ---------------------------------------------------------------------------
# 3. SFT label masking
# ---------------------------------------------------------------------------

def build_sft_labels(
    tokenized_segments: list[tuple[list[int], str]],
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Build (input_ids, labels) for next-token SFT on a ReAct trajectory.

    tokenized_segments: list of (token_ids, role); role in
        {"prompt", "agent", "observation"}.
      - "agent"            -> the model produced these; KEEP the id as the label
                              (train the policy on its own actions/reasoning).
      - "prompt"/"observation" -> question + tool outputs were INJECTED;
                              set the label to ignore_index so no gradient flows.

    Returns (input_ids, labels). input_ids is the full concatenation (the model
    still ATTENDS to observations); only the labels are masked, so the loss is
    computed solely on agent tokens. This is the discrete-SFT analogue of the
    RL action-mask: train on what the policy controls, not on the environment.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    for token_ids, role in tokenized_segments:
        input_ids.extend(token_ids)
        if role == "agent":
            labels.extend(token_ids)
        else:
            labels.extend([ignore_index] * len(token_ids))
    return input_ids, labels


def masked_cross_entropy(
    logits: torch.Tensor,   # (N, V) next-token logits
    labels: torch.Tensor,   # (N,)   target ids, ignore_index where masked
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """From-scratch cross-entropy that averages ONLY over non-ignored positions.

    Implemented by hand (log_softmax -> gather -> mask -> mean) so the masking
    is explicit and can be checked against F.cross_entropy(ignore_index=...).
    Positions whose label == ignore_index contribute neither to the numerator
    nor the denominator — exactly the SFT loss-masking semantics.
    """
    keep = labels != ignore_index
    if keep.sum() == 0:
        return logits.new_zeros(())
    log_probs = F.log_softmax(logits, dim=-1)          # (N, V)
    safe_labels = labels.clamp(min=0)                  # avoid gather on -100
    picked = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)  # (N,)
    nll = -picked[keep]                                # only kept positions
    return nll.mean()


# ---------------------------------------------------------------------------
# Runnable toy example: a 2-step ReAct episode against a scripted mock LLM.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A deterministic "model": it looks up a digit, then answers. In reality
    # this is a trained LLM; here it is a scripted list so the loop is exact.
    scripted = iter([
        "Thought: I should look up the population.\n"
        "Action: kb\nAction Input: pop_of_X\n",
        "Thought: The KB says 42. I can answer now.\n"
        "Final Answer: 42\n",
    ])
    tools = {"kb": lambda arg: "42" if arg == "pop_of_X" else "not found"}

    traj = run_react_loop(lambda _prompt: next(scripted), tools,
                          question="What is the population of X?")
    print("stop_reason :", traj.stop_reason)
    print("tool calls  :", traj.calls)
    print("final answer:", traj.final_answer)
    print("--- transcript ---")
    print(traj.transcript)

    # Label masking: only the two agent turns are trained on; the question and
    # the "Observation: 42" line are masked. Token ids here are toy stand-ins.
    seg = [([0, 1], "prompt"), ([2, 3], "agent"), ([4], "observation"),
           ([5, 6], "agent")]
    ids, labels = build_sft_labels(seg)
    print("input_ids :", ids)
    print("labels    :", labels, "(-100 = masked)")
