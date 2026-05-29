# Self-improving LLMs

> How LLMs use **self-generated signals** to "score → filter → train" themselves, enabling continuous iteration without massive human annotation.

> ⚠️ **Learning notes, not original research** (see README for integrity statement). Figures and conclusions are based on original papers; uncertainties are marked.

## 0. The core loop

```
Generate → Filter / Score → Train → Repeat
```

Each round, **the current policy** produces candidate answers or preference pairs; some filtering mechanism (rules, another model, self-scoring) culls low-quality outputs; the remaining high-quality samples are used to update the weights; the next round runs with the new model. This **self-improvement loop** is the common skeleton for all methods.

---

## 1. Bootstrap-then-Train: Bootstrapping from correct traces

### 1.1 STaR — Rejection sampling + iterative fine-tuning

STaR<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Iterative fine-tuning on "generated correct answer" chain-of-thought, without a large-scale rationale dataset.<a href="https://arxiv.org/abs/2203.14465">Zelikman 2022 ↗</a></span></span> (Self-Taught Reasoner) is the foundational scheme for LLM chain-of-thought bootstrap fine-tuning:

1. **Rollout**: Sample $K$ chain-of-thought rationales for each problem.
2. **Filter**: Keep the rationales whose final answer is correct (rejection sampling).
3. **Fine-tune**: SFT on the kept set, updating the model.
4. **Hint-retry**: For problems where all answers are wrong, give the correct answer and let the model "reinterpret", then mix into training (to prevent easy problems from dominating the training set).

After $T$ iterations, the model is both the data generator and the data filter.

### 1.2 RFT — Rejection Sampling Fine-tuning

RFT is a simplified variant of STaR: omit hint-retry and directly keep the correct answers from $K$ samples for the same problem, assembling a richer fine-tuning set. Core finding: **multiple correct solutions for the same problem** are more diverse than a single solution, aiding generalization.

### 1.3 ReST — Grow-Improve offline RL loop

ReST<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">First generate a large-scale dataset with the current policy (Grow), then filter by a reward threshold and fine-tune (Improve), more sample-efficient than online RLHF.<a href="https://arxiv.org/abs/2308.08998">Gulcehre 2023 ↗</a></span></span> splits the loop into two stages:

- **Grow**: Sample from the current policy $\pi_\theta$ to construct an offline dataset $\mathcal{D}$, scored by a reward function $r(\cdot)$.
- **Improve**: Fine-tune $\pi_\theta$ on the subset $\mathcal{D}_{\ge\tau}$ where the reward exceeds threshold $\tau$.

Key point: **The Improve stage can be repeated multiple times** (increasing $\tau$ to gradually filter more strictly), but Grow only needs to be refreshed occasionally — compared to online RLHF's per-step sampling, computation is more concentrated.

| Method | Filtering Basis | Online? | Training Method |
|---|---|---|---|
| STaR / RFT | Answer correctness (rules) | Quasi-online (iterative) | SFT |
| ReST | Reward function threshold | Offline batch | SFT / best-of-N distillation |

---

## 2. Self-Rewarding: The model acts as its own judge

Self-Rewarding Language Models<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">The same model both generates responses and scores them using LLM-as-a-Judge; iterative DPO co-improves generation and judgment capabilities.<a href="https://arxiv.org/abs/2401.10020">Yuan 2024 ↗</a></span></span> breaks the assumption that "an external reward model is needed":

1. Sample multiple responses for the same prompt.
2. **The same model** scores each response using the LLM-as-a-Judge format (score + rationale).
3. Construct preference pairs $(y_w, y_l)$ based on scores, update using DPO.
4. In the next round, the scoring ability also improves — **the two capabilities share the same parameters, co-evolving**.

The premise for this path: the model's **generation capability** and **judgment capability** should promote each other without contaminating each other. Experiments show this holds for several iterations, but whether it degrades long-term remains an open question (see §6 Failure modes).

---

## 3. Self-Play: Using the "previous version of oneself" as an opponent

SPIN<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Current model vs previous round model: the latter generates negative samples, the former learns to distinguish, enabling self-improvement using only SFT data.<a href="https://arxiv.org/abs/2401.01335">Chen 2024 ↗</a></span></span> (Self-Play Fine-Tuning) is inspired by game theory:

- **Positive samples**: Human responses $y^*$ from the original SFT dataset.
- **Negative samples**: Outputs $\tilde{y}$ of the previous model $\pi_{\theta_{t-1}}$ for the same prompt.
- **Objective**: The current model $\pi_{\theta_t}$ learns to **distinguish** real human responses from outputs of the "old self", updated using a DPO-like loss.

$$\mathcal{L}_{\text{SPIN}}(\theta_t) = -\mathbb{E}\left[\log\sigma\!\left(\lambda\log\frac{\pi_{\theta_t}(y^*|x)}{\pi_{\theta_{t-1}}(y^*|x)} - \lambda\log\frac{\pi_{\theta_t}(\tilde{y}|x)}{\pi_{\theta_{t-1}}(\tilde{y}|x)}\right)\right].$$

Key point: No additional human preference annotations are needed — negative samples are entirely provided by **its own historical versions**. With each iteration, $\pi_{\theta_t}$ continuously approximates the human distribution until convergence when the two become indistinguishable.

---

## 4. AI Feedback: Using AI to replace human preference labeling

Constitutional AI<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Uses a set of "constitutional" principles for the model to self-critique and revise its output; AI-generated preference data replaces human harmlessness annotations (RLAIF).<a href="https://arxiv.org/abs/2212.08073">Bai 2022 ↗</a></span></span> (CAI / RLAIF) is currently the most influential scheme for "using AI to replace human preference":

**SL-CAI (Supervised Stage)**:
1. The model generates a draft of a harmful response.
2. Provide a constitutional principle (e.g., "avoid discriminatory content"), letting the model **self-critique**.
3. Let the model **revise** the response based on the critique.
4. Use the revised response for SFT.

**RL-CAI (Reinforcement Stage)**:
5. Let the model score a pair of responses with AI (which one better adheres to the constitution), constructing preference data.
6. Train a reward model on the AI-labeled preferences, then iterate with RL.

Difference from STaR/ReST: **The filtering signal comes from constitutional principles**, not task answer correctness — oriented towards alignment, not reasoning capability.

---

## 5. Inference-time self-correction (Training-free)

The following two methods do not update weights; they are **inference-time self-improvement**, different from the training loops above, but conceptually rooted similarly:

### 5.1 Reflexion — Language Reinforcement Learning

Reflexion<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">The agent converts task feedback into natural language reflections, stored in episodic memory, and referenced in the next attempt — without gradient updates.<a href="https://arxiv.org/abs/2303.11366">Shinn 2023 ↗</a></span></span> lets the agent go through multiple **trial-and-error loops**:

- Execute the task → receive environment feedback (success/failure/error message).
- Generate a **verbal reflection**: summarize in natural language "what went wrong, how to improve next time".
- Store the reflection in **episodic memory**, inject it into context in the next round.

After a few iterations, the success rate improves significantly — but the improvement **exists only within the current session's context** and is lost upon restart.

### 5.2 Self-Refine — Generate-Critique-Refine Loop

Self-Refine<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">A frozen LLM loops: generate output → self-critique → revise based on critique, requiring no training or extra supervision, showing consistent gains across tasks.<a href="https://arxiv.org/abs/2303.17651">Madaan 2023 ↗</a></span></span> has a fixed three-step loop:

$$\text{output}_0 \xrightarrow{\text{critique}} \text{feedback}_0 \xrightarrow{\text{refine}} \text{output}_1 \xrightarrow{\cdots}$$

Requires no training, no extra supervision — directly leveraging the **pre-trained model's self-critique capability**. Experiments show gains across multiple tasks (code, summarization, dialogue, math), but the gain ceiling is limited by the model's initial judging ability.

| Method | Where Improvement Happens | Weights Updated? | Can be Persisted? |
|---|---|---|---|
| Reflexion | inference-time, multiple trial-and-error | No | No (within context) |
| Self-Refine | inference-time, single loop | No | No |
| STaR / ReST / SPIN / CAI | training-time | Yes | Yes |

---

## 6. Failure modes

The self-improvement loop seems promising, but has three structural risks:

### 6.1 Reward Hacking

When the filtering signal (reward model, LLM scoring, rule-based filtering) is imperfect, the model can learn strategies that **score high but are not truly correct**: shortcut answers, superficially fluent but content-incorrect rationales, outputs specifically tailored to please the scoring template.

- Root cause: Gap between the optimization objective (proxy reward) and the true objective (task quality) — **Goodhart's Law**.
- Mitigation: Use diverse, independent evaluation signals; limit the magnitude of single RL updates (KL constraint).

### 6.2 Model Collapse / Distribution Narrowing

Only "high-scoring" samples are kept each round, culling low-score diversity. After multiple rounds, the training set becomes homogenous, model output diversity drops, and generalization worsens. This is especially severe in schemes like Self-Rewarding where "the model scores itself": the model's blind spots are **systematically inherited** in the preference annotations.

$$\text{Diversity}(\pi_{\theta_t}) \le \text{Diversity}(\pi_{\theta_{t-1}}) \quad \text{(if only top-}k\text{ kept per round)}$$

### 6.3 Reward Model Over-optimization (RM Over-optimization)

The reward model in the RL stage is itself an **approximation**; when the policy is continuously optimized, the score curve eventually decouples from true quality (the out-of-distribution region of the reward model is exploited). The KL divergence penalty term is a standard mitigation:

$$\mathcal{J}(\theta) = \mathbb{E}[r(y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}].$$

Larger $\beta$ keeps the policy closer to the reference policy, but the improvement magnitude is also more conservative.

---

## 7. From-scratch code: STaR-style rejection sampling fine-tuning loop

```python
"""
STaR-style rejection-sampling fine-tuning loop (illustrative).
Dependencies: transformers, torch — uses GPT-2 for demonstration; replace with a larger model for real training.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ---------- Hypothetical Q&A data ----------
PROBLEMS = [
    {"question": "What is 3 + 5?",  "answer": "8"},
    {"question": "What is 7 * 6?",  "answer": "42"},
    {"question": "What is 12 - 4?", "answer": "8"},
]

# ---------- Helper: Simple answer extraction ----------
def extract_answer(text: str) -> str:
    """Extracts the last number from generated text (for demonstration)."""
    import re
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else ""

# ---------- 1. Rollout: Sample K rationales per problem ----------
def rollout(model, tokenizer, problems, K=4, max_new=64, device="cpu"):
    """Returns a list of (question, rationale, is_correct)."""
    results = []
    model.eval()
    for prob in problems:
        prompt = f"Question: {prob['question']}\nLet's think step by step:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.8,
                num_return_sequences=K, pad_token_id=tokenizer.eos_token_id,
            )
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            rationale = text[len(prompt):]
            correct = extract_answer(rationale) == prob["answer"]
            results.append({"prompt": prompt, "rationale": rationale, "correct": correct})
    return results

# ---------- 2. Filter: Keep only rationales with correct answers ----------
def filter_correct(results):
    return [r for r in results if r["correct"]]

# ---------- 3. Dataset wrapper ----------
class RationaleDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.data = []
        for s in samples:
            text = s["prompt"] + s["rationale"]
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ---------- 4. Train: SFT on correct rationales ----------
def finetune(model, tokenizer, samples, output_dir="./star-ckpt"):
    ds = RationaleDataset(samples, tokenizer)
    if len(ds) == 0:
        print("No correct samples — skip this iteration.")
        return
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=1,
        per_device_train_batch_size=2, logging_steps=5,
        save_strategy="no", report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()

# ---------- 5. STaR main loop ----------
def star_loop(model_name="gpt2", n_iters=3, K=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for t in range(n_iters):
        print(f"\n=== Iteration {t+1}/{n_iters} ===")
        all_results = rollout(model, tokenizer, PROBLEMS, K=K, device=device)
        correct = filter_correct(all_results)
        print(f"  Correct rationales: {len(correct)} / {len(all_results)}")
        finetune(model, tokenizer, correct)

    return model

if __name__ == "__main__":
    star_loop(n_iters=2, K=4)
```

> The code above is purely illustrative for principles: real STaR uses larger models, longer rationales, and hint-retry as a fallback. The core flow (sample → filter → finetune → repeat) is consistent with the paper.

---

## Stratified follow-ups

### L1 Basic

<details class="qa"><summary>1. What is the "Generate-Filter-Train" loop of self-improvement? Why is a loop needed instead of a one-shot process?</summary>

Answer: The loop skeleton is: current policy generates candidate outputs → filtering/scoring mechanism culls low-quality samples → update weights on the kept set → re-run with the new model. The problem with one-shot: the initial model has limited capability, and correct samples from one generation have narrow coverage; after looping, each new model can solve problems the previous one could not, gradually expanding training signal coverage, achieving bootstrapped capability improvement.

</details>

<details class="qa"><summary>2. How does STaR train chain-of-thought without rationale annotations? What problem does <strong>hint-retry</strong> solve?</summary>

Answer: STaR samples $K$ chain-of-thoughts per problem, keeping only those with correct final answers for SFT (rejection sampling), thus requiring no human rationale annotations. Hint-retry addresses "problems where the model is completely wrong" — provide the correct answer, let the model regenerate an explanation, then mix into the training set, preventing easy problems from monopolizing the training set and difficult problems from receiving no gradient updates.

</details>

<details class="qa"><summary>3. Why are Reflexion and Self-Refine called "training-free"? Can their improvements be persisted?</summary>

Answer: Neither updates model weights — Reflexion stores natural language reflections in episodic memory injected into context; Self-Refine loops "generate → critique → revise" within a single conversation. Improvements cannot be persisted: Reflexion's improvement exists only in the current session's context and is lost upon restart; Self-Refine is similar, starting from scratch on the next invocation.

</details>

<details class="qa"><summary>4. What role does the "constitution" play in Constitutional AI? How does AI feedback replace human preference annotations?</summary>

Answer: The "constitution" is a list of principles (e.g., "avoid discriminatory content"). In the SL-CAI stage, it guides the model to self-critique and revise harmful drafts; the revised output is used for SFT. In the RL-CAI stage, the model uses AI scoring (which response better adheres to the constitution) to construct preference pairs; AI-labeled preferences train a reward model for RL, thereby replacing massive human harmlessness annotations (RLAIF).

</details>

### L2 Advanced

<details class="qa"><summary>5. How are the Grow and Improve phases of ReST divided? Why is it more sample-efficient than online RLHF?</summary>

Answer: The Grow phase uses the current policy $\pi_\theta$ for large-scale sampling and scores them with a reward function, constructing an offline dataset $\mathcal{D}$; the Improve phase fine-tunes on the subset where the reward exceeds threshold $\tau$, and can raise $\tau$ to repeat Improve multiple times. It is sample-efficient because: Grow only needs to be refreshed occasionally, and Improve can be repeatedly reused on the same batch of data; online RLHF requires sampling new data at each step, making computation more dispersed.

</details>

<details class="qa"><summary>6. SPIN uses the "previous self" as negative samples. Compared to DPO using human preference pairs, what are its pros and cons?</summary>

Answer: SPIN's advantage is that it requires no additional human preference annotations; negative samples are entirely provided by the historical version $\pi_{\theta_{t-1}}$, lowering cost. Its disadvantage is that the theoretical upper bound is locked by SFT data quality — SPIN's convergence condition is $\pi_{\theta_t} = p_\text{data}$, incapable of surpassing human SFT data; moreover, as iteration progresses, negative sample quality approaches positive sample quality, making the contrast signal weaker. DPO's human preferences can cover alignment dimensions beyond SFT data, but annotation costs are high.

</details>

<details class="qa"><summary>7. In Self-Rewarding, what problems arise from "generation" and "judgment" sharing the same parameters?</summary>

Answer: The generator's blind spots are inherited by the judge — if the model is poor at a certain type of reasoning, its probability of scoring such reasoning highly is also below true levels, so preference data systematically underestimates this capability. There is also a self-confirmation bias: the model tends to give high scores to answers that "sound like its own style," creating a positive feedback loop that accumulates and amplifies the bias rather than regressing to the mean (see Deep-dive Q3).

</details>

<details class="qa"><summary>8. Are reward hacking and RM over-optimization the same thing? How does KL constraint mitigate it?</summary>

Answer: RM over-optimization is a specific form of reward hacking: after the policy is continuously optimized, it finds outputs in the RM's out-of-distribution region that have high proxy reward but low true quality — a quantitative manifestation of Goodhart's Law. KL constraint limits the degree to which the policy deviates from the reference model via $\mathcal{J}(\theta) = \mathbb{E}[r(y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}]$; larger $\beta$ is more conservative, preventing the policy from entering high-scoring OOD regions of the RM.

</details>

### L3 Deep Dive

<details class="qa"><summary>9. How is model collapse / distribution narrowing characterized mathematically? What are mitigation measures (temperature sampling, diversity constraints, data mixing)?</summary>

Answer: Keeping only top-$k$ samples each round is equivalent to truncated sampling, with entropy decreasing monotonically: $H(\pi_{\theta_t}) \le H(\pi_{\theta_{t-1}})$. Shumailov et al. point out that statistical approximation errors and function approximation errors accumulate in iterations, causing distribution tails (the source of diversity) to systematically disappear. Mitigation measures: increase sampling temperature to retain low-probability paths; add a diversity reward term to incentivize output diversity; periodically mix in original human data as a distribution anchor to prevent uncontrolled drift.

</details>

<details class="qa"><summary>10. What selection bias does STaR introduce by only keeping correct samples each round? How can it be mitigated?</summary>

Answer: Filtering only looks at the final answer, equivalent to $p_{\text{train}}(r|x) \propto p_\theta(r|x)\cdot\mathbf{1}[\text{answer}(r)=a^*]$, leading to three types of bias: ① Incorrect reasoning paths are mixed into the training set as long as the final answer is coincidentally correct; ② The training set comes from $\pi_{\theta_{t-1}}$, deviating from the true reasoning distribution each round; ③ For difficult problems, the training set is empty after filtering, preventing the model from bootstrapping improvement. Mitigation directions: Use a process reward model (PRM) to score each step (Lightman et al.) to reduce step-level errors; mix in original SFT data to prevent distribution from drifting completely.

</details>

<details class="qa"><summary>11. Combining Self-Rewarding's LLM-as-Judge with an external reward model, what are the information contributions of each? How to prevent them from "colluding"?</summary>

Answer: The LLM-as-Judge contributes the generator's own semantic understanding and style judgment (broad coverage but with self-confirmation bias); the external RM contributes an independent parameter's preference estimate (initially uncorrelated with the generator's bias direction, but risks failure in out-of-distribution generalization). The key to preventing collusion is to keep their parameters independent and training data uncontaminated by each other; also use held-out verifiable answers or human evaluation as a third-party signal for periodic calibration to avoid both approximate signals accumulating error in the same direction.

</details>

<details class="qa"><summary>12. If the self-improvement loop converges to a local optimum (the model cannot produce data better than itself), what are some ideas to break the deadlock?</summary>

Answer: According to Deep-dive Q7, stagnation has three roots — filtering signal saturation, insufficient exploration after distribution narrowing, and task difficulty exceeding bootstrapping capability. Corresponding breakthrough ideas: ① Curriculum learning: introduce harder or more diverse problems to expand signal coverage; ② Increase sampling temperature or add diversity rewards to restore exploratory capability; ③ Introduce an external stronger teacher model (or RLVR verifier) to provide training signals independent of the current model; ④ Switch methods, from bootstrapping methods like SPIN/STaR to RL methods with external verification signals.

</details>

---

---

## Deep-dive

> Detailed analysis of advanced interview questions. ⚠️ Learning notes, not original research. Figures based on original papers.

---

### Q1. STaR only keeps "correct answer" samples: What selection bias does this introduce? What is the formal impact on the learned distribution?

**Core Bias**: STaR<a class="cite" href="#ref-1">1</a>, in each iteration, only includes chain-of-thoughts with correct final answers in the training set. Formally, this is equivalent to:

$$p_{\text{train}}(r \mid x) \propto p_\theta(r \mid x) \cdot \mathbf{1}[\text{answer}(r) = a^*]$$

where $r$ is the rationale, $x$ is the problem, and $a^*$ is the reference answer.

Three structural consequences:

1. **Correctness ≠ Reasoning Quality**: A rationale might get the correct answer by luck, a shortcut, or "working backward," while the reasoning steps themselves are incorrect. Since filtering only looks at the final answer, **incorrect reasoning paths are systematically mixed into the training set**. This aligns with the motivation for Lightman et al.<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a></span> proposing process reward models (PRMs): outcome supervision cannot distinguish "correct answer with good reasoning" from "correct answer with flawed reasoning."

2. **Accumulation of Distribution Shift**: The training set at iteration $t$ is drawn from the conditional distribution of $\pi_{\theta_{t-1}}$, not the true reasoning distribution $p^*(r \mid x)$. After each iteration, $\pi_{\theta_t}$ deviates further from $p^*$, and the filter's "accuracy" signal becomes increasingly self-referential.

3. **Blind Spots on Hard Problems**: For difficult problems the model cannot answer correctly at all, the training set after filtering is empty (hint-retry helps partially, but cannot fully compensate). The model receives neither gradient updates nor bootstrapped improvement on these problems, creating a Matthew effect where "the strong get stronger, and hard problems stagnate."

**Mitigation Directions**: PRM scoring each step (rather than only looking at the final answer) can reduce step-level errors; data mixing (retaining original SFT data) prevents complete distribution drift.

---

### Q2. Why does iterative self-training narrow the distribution (model collapse)? Intuition + When does it bite?

**Intuition**: Each round of "only keeping high-scoring samples" is statistically equivalent to truncated sampling — only taking the high-density region of the distribution each time. Over multiple rounds, tail events with low probability (but high diversity) are systematically eliminated.

Shumailov et al.<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a></span> analyzed theoretically and experimentally the consequences of **recursive training on self-generated data**:

- **Statistical Approximation Error**: The dataset produced by sampling each time is finite, underestimating or missing tail events entirely.
- **Function Approximation Error**: Limited model capacity further compresses the expression of low-frequency patterns.

The two errors **compound** in iterations, causing the distribution to continuously narrow. Intuitively captured by an inequality:

$$H(\pi_{\theta_t}) \le H(\pi_{\theta_{t-1}}) \quad \text{(if only top-}k\text{ samples kept per round)}$$

Entropy decreases monotonically, outputs trend towards repetition and homogeneity.

**When it really bites**:

| Scenario | Why it's severe |
|---|---|
| Self-Rewarding (model scores itself) | The judge itself is drifting, the gap between preference data and true preferences grows |
| Long-chain chain-of-thought tasks | High sampling variance at each step; tail solutions (unconventional reasoning paths) disappear first in filtering |
| Multi-turn dialogue / agent loop | History in context is also self-generated data, strengthening recursive contamination |
| Only using self-generated data, no human data mixed in | No anchor, distribution drift is uncontrolled |

**Mitigation**: Periodically mix in original human data (an anchor against drift); increase sampling temperature to preserve diversity; use a diversity reward term to explicitly incentivize output diversity.

---

### Q3. Why does the judge-generator coupling in Self-Rewarding fail?

In Self-Rewarding<a class="cite" href="#ref-3">3</a>, **the same set of parameters** acts as both the generator (producing answers) and the judge (scoring answers). This creates a structural problem: **the generator's blind spots are inherited by the judge**.

Specific mechanisms:

1. **Shared Blind Spots**: If the generator is poor at a certain type of reasoning (e.g., counterfactual reasoning), its probability of scoring such reasoning highly is also below the true level — because judging ability and generating ability share the same knowledge base. Preference data therefore systematically underestimates this capability.

2. **Self-Confirmation Bias**: The model tends to give high scores to answers that "sound like its own style." This is not a random hallucination error, but a **systematic bias** — the preference signal itself is pulling the model towards its own existing style, forming a positive feedback loop.

3. **Error Co-Drift**: After each DPO update, the generator and judge move synchronously in the direction indicated by the preference data. If the preference data itself is flawed (coming from a biased judge), the next round's judge will also amplify the bias in the same direction — errors do not regress to the mean but **accumulate and amplify**.

Formally, let $J_\theta$ be the judgment score function, $G_\theta$ be the generation function, both sharing $\theta$. The true quality function is $q^*$. Then:

$$\mathbb{E}[J_\theta(y) - q^*(y)] \ne 0 \quad \text{and correlated with the bias direction of } G_\theta$$

**Comparison**: An external reward model (independent parameters) is at least initially uncorrelated with the generator's deviation direction. But it has another problem: failure in out-of-distribution generalization (see Q5).

---

### Q4. SPIN converges to the SFT data distribution — why is this an upper bound? What does it mean in practice?

SPIN<a class="cite" href="#ref-4">4</a>'s theoretical convergence condition is: if and only if $\pi_{\theta_t} = p_\text{data}$ (the current model is identical to the human SFT data distribution) does the loss gradient vanish and training stop.

This mathematically establishes a **strict capability upper bound**:

$$\text{SPIN limit policy} = p_\text{data} \quad \text{(SFT data distribution)}$$

Implications:

1. **Cannot surpass SFT data quality**: If the SFT data contains errors, biases, or capability blind spots, the converged SPIN model will also inherit these flaws. SPIN only makes the model "more like human SFT data"; it cannot discover new capabilities beyond that data.

2. **Negative sample quality degrades with iteration**: Negative samples at iteration $t$ are generated by $\pi_{\theta_{t-1}}$; as $\pi_{\theta_t} \to p_\text{data}$, negative sample quality approaches positive sample quality, **weakening the contrast signal**. In practice, this means: early iterations show significant effects, later iterations show diminishing or zero marginal gains.

3. **Fundamental difference from STaR/ReST**: STaR-like methods use **task correctness** as the filtering signal and can theoretically surpass SFT data (as long as correct rationales exist, they can be learned). SPIN aims for **discriminability from human data**, with the ceiling determined by human data quality.

**Practical Implications**: SPIN is suitable as a tool to "close the SFT gap" (eliminating the gap between model and human data distribution), but not for an infinite loop of continuous self-improvement — in later iterations, one should switch to methods with external verification signals.

---

### Q5. Reward Model Over-optimization: What is the scaling-law shape of reward increase / true quality decrease?

Gao et al.<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a></span> systematically studied the relationship between **optimization degree against RM** (measured by KL divergence $d$) and **true quality**, finding the curve shape depends on the optimization method:

- **Best-of-$N$ sampling**: Proxy reward grows roughly with $\sqrt{\log N}$ (paper notes fitting difficulty, this is an approximate description), true quality first increases then plateaus — over-optimization effect is relatively mild.
- **RL (policy gradient)**: Proxy reward can continuously rise, but true quality **decreases monotonically** after some $d^*$.

Approximate functional form (fitted in the paper):

$$\text{gold reward} \approx d\,(\alpha - \beta \ln d)$$

where $\alpha, \beta > 0$, $d = \sqrt{D_{\mathrm{KL}}}$, the optimal point $d^* = \exp\!\left(\dfrac{\alpha - \beta}{\beta}\right)$ (obtained by setting $\mathrm{d}R/\mathrm{d}d = 0$). Beyond $d^*$, true quality decreases as KL increases.

**Key Scaling Conclusion**: The coefficients $\alpha, \beta$ **change smoothly** with RM parameter count — larger RMs have higher $d^*$, meaning the over-optimization "critical point" arrives later, but the critical point definitely exists. This shows that **increasing RM size cannot eliminate over-optimization risk, only delay it**.

**Intuition**: The RM is an approximation fitted on limited data; the policy finds shortcuts in the RM's out-of-distribution region (high KL). This is a quantitative statement of Goodhart's Law in RL.

**Practical Defense**: Set a KL penalty coefficient $\beta$; periodically check with held-out gold reward (e.g., human evaluation or verifiable answers); avoid too many iterative updates on a single RM.

---

### Q6. Why are ground-truth verifiers (RLVR) safer than learned judges?

RLVR (Reinforcement Learning with Verifiable Rewards) was systematically applied to mathematical reasoning by DeepSeekMath<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a></span>: for tasks with deterministic answers (math problems, code unit tests), directly use programmatic checks of result correctness as the reward signal, rather than training a reward model.

Safety comparison:

| Dimension | Learned Judge / RM | Ground-truth Verifier |
|---|---|---|
| Signal Truthfulness | Approximate (has fitting error) | Exact (rules/symbolic execution) |
| Over-optimization Risk | High (can be "hacked" by the policy) | Extremely low (answer correctness is a binary fact) |
| Out-of-distribution Generalization | Unreliable outside training distribution | Independent of policy distribution, always trustworthy |
| Blind Spot Inheritance | May share blind spots with the generator | No parameters, no blind spots |
| Applicable Scope | Broad (but imprecise) | Only for mechanistically verifiable tasks |

**Why "hacking" is easy for RMs but hard for verifiers**: The RM's out-of-distribution behavior is unconstrained; the policy can find "high-score but low-quality" outputs unseen by the RM. A programmatic verifier only checks if the final result conforms to the specification — the policy cannot cheat on "the specification itself" (the specification is exogenous).

**Limitations**: RLVR's applicable scope depends on task verifiability. Natural language generation, summarization, creative writing, etc., do not have a single correct answer and cannot be directly applied. Therefore, RLVR and learned rewards are not substitutes but complementary — prefer RLVR on verifiable tasks; on open-ended generation tasks, one must rely on RM + KL constraints.

---

### Q7. When does self-improvement stagnate? What is the role of exploration/diversity? How to empirically distinguish true improvement from reward hacking?

**Three roots of stagnation**:

1. **Filtering signal saturation**: When the model can answer all problems in the training set correctly, the correct samples from rejection sampling are nearly duplicates of existing training data — the gradient signal approaches zero.
2. **Insufficient exploration due to distribution narrowing**: As described in Q2, after distribution entropy decreases, the model no longer samples sufficiently diverse rationales, making it hard to recover from error paths or discover new strategies.
3. **Task difficulty exceeds bootstrapping capability**: For problems the model is completely incapable of, no correct samples can be produced via rejection sampling; an external curriculum (simpler sub-problems, a stronger teacher model) is needed.

**Role of exploration/diversity**: Self-improvement is essentially an **exploitation-exploration tradeoff**. Only keeping correct samples each round is pure exploitation; to maintain improvement, one needs:
- **Increase temperature**: Sample more diverse paths, trading higher variance for a higher probability of hitting new correct paths.
- **Diversity reward**: Add an entropy regularization or diversity term to the optimization objective to prevent mode collapse.
- **Curriculum learning**: Progressively introduce harder problems instead of repeatedly iterating on a fixed set.

**Empirical methods to distinguish true improvement from reward hacking**:

| Metric | Signal of True Improvement | Signal of Reward Hacking |
|---|---|---|
| Proxy reward vs Gold reward | Both rise synchronously | Proxy rises, but Gold reward plateaus or drops |
| Held-out evaluation set | Gains on **unseen problem types** as well | Gains only within training distribution, drops outside |
| Manual spot-check of output quality | Visible improvement in reasoning step quality | Surface fluency, but more logical loopholes in steps |
| Output diversity | Distribution entropy maintained or slightly decreased | Distribution entropy collapses rapidly, outputs highly repetitive |
| KL divergence trend | Slow increase, positively correlated with Gold | KL increases rapidly, exceeding Gao et al.'s $d^*$ |

The golden standard is always: **retain a held-out evaluation set completely untouched by the self-training process, and periodically score it with a credible oracle (human or verifiable answers).** Only if this score continuously rises can it be considered true improvement.

---

## References

> All are original sources of foundational methods, individually verified (title + arXiv ID). Click superscript to jump, click ↩ to return.

<ol>
<li id="ref-1">Zelikman et al. <em>STaR: Bootstrapping Reasoning With Reasoning</em>. 2022. <a href="https://arxiv.org/abs/2203.14465">arXiv:2203.14465</a> — Iterative fine-tuning on correct chain-of-thought, without large-scale rationale annotations. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Gulcehre et al. <em>Reinforced Self-Training (ReST) for Language Modeling</em>. 2023. <a href="https://arxiv.org/abs/2308.08998">arXiv:2308.08998</a> — Grow-Improve offline RL loop, more sample-efficient than online RLHF. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Yuan et al. <em>Self-Rewarding Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.10020">arXiv:2401.10020</a> — Same model acts as generator and LLM-as-Judge, co-improving both via iterative DPO. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Chen et al. <em>Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.01335">arXiv:2401.01335</a> — SPIN: using the previous self as an opponent, self-improvement using only SFT data. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Bai et al. <em>Constitutional AI: Harmlessness from AI Feedback</em>. 2022. <a href="https://arxiv.org/abs/2212.08073">arXiv:2212.08073</a> — Constitution-guided self-critique and revision; RLAIF replaces human harmlessness annotations with AI preferences. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Shinn et al. <em>Reflexion: Language Agents with Verbal Reinforcement Learning</em>. 2023. <a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a> — Language reflection stored in episodic memory, multi-round self-correction without weight updates. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Madaan et al. <em>Self-Refine: Iterative Refinement with Self-Feedback</em>. 2023. <a href="https://arxiv.org/abs/2303.17651">arXiv:2303.17651</a> — Frozen model self-loop: generate → critique → revise, gains across tasks without training. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — Divergence curve of proxy reward and gold reward with increasing KL; RM scaling laws. <a href="#fnref-8">↩</a></li>
<li id="ref-9">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — Process supervision (PRM) outperforms outcome supervision (ORM); PRM800K dataset. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Shumailov et al. <em>The Curse of Recursion: Training on Generated Data Makes Models Forget</em>. 2023. <a href="https://arxiv.org/abs/2305.17493">arXiv:2305.17493</a> — Recursive training on generated data causes distribution tails to disappear (model collapse). <a href="#fnref-10">↩</a></li>
<li id="ref-11">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — RLVR (Verifiable Reward RL) + GRPO; programmatic verification replaces learned RM. <a href="#fnref-11">↩</a></li>
</ol>
