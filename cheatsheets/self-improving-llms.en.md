# Self-improving LLMs

> How LLMs use **self-generated signals** to "score → filter → train" themselves, iterating continuously without large-scale human annotation.

> ⚠️ **Study notes, not the authors' research** (see README integrity statement). Numbers / conclusions follow the original papers; uncertain points are noted.

## 0. The core loop

```
Generate → Filter / Score → Train → Repeat
```

Each round, the **current policy** produces candidate answers or preference pairs; some filtering mechanism (rules, another model, self-scoring) eliminates low-quality outputs; the remaining high-quality samples are used to update weights; the next round reruns with the new model. This **self-improvement loop** is the shared skeleton of all methods.

---

## 1. Bootstrap-then-Train: bootstrapping from correct traces

### 1.1 STaR — Rejection Sampling + Iterative Fine-tuning

STaR<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Iteratively fine-tunes on chain-of-thought where the correct answer was produced, without requiring a large-scale rationale dataset. <a href="https://arxiv.org/abs/2203.14465">Zelikman 2022 ↗</a></span></span> (Self-Taught Reasoner) is the foundational scheme for bootstrapped fine-tuning of LLM chain-of-thought:

1. **Rollout**: sample $K$ chain-of-thought rationales per problem.
2. **Filter**: retain only those rationales whose final answer is correct (rejection sampling).
3. **Fine-tune**: SFT on the retained set, update the model.
4. **Hint-retry**: for problems where all answers are wrong, give the correct answer and ask the model to "re-explain", then mix those into training (prevents easy problems from dominating the training set).

After $T$ iterations, the model is simultaneously the data generator and the data filter.

### 1.2 RFT — Rejection Sampling Fine-tuning

RFT is a simplified variant of STaR: it skips hint-retry, and directly retains the correctly-answered samples from $K$ samples per problem, aggregating them into a richer fine-tuning set. Key finding: **multiple correct solutions to the same problem** have higher diversity than a single solution, which helps generalization.

### 1.3 ReST — Grow-Improve Offline RL Loop

ReST<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Generates a large dataset from the current policy (Grow), then filters by reward threshold and fine-tunes (Improve), more sample-efficient than online RLHF. <a href="https://arxiv.org/abs/2308.08998">Gulcehre 2023 ↗</a></span></span> splits the loop into two phases:

- **Grow**: sample from the current policy $\pi_\theta$, build an offline dataset $\mathcal{D}$, score with reward function $r(\cdot)$.
- **Improve**: fine-tune $\pi_\theta$ on the subset $\mathcal{D}_{\ge\tau}$ where reward exceeds threshold $\tau$.

Key point: the **Improve phase can be repeated multiple times** (progressively raising $\tau$ for stricter filtering), while Grow only needs occasional refresh — computation is more concentrated compared to online RLHF's per-step sampling.

| Method | Filter criterion | Online? | Training method |
|---|---|---|---|
| STaR / RFT | Answer correctness (rule) | Quasi-online (iterative) | SFT |
| ReST | Reward function threshold | Offline batches | SFT / best-of-N distillation |

---

## 2. Self-Rewarding: the model as its own judge

Self-Rewarding Language Models<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">The same model both generates responses and scores them using LLM-as-a-Judge; uses iterative DPO to jointly improve generation and judgment capabilities. <a href="https://arxiv.org/abs/2401.10020">Yuan 2024 ↗</a></span></span> breaks the assumption of "requiring an external reward model":

1. Sample multiple responses to the same prompt.
2. The **same model** scores each response using LLM-as-a-Judge format (score + rationale).
3. Construct preference pairs $(y_w, y_l)$ by score, update with DPO.
4. In the next round, judging ability also improves — **both abilities share the same parameters and co-evolve**.

The prerequisite of this approach: the model's **generation ability** and **judgment ability** must mutually promote rather than contaminate each other. Experiments show this holds for several iterations, but whether long-term degradation occurs remains an open question (see §6 Failure modes).

---

## 3. Self-Play: using "the previous-round self" as opponent

SPIN<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Current model vs. previous-round model: the latter generates negative samples, the former learns to distinguish them; self-improvement using only SFT data. <a href="https://arxiv.org/abs/2401.01335">Chen 2024 ↗</a></span></span> (Self-Play Fine-Tuning) is inspired by game theory:

- **Positive samples**: human responses $y^*$ in the original SFT dataset.
- **Negative samples**: outputs $\tilde{y}$ of the previous-round model $\pi_{\theta_{t-1}}$ on the same prompts.
- **Objective**: the current model $\pi_{\theta_t}$ learns to **distinguish** genuine human responses from "old-self" outputs, updated with a DPO-like loss.

$$\mathcal{L}_{\text{SPIN}}(\theta_t) = -\mathbb{E}\left[\log\sigma\!\left(\lambda\log\frac{\pi_{\theta_t}(y^*|x)}{\pi_{\theta_{t-1}}(y^*|x)} - \lambda\log\frac{\pi_{\theta_t}(\tilde{y}|x)}{\pi_{\theta_{t-1}}(\tilde{y}|x)}\right)\right].$$

Key point: no additional human preference annotation required — negative samples are entirely provided by **the model's own historical versions**. As iterations proceed, $\pi_{\theta_t}$ continually approaches the human distribution until convergence when the two become indistinguishable.

---

## 4. AI Feedback: letting AI replace human preference labeling

Constitutional AI<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Uses a set of "constitutional" principles to guide the model to self-critique and revise outputs; AI-generated preference data replaces human harmlessness annotation (RLAIF). <a href="https://arxiv.org/abs/2212.08073">Bai 2022 ↗</a></span></span> (CAI / RLAIF) is currently the most influential approach to "AI replacing human preference":

**SL-CAI (supervised phase)**:
1. Model generates a harmful draft response.
2. Given a constitutional principle (e.g., "avoid discriminatory content"), the model **self-critiques**.
3. The model **revises** its response based on the critique.
4. The revised response is used for SFT.

**RL-CAI (reinforcement phase)**:
5. The model scores a pair of responses using AI judgment (which better conforms to the constitution), constructing preference data.
6. Train a reward model with AI-labeled preferences, then iterate with RL.

Difference from STaR/ReST: **the filtering signal comes from constitutional principles**, not task answer correctness — targeting alignment rather than reasoning ability.

---

## 5. Inference-time Self-correction (Training-free)

The following two methods do not update weights; they belong to **inference-time self-improvement**, conceptually related to the training loops above but different:

### 5.1 Reflexion — Verbal Reinforcement Learning

Reflexion<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">Agent converts task feedback into natural language reflections, stores them in episodic memory, and references them on the next attempt — no gradient updates needed. <a href="https://arxiv.org/abs/2303.11366">Shinn 2023 ↗</a></span></span> lets the agent in multiple **trial-and-error loops**:

- Execute task → receive environment feedback (success / failure / error message).
- Generate **verbal reflection**: summarize in natural language "what went wrong and how to improve next time".
- Store reflection in **episodic memory**, inject into context in the next round.

Success rate improves significantly after a few iterations — but improvement **exists only in the current session's context**, and is lost on restart.

### 5.2 Self-Refine — Generate-Critique-Revise Loop

Self-Refine<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">The same frozen LLM loops: generate output → self-critique → revise based on critique, no training or additional supervision required, consistently gains across tasks. <a href="https://arxiv.org/abs/2303.17651">Madaan 2023 ↗</a></span></span> has a fixed three-step loop:

$$\text{output}_0 \xrightarrow{\text{critique}} \text{feedback}_0 \xrightarrow{\text{refine}} \text{output}_1 \xrightarrow{\cdots}$$

No training, no additional supervision — directly leverages the **pretrained model's self-critique capability**. Experiments show gains across multiple tasks (code, summarization, dialogue, math), but the ceiling is limited by the model's initial judgment ability.

| Method | Improvement occurs at | Updates weights? | Persistent? |
|---|---|---|---|
| Reflexion | inference-time, multiple attempts | No | No (within context) |
| Self-Refine | inference-time, single loop | No | No |
| STaR / ReST / SPIN / CAI | training-time | Yes | Yes |

---

## 6. Failure modes

The self-improvement loop looks appealing, but has three structural risks:

### 6.1 Reward Hacking

When the filtering signal (reward model, LLM scoring, rule filter) is imperfect, the model learns strategies that **score high but are not truly correct**: shortcut answers, surface-fluent but content-wrong rationales, outputs specifically designed to please the scoring template.

- Root cause: the gap between the optimization target (proxy reward) and the true target (task quality) — **Goodhart's Law**.
- Mitigation: use diverse, independent evaluation signals; limit the magnitude of a single RL update (KL constraint).

### 6.2 Model Collapse / Distribution Narrowing

Each round only retains "high-score" samples, eliminating the diversity of low-score samples. After multiple rounds, the training set tends toward uniformity, model output diversity decreases, and generalization worsens. This is especially severe in Self-Rewarding-style "model scores itself" schemes: the model's blind spots are **systematically inherited** in preference labeling.

$$\text{Diversity}(\pi_{\theta_t}) \le \text{Diversity}(\pi_{\theta_{t-1}}) \quad \text{(if only top-}k\text{ kept per round)}$$

### 6.3 Reward Model Over-optimization (RM Over-optimization)

The reward model in the RL phase is itself an **approximation**; as the policy is continuously optimized, the score curve eventually decouples from true quality (the out-of-distribution regions of the reward model are exploited). A KL divergence penalty is the standard mitigation:

$$\mathcal{J}(\theta) = \mathbb{E}[r(y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}].$$

Larger $\beta$ keeps the policy closer to the reference policy, but at the cost of more conservative improvement.

---

## 7. From-scratch code: STaR-style rejection-sampling fine-tuning loop

```python
"""
STaR-style rejection-sampling fine-tuning loop (illustrative).
Dependencies: transformers, torch — uses GPT-2 for pedagogical demonstration; replace with a larger model for real training.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ---------- Hypothetical QA data ----------
PROBLEMS = [
    {"question": "What is 3 + 5?",  "answer": "8"},
    {"question": "What is 7 * 6?",  "answer": "42"},
    {"question": "What is 12 - 4?", "answer": "8"},
]

# ---------- Helper: simple answer extraction ----------
def extract_answer(text: str) -> str:
    """Extract the last number from generated text (for demonstration)."""
    import re
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else ""

# ---------- 1. Rollout: sample K rationales per problem ----------
def rollout(model, tokenizer, problems, K=4, max_new=64, device="cpu"):
    """Returns list of (question, rationale, is_correct)."""
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

# ---------- 2. Filter: retain only correct rationales ----------
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

> The above code is for illustrative purposes only: real STaR uses larger models, longer rationales, and hint-retry as a fallback. The core workflow (sample → filter → finetune → repeat) is consistent with the paper.

---

## Stratified follow-ups

### L1 Foundational

<details class="qa"><summary>1. What is the "generate-filter-train" loop for self-improvement? Why does it need to loop rather than run once?</summary>

Answer: The loop skeleton is: the current policy generates candidate outputs → a filtering/scoring mechanism eliminates low-quality samples → weights are updated on the retained set → the new model reruns. The problem with a single pass is: the initial model has limited capability, so the correct samples from one round cover a narrow range; as the loop runs, each new model can solve problems the previous round got wrong, gradually expanding training signal coverage and achieving bootstrapped capability improvement.

**Follow-up:** Under what conditions will this loop stop iterating? Does it stop because the model has converged, or because it has hit an insurmountable bottleneck? → The loop stopping usually means the filtering signal has saturated (the current model answers all training set problems correctly, rejection sampling produces new samples nearly identical to existing data, and gradients approach zero), or the task difficulty exceeds the model's bootstrapping capability (no correct samples can be produced for completely unsolvable problems). Neither is true "convergence" — both are stagnation; true convergence requires that the model also stops improving on a held-out set.

</details>

<details class="qa"><summary>2. How does STaR train chain-of-thought without rationale annotations? What problem does <strong>hint-retry</strong> solve?</summary>

Answer: STaR samples $K$ chain-of-thought traces per problem and retains only the rationales with a correct final answer for SFT (rejection sampling), thus requiring no human rationale annotations. Hint-retry handles "problems where the model gets everything wrong" — it gives the correct answer and asks the model to re-generate an explanation, then mixes those into the training set, preventing easy problems from monopolizing the training set and leaving hard problems without any gradient updates.

**Follow-up:** Hint-retry introduces the correct answer as a hint — what bias does this bring? → The rationale the model re-generates may contain "backward reasoning from the answer" — on the surface the steps look reasonable, but they actually rely on information that shouldn't have been known. When such samples are mixed into the training set, they may teach the model to imitate backward-reasoning patterns rather than genuine forward reasoning, harming generalization to new problems; this is precisely one motivation for proposing PRM (process reward model).

</details>

<details class="qa"><summary>3. Why are Reflexion and Self-Refine called "training-free"? Can their improvements persist?</summary>

Answer: Neither updates model weights — Reflexion stores natural language reflections in episodic memory and injects them into context, while Self-Refine loops "generate → critique → revise" within a single conversation. Improvements cannot persist: Reflexion's improvement exists only within the current session's context and is lost on restart; Self-Refine similarly starts from scratch on each new call.

**Follow-up:** To make Reflexion or Self-Refine improvements persistent, what system architecture design is needed? → The high-quality "reflections" or "revised outputs" extracted after multiple rounds of inference can be used as new training data for periodic offline SFT or DPO updates, forming a closed loop from inference-time improvement to training; but the core challenge is filtering the quality of this self-produced data — erroneous reflections that are made persistent are harder to correct than one-time errors.

</details>

<details class="qa"><summary>4. What role does the "constitution" play in Constitutional AI? How does AI feedback replace human preference annotation?</summary>

Answer: The "constitution" is a list of principles (e.g., "avoid discriminatory content"); in the SL-CAI phase it guides the model to self-critique and revise harmful drafts, and the revised outputs are used for SFT. In the RL-CAI phase, the model uses AI scoring (which response better conforms to the constitution) to construct preference pairs, trains a reward model with AI-labeled preferences, and then performs RL — thus replacing large-scale human harmlessness annotation with AI feedback (RLAIF).

**Follow-up:** Beyond a preset static list of constitutional principles, what directions exist for making feedback principles more dynamic and adaptive? → One direction is meta-reward models: dynamically retrieving or generating the most relevant principles for a given input and harmful output for critique, rather than applying the same rules to all prompts; another direction is automatically distilling "implicit constitutions" from large amounts of annotated data using clustering or inductive learning, allowing principles themselves to evolve with task distribution rather than being fixed by humans.

</details>

### L2 Intermediate

<details class="qa"><summary>5. How do the Grow and Improve phases of ReST divide their roles? Why is it more sample-efficient than online RLHF?</summary>

Answer: The Grow phase uses the current policy $\pi_\theta$ to sample at large scale and scores with a reward function to build offline dataset $\mathcal{D}$; the Improve phase fine-tunes on the subset where reward exceeds threshold $\tau$, and can raise $\tau$ to repeat Improve multiple times. The reason for sample efficiency is: Grow only needs occasional refresh, and Improve can reuse the same batch of data for multiple rounds; online RLHF must sample new data every step, dispersing computation.

**Follow-up:** In what scenarios is ReST's offline batch mechanism actually inferior to online RLHF? → When the task distribution or environment changes dynamically, the dataset built in the offline Grow phase becomes rapidly outdated, and its reward signals reflect the distribution under the old policy; online RLHF samples in real-time at every step, can track distribution drift, and is better suited for non-stationary environments (e.g., user preference drift in dialogue systems, external dependency changes in code execution) — at the cost of higher computational expense.

</details>

<details class="qa"><summary>6. SPIN uses "the previous-round self" as negative samples — what are the advantages and disadvantages compared to DPO using human preference pairs?</summary>

Answer: SPIN's advantage is requiring no additional human preference annotation, with negative samples entirely provided by the historical version $\pi_{\theta_{t-1}}$, at low cost. The disadvantage is that the theoretical upper bound is locked by SFT data quality — SPIN's convergence condition is $\pi_{\theta_t} = p_\text{data}$, and it cannot surpass the human SFT data; moreover, as iterations progress, negative sample quality approaches positive sample quality, and the contrastive signal grows progressively weaker. DPO's human preferences can cover alignment dimensions beyond SFT data, but annotation costs are high.

**Follow-up:** The contrastive signal in SPIN vanishes with iteration — what analogy does this have with GAN training dynamics, and what does it imply for choosing iteration count in practice? → The discriminator loss in SPIN is analogous to the GAN discriminator's loss approaching zero as the generator approaches the real distribution: when $\pi_{\theta_t} \approx p_\text{data}$, positive and negative samples are nearly indistinguishable, and gradients approach zero — analogous to GAN training saturation. The practical implication is: SPIN is well-suited for early iterations to close the SFT distribution gap, but should be switched to methods with external validation signals (such as RLVR) in later stages; otherwise extra iteration rounds yield neither benefit nor protection against distribution drift.

</details>

<details class="qa"><summary>7. What problems arise when "generation" and "judgment" share the same parameters in Self-Rewarding?</summary>

Answer: The generator's blind spots are inherited by the judge — if the model is weak at a certain type of reasoning (e.g., counterfactual reasoning), its probability of scoring that reasoning highly is also lower than the true level, because judgment and generation capability share the same knowledge base. Preference data therefore systematically underestimates this type of capability. There is also self-confirmation bias: the model tends to give high scores to answers that "sound like its own style". This is not the random error of hallucination but a **systematic bias** — the preference signal itself pulls the model toward its own existing style, forming a positive feedback loop that accumulates and amplifies errors rather than mean-reverting (see Deep-dive Q3).

**Follow-up:** On what kinds of task distributions will self-confirmation bias be most severe? → It is most severe on tasks requiring divergent/creative thinking (e.g., story creation, brainstorming) — the judge tends to reward answers similar in style and logical path to its own, systematically suppressing novel but "atypical" high-quality outputs; conversely, on tasks with objective correct/wrong criteria (e.g., math, code), self-confirmation bias is relatively weakened by external verifiable signals as a constraint.

</details>

<details class="qa"><summary>8. Are reward hacking and RM over-optimization the same thing? How does KL constraint mitigate it?</summary>

Answer: RM over-optimization is a specific form of reward hacking: after continuous policy optimization, outputs that achieve high proxy reward but low true quality are found in the RM's out-of-distribution regions — a quantitative manifestation of Goodhart's Law. KL constraint limits the extent to which the policy deviates from the reference model via $\mathcal{J}(\theta) = \mathbb{E}[r(y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}]$; larger $\beta$ is more conservative, preventing the policy from entering the RM's high-scoring out-of-distribution regions.

**Follow-up:** What cost does KL constraint impose while mitigating RM over-optimization, and is this cost worth it? → The cost is limiting the exploration range: the policy cannot deviate far from the reference model, even if the RM points in a genuinely better direction. When RM quality is high, this cost is worth it (stable improvement prioritized over risky exploration); when RM quality is poor, the KL constraint locks the policy near a suboptimal region, unable to either improve or discover the true optimum — in this case, improving the RM or switching to verifiable signals should be prioritized over increasing $\beta$.

</details>

### L3 Deep-dive

<details class="qa"><summary>9. How is model collapse / distribution narrowing characterized mathematically? What mitigation strategies exist (temperature sampling, diversity constraints, data mixing)?</summary>

Answer: Retaining only top-$k$ samples per round is statistically equivalent to truncated sampling — each time only the high-density regions of the distribution are taken, causing entropy to monotonically decrease over multiple rounds: $H(\pi_{\theta_t}) \le H(\pi_{\theta_{t-1}})$. Distribution narrowing has two root layers, analyzed by Shumailov et al.: the first is **statistical approximation error** — each sampled dataset is finite, low-probability tail events are underestimated or missing, and the next-round model learns from this finite sample and cannot recover missing tails regardless of model capacity; the second is **function approximation error** — limited model capacity further compresses representation of already-low-frequency patterns. Both errors **accumulate additively** across iterations: statistical error provides "worse raw material" for function error, while function error makes the base distribution for the next round's sampling narrower than the last, forming a negative spiral. In Self-Rewarding settings, the situation is more severe: the judge itself is also drifting, the gap between preference data and true preferences grows with each round, and collapse signals and bias signals amplify simultaneously. In long-chain chain-of-thought tasks, tail solutions (unconventional reasoning paths) disappear first in the initial rounds of filtering, yet these paths are often precisely what is needed to handle out-of-distribution problems. Mitigation strategies fall into three categories: raising sampling temperature to preserve low-probability paths (trading higher variance for higher probability of hitting diverse correct paths); adding a diversity reward term to explicitly reward output diversity against the main loss; periodically mixing in original human data as a distributional anchor to prevent unconstrained drift. Among the three, "mixing human data" is most fundamental, because it directly blocks the accumulation source of both errors — having an anchor provides tail replenishment.

**Follow-up:** Can the above mitigation strategies theoretically fully prevent distribution narrowing? → No: raising temperature only increases sampling variance, while the training objective (MLE in SFT or preference loss in DPO) itself still pushes the model to fit high-density regions of the data; diversity reward is an additive term, and its tradeoff with the main loss requires tuning and cannot precisely cover all tail patterns; only continuously mixing external data can theoretically break the additive spiral of both errors.

</details>

<details class="qa"><summary>10. What selection bias does retaining only correct samples each round in STaR introduce? How can it be mitigated?</summary>

Answer: Filtering only on final answers is equivalent to $p_{\text{train}}(r|x) \propto p_\theta(r|x)\cdot\mathbf{1}[\text{answer}(r)=a^*]$, leading to three types of bias: ① incorrect reasoning paths enter the training set as long as the answer happens to be correct; ② the training set comes from $\pi_{\theta_{t-1}}$, deviating from the true reasoning distribution each round; ③ for hard problems that are filtered out entirely, the model receives no gradient updates and cannot improve by bootstrapping. Mitigation directions: use a process reward model (PRM) to score each step (Lightman et al.) to reduce step-level errors; mix original SFT data to prevent complete distribution drift.

**Follow-up:** PRM was proposed to address outcome bias — but does PRM itself have similar limitations or new risks? → Yes: PRM requires step-level annotations as supervision, still relying initially on human or strong-model annotation, whose distribution is equally subject to annotation bias; more importantly, when using PRM scores as the optimization objective, the same "PRM over-optimization" risk applies — the policy may learn to generate step sequences that cater to PRM scoring patterns but contain actual reasoning errors, essentially the same structure as RM over-optimization, just with the granularity shifted from outcome to step.

</details>

<details class="qa"><summary>11. Combining Self-Rewarding's LLM-as-Judge with an external reward model — what information does each contribute? How to prevent the two from "colluding"?</summary>

Answer: LLM-as-Judge contributes the generator's own semantic understanding and stylistic judgment (broad coverage but subject to self-confirmation bias); the external RM contributes independent-parameter preference estimation (initially uncorrelated in bias direction with the generator, but subject to out-of-distribution generalization failure). The key to preventing collusion is maintaining parameter independence and ensuring training data is not cross-contaminated; simultaneously, using held-out verifiable answers or human evaluation as third-party signals for periodic calibration, to avoid two approximation signals accumulating errors in the same direction.

**Follow-up:** If labels generated by LLM-as-Judge are used to train the external RM, can the two still be considered "independent"? What effect does this have on collusion prevention? → No longer independent: the RM's training data already carries the bias direction of LLM-as-Judge; although the parameters are separate, the information is already contaminated, and the two will accumulate errors in the same blind-spot direction rather than correcting each other — this is the most common "false complementarity" trap. True independence requires the RM's annotation data source to be independent from LLM-as-Judge (e.g., human annotation or programmatic judgment of verifiable tasks), and uses held-out third-party signals to periodically verify whether the divergence directions of the two are correlated.

</details>

<details class="qa"><summary>12. If the self-improvement loop converges to a local optimum (the model cannot produce data better than itself), what are the ways to break out?</summary>

Answer: Based on Deep-dive Q7, stagnation has three root causes — filtering signal saturation, insufficient exploration after distribution narrowing, and task difficulty exceeding bootstrapping capability. Corresponding breakout strategies: ① curriculum learning: introduce harder or more diverse problems to expand signal coverage; ② raise sampling temperature or add diversity reward to restore exploration capability; ③ introduce an external stronger teacher model (or RLVR verifier) to provide training signals independent of the current model; ④ switch methods, from bootstrapping methods like SPIN/STaR to RL methods with external verification signals.

**Follow-up:** After introducing an external teacher model or RLVR verifier, what new failure modes may still arise? → Three main risk categories: ① over-optimization on verifiable tasks causing degradation on open tasks — the model specializes in fitting the verifier's judgment rules, with declining generalization to tasks without a unique answer; ② teacher-student distribution mismatch — if the teacher model or verifier's task distribution does not match the target distribution, the provided signals are ineffective or even harmful; ③ shortcut learning — the model learns to guess the teacher's output patterns or the verifier's rule boundaries rather than internalizing general reasoning capabilities, immediately failing when the verifier set is changed.

</details>

---

---

## Deep-dive

> Detailed analysis of advanced interview questions. ⚠️ Study notes, not the authors' research. Numbers follow the original papers.

---

### Q1. STaR only retains "correctly answered" samples: what selection bias does this introduce? What is the formal impact on the learned distribution?

**Core bias**: STaR<a class="cite" href="#ref-1">1</a> in each round of iteration includes only chain-of-thought traces with a correct final answer in the training set. Formally this is equivalent to:

$$p_{\text{train}}(r \mid x) \propto p_\theta(r \mid x) \cdot \mathbf{1}[\text{answer}(r) = a^*]$$

where $r$ is the rationale, $x$ is the problem, and $a^*$ is the reference answer.

Three structural consequences:

1. **Correctness ≠ reasoning quality**: a rationale may arrive at the correct answer through luck, shortcuts, or "backward reasoning from the answer", yet the reasoning steps themselves are wrong. Since filtering only looks at the final answer, **incorrect reasoning paths are systematically mixed into the training set**. This aligns with the motivation for Lightman et al.<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a></span> proposing the process reward model (PRM): outcome supervision cannot distinguish "correct reasoning getting the right answer" from "incorrect reasoning getting the right answer".

2. **Accumulated distribution shift**: the round-$t$ training set is drawn from the conditional distribution of $\pi_{\theta_{t-1}}$, not the true reasoning distribution $p^*(r \mid x)$. After each iteration, $\pi_{\theta_t}$ further deviates from $p^*$, and the "correctness rate" signal of the filter becomes increasingly self-referential.

3. **Hard-problem blind spots**: for problems the model consistently fails, the filtered training set is empty (hint-retry covers some, but cannot fully compensate). The model receives neither gradient updates nor bootstrapped improvement on these problems, creating a "Matthew effect" — the strong get stronger, hard problems stagnate.

**Mitigation directions**: PRM scoring each step (rather than only looking at the final answer) can reduce step-level errors; data mixing (retaining original SFT data) prevents complete distribution drift.

---

### Q2. Why does iterative self-training narrow the distribution (model collapse)? Intuition + when does it bite?

**Intuition**: "retaining only high-score samples" each round is statistically equivalent to truncated sampling — only taking the high-density region of the distribution each time. Over multiple rounds, tail low-probability (but high-diversity) outputs are systematically eliminated.

Shumailov et al.<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a></span> analyzed the consequences of **recursively training on self-generated data** at both theoretical and experimental levels:

- **Statistical approximation error**: each sampled dataset is finite; tail events are underestimated or missing.
- **Function approximation error**: limited model capacity further compresses representation of low-frequency patterns.

The two errors **accumulate** through iteration, causing the distribution to continuously narrow. Intuitively characterized with an inequality:

$$H(\pi_{\theta_t}) \le H(\pi_{\theta_{t-1}}) \quad \text{(if only top-}k\text{ samples kept per round)}$$

Entropy monotonically decreases; outputs trend toward repetition and uniformity.

**When it truly bites**:

| Scenario | Why it's severe |
|---|---|
| Self-Rewarding (model scores itself) | The judge itself is drifting; the gap between preference data and true preferences keeps growing |
| Long-chain chain-of-thought tasks | Per-step sampling variance is high; tail solutions (unconventional reasoning paths) disappear first in filtering |
| Multi-turn dialogue / agent loop | History in context is also self-generated data; recursive contamination effect is stronger |
| Using only self-generated data, without mixing human data | No anchor; distribution drift is unconstrained |

**Mitigation**: periodically mix in original human data (anchor to prevent drift); raise sampling temperature to preserve diversity; use a diversity reward term to explicitly reward output diversity.

---

### Q3. Why does the judge-generator coupling in Self-Rewarding fail?

In Self-Rewarding<a class="cite" href="#ref-3">3</a>, **the same set of parameters** acts both as the generator (producing answers) and as the judge (scoring answers). This creates a structural problem: **the generator's blind spots are inherited by the judge**.

Specific mechanism:

1. **Shared blind spots**: if the generator is weak at a certain type of reasoning (e.g., counterfactual reasoning), its probability of scoring that reasoning highly is also lower than the true level — because judgment capability and generation capability share the same knowledge foundation. Preference data therefore systematically underestimates this type of capability.

2. **Self-Confirmation Bias**: the model tends to score answers that "sound like its own style" higher. This is not a random error from hallucination, but a **systematic bias** — the preference signal itself is pulling the model toward its existing style, forming a positive feedback loop.

3. **Correlated error drift**: after each DPO update, the generator and judge move synchronously toward the direction of the preference data. If the preference data itself is erroneous (coming from a judge with blind spots), the next round's judge will aggravate the bias in the same direction — errors do not mean-revert but **accumulate and amplify**.

Formally, let $J_\theta$ be the judgment score function, $G_\theta$ the generation function, both sharing $\theta$. The true quality function is $q^*$. Then:

$$\mathbb{E}[J_\theta(y) - q^*(y)] \ne 0 \quad \text{and is correlated with the bias direction of } G_\theta$$

**Contrast**: an external reward model (with independent parameters) is at least initially uncorrelated in bias direction with the generator. But it has another problem: out-of-distribution generalization failure (see Q5).

---

### Q4. SPIN converges to the SFT data distribution — why is this an upper bound? What does it mean in practice?

SPIN<a class="cite" href="#ref-4">4</a>'s theoretical convergence condition is: if and only if $\pi_{\theta_t} = p_\text{data}$ (the current model is identical to the human SFT data distribution), the loss gradient vanishes and training stops.

This mathematically provides a **strict capability upper bound**:

$$\text{SPIN limit policy} = p_\text{data} \quad \text{(SFT data distribution)}$$

Corollaries:

1. **Cannot surpass SFT data quality**: if the SFT data contains errors, biases, or capability blind spots, the model after SPIN convergence will also inherit these defects. SPIN only makes the model "more like the human SFT data" — it cannot discover new capabilities beyond that data.

2. **Negative sample quality degrades with iteration**: the round-$t$ negative samples are generated by $\pi_{\theta_{t-1}}$; as $\pi_{\theta_t} \to p_\text{data}$, negative sample quality increasingly approaches positive sample quality, and the **contrastive signal grows progressively weaker**. In practice, this manifests as: large gains in early iterations, diminishing marginal returns in later iterations approaching zero.

3. **Fundamental difference from STaR/ReST**: STaR-type methods use **task correctness** as the filtering signal, and can theoretically surpass SFT data (as long as a correct rationale exists, it can be learned). SPIN targets the ability to **distinguish from human data**, and its ceiling is determined by human data quality.

**Practical implication**: SPIN is suitable as a tool to "close the SFT gap" (eliminating the gap between the model distribution and human data distribution), but is not suitable as an infinite loop for continuous self-improvement — in later iterations, methods with external verification signals should be switched to.

---

### Q5. Reward Model over-optimization: what is the scaling-law shape of reward rising / true quality falling?

Gao et al.<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a></span> systematically studied the relationship between **degree of RM optimization** (measured by KL divergence $d$) and **true quality**, finding the curve shape depends on the optimization method:

- **Best-of-$N$ sampling**: proxy reward roughly grows as $\sqrt{\log N}$ (the paper notes fitting is difficult; this is an approximate description); true quality first rises then plateaus — over-optimization effect is relatively mild.
- **RL (policy gradient)**: proxy reward can continue rising, but true quality **monotonically decreases** after some $d^*$.

Approximate functional form (fitted in the paper):

$$\text{gold reward} \approx d\,(\alpha - \beta \ln d)$$

where $\alpha, \beta > 0$, $d = \sqrt{D_{\mathrm{KL}}}$, optimal point $d^* = \exp\!\left(\dfrac{\alpha - \beta}{\beta}\right)$ (derived by setting $\mathrm{d}R/\mathrm{d}d = 0$). Beyond $d^*$, true quality decreases as KL increases.

**Key scaling conclusion**: the coefficients $\alpha, \beta$ **smoothly change** with RM parameter count — larger RMs have a higher $d^*$, meaning the over-optimization "critical point" arrives later, but a critical point always exists. This means **scaling up the RM cannot eliminate over-optimization risk, only delay it**.

**Intuition**: the RM is an approximation fitted on limited data; the policy finds shortcuts in the RM's out-of-distribution regions (high KL regions) that achieve high proxy reward but low true quality. This is Goodhart's Law expressed quantitatively in RL.

**Practical defenses**: set KL penalty coefficient $\beta$; periodically check with held-out gold reward (e.g., human evaluation or verifiable answers); avoid too many iteration rounds on a single RM.

---

### Q6. Why is a ground-truth verifier (RLVR) safer than a learned judge?

RLVR (Reinforcement Learning with Verifiable Rewards) was systematically applied to mathematical reasoning by DeepSeekMath<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a></span>: for tasks with deterministic answers (math problems, code unit tests), correctness is checked programmatically as the reward signal, rather than training a reward model.

Safety comparison:

| Dimension | Learned Judge / RM | Ground-truth Verifier |
|---|---|---|
| Signal authenticity | Approximate (has fitting error) | Exact (rule / symbolic execution) |
| Over-optimization risk | High (policy can exploit loopholes) | Very low (answer correctness is binary fact) |
| Out-of-distribution generalization | Unreliable outside training distribution | Independent of policy distribution, always trustworthy |
| Blind spot inheritance | May share blind spots with generator | No parameters, no blind spots |
| Applicable scope | Broad (but imprecise) | Limited to mechanically verifiable tasks |

**Why exploiting loopholes is easy for RM but hard for verifiers**: the RM's out-of-distribution behavior is unconstrained; the policy can find "high-scoring but low-quality" outputs unseen by the RM. Programmatic verifiers only check whether the final result conforms to the specification — the policy cannot cheat on "the specification itself" (specifications are exogenous).

**Limitations**: the applicable scope of RLVR depends on task verifiability. Tasks like natural language generation, summarization, and creative writing have no unique correct answer and cannot directly use RLVR. Therefore RLVR and learned rewards are not substitutes but complements — prefer RLVR for verifiable tasks; for open generation tasks, RM + KL constraint is necessary.

---

### Q7. When does self-improvement stagnate? What is the role of exploration / diversity? How to empirically distinguish genuine improvement from reward hacking?

**Three root causes of stagnation**:

1. **Filtering signal saturation**: when the model can answer all training set problems correctly, rejection sampling produces correct samples nearly identical to existing training data — gradient signal approaches zero.
2. **Insufficient exploration after distribution narrowing**: as described in Q2, after distribution entropy decreases, the model no longer samples sufficiently diverse rationales, making it difficult to recover from erroneous paths or discover new strategies.
3. **Task difficulty exceeds bootstrapping capability**: for problems completely beyond the model's ability, no correct samples can be produced through rejection sampling; external curriculum (simpler subproblems, stronger teacher models) is needed.

**Role of exploration / diversity**: self-improvement is fundamentally an **exploitation-exploration tradeoff**. Retaining only correct samples each round is pure exploitation; to sustain improvement, the following are needed:
- **Higher temperature**: sample more diverse paths, trading higher variance for higher probability of hitting new correct paths.
- **Diversity reward**: add entropy regularization or a diversity term to the optimization objective to prevent mode collapse.
- **Curriculum learning**: progressively introduce harder problems rather than repeatedly iterating on a fixed set.

**Empirical methods for distinguishing genuine improvement from reward hacking**:

| Metric | Signal of genuine improvement | Signal of reward hacking |
|---|---|---|
| Proxy reward vs Gold reward | Both rise synchronously | Proxy rises but Gold reward is flat or falls |
| Held-out evaluation set | Also gains on **unseen problem types** | Only gains within training distribution; drops out-of-distribution |
| Manual spot-check of output quality | Reasoning step quality visibly improves | Surface fluency, but logical gaps in steps increase |
| Output diversity | Distribution entropy is maintained or slightly decreases | Distribution entropy collapses rapidly; outputs highly repetitive |
| KL divergence trend | Grows slowly and positively correlated with Gold | KL grows rapidly, exceeding Gao et al.'s $d^*$ |

The gold standard is always: **maintain a held-out evaluation set completely untouched by the self-training process, and regularly score with a trusted oracle (human evaluation or verifiable answers).** Only if this score consistently rises can genuine improvement be confirmed.

---

## References

> All are original sources of foundational load-bearing methods, individually verified (title + arXiv ID). Click superscripts to jump; click ↩ to return.

<ol>
<li id="ref-1">Zelikman et al. <em>STaR: Bootstrapping Reasoning With Reasoning</em>. 2022. <a href="https://arxiv.org/abs/2203.14465">arXiv:2203.14465</a> — Iteratively fine-tunes on correct chain-of-thought, without large-scale rationale annotation. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Gulcehre et al. <em>Reinforced Self-Training (ReST) for Language Modeling</em>. 2023. <a href="https://arxiv.org/abs/2308.08998">arXiv:2308.08998</a> — Grow-Improve offline RL loop, more sample-efficient than online RLHF. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Yuan et al. <em>Self-Rewarding Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.10020">arXiv:2401.10020</a> — Same model acts as both generator and LLM-as-Judge; jointly improves both via iterative DPO. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Chen et al. <em>Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.01335">arXiv:2401.01335</a> — SPIN: uses the previous-round self as opponent; self-improvement using only SFT data. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Bai et al. <em>Constitutional AI: Harmlessness from AI Feedback</em>. 2022. <a href="https://arxiv.org/abs/2212.08073">arXiv:2212.08073</a> — Constitution-guided self-critique and revision; RLAIF replaces human harmlessness annotation with AI preferences. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Shinn et al. <em>Reflexion: Language Agents with Verbal Reinforcement Learning</em>. 2023. <a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a> — Verbal reflections stored in episodic memory; multi-round self-correction without weight updates. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Madaan et al. <em>Self-Refine: Iterative Refinement with Self-Feedback</em>. 2023. <a href="https://arxiv.org/abs/2303.17651">arXiv:2303.17651</a> — Frozen model self-loop: generate → critique → revise; no training, consistent gains across tasks. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — Separation curve of proxy reward vs gold reward as KL increases; RM scaling laws. <a href="#fnref-8">↩</a></li>
<li id="ref-9">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — Process supervision (PRM) outperforms outcome supervision (ORM); PRM800K dataset. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Shumailov et al. <em>The Curse of Recursion: Training on Generated Data Makes Models Forget</em>. 2023. <a href="https://arxiv.org/abs/2305.17493">arXiv:2305.17493</a> — Recursive training on self-generated data causes distribution tails to vanish (model collapse). <a href="#fnref-10">↩</a></li>
<li id="ref-11">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — RLVR (RL with verifiable rewards) + GRPO; programmatic verification replaces learned RM. <a href="#fnref-11">↩</a></li>
</ol>
