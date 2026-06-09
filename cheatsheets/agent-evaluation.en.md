# Agent Evaluation

> The **metrology** of agent post-training: how to tell whether an agent actually works. Read [agent-foundations](cheatsheet-agent-foundations-en.html) first to know what an agent is and which benchmarks / human baselines exist; then use this page to learn **contamination / saturation / harness / trajectory evaluation** — the literacy you need to *read technical-report numbers correctly*. Afterwards head to [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl-en.html) to wire verifiable rewards into RL.

> ⚠️ **Study notes, not the author's own research** (see README disclaimer). Benchmark numbers move fast and are contamination-prone, so this page records **only original-paper human baselines + what is tested + the evaluation methodology**, never model SOTA; every concrete score carries a date / source / caveat.

## 0. TL;DR

- **Agent eval ≠ single-turn QA eval**: it measures goal attainment over a **multi-step trajectory** (execution-based terminal-state checking), not a single answer matched against a reference string; three extra axes: **reliability (pass^k), trajectory quality, resistance to time-decay**.
- Benchmarks split along **two orthogonal axes**: capability domain (coding / web / GUI / tool-user / general / ML-eng) × judging method (execution / trajectory / LLM-judge). Human baselines live in foundations §8; this page does not repeat them, only classifies + adds methodology.
- **Contamination = the #1 killer of agent eval**: test samples (GitHub issues, solutions) leak into the training set → you measure **memorization, not generalization**, and scores inflate. **LiveCodeBench** uses a **release-time window** to evaluate only problems published after the cutoff, cutting contamination at the root.
- **Saturation example (2026-02, verified)**: OpenAI officially **stopped reporting SWE-bench Verified** — scores "no longer reflect real development ability, and increasingly reflect how much of the benchmark the model has seen"; it pivoted to the contamination-resistant **SWE-bench Pro** (the same cohort drops from ~70–80% → ~23–58% magnitude, varying by shard/source/scaffold; not a stable ranking, see §3.2).
- **Harness decides the score**: swap scaffold / prompt / toolset on the same model and the score moves materially (public reports often show swings on the order of ten points, varying by scaffold/benchmark, no fixed value); a reported score MUST cite the **harness version**, otherwise you are comparing model×harness, not model.
- **Variance**: `pass@k` is an **estimator** with variance (temperature / seed / environment randomness all matter); a single number is not reproducible — report $n$ samples + an unbiased estimator.
- **Trajectory vs outcome**: outcome-only misses "lucky guess / reward hacking"; trajectory eval checks per-step plausibility, but using an LLM to judge trajectories has its own reliability pitfalls (calibration belongs to the sister page [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html); only touched here).
- **Verifiable ≠ perfect**: an execution-based verifier still false-positives (weak unit tests pass a wrong solution) / false-negatives (over-strict tests reject a correct solution — exactly SWE-bench Verified's defective items).
- **Safety eval is orthogonal to capability**: sabotage / privilege escalation / prompt-injection must be listed separately; numbers like a "multi-agent orchestrator's malicious-execution rate under injection attack" or "sabotage" **must carry a threat-model + attack-scenario caveat**, never read as general deployment behavior.

## 1. Why agent-eval ≠ single-turn eval

Single-turn QA eval: input → one output → compare to a reference (EM / F1 / accuracy), one-shot and near-deterministic. Agent eval measures **a multi-step trajectory**: actions change environment state, "success" = terminal state satisfies the goal (execution-based), not string matching. This brings several dimensions single-turn eval lacks:

| Dimension | Single-turn QA | Agent |
|---|---|---|
| Success criterion | match a reference answer (EM/F1) | **terminal state / unit test / environment check** (execution-based, verifiable) |
| Stochasticity | low (single decode) | high (multi-step sampling + environment randomness) → must run many times for a distribution |
| Reliability | accuracy | **pass^k** (all k succeed), see foundations §9 |
| Process | ignored | **trajectory quality**: step count, right tool, any reward hacking |
| Time stability | static | **contamination + saturation** make the same score decay over time |

> ❌ **Misconception:** "agent eval = run the benchmark's accuracy." Accuracy is just a point estimate; what is actually hard in agent eval is **designing the execution-based criterion** (anti-cheat), **reliability across multiple runs** (pass^k), and **whether the score is still trustworthy over time** (contamination / saturation). Ignore these three and the leaderboard number is misleading.

## 2. Benchmark taxonomy

Human baselines and "what is tested" are already listed in [agent-foundations §8](cheatsheet-agent-foundations-en.html), so this page **does not repeat** them — instead it locates each benchmark methodologically along **two orthogonal axes**:

- **Axis A · capability domain**: coding / web / GUI (computer-use) / multi-turn tool-user / general assistant / ML engineering.
- **Axis B · judging method × interactivity**: execution-based outcome (run unit tests / check terminal state) vs trajectory vs LLM-judge; static one-shot vs interactive environment.

| Benchmark | Domain | Judging | Interaction | Key design point |
|---|---|---|---|---|
| **SWE-bench**<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">2294 real GitHub issues; edit the repo so hidden unit tests pass. <a href="https://arxiv.org/abs/2310.06770">Jimenez 2023 ↗</a></span></span> | coding | execution (hidden tests) | repo interaction | real issue→PR, FAIL_TO_PASS tests as criterion |
| **WebArena**<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">812 long-horizon tasks across 4 self-hosted domains (e-commerce/forum/GitLab/CMS); functional terminal-state check. <a href="https://arxiv.org/abs/2307.13854">Zhou 2023 ↗</a></span></span> | web | execution (terminal-state) | self-hosted sandbox web | reproducible, resettable real-site replicas |
| **OSWorld**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">369 computer-use tasks on a real OS, scripted terminal-state checking. <a href="https://arxiv.org/abs/2404.07972">Xie 2024 ↗</a></span></span> | GUI | execution (terminal-state script) | real OS interaction | multimodal, cross-app, execution-based not multiple-choice |
| **τ-bench**<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Tool-agent-user multi-turn with policy constraints; introduces pass^k reliability. <a href="https://arxiv.org/abs/2406.12045">Yao 2024 ↗</a></span></span> | tool-user | terminal DB + **pass^k** | simulated-user multi-turn | policy constraints + reliability as a first-class metric |
| **GAIA**<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">General assistant: reasoning + multimodal + web + tools, three difficulty tiers, uniquely judgeable answers. <a href="https://arxiv.org/abs/2311.12983">Mialon 2023 ↗</a></span></span> | general | exact-match answer | static (with tools) | unique answer, auto-judged; tiered difficulty |
| **MLE-bench**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">75 Kaggle ML-engineering competitions, agents graded by medal rate. <a href="https://arxiv.org/abs/2410.07095">Chan 2024 ↗</a></span></span> | ML-eng | Kaggle scoring (medals) | long-horizon interaction | reuses existing leaderboards as an objective ruler |
| **LiveCodeBench**<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a><span class="cite-note">Rolls in problems by competition **release window**, evaluating only post-cutoff items to prevent contamination. <a href="https://arxiv.org/abs/2403.07974">Jain 2024 ↗</a></span></span> | coding (competitive) | execution | static-but-windowed | **time-window anti-contamination**, living |

> 💡 **Selection principle (echoing foundations Q13):** look at "does the action space + task distribution match the target scenario." Domains overlap (SWE-bench and LiveCodeBench both test coding, but the former tests repo-level issue fixing, the latter algorithmic contests + anti-contamination); the **judging method** decides "can I trust this score" more than the domain — execution-based + hidden criteria are hardest to cheat, LLM-judge is easiest to manipulate.

## 3. Contamination · Saturation · Living benchmarks

This is the section agent eval **most often crashes on**, and the first gate when reading report numbers.

### 3.1 Contamination

**Mechanism**: test samples (problems, even official solutions) enter the training corpus → the model "remembers" rather than "solves," and the score measures memory. Two paths:

1. **Direct**: the problem + solution text is crawled into the training set (SWE-bench's issues and fix PRs are all on public GitHub — inherently high-risk).
2. **Indirect**: the solution is paraphrased in blogs / forums and then enters the training set.

**How to detect contamination**: ① n-gram / substring overlap scan of the training corpus; ② **memorization-recall test** — give only the problem statement, no executable environment, and see whether the model can directly emit the official fix (if it can, strong hint it has seen it); ③ **time-window cliff** — compare scores on problems before vs after the cutoff; if older problems score far higher than structurally identical new ones, that is a contamination signal.

> 💡 **LiveCodeBench's solution: the release window.** Evaluate only on contest problems published **after** the model's training cutoff, guaranteeing at the root that "the model could not have seen it during training"; "living" = continuously rolling in new problems and retiring old ones. This is currently one of the cleanest anti-contamination paradigms (cost: problem types skew to algorithmic contests, narrower coverage than real engineering).

### 3.2 Saturation — the retirement of SWE-bench Verified (2026-02, verified)

**Saturation** = top models' scores approach the ceiling, discrimination vanishes, and the benchmark loses its discriminative power. The most representative contemporary case:

On **2026-02-23** OpenAI officially announced it would **stop evaluating frontier models on SWE-bench Verified**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">OpenAI's 2024-08 human-verified clean 500-problem subset. <a href="https://openai.com/index/introducing-swe-bench-verified/">OpenAI 2024 ↗</a></span></span><span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">OpenAI 2026-02-23: Verified is saturated + contaminated, no longer reflecting real development ability. <a href="https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/">OpenAI 2026 ↗</a></span></span>, for two reasons:

- **Contamination**: in OpenAI's elicitation experiments, frontier models on **some tasks** reproduce the official fix or task-specific details from memory given only the problem statement — a strong signal they saw the solutions during training.
- **Defective problems**: OpenAI audited the **138 problems** its o3 **did not consistently solve across 64 independent runs** (**27.6%** of the 500) and found **59.4% of them have substantial defects in test design / problem statement** — **35.5% (≈49) too narrow** (strict tests reject functionally correct solutions), **18.8% (≈26) too wide** (testing extra functionality the statement never required), the remaining ~5% description defects. I.e. part of the "failures" are benchmark misjudgments, not model incapacity.

> ✅ **OpenAI's conclusion (paraphrased):** gains on Verified "no longer reflect progress in real software-development ability, and increasingly reflect how much of this benchmark the model saw during training." This is a living textbook of "saturation + contamination" jointly killing a former gold standard.

**Replacement**: OpenAI pivoted to the contamination-resistant **SWE-bench Pro**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Scale AI 2025: a harder, contamination-resistant SWE-bench variant; public/private/commercial shards. <a href="https://labs.scale.com/leaderboard/swe_bench_pro_public">Scale AI 2025 ↗</a></span></span>, and argues for moving to **expert-authored private** benchmarks (e.g. GDPval). The magnitude gap is striking: the same cohort at ~**70–80%** on Verified drops to **~23–58% magnitude** on Pro (concrete scores shift with shard / leaderboard / scaffold, see caveat below).

> ⚠️ **caveat (integrity gate):** SWE-bench Pro's **concrete model scores** are a single 2025–26 source (the Scale leaderboard) and shift with the board; this page uses only the "~70–80% → ~23–58% magnitude gap" to illustrate the **methodological phenomenon** of saturation/contamination, not as a stable ranking or a fixed capability value for any model.

### 3.3 The living-benchmark paradigm

The general remedy for contamination + saturation: **rotate problems periodically / roll a release window (LiveCodeBench) / hold out a private set (GDPval-style expert authoring) / commercial closed-source shards (SWE-bench Pro private)**. The core is to invalidate the "memorize the problem" shortcut.

> ❌ **Misconception:** "leaderboard scores keep rising = agent ability is improving." High scores may come from **contamination** (memorizing), **scaffold engineering** (see §4), or **overfitting** to that benchmark; real progress is whether **newly released, contamination-resistant** living benchmarks + **pass^k** reliability rise in step — and verify the harness and date.

## 4. Harness · Reproducibility · Variance

### 4.1 The harness decides the score

**Harness (the eval scaffold)** = all the engineering that connects the model to the benchmark: prompt template, toolset, output parser, retry / step budget, retrieval strategy, whether environment feedback is given. **Swap the harness on the same model and the score can move materially (public reports often show swings on the order of ten points, varying by benchmark/scaffold, no fixed value)** — much of SWE-bench's early leaderboard gains came from scaffold engineering, not the model.

> 💡 **Corollary:** a benchmark compares **model × harness**, not pure model. A reported score **must** attach the harness version + config (toolset, max steps, temperature, retrieval on/off); otherwise two numbers are simply incomparable. When a report says "we reached X%" without describing the harness, discount that number's credibility.

### 4.2 Variance: pass@k is an estimator

Agent-eval randomness comes from three places: ① **sampling** (temperature > 0); ② **environment randomness** (web / tool returns vary, simulated users are random); ③ **the estimator itself** — `pass@k` is not the truth but an estimate from $n$ samples; small $n$ means high variance.

Codex / HumanEval<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a><span class="cite-note">Chen 2021 introduces HumanEval and the unbiased pass@k estimator; repeated sampling is a strong baseline. <a href="https://arxiv.org/abs/2107.03374">Chen 2021 ↗</a></span></span> gives the **unbiased estimator** for pass@k: draw $n$ samples per problem, of which $c$ are correct, then

$$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right]$$

Intuition: $\binom{n-c}{k}/\binom{n}{k}$ is the probability of "drawing $k$ samples that are all wrong," and $1-$ it is "drawing at least one correct." **Running a single group of $k$ samples and checking whether ≥1 succeeds is unbiased but extremely high-variance (one group is only 0/1); this combinatorial form uses all $n$ samples — equally unbiased but far lower-variance.** Stable only when $n \gg k$.

> ⚠️ **Reproducibility red line:** reporting only "pass@1 = 41%" without $n$, temperature, seed, harness means **nobody can reproduce it**. The responsible report: fixed seed + the mean ± interval over $n$ samples + the harness version. A lone number without these defaults to noisy variance.

**From-scratch implementation** (core agent evaluation metrics: unbiased pass@k + pass^k reliability):

```python
import numpy as np

def unbiased_pass_at_k(n, c, k):
    """n samples, c correct — unbiased probability of ≥1 correct in k draws (Chen et al.)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_agent_metrics(results, k=5):
    """results: list[dict] with {success: bool, traj_len: int, traj_cost: float}.
    Returns pass@k (capability ceiling), pass^k (reliability), avg steps/cost."""
    n = len(results)
    c = sum(1 for r in results if r["success"])
    pass_at_k = unbiased_pass_at_k(n, c, k) if n >= k else float('nan')
    # pass^k: sliding-window estimate — fraction of length-k windows where all succeeded
    successes = [r["success"] for r in results]
    windows_all_pass = sum(
        all(successes[i:i+k]) for i in range(len(successes) - k + 1)
    )
    pass_pow_k = windows_all_pass / max(len(successes) - k + 1, 1)
    avg_steps = np.mean([r["traj_len"] for r in results])
    avg_cost  = np.mean([r["traj_cost"] for r in results])
    return {"pass@k": pass_at_k, "pass^k": pass_pow_k,
            "avg_steps": avg_steps, "avg_cost": avg_cost, "success_rate": c/n}
```
# Key: ① pass@k = capability ceiling; pass^k = reliability (what deployment cares about)
# ② Report trajectory efficiency (steps/cost) alongside success rate — "20 steps to succeed" ≠ "3 steps"

## 5. Trajectory vs Outcome eval

| | outcome-only (execution-based) | trajectory (process) |
|---|---|---|
| What it looks at | only **terminal-state** success | whether **each step** is plausible |
| Strength | objective, verifiable; **hard to cheat** when the criterion is hidden | can diagnose **which step failed**, give partial credit, feed a PRM |
| Blind spot | misses "lucky guess / lucky path" and **reward hacking** (edit the test, empty impl passes CI); no diagnosis | **who judges** is hard: rules miss coverage, LLM-judge carries bias |

**Who judges the trajectory**: ① **programmatic rules** (step count, whether a forbidden tool was called, whether a dangerous action fired) — reliable but narrow; ② **LLM-as-judge** — flexible but with position bias, self-preference, easily "talked into it" reliability issues.

> 📝 **Scope boundary:** LLM-judge **calibration / reliability / de-biasing** is the home turf of the sister page [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html); this page only points at it: when judging agent trajectories, the judge's bias directly contaminates the process reward → credit assignment — details there.

**Agent-specific verifier reliability**: even execution-based "verifiable rewards" err — tests **too weak** → **false positive** (a wrong solution sneaks through), tests **too strict** → **false negative** (a functionally correct solution is rejected, exactly §3.2's 49 "too narrow" defective items in SWE-bench Verified).

> ❌ **Misconception:** "execution-based / verifiable reward = objective and reliable." Verifiable only means **harder to cheat than an LLM-judge**, not **correct**: the judge's quality ceiling is the quality of that unit-test / check script; a weak test suite passes reward hacking straight through as success, and in RL it gets amplified.

## 6. Safety & sabotage eval

Capability eval asks "can it do it right"; safety eval asks "will it do harm / overreach / get hijacked" — **the two are orthogonal**, high capability does not imply safety. Three categories to list separately:

1. **sabotage / sandbagging**: the model deliberately underperforms or covertly subverts. Anthropic's sabotage evaluations<span class="cite-wrap"><a class="cite" id="fnref-13" href="#ref-13">13</a><span class="cite-note">An Anthropic **evaluation framework**: code sabotage / sandbagging / undermining oversight / decision sabotage. <a href="https://arxiv.org/abs/2410.21514">Anthropic 2024 ↗</a></span></span> are an **evaluation framework** (code sabotage, sandbagging, undermining oversight, decision sabotage) — this page **describes only the framework, cites no specific percentage** (a commonly circulated number could not be verified, dropped per the integrity gate).
2. **prompt injection / tool poisoning**: malicious instructions in external content hijack the agent (see foundations §10 + Greshake<span class="cite-wrap"><a class="cite" id="fnref-14" href="#ref-14">14</a><span class="cite-note">Indirect prompt injection: instructions in external content hijack an LLM application. <a href="https://arxiv.org/abs/2302.12173">Greshake 2023 ↗</a></span></span>).
3. **privilege escalation / dangerous actions**: high-privilege tools are misused (drop a database, make outbound requests).

> 🚨 **Integrity gate for citing safety numbers (a multi-agent injection-attack case):** A multi-agent-system security study<span class="cite-wrap"><a class="cite" id="fnref-12" href="#ref-12">12</a><span class="cite-note">Triedman 2025: under a specific web injection attack, the attack success rate for a multi-agent orchestrator executing arbitrary malicious code (scope-locked). <a href="https://arxiv.org/abs/2503.12188">Triedman 2025 ↗</a></span></span> reports: under a **specific web injection attack**, the **attack success rate** for getting a multi-agent orchestrator to execute arbitrary malicious code reaches **58–90%** (varying by orchestrator), **up to 100% in individual configurations**. It is the **success rate of a control-flow-hijack attack under that threat model**, **not** "agents will misbehave at this rate in normal deployment." Cite it with that scope, or you inflate an attack experiment into a general fact.

> ❌ **Misconception:** "high benchmark safety score = safe deployment." Safety eval is mostly run under a **specific threat model**; change the attack surface / privilege config and the conclusion changes. Safety is **defense in depth** (least privilege + host policy gate + monitoring), not a property that a single benchmark score can "pass" (expanded in the planned agent-safety page).

---

## Stratified follow-ups

> ⚠️ The "would be asked" items below are **inferred from public JDs + technical reports**, not real interview questions.

### L1 Basics

<details class="qa"><summary>1. What is the essential difference between agent eval and single-turn QA eval? Why can't you just look at accuracy?</summary>

A: Single-turn QA is "input → one output → compare to a reference," one-shot and near-deterministic; agent eval measures **a multi-step trajectory** whose success criterion is the **terminal state satisfying the goal** (execution-based: run unit tests / check environment state), not string matching. Looking only at accuracy misses three agent-specific things: **reliability across runs** (pass^k), **trajectory quality** (any reward hacking / lucky guess), and **score decay over time** (contamination + saturation).

**Follow-up:** Why do agent benchmarks prefer execution-based criteria over answer matching? → Because the agent's "correct" is often not a unique string (code edits have many forms, web operations have many paths); it can only be judged by "is the final executable goal state reached"; execution-based is also harder to cheat (if the criterion is hidden).

</details>

<details class="qa"><summary>2. What is benchmark contamination? Why is it especially severe for agent benchmarks?</summary>

A: Contamination = test samples (problems, even official solutions) enter the training corpus, so the model "remembers" rather than "solves," and the score measures memory. It is especially severe for agent benchmarks because many tasks originate from **public GitHub** (SWE-bench's issues + fix PRs are public and crawled), so problems and gold solutions are naturally in the training set; add frontier models repeatedly farming these boards and overfitting + memorization compound.

**Follow-up:** How do you detect that a model is contaminated on a benchmark? → ① n-gram overlap scan of the training corpus; ② give only the statement, no environment, and see if it reproduces the official fix from memory; ③ compare score cliffs on structurally identical problems before vs after the training cutoff.

</details>

<details class="qa"><summary>3. What is the difference between pass@k and pass^k, and what does each one's "estimator" meaning entail?</summary>

A: `pass@k` = **at least one** of k attempts succeeds (capability upper bound, optimistic); `pass^k` = **all** k succeed (reliability, see foundations §9). Neither is the truth — both are **estimators**: in practice you draw $n$ samples per problem to estimate, and small $n$ means high variance. pass@k has the unbiased estimator $1-\binom{n-c}{k}/\binom{n}{k}$ ($c$ = number correct among $n$), lower-variance than "count one hit/miss."

**Follow-up:** Why must reporting pass@k also report $k$, temperature, and $n$? → pass@k's value varies with **$k$ and the sampling distribution (temperature)** — temperature changes per-sample success rate and diversity; whereas **$n$ (the number of samples used to estimate it) only sets the variance / confidence interval and does NOT change pass@k's expected value** (unless you actually change $k$ or do best-of-$n$). So missing $k$ / temperature → not comparable; missing $n$ / seed → not reproducible.

</details>

### L2 Intermediate

<details class="qa"><summary>4. Why was SWE-bench Verified deprecated by OpenAI in 2026? Explain along the contamination and saturation lines.</summary>

A: Two lines compound. **Contamination line**: OpenAI's elicitation experiments show frontier models on some tasks reproduce the official fix from memory given only the statement → scores reflect "how much benchmark was seen" rather than real ability. **Saturation/defect line**: OpenAI audited the 138 problems its o3 did not consistently solve across 64 runs (27.6% of 500) and found 59.4% have substantial test/description defects — 35.5% (≈49) too narrow (reject functionally correct solutions), 18.8% (≈26) too wide (demand functionality the statement never mentioned), the rest ~5% description defects; i.e. part of the "failures" are benchmark misjudgments. Conclusion: gains on Verified no longer reflect real development ability. OpenAI pivoted to the contamination-resistant SWE-bench Pro (same cohort ~70–80%→~23–58% magnitude).

**Follow-up:** What to watch when citing "drops to ~23–58% on SWE-bench Pro"? → That is a single 2025–26 source (Scale leaderboard), a shifting concrete score; use it only to illustrate the "magnitude gap from saturation/contamination," not as a stable ranking or fixed capability value.

</details>

<details class="qa"><summary>5. The same model can score materially differently (on the order of ten points) under different harnesses — what does that say? What info must a reported benchmark score carry?</summary>

A: It says a benchmark compares **model × harness**, not pure model — prompt template, toolset, parser, step budget, retrieval strategy all materially move the score (SWE-bench's early gains were largely scaffold engineering). A responsible report must carry: **harness version + toolset + max steps + temperature + seed + sample count $n$ + date**. A lone number missing these is incomparable and irreproducible.

**Follow-up:** What does this mean for "reading others' technical reports"? → Seeing "we reached X%" without the harness and date, discount comparability; before comparing scores across reports, first confirm same harness, same benchmark version, whether contamination-resistant.

</details>

<details class="qa"><summary>6. How does LiveCodeBench's "release window" prevent contamination at the root? What are its limits?</summary>

A: It evaluates only on contest problems newly published **after** the model's training cutoff — since the problems appeared after training, they **cannot** have been seen during training, cutting contamination mechanistically; "living" = continuously roll in new problems and retire old ones, staying fresh long-term. Limits: ① problem types skew **algorithmic contests**, narrower than real software engineering (no repo-level issues); ② requires ongoing operational effort; ③ different models have different cutoffs, so strict comparability requires windowing per each cutoff.

**Follow-up:** Time-window anti-contamination vs "private hold-out / expert authoring" (GDPval, SWE-bench Pro private) — what suits each? → Time windows suit domains with a steady stream of new problems (contests); private / expert authoring suits domains with no natural new-problem stream, or that need closeness to real professional tasks, at the cost of human effort for authoring + grading.

</details>

<details class="qa"><summary>7. For a concrete agent application (code-fixing, web browsing, customer-service tools, etc.), how do you choose benchmarks? What are the applicability ranges and key limitations?</summary>

A: Filter by three matching criteria in order — ① **Action-space match**: what is the agent's operation granularity? Fixing code → SWE-bench Pro / LiveCodeBench; operating websites → WebArena; controlling an OS GUI → OSWorld; multi-turn tool + user interaction → τ-bench; ML engineering → MLE-bench; general-purpose QA + tool use → GAIA. ② **Judging-method reliability**: can you tolerate LLM-judge bias? High-stakes scenarios (code deployment, financial transactions) → prefer execution-based + hidden-criterion benchmarks; low-stakes exploratory scenarios → LLM-judge acceptable for initial filtering. ③ **Maintenance status**: is it actively maintained / living? A saturated benchmark (SWE-bench Verified, retired) cannot be the sole basis; living benchmarks (LiveCodeBench) or recently released, community-active ones are more trustworthy.

Concrete mapping:

| Application | Recommended benchmarks (in priority order) | Key limitation |
|---|---|---|
| Code-fixing / PR agent | SWE-bench Pro (contamination-resistant, 41 repos/multi-language) > LiveCodeBench (general coding anti-contamination screen) | Single-repo issue-level tasks; no product/requirements context; not multi-repo coordinated changes |
| Web-operating agent | WebArena (self-hosted, reproducible) | 4 self-hosted site categories + auxiliary knowledge/tool sites (map/wiki/calculator); not live-web OOD |
| GUI / desktop-operating agent | OSWorld (real OS, scripted terminal-state check) | 369 tasks, desktop OS not mobile; limited cross-app coverage |
| Tool + dialogue agent (customer service / assistant) | τ-bench (pass^k reliability) + GAIA (general) | τ-bench domains fixed (retail/airline); GAIA only looks at terminal state |
| ML-engineering agent | MLE-bench (Kaggle medal rate) | Skews to contest style; does not test day-to-day ML ops |

> 💡 **Core principle:** there is no "best" benchmark — only a benchmark whose "action space + risk tolerance" match. Define your success criteria first — what does "success" mean concretely in this scenario — then map through the three-axis framework.

**Follow-up:** What if no existing benchmark fits your scenario? → ① Check whether a general-purpose benchmark (GAIA / τ-bench) can serve for an initial screen; ② if not, **build a domain-specific eval**: the key is making the success criterion **executable** (script / unit test / terminal-state check) and **keeping it invisible to the agent**; ③ estimate the construction cost: typically dozens to hundreds of items depending on baseline success rate + effect size + desired CI width (do a power analysis to determine item count; as a rough heuristic often ≥100), plus ongoing maintenance (item rotation to prevent overfitting).

</details>

<details class="qa"><summary>8. What are the known biases of LLM-as-judge for agent trajectories? When is it usable and when is it not?</summary>

A: LLM-as-judge for agent trajectories carries **three categories of bias risk that have been repeatedly documented in the literature**:

1. **Position / order bias**: the order in which answers appear in the prompt, and whether content sits near the prompt's end, significantly affects judge scores. Gets worse with longer trajectories — agent multi-step trajectories are inherently longer than single-turn QA, so the last steps dominate the judge far more than the first.
2. **Self-preference**: a judge from the same model family tends to give higher scores to same-family agents; this has been observed across GPT, Gemini, and Claude families in multiple studies — directionally consistent, though the magnitude varies by model/task rather than being a fixed value.
3. **Style / verbosity / authority bias**: if the agent's trajectory includes confident, detailed, or authoritative-sounding explanation text, the LLM judge is more inclined to mark it correct even when the underlying action is wrong — the judge is influenced by rhetoric rather than facts (sometimes called verbosity bias, authority bias, or rationale bias).

**When LLM-judge is usable:**
- **Relative ranking** (A vs B which trajectory is better) — more reliable than absolute scoring; biases partially cancel in pairwise comparison.
- **Coarse filtering** — screen out obviously bad trajectories (didn't call a tool / infinite loop / empty response), with rule-based criteria as the primary gate and LLM as the fallback.
- **Assistive pre-labeling** — pre-annotate before human review to accelerate manual checking.

**When it is NOT usable / requires extreme caution:**
- **As the reward source for RL training** — biases will be actively searched and amplified by the policy, turning the objective into "please the judge" rather than "complete the task."
- **As an absolute pass/fail criterion** — reliability insufficient for a high-stakes single decision.
- **Cross-model-family capability ranking** — self-preference makes the ranking systematically incomparable.

> ❌ **Misconception:** "Pick the 'strongest' LLM as judge and it will be objective." Stronger ≠ less biased; on some dimensions (e.g. verbosity bias — judging longer outputs as better) stronger LLMs can even be **more** biased, and cross-model-family self-preference does not vanish with capability gains. **Dedicated judge calibration methodology** (position randomization, cross-family judge ensembles, pairwise rather than absolute scoring) lives in the sister page [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html).

**Follow-up:** Is there a middle ground more reliable than LLM-judge but more flexible than execution-based? → **Programmatic rules + LLM combinations**: dimensions that can be ruled (step budget, forbidden-tool calls, action-format validity) use deterministic rules; dimensions that need "semantic judgment" (is this step directionally correct) use LLMs, but only in a statistical sense (mean + variance across multiple judges), never relying on any single judge for a single decision.

</details>

### L3 Deep

<details class="qa"><summary>9. To design a contamination-resistant, reward-hacking-resistant, reproducible coding-agent eval, what elements are needed?</summary>

A: Synthesizing §3–§5: ① **anti-contamination** — use problems after the training cutoff (time window) or a private hold-out, and audit contamination (memorization-recall test); ② **anti-reward-hacking** — use **hidden extra unit tests + environment terminal-state checks** (invisible to the agent) as the success criterion, preventing test-editing / empty-impl-passes-CI; audit the unit-test quality itself (avoid too-weak→false-positive, too-strict→false-negative); ③ **reproducibility** — fixed seed, public harness, report the unbiased pass@k estimate over $n$ samples + interval + version date; ④ **reliability** — report **pass^k** not just pass@1, exposing multi-run stability; ⑤ **living** — rotate problems periodically to prevent long-term overfitting.

**Follow-up:** Why is "hidden criteria" the key to anti-hacking? → Once the agent can see the criterion (assertions / unit tests), it satisfies the criterion rather than truly solving (edit the test file, hard-code the expected output); hiding the criterion where the agent can't see it forces it to actually reach the goal state.

</details>

<details class="qa"><summary>10. What are the blind spots of outcome-only vs trajectory eval? Execution-based verifiers also false-negative/false-positive — how to mitigate?</summary>

A: **outcome-only** blind spot — misses "lucky guess / lucky path" and reward hacking, and does not diagnose which step failed or give partial credit; **trajectory** blind spot — "who judges each step" is hard, rules miss coverage, LLM-judge carries position/self-preference. **execution-based verifier false-negative/false-positive**: tests too weak → a wrong solution sneaks through (false positive), too strict → a functionally correct solution is rejected (false negative, i.e. SWE-bench Verified's defective items). Mitigations: ① audit + strengthen the test suite (diverse inputs, hidden extra assertions); ② cross multiple criteria (unit tests + terminal-state check + human review of a subset when needed); ③ keep human-in-the-loop review for high-risk items; ④ handle verifier noise robustly in RL (don't amplify a single weak signal as truth).

**Follow-up:** Why is a weak verifier more dangerous in RL training than in static eval? → In static eval a weak verifier just mismeasures; in RL it is the **reward source**, and reward hacking is actively searched and amplified by the policy — the agent optimizes toward "fooling the weak test" rather than "truly solving," getting worse as it trains.

</details>

<details class="qa"><summary>11. (Integrity) How do you cite safety numbers like a multi-agent injection-attack's "malicious-execution rate" or sabotage honestly?</summary>

A: Carry the **precise scope**. That "multi-agent orchestrator executes arbitrary malicious code" number (a study reports **58–90%**, up to 100% in individual configs) is "the **attack success rate** under a **specific web injection attack**," i.e. the **success rate of a control-flow hijack**, not "agents misbehave at this rate in normal deployment." Sabotage is similar: Anthropic's sabotage evaluations are an **evaluation framework**; cite that it is a framework / specific setting, and if a circulated percentage can't be traced to the original setting, **drop it**. Principle: safety numbers are almost always bound to a **threat model + experimental setting**; citing one detached from the setting distorts it.

**Follow-up:** Why are safety-eval numbers more easily misquoted than capability-eval numbers? → Safety numbers are often upper bounds under "worst case / a specific attack," naturally eye-catching and easy to quote out of context; and they are extremely sensitive to the **threat model** (change the attack surface and the conclusion changes), so quoting one detached from the setting almost inevitably inflates it.

</details>

<details class="qa"><summary>12. How does evaluation drive agent RL training? What are the key traps in the eval-signal → RL-reward pipeline?</summary>

A: The standard pipeline is **benchmark eval → reward design → RL training → re-eval**. Every step can fail:

1. **Same benchmark for both eval and reward**: if the same verifier / test suite serves as both the RL reward and the final evaluation, the agent learns to "game this specific verifier" rather than truly solve. You must **separate eval and train verifiers** — the criterion used during training and the one used for final evaluation are two independent sets.
2. **Weak verifier amplified in RL** (echoing Q10): in static eval a weak verifier merely "mismeasures"; in RL it is the **reward source**, and reward hacking is actively searched and amplified by the policy — the agent optimizes toward "fool the weak test" (edit the test file, empty impl passes CI), getting worse as it trains.
3. **Distribution shift**: RL changes the policy → trajectory distribution changes → a verifier calibrated on pre-RL trajectories may no longer be valid post-RL — the agent enters behavior regions the verifier has never seen, and the false-positive/false-negative rate becomes unknown.
4. **Proxy reward gap**: the benchmark measures "task success," but the RL reward is a **proxy** for task success (unit tests passing / terminal-state check passing / LLM-judge score). The gap "success ↔ proxy" is where reward hacking thrives — the agent optimizes the proxy, not the true objective.
5. **RL runs contaminate the benchmark**: if RL training rounds use benchmark problems for environment interaction (not just final eval), those problems cannot be reused — the agent has already trained on them; testing them again is contamination.

Remedies: ① **physically isolate** eval and training problem sets (independently sampled, not a random split — agent-eval problems are naturally few, random splits are not safe enough); ② use **training-dedicated hidden** criteria for RL reward (verifiers independent of final eval, with zero overlap); final-eval criteria must be **at least equally strong** and fully held-out; ③ periodically spot-check agent trajectories manually for reward-hacking signals; ④ use multiple heterogeneous reward sources for cross-validation rather than relying on a single signal.

> ❌ **Misconception:** "After RL training, run the benchmark once more and that's evaluation." If any part of that benchmark (problems / criteria / environment) was used during the RL training loop, it is already contaminated — what you are measuring is **overfitting on the training set**, not generalization.

**Follow-up:** Are eval/train verifier split and eval/train problem split the same thing? → No. The problem split is "which problems are for training vs evaluation." The verifier split is using **different criteria** — during training expose only a training-dedicated weak criterion (few visible unit tests), and during evaluation use a held-out full-strength criterion set (zero overlap with training criteria). Both layers are needed; one layer alone is insufficient.

</details>

<details class="qa"><summary>13. Beyond pass@k/pass^k, what metrics matter for production agent systems? Why might an agent with high academic benchmark scores perform poorly after deployment?</summary>

A: Academic benchmark scores measure **capability ceiling under ideal conditions**. A production agent must also be assessed along five additional dimensions:

| Metric dimension | What it measures | Why academic benchmarks don't test it |
|---|---|---|
| **Per-task cost** | token count × unit price / successful task (including retries) | Benchmarks don't impose a budget constraint |
| **Latency / steps** | average steps to completion + P99 latency | Academic evals don't penalize step count (most benchmarks) |
| **Budget-constrained reliability** | task success rate under a given token / time budget | standard pass@k treats k attempts as free; does not reflect cost-effectiveness under token/time constraints |
| **User-experience metrics** | task abandonment rate, user satisfaction, human-handoff rate | Benchmarks have no real users |
| **Safety-incident rate** | privilege escalations / injections / erroneous actions per task | Capability benchmarks ignore safety |

> 💡 **The "last mile" — high academic scores then deployment crash:** a common cause is the agent carrying the academic benchmark's assumption of infinite retries + no budget into production — pass@k is high but every step is expensive (unbounded search / retries), and once a token budget or timeout is imposed in production, the real success rate collapses. **Remedy:** add a "pass@k under budget" curve to eval reports, treating cost as a first-class citizen.

> ❌ **Misconception:** "High benchmark score = deployable." A benchmark is **necessary but not sufficient** — it only proves "can do it under ideal conditions," not "can do it stably under budget / safety / UX constraints." Before deployment, run at least one round of **shadow deployment** (a parallel shadow deploy where the agent executes on real traffic but doesn't take over), using the real distribution as the final gate.

**Follow-up:** How is shadow deployment done concretely, and what do you watch? → Deploy the agent in an **isolated environment with the same distribution as production** (production traffic mirroring / production snapshot replay / dry-run mode), with **writes routed through no-op / sandbox adapters so there are no real side effects** (no payments triggered, no external messages sent, not visible to real users); run silently for days to weeks, tracking:
- **Shadow metrics** (measurable offline): real-distribution task success rate vs academic benchmark gap, per-task cost distribution, safety red-line trigger rate, latency/step distribution
- **Live metrics** (require canary / A-B / assisted-human deployment to measure): user abandonment rate, human-handoff rate, satisfaction — these are NOT measurable in shadow mode and must be called out separately

The gap between these numbers and the academic benchmark scores is the chasm between "can solve problems" and "can do the job." Before shadow deployment, confirm: privacy/PII compliance (whether mirrored traffic contains sensitive data), write-operation isolation completeness, and that an approval process is in place.

</details>

---

## References

> All are load-bearing methods / primary sources, web-checked item by item (title + arXiv ID / official URL; the 2026 SWE-bench Verified retirement and Pro scores were re-checked 2026-06). Click a superscript to jump, click ↩ to return.

<ol>
<li id="ref-1">Jimenez et al. <em>SWE-bench: Can Language Models Resolve Real-World GitHub Issues?</em> ICLR 2024. <a href="https://arxiv.org/abs/2310.06770">arXiv:2310.06770</a> — real code-fixing eval. <a href="#fnref-1">↩</a></li>
<li id="ref-2">OpenAI. <em>Introducing SWE-bench Verified</em>. 2024-08. <a href="https://openai.com/index/introducing-swe-bench-verified/">openai.com</a> — 500-problem human-verified subset. <a href="#fnref-2">↩</a></li>
<li id="ref-3">OpenAI. <em>Why we no longer evaluate SWE-bench Verified</em>. 2026-02-23. <a href="https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/">openai.com</a> — retired due to saturation+contamination; 138-problem audit ~59% tests defective. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Scale AI. <em>SWE-bench Pro Leaderboard</em>. 2025. <a href="https://labs.scale.com/leaderboard/swe_bench_pro_public">labs.scale.com</a> — contamination-resistant SWE-bench variant (public/private/commercial shards). <a href="#fnref-4">↩</a></li>
<li id="ref-5">Zhou et al. <em>WebArena: A Realistic Web Environment for Building Autonomous Agents</em>. ICLR 2024. <a href="https://arxiv.org/abs/2307.13854">arXiv:2307.13854</a> — self-hosted reproducible web eval. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Xie et al. <em>OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments</em>. NeurIPS 2024. <a href="https://arxiv.org/abs/2404.07972">arXiv:2404.07972</a> — computer-use terminal-state checking. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Mialon et al. <em>GAIA: a benchmark for General AI Assistants</em>. ICLR 2024. <a href="https://arxiv.org/abs/2311.12983">arXiv:2311.12983</a> — general assistant, uniquely judgeable answers. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Yao et al. <em>τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains</em>. 2024. <a href="https://arxiv.org/abs/2406.12045">arXiv:2406.12045</a> — multi-turn tool-user + pass^k reliability. <a href="#fnref-8">↩</a></li>
<li id="ref-9">Chan et al. <em>MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering</em>. ICLR 2025. <a href="https://arxiv.org/abs/2410.07095">arXiv:2410.07095</a> — Kaggle ML engineering, medal rate. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Jain et al. <em>LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code</em>. ICLR 2025. <a href="https://arxiv.org/abs/2403.07974">arXiv:2403.07974</a> — release-window anti-contamination. <a href="#fnref-10">↩</a></li>
<li id="ref-11">Chen et al. <em>Evaluating Large Language Models Trained on Code</em>. 2021. <a href="https://arxiv.org/abs/2107.03374">arXiv:2107.03374</a> — HumanEval + unbiased pass@k. <a href="#fnref-11">↩</a></li>
<li id="ref-12">Triedman, Jha, Shmatikov. <em>Multi-Agent Systems Execute Arbitrary Malicious Code</em>. 2025. <a href="https://arxiv.org/abs/2503.12188">arXiv:2503.12188</a> — under specific injection attacks, multi-agent orchestrators execute malicious code at 58–90% ASR (scope-locked). <a href="#fnref-12">↩</a></li>
<li id="ref-13">Anthropic (Benton et al.). <em>Sabotage Evaluations for Frontier Models</em>. 2024. <a href="https://arxiv.org/abs/2410.21514">arXiv:2410.21514</a> — sabotage/sandbagging eval framework (this page cites the framework only). <a href="#fnref-13">↩</a></li>
<li id="ref-14">Greshake et al. <em>Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection</em>. 2023. <a href="https://arxiv.org/abs/2302.12173">arXiv:2302.12173</a> — indirect prompt injection. <a href="#fnref-14">↩</a></li>
</ol>
