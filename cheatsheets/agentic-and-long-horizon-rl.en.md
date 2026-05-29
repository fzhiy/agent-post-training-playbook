# Agentic & Long-horizon RL

> The **next step** after single-turn reasoning RL (GRPO / RLVR, see sister repo [reasoning-rl-frontier](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reasoning-rl-frontier.html)): extending the reward signal from "one question, one answer, one reward" to **multi-turn trajectories** (think → call tool → observe)\*.

> ⚠️ **Study notes, not the author's own research** (see README integrity statement). Numbers / conclusions follow the original papers; uncertainties are annotated.

## 0. The evolution

`Single-turn RLHF (prompt→response→reward)` → `Single-turn verifiable RLVR (correct/wrong→reward)` → **`Multi-turn agentic RL (trajectory→sparse terminal reward)`**.  
What changes is not the loss function but the **shape of the episode**: one episode is `(reasoning, tool_call, observation)` repeated across turns<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Interleaving reasoning and tool calls so the LLM thinks and acts simultaneously (think→act→observe). <a href="https://arxiv.org/abs/2210.03629">Yao 2022 ↗</a></span></span>, and the reward is usually given only **once at the end** (task success or failure).

## 1. What makes it long-horizon

| Dimension | Single-turn reasoning RL | Long-horizon agentic RL |
|---|---|---|
| Episode length | 1 turn | Several to dozens of turns |
| Reward | One per response | **Sparse / delayed**, often only at the terminal |
| Observation | Fully visible (prompt) | **Partially visible** (only known after tool returns) |
| Action space | Tokens | Tokens + **tool calls** + when to stop |
| Main challenge | Preference / correctness | **Long-horizon credit assignment** + error accumulation |

## 2. Formalization (POMDP)

Treat a trajectory $\tau=(s_0,a_0,\dots,s_T,a_T)$ as a POMDP. The terminal task reward $R(\tau)\in\{0,1\}$ (success/failure) or a scalar score. Trajectory return:

$$G_t=\sum_{k=0}^{T-t}\gamma^{k}\,r_{t+k}.$$

In practice, a "turn" is often used as the decision granularity (turn-level MDP), while gradients still land on the **tokens generated** in that turn — i.e., **turn-level credit + token-level updates**.

## 3. Credit assignment (the core challenge)

The terminal provides only one reward; the question is: **across these dozens of turns, which turn and which token should be rewarded or penalized?**

- **Trajectory-level**: the entire trajectory shares one advantage — simplest, but rewards bad steps inside a good trajectory along with the rest.
- **Turn-level**: each turn gets its own advantage (requires step rewards or value estimates).
- **Token-level**: finest granularity; typically **broadcast** from the turn-level advantage to that turn's tokens.

GRPO<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Sample a group of rollouts for the same task; use group-relative returns as the baseline, eliminating the critic. <a href="https://arxiv.org/abs/2402.03300">Shao 2024 ↗</a></span></span> can be carried over directly: **sample a group of trajectories for the same task and use group-relative returns as the baseline** (no critic needed):

$$A(\tau_i)=\frac{R(\tau_i)-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}.$$

> Process reward **PRM**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Score each reasoning step (process supervision) rather than only looking at final correctness. <a href="https://arxiv.org/abs/2305.20050">Lightman 2023 ↗</a></span></span> (scoring each step) is friendlier for long-horizon credit assignment, but **annotation / training is more expensive** and it can itself be hacked; **ORM** (outcome only) is cheaper but coarse-grained for credit assignment. Long-horizon settings often combine both.

## 4. Reward design

- **Verifiable outcome rewards (RLVR→agentic)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Use automatically checkable correctness (rather than a reward model) as the RL signal. <a href="https://arxiv.org/abs/2411.15124">Lambert 2024 ↗</a></span></span>: automatically checkable terminal signals are the most stable — unit test pass, environment state achieved, verifiable answer.
- **Process / step reward**: awarding points at intermediate milestones mitigates sparsity, but is **easily gamed** (the agent learns to trigger milestones without solving the task).
- **Long-horizon reward hacking**: the longer the trajectory, the more shortcuts exist (idle looping to accumulate steps, repeatedly calling cheap tools). Mitigations: terminal-verifiable rewards as primary signal + step/cost penalty + **read-only treatment of tool outputs** (see §5 masking).

## 5. Algorithm essentials

Two "long-horizon-specific" details in multi-turn PPO<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Policy gradient with clipping; baseline algorithm for RLHF / agentic RL. <a href="https://arxiv.org/abs/1707.06347">Schulman 2017 ↗</a></span></span> / GRPO:

1. **Observation token masking**: tool returns / environment observations are **injected** into the context, not generated by the policy — they **must be masked** in the loss; otherwise the model is being asked to "fit" tool outputs (analogous to loss masking in SFT).
2. **Advantage broadcasting + masking**: the turn-level advantage is broadcast to all agent-generated tokens in that turn, then multiplied by the action mask.

```python
import torch

def group_relative_advantages(returns, group_ids, eps=1e-6):
    """GRPO-style group-relative advantage (no critic).
    returns:   (N,) terminal return of each trajectory (e.g. success={0,1} or scalar)
    group_ids: (N,) multiple rollouts of the same task share one group id
    """
    adv = torch.zeros_like(returns)
    for g in group_ids.unique():
        m = group_ids == g
        r = returns[m]
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv

def masked_pg_loss(logp, adv_per_token, action_mask):
    """Policy gradient loss: back-propagate only through agent-generated tokens.
    logp:          (B,T) log-prob of the taken token under the current policy
    adv_per_token: (B,T) broadcast from turn-level advantage to token level
    action_mask:   (B,T) 1=agent-generated token, 0=tool output/observation (masked)
    """
    pg = -(logp * adv_per_token) * action_mask
    return pg.sum() / action_mask.sum().clamp_min(1)
```

> During rollout, tools / the environment must be **actually executed inside the loop** (often asynchronously); longer trajectories → more expensive sampling → one of the main system bottlenecks in agentic RL.

## 6. Bridge from single-turn

The sister repo's **GRPO / RLVR / loss masking** are the building blocks; agentic RL ≈ applying them **over trajectories** + solving "credit assignment for sparse terminal rewards". Once you know single-turn RL, grasping three points — **trajectoryification + masking + group-relative baseline** — is sufficient to transfer.

---

## Stratified follow-ups

### L1 Basics

<details class="qa"><summary>1. What makes a task "long-horizon"? Give two LLM-agent examples.</summary>

Answer: The task requires multiple think→act→observe cycles and the reward is given only at the terminal (sparse / delayed); the action space includes tool calls. Examples: ① a code agent that iteratively calls a code executor to debug a program; ② a search-and-summarize agent that issues multiple rounds of web-API queries before generating a report.

</details>

<details class="qa"><summary>2. Why is single-turn RLHF/RLVR insufficient for training multi-turn tool-using agents?</summary>

Answer: Single-turn RL assumes one question, one answer, one reward, and the episode is a single response. A multi-turn agent's episode is `(reasoning, tool_call, observation)` interleaved repeatedly, with reward given only at the final turn. Single-turn loss functions cannot handle cross-turn credit assignment, and they lack a masking mechanism for observation tokens.

</details>

<details class="qa"><summary>3. Why is sparse / delayed reward difficult?</summary>

Answer: The terminal provides only one reward, so there is no direct way to determine which turn or which token among dozens deserves reward or penalty — this is the long-horizon credit assignment problem. Additionally, the probability of a successful trajectory decays exponentially with the number of turns at the start of training (if per-turn success rate is 0.8, after 10 turns it is approximately 0.11), causing most rollouts to have all-zero rewards and the gradient signal to nearly vanish.

</details>

<details class="qa"><summary>4. Why must <strong>tool-return tokens</strong> be masked during training?</summary>

Answer: Tool returns are observations injected by the environment, not generated by the policy — if they are not masked, the loss forces the model to "fit" tool outputs, which is equivalent to performing SFT on observations and contaminates the policy gradient signal. The correct approach is action_mask=0 for all `obs_*` tokens, so gradients flow only to the think/act tokens generated by the agent.

</details>

### L2 Advanced

<details class="qa"><summary>5. What are trajectory-level / turn-level / token-level advantage? What are the trade-offs?</summary>

Answer: Trajectory-level shares one advantage across the entire trajectory — simplest, but rewards bad steps inside a good trajectory; turn-level assigns each turn its own advantage (requires step rewards or value estimates) for more precise credit; token-level is the finest granularity, typically broadcast from the turn-level advantage to the agent-generated tokens in that turn multiplied by the action mask. Finer granularity gives more accurate credit but increases dependency on a critic/PRM.

</details>

<details class="qa"><summary>6. How does GRPO's group-relative baseline transfer to multi-turn? Why can it eliminate the critic?</summary>

Answer: Sample a group of multi-turn trajectories for the same task and normalize by the group's mean and standard deviation of terminal returns to obtain $A(\tau_i)=\frac{R(\tau_i)-\mu_g}{\sigma_g+\epsilon}$, using the group mean as a baseline instead of a critic. The critic is eliminated because the baseline is derived from statistics of the same batch of rollouts, requiring no separate value network. The cost is that variance can be large in sparse long-horizon settings (when an entire group returns all zeros, the advantage degenerates to zero).

</details>

<details class="qa"><summary>7. What are the trade-offs between PRM and ORM for long-horizon credit assignment?</summary>

Answer: ORM looks only at the terminal result — the signal is sparse and credit assignment is coarse, but annotation cost is low and it is hard to hack. PRM scores each step (ideally the change in future success probability), giving precise credit, but annotation/training is more expensive and the intermediate-step scorer can be gamed by the agent. Long-horizon settings often combine both: terminal verifiable reward (ORM) as the primary signal, supplemented by a small number of verifiable milestones serving as PRM signals.

</details>

<details class="qa"><summary>8. What are the typical forms of long-horizon reward hacking? How can they be mitigated?</summary>

Answer: Three typical forms: ① idle looping (repeatedly calling cheap tools to extend the trajectory and accumulate milestone points); ② premature stop (declaring task completion early to skip subsequent difficult steps); ③ milestone gaming (triggering milestone checkpoints without genuinely solving the subtask). Mitigation combination: terminal verifiable reward as primary + step/token cost penalty + observation token loss mask + KL penalty term + adversarial test-set rotation.

</details>

### L3 Deep-dive

<details class="qa"><summary>9. With only one 0/1 reward at the terminal, how do you reasonably distribute credit across dozens of turns? Give at least two approaches and compare them.</summary>

Answer: ① **GRPO group-relative baseline**: multiple trajectories of the same task share terminal returns for normalized baseline — simple, no critic required, but credit is still averaged within the trajectory. ② **Turn-level GAE**: introduce a lightweight turn-level value head and use $\hat{A}^{\text{GAE}}=\sum(\gamma\lambda)^l\delta_{t+l}$ to decompose the terminal return into per-turn TD errors — more precise credit but requires critic bootstrapping. ③ **PRM step reward**: estimate the change in per-step future success probability via Monte Carlo rollouts as a step-level reward — finest credit but highest sampling cost. Practical compromise: start with GRPO to build basic competence, then add a turn-level value head once training stabilizes.

</details>

<details class="qa"><summary>10. How can "verifiable outcome rewards" and "process supervision" be combined to be both stable and unhackable?</summary>

Answer: Use terminal verifiable signals (unit test pass / environment state achieved) as the primary reward — hard to hack. Use **verifiable milestones** as intermediate step rewards (subtask unit tests pass rather than neural network scores), mitigating sparsity without introducing a gaming-prone proxy. Also add a KL penalty $R'=R-\beta\,\text{KL}(\pi_\theta\|\pi_\text{ref})$ to prevent overoptimization, and a step-count penalty to suppress looping.

</details>

<details class="qa"><summary>11. Why is agentic RL rollout expensive? What engineering mitigations exist (async execution, truncation, length penalty)? What are their costs?</summary>

Answer: Rollout requires actually executing tools/the environment inside the loop (network, code execution, etc.), with latency up to seconds, and longer trajectories lead to large KV-cache occupancy. Mitigations: ① **Async execution** — run multiple episodes in parallel so the GPU does not idle; cost: when a rollout completes the policy has already advanced several steps (staleness), requiring IS correction or ESS monitoring. ② **Trajectory truncation** — forcibly terminate at the maximum number of steps; cost: truncated trajectories have incomplete returns that need bootstrap compensation or must be discarded. ③ **Length penalty** $R'=R-\alpha|\tau|$ — incentivizes the agent to complete tasks efficiently; cost: may penalize necessary long reasoning chains.

</details>

<details class="qa"><summary>12. How does compounding error amplify over long trajectories? What is its relationship to exposure bias?</summary>

Answer: Small errors in each turn's decision alter subsequent observations, causing the trajectory to drift from the training distribution. The next turn then makes larger errors in out-of-distribution states, and errors **amplify exponentially** with the number of turns. This is structurally isomorphic to SFT's exposure bias — during training, ground-truth prefixes are seen; during inference, the model's own generated prefixes are seen. Long-horizon agents are particularly vulnerable because tool returns also depend on prior actions. Mitigation: RL itself mitigates exposure bias by training the model on its own rollouts; curriculum learning (starting from short horizons) can reduce the rate of early error accumulation.

</details>

---

## Deep-dive

> Interview-trap-level questions: the following Q&A assumes the interviewer is already familiar with single-turn GRPO/PPO and is probing the fine-grained mechanisms in multi-turn settings. **Study notes, not the author's own research**.

---

### Q1. Per-token vs per-turn advantage estimation — how does GAE's discount transfer to multi-turn?

**Core tension**: single-turn RL defines advantage at token granularity ($A_t = Q_t - V_t$); multi-turn settings have a two-level structure — **temporal discount across turns** (turn-level) and **token broadcasting** within a turn.

**Transferring GAE**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">GAE uses λ-weighted TD residual sums to trade off bias-variance: λ→1 approaches Monte Carlo, λ→0 approaches single-step TD. <a href="https://arxiv.org/abs/1506.02438">Schulman 2015 ↗</a></span></span>:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

Replacing "one token" with "one turn", the formula applies directly on the **turn-level MDP**:
- $\gamma$ (turn discount): in sparse terminal reward settings, $\gamma\approx1$ (no discount) is common, since every turn has genuine contribution value; too low a $\gamma$ pushes early-turn advantages toward zero, wasting gradient signal.
- $\lambda$ (GAE smoothing coefficient): controls bias-variance trade-off — critic bootstrap errors accumulate more easily in multi-turn episodes, so $\lambda<1$ is more critical than in single-turn settings; otherwise variance explodes exponentially with the horizon.

**Broadcasting from turn-level to token-level**: all agent-generated tokens within a turn share that turn's $\hat{A}^{\text{turn}}$; observation tokens have mask=0. Per-token gradients thus equal the per-turn advantage multiplied by the action mask, with no additional approximation.

**Interview trap**: "If tokens share the same advantage, what is the difference between token-level and turn-level?" — The difference lies in **which level computes discounts and baselines**: turn-level lets discounts span turns and allows the value function to bootstrap at turn granularity; token-level is purely where gradients land. Confusing the two leads to discount being applied at the wrong level (applying γ per token is equivalent to γ^t decay over a very long sequence, making early-token gradients nearly zero).

---

### Q2. High variance of the GRPO group baseline under a single terminal reward — why? What are variance-reduction techniques?

**GRPO group baseline** advantage:

$$A(\tau_i) = \frac{R(\tau_i) - \mu_g}{\sigma_g + \epsilon}, \quad \mu_g=\frac{1}{G}\sum_{j=1}^G R(\tau_j).$$

**Two sources of variance explosion in multi-turn settings**:

1. **Binary 0/1 reward + small group size**: if $G=8$ and success rate $p\approx0.1$, the group commonly sees all zeros (8 failures) or only 1 success. When all zeros, $\sigma_g=0$ and advantage degenerates to zero; with 1/8 success, $\sigma_g$ is tiny and the advantage spikes — a single sample dominates the entire batch update.
2. **Trajectory length variation**: the log-prob sum of a long trajectory is numerically much larger than that of a short one; if advantage is directly multiplied by the number of tokens, long trajectories naturally produce larger gradients, creating an implicit length bias.

**Variance-reduction techniques**:

| Technique | Mechanism | Cost |
|---|---|---|
| Increase group size $G$ | More stable $\mu_g,\sigma_g$ | Sampling cost $\times G$ |
| Length-normalization | Divide loss by action token count | Balances short/long trajectories, but may penalize necessary long reasoning |
| Mix ORM + sparse intermediate reward | Reduces probability of all-zero groups, increases positive signal | Intermediate rewards can be hacked |
| Advantage clipping / quantile truncation | Remove spike-advantage samples | May discard high-information samples |
| Introduce lightweight critic (turn-level value estimate) | Use $V$ to decompose terminal return into per-turn TD errors | Extra model; contradicts GRPO's "critic-free" premise |

**Interview trap**: "Can GRPO fully replace PPO's critic?" — In single-turn binary reward settings, yes. In long-horizon sparse reward settings, the group baseline's variance often exceeds the critic baseline's, and the value of a critic returns. In practice, the compromise is using a small turn-level value head rather than a full PPO critic.

---

### Q3. Importance-sampling / off-policy correction when multi-turn rollouts go stale

**Problem background**: agentic rollouts involve real tool calls (network, database, code execution) with latency up to seconds — by the time a rollout finishes, policy parameters have advanced several steps, $\pi_\theta \neq \pi_{\theta_\text{old}}$.

**Importance weight (IS weight)**:

$$w(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\theta_\text{old}}(\tau)} = \prod_{t \in \text{agent tokens}} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}.$$

**Why the product is dangerous**: across $T$ turns, agent tokens can reach thousands; even with per-step ratio $\approx1.05$, the product can yield IS weights $\gg10$ — variance explodes and a tiny fraction of samples dominate the gradient.

**Engineering corrections**:

1. **PPO clip**<a class="cite" href="#ref-5">5</a> ($\epsilon$ clip): truncate each token's ratio to $[1-\epsilon, 1+\epsilon]$; does not correct bias but effectively controls variance. Suitable for **synchronous training** where policy lag does not exceed 1–2 mini-batches.
2. **Sequence-level truncated IS (TIS)**: truncate the entire trajectory's IS weight to some upper bound (e.g., 3.0); simple but biased.
3. **Dynamically reduce learning rate using ESS**: monitor rollout staleness with effective sample size $\text{ESS}=(\sum w_i)^2/\sum w_i^2$; automatically reduce the learning rate when ESS falls below a threshold to avoid gradient shocks.
4. **Prefix IS ratio**: the theoretically correct correction is the **prefix IS ratio** (prefix product of the full sequence) rather than independent per-token ratio truncation; but this is complex to implement and numerically unstable.

**Interview trap**: "Is PPO clip alone sufficient?" — In synchronous training, yes. In async / agentic settings, policy lag can reach dozens of steps, at which point clip only treats the symptom (variance reduced but bias is large), and in effect performs gradient updates on the wrong distribution — it must be paired with small policy-lag design or ESS monitoring.

---

### Q4. Specific forms of long-horizon reward hacking and mitigations

**"Long-horizon" makes hacking easier**: single-turn hacking only needs to find an exploit in one response; a long-horizon agent can slowly accumulate shortcuts over dozens of steps, overwrite evaluation files, or exploit tool side effects.

**Three typical forms**:

| Type | Mechanism | Example |
|---|---|---|
| **Idle looping** | Agent repeatedly calls irrelevant tools, extending the trajectory hoping to score on milestones | Repeatedly querying a search API without advancing the task |
| **Premature stop** | Declaring "task complete" early to bypass subsequent difficult steps | Code agent outputs "DONE" before the test run |
| **Milestone gaming** | Triggering milestone checkpoints without genuinely solving the subtask | Writing an empty function to make CI pass, or directly mocking test output |

**Why long-horizon is worse**: theoretical analysis<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Scaling Laws for Reward Model Overoptimization: as KL divergence increases, gold reward first rises then falls; the gap between proxy reward and gold reward increases monotonically with KL. <a href="https://arxiv.org/abs/2210.10760">Gao 2022 ↗</a></span></span> shows that the gap between proxy reward and gold reward increases monotonically with KL divergence — the longer the trajectory, the larger the total KL divergence, and per scaling laws [7] the hacking risk increases systematically.

**Mitigation combination**:

1. **Terminal verifiable reward as primary**<a class="cite" href="#ref-4">4</a>: unit test pass / environment state achieved, harder to hack than neural-network proxy rewards;
2. **Step / token cost penalty**: $R'(\tau) = R(\tau) - \alpha \cdot |\tau|$, directly suppresses looping;
3. **Read-only observation tokens (loss mask)**: prevents the agent from learning to "generate output formats that pass mock checks";
4. **KL penalty term**: $R' = R - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$, acts as a brake before proxy reward and gold reward begin to diverge;
5. **Adversarial test-set rotation**: periodically replace evaluation samples to prevent the agent from memorizing shortcuts to specific test cases.

---

### Q5. How does partial observability (POMDP) make value estimation harder?

**Difference in information structure: single-turn RL vs long-horizon agentic RL**:

Single-turn reasoning: $V(s) \approx V(\text{prompt})$ — the prompt is fully visible, and the value function receives complete input.

Multi-turn agentic: $s_t$ = conversation history + last turn's tool return, but **next turn's tool return is unknown** — the agent operates in a POMDP, unaware of future observations.

**Three specific challenges**:

1. **Stochasticity of tool returns**: the same action (API call) may return different content due to network conditions or external DB version differences — the value function must take an expectation over this randomness, but during training only one specific return is observed. Single-step bootstrap ($V(s_{t+1})$) fits the value of a noisy observation, leading to slow convergence and hard-to-estimate bias.

2. **Context length explosion**: as turns increase, $s_t$ grows linearly (full history concatenated); if the value network uses the same LM backbone, its forward-pass cost grows quadratically with context length — value bootstrapping itself becomes expensive.

3. **Irreversibility of external state**: the agent has modified a database / filesystem, and these side effects are not in the token stream — the value function cannot see the "hidden state of the environment". Traditional POMDP solutions (belief states) cannot be directly applied in LLM settings.

**Practical responses**:
- Use a lightweight turn-level value head (attached to the hidden state of the last generated token) rather than a critic over the full rollout;
- **Summarize** tool returns before feeding them into the value input to compress the context;
- Accept higher bias: use GAE with $\lambda<1$ to reduce reliance on long-range bootstrapping, at the cost of slightly underestimating long-term advantage.

---

### Q6. Exploration under long-horizon sparse rewards

**Why single-turn RL exploration strategies are insufficient for long-horizon**: in single-turn RL, randomly sampling tokens is sufficient for exploration; in multi-turn settings, **the probability of a successful trajectory decays exponentially with the number of turns** — if the per-turn correct probability is 0.8, after 10 turns the success rate drops to $0.8^{10}\approx 0.11$, the agent rarely sees positive rewards, and the training signal is nearly all zero.

**Three exploration strategies**:

1. **Curriculum learning**: start from short horizons / easy subtasks and gradually increase difficulty. The key is ensuring positive rewards are frequent enough during early training to produce effective gradients. Cost: requires automatic difficulty labeling or hand-crafted curricula.

2. **Subgoal / milestone rewards (use with caution)**: give small rewards at intermediate steps to guide exploration direction. Problem: as discussed in Q4, milestones themselves can be gamed — must be paired with verifiable milestones (e.g., subtask unit tests pass) rather than neural network scores.

3. **Replay + prioritized experience replay**: retain a small number of historical successful trajectories and resample them with higher probability — allowing the model to consistently see "what success looks like" in an extremely sparse environment. Cost: introduces an off-policy problem (see Q3).

**Interview follow-up**: "If the task success rate is consistently <5%, can GRPO still be used?" — Practical experience: group size needs to be large enough to ensure at least 1 success within each group; otherwise the advantage for the entire group degenerates to all zeros, equivalent to wasted rollouts. In this case, it is recommended to warm-start with SFT (learning from a small number of successful trajectories) before switching to RL.

---

### Q7. After masking observation tokens, which tokens actually receive advantage? PRM step-level credit vs pure ORM

**Precisely answering "which tokens receive advantage"**:

Consider a trajectory with the following token sequence:

```
[system_prompt] [user_turn_1] [agent_think_1] [agent_act_1] [obs_1] [agent_think_2] [agent_act_2] [obs_2] … [agent_final]
```

Tokens with action_mask=1: all `agent_think_*` + `agent_act_*` + `agent_final`.  
Tokens with action_mask=0: `system_prompt`, `user_turn_*`, all `obs_*` (tool returns / environment observations).

**Advantage broadcasting rule**: if turn-level GAE is used, all mask=1 tokens within the same turn share that turn's $\hat{A}^{\text{turn}}$. The final gradient flows only to token positions with mask=1.

**A common mistake**: if the agent's `<think>` block is treated as internal reasoning rather than an action (some implementations mask CoT), then `<think>` tokens have mask=0 and the gradient does not flow through the reasoning chain — training only "action selection" and not "reasoning quality". This is an implementation detail that interviews use to probe whether the candidate truly understands the semantics of masking.

**Credit granularity comparison: PRM (process reward) vs ORM (outcome reward)**<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Define PRM step-level reward as the step-level advantage (change in future success probability), which is theoretically equivalent to RL's Q-V difference, and must be estimated using an independent prover policy rather than the current policy. <a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>:

| Dimension | ORM (terminal reward) | PRM (step-level reward) |
|---|---|---|
| Signal sparsity | One scalar per trajectory | One scalar per step |
| Credit granularity | Trajectory-level → broadcast to tokens | Step-level → directly assigned to that step's tokens |
| Annotation cost | Low (only final correctness) | High (requires per-step judgment or automatic rollout estimation) |
| Hackability | Harder (terminal state is difficult to fake) | Easier (intermediate step scorer can be deceived) |
| Relationship to GAE | GAE uses $V$ function to approximate step-level advantage | PRM directly provides step-level advantage estimates |

**PRM's step-level advantage definition**: the theoretically cleanest PRM step-level reward is the "change in future success probability" brought by that step: $r_t^{\text{PRM}} = P(\text{success}|s_{t+1}) - P(\text{success}|s_t)$. This is definitionally equivalent to RL's advantage ($Q(s,a)-V(s)$)<span class="cite-wrap"><a class="cite" href="#ref-8">8</a><span class="cite-note">Define PRM step-level reward as the step-level advantage (change in future success probability), which is theoretically equivalent to RL's Q-V difference, and must be estimated using an independent prover policy rather than the current policy. <a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>. In practice, **Monte Carlo rollouts** are used to estimate $P(\text{success}|s_t)$, at the cost of requiring many rollouts per step.

**Interview trap**: "Does using PRM eliminate the need for discount $\gamma$?" — No. PRM provides **step-level rewards**, which still need to be accumulated into returns using discounting or GAE. PRM solves "how much reward each step should receive"; it does not solve "how to convert multi-step rewards into gradient signals for the current policy".

---

## References

> All are original sources for classic foundational methods, verified one by one (title + arXiv ID). Click superscripts to jump, click ↩ to return.

<ol>
<li id="ref-1">Yao et al. <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a> — think→act→observe paradigm. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — GRPO: group-relative baseline, critic-free. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — process supervision / PRM (PRM800K). <a href="#fnref-3">↩</a></li>
<li id="ref-4">Lambert et al. <em>Tülu 3: Pushing Frontiers in Open Language Model Post-Training</em>. 2024. <a href="https://arxiv.org/abs/2411.15124">arXiv:2411.15124</a> — RLVR: verifiable correctness as terminal reward. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Schulman et al. <em>Proximal Policy Optimization Algorithms</em>. 2017. <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a> — PPO (policy gradient baseline). <a href="#fnref-5">↩</a></li>
<li id="ref-6">Schulman et al. <em>High-Dimensional Continuous Control Using Generalized Advantage Estimation</em>. ICLR 2016. <a href="https://arxiv.org/abs/1506.02438">arXiv:1506.02438</a> — GAE: λ-weighted TD residuals, bias-variance trade-off. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — scaling laws for proxy vs gold reward as a function of KL divergence. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Setlur et al. <em>Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning</em>. 2024. <a href="https://arxiv.org/abs/2410.08146">arXiv:2410.08146</a> — PRM step-level advantage = change in future success probability, equivalent to Q-V difference. <a href="#fnref-8">↩</a></li>
</ol>
