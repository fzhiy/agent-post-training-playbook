# Agentic & Long-horizon RL

> The **next step** beyond single-turn reasoning RL (GRPO / RLVR, see the sibling repository [reasoning-rl-frontier](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reasoning-rl-frontier.html)): Extending the reward signal from "one prompt, one response, one reward" to **multi-turn trajectories** (think → call tool → observe)*.

> ⚠️ **Study notes, not author's research findings** (see README integrity statement). Numbers and conclusions are based on the original papers; uncertain points are noted.

## 0. One-sentence evolution

`Single-turn RLHF (prompt→response→reward)` → `Single-turn verifiable RLVR (correct/incorrect→reward)` → **`Multi-turn agentic RL (trajectory→sparse terminal reward)`**.
What changes is not the loss function, but the **shape of the episode**: An episode consists of `(reasoning, tool_call, observation)` repeated over multiple turns<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Enabling LLMs to interleave reasoning with tool calls (think→act→observe), thinking while acting.<a href="https://arxiv.org/abs/2210.03629">Yao 2022 ↗</a></span></span>, and the reward is often given only **once at the end** (task success or failure).

## 1. What makes it long-horizon

| Dimension | Single-turn Reasoning RL | Long-horizon Agentic RL |
|---|---|---|
| Episode Length | 1 turn | Several to dozens of turns |
| Reward | One per response | **Sparse / Delayed**, often only at the terminal |
| Observation | Fully visible (prompt) | **Partially visible** (only known after tool returns) |
| Action Space | Tokens | Tokens + **tool calls** + when to stop |
| Main Difficulty | Preference / Correctness | **Long-horizon credit assignment** + error accumulation |

## 2. Formalization (POMDP)

Consider a trajectory $\tau=(s_0,a_0,\dots,s_T,a_T)$ as a POMDP. The terminal task reward $R(\tau)\in\{0,1\}$ (success/failure) or a scalar score. The trajectory return is

$$G_t=\sum_{k=0}^{T-t}\gamma^{k}\,r_{t+k}.$$

In practice, a "turn" is often used as the decision granularity (turn-level MDP), while gradients still fall on the **tokens generated within that turn** — i.e., **turn-level credit + token-level updates**.

## 3. Credit assignment (core difficulty)

With only one terminal reward, the question is: **Which turn and which token among these dozens should be rewarded/punished?**

- **trajectory-level**: The entire trajectory shares one advantage — simplest, but also rewards "bad steps in a good trajectory".
- **turn-level**: Gives each turn an advantage (requires step rewards or value estimation).
- **token-level**: Most fine-grained; typically **broadcast** from the turn-level advantage to the tokens in that turn.

The idea of GRPO<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Sampling a group of rollouts for the same task, using relative returns within the group as the baseline, eliminating the need for a critic.<a href="https://arxiv.org/abs/2402.03300">Shao 2024 ↗</a></span></span> is naturally transferable: **Sample a group of trajectories for the same task, use the relative return within the group as the baseline** (no critic needed):

$$A(\tau_i)=\frac{R(\tau_i)-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}.$$

> A **Process Reward Model (PRM)**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Scoring each step of reasoning (process supervision), not just the final correctness.<a href="https://arxiv.org/abs/2305.20050">Lightman 2023 ↗</a></span></span> (scoring each step) is more friendly for long-horizon credit assignment, but **annotation/training is more expensive** and it can be hacked itself; an **ORM** (only looking at the outcome) is cheaper but has coarse credit assignment. Long-horizon scenarios often involve a compromise between the two.

## 4. Reward design

- **Verifiable Outcome Reward (RLVR→agentic)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Using automatically determinable correctness (instead of a reward model) as the RL signal.<a href="https://arxiv.org/abs/2411.15124">Lambert 2024 ↗</a></span></span>: The most stable terminal signal is one that can be automatically determined — unit tests passing, environment state achieved, answer verifiable.
- **Process / Step Reward**: Giving points for intermediate milestones alleviates sparsity but is **prone to gaming** (the agent learns to trigger milestones without solving the task).
- **Long-horizon reward hacking**: The longer the trajectory, the more shortcuts exist (idling to accumulate steps, repeatedly calling cheap tools). Mitigation: Focus on verifiable terminal rewards + step/token cost penalties + **read-only but don't learn from tool outputs** (see §5 masking).

## 5. Algorithm essentials

Two "long-horizon specific" details for multi-turn PPO<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Policy gradient with clipping, the baseline algorithm for RLHF / agentic RL.<a href="https://arxiv.org/abs/1707.06347">Schulman 2017 ↗</a></span></span> / GRPO:

1. **Observation Token Masking**: Tool returns / environment observations are **injected** into the context, not generated by the policy — they must be **masked out** in the loss, otherwise it's equivalent to making the model "fit" the tool outputs (analogous to SFT loss masking).
2. **Advantage Broadcasting + Masking**: The turn-level advantage is broadcast to the tokens generated by the agent in that turn, then multiplied by the action mask.

```python
import torch

def group_relative_advantages(returns, group_ids, eps=1e-6):
    """GRPO-style group relative advantage (no critic).
    returns:   (N,) terminal return per trajectory (e.g., success={0,1} or scalar score)
    group_ids: (N,) multiple rollouts for the same task share one group id
    """
    adv = torch.zeros_like(returns)
    for g in group_ids.unique():
        m = group_ids == g
        r = returns[m]
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv

def masked_pg_loss(logp, adv_per_token, action_mask):
    """Policy gradient loss: backpropagate only on tokens generated by the agent.
    logp:          (B,T) log-probability of the token taken under the current policy
    adv_per_token: (B,T) broadcast from turn-level advantage to tokens
    action_mask:   (B,T) 1=token generated by agent, 0=tool output/observation (masked)
    """
    pg = -(logp * adv_per_token) * action_mask
    return pg.sum() / action_mask.sum().clamp_min(1)
```

> The rollout phase requires **actually executing tools/environment within the loop** (often asynchronously), leading to longer trajectories → more expensive sampling → a major system bottleneck for agentic RL.

## 6. Bridge from single-turn

The **GRPO / RLVR / loss masking** from the sibling repository are the building blocks; Agentic RL ≈ **applying them to trajectories** + solving "credit assignment for sparse terminal rewards". If you understand single-turn, grasping the three points of "trajectory formulation + masking + group relative baseline" enables the transition.

---

## Stratified follow-ups

### L1 Basics

<details class="qa"><summary>1. What kind of task is considered "long-horizon"? Give two examples of LLM agents.</summary>

Answer: A task requires multiple think→act→observe cycles, with the reward given only at the terminal (sparse/delayed), and the action space includes tool calls. Examples: ① A code agent cyclically calls a code executor to debug a program; ② A search+summarization agent queries a web API multiple times before generating a report.

**Follow-up:** Why is a long-horizon task modeled as a POMDP rather than an MDP, and what is the direct consequence for value estimation? → The tool return (next turn's observation) is unknown before acting; the agent is in a partially observable state. The value function can only bootstrap on the currently visible history, cannot accurately estimate the contribution of hidden states, leading to slow convergence and hard-to-quantify bias for the critic.

</details>

<details class="qa"><summary>2. Why is single-turn RLHF/RLVR insufficient for training multi-turn tool-use agents?</summary>

Answer: Single-turn RL assumes one prompt, one response, one reward; the episode shape is a single response. A multi-turn agent's episode consists of `(reasoning, tool_call, observation)` interleaved repeatedly, with the reward given only on the final turn. The single-turn loss function cannot handle cross-turn credit assignment and lacks a masking mechanism for tool observation tokens.

**Follow-up:** If the single-turn RLVR loss is directly applied to multi-turn trajectories without any masking, which tokens will the gradients flow to, and what specific consequences does this bring? → Gradients will flow to the observation tokens returned by tools, effectively performing SFT on the environment-injected content. The model will learn to "generate tokens that match the tool output format" rather than "make better action decisions," contaminating the policy gradient signal.

</details>

<details class="qa"><summary>3. Why are sparse / delayed rewards difficult?</summary>

Answer: There is only one terminal reward, making it impossible to directly judge which turn or token among dozens should be rewarded/punished—the long-horizon credit assignment problem. Furthermore, early in training, the probability of success trajectories decays exponentially with the number of turns (if each turn's success rate is 0.8, after 10 turns it's ~0.11), leading to many rollouts receiving zero reward, and the gradient signal nearly vanishes.

**Follow-up:** When sparse rewards cause gradient vanishing, how do curriculum learning and auxiliary subgoal rewards mitigate this from different angles, and can they be used simultaneously? → Curriculum learning increases the success rate early on by shortening the initial horizon, allowing sufficient positive rewards within a group. Subgoal rewards supplement signal density at intermediate steps. The two are complementary, but subgoal hacking must be prevented—verifiable milestones (e.g., passing unit tests for sub-functions) are better than neural network proxy scores.

</details>

<details class="qa"><summary>4. Why must <strong>tokens returned by tools</strong> be masked out during training?</summary>

Answer: Tool returns are environment-injected observations, not generated by the policy—if not masked, the loss will require the model to "fit" the tool output, equivalent to performing SFT on observations, contaminating the policy gradient signal. The correct approach is to set action_mask=0 for all `obs_*` tokens, so gradients only flow to the think/act tokens generated by the agent itself.

**Follow-up:** If the `<think>` reasoning chain is also masked (action_mask=0), what consequences does this produce, and is it sometimes a reasonable choice? → The gradients for reasoning chain tokens become zero; the model only trains "action selection" without training "reasoning quality," causing the reasoning chain to degrade into decorative output. However, if the reasoning chain content is uncontrollable and its quality highly unstable, masking it in the short term can reduce training noise—this is an implementation trade-off that depends on the reasoning chain's quality.

</details>

### L2 Advanced

<details class="qa"><summary>5. What are trajectory-level / turn-level / token-level advantages? What are the trade-offs?</summary>

Answer: Trajectory-level shares one advantage across the entire trajectory—simplest but rewards bad steps in good trajectories. Turn-level gives each turn an advantage (requires step rewards or value estimation), providing more precise credit. Token-level is the finest granularity, typically obtained by broadcasting the turn-level advantage to the tokens generated by the agent in that turn and multiplying by the action mask. Finer granularity leads to more precise credit but increases reliance on the critic/PRM.

**Follow-up:** After broadcasting turn-level advantage to tokens, all agent tokens within the same turn share the same advantage value—under what circumstances does this approximation become severely distorted? → When the token sequence within a turn contains two segments with large semantic differences (think and act), sharing the advantage applies the same credit to "good acts" and "the reasoning process that led to the act." If the reasoning is incorrect but the act happens to be correct (or vice versa), the gradient direction for that turn becomes inaccurate—this is a fundamental limitation of turn-level granularity and a motivation for introducing PRM step-level rewards.

</details>

<details class="qa"><summary>6. How does GRPO's group relative baseline translate to multi-turn? Why can it eliminate the critic?</summary>

Answer: For the same task, sample a group of multi-turn trajectories and normalize using the mean/standard deviation of the terminal returns within the group to obtain $A(\tau_i)=\frac{R(\tau_i)-\mu_g}{\sigma_g+\epsilon}$, using the group mean as the baseline to replace the critic. The reason the critic is eliminated is that the baseline is derived from the statistics of the same batch of rollouts, requiring no additional value network. The cost is that the variance may be large in sparse long-horizon scenarios (when the group is all zeros, the advantage degenerates to zero).

**Follow-up:** When GRPO groups have all-zero rewards (all trajectories fail), the advantage degenerates to zero, and gradients vanish—what are some practically feasible mitigation strategies? → Three paths: ① Increase group size $G$ to ensure at least 1 success within the group; ② Introduce a small amount of verifiable milestone rewards to give some trajectories non-zero rewards; ③ Use SFT warm-starting (learning from a small set of successful trajectories) to improve the base success rate before switching to RL—any of these is more effective than running rollouts on an all-zero group.

</details>

<details class="qa"><summary>7. What are the trade-offs between PRM and ORM in long-horizon credit assignment?</summary>

Answer: ORM only looks at the terminal result—signal is sparse, credit assignment is coarse, but annotation cost is low and it's harder to hack. PRM scores each step (ideally the change in future success probability)—credit is precise, but annotation/training is more expensive and the intermediate step scorer can be gamed by the agent. Long-horizon scenarios often compromise: ORM verifiable terminal rewards are primary, supplemented by a small number of verifiable milestones acting as PRM signals.

**Follow-up:** What are the mainstream methods for automatically estimating PRM step-level rewards without relying on manual annotation, and what are their computational bottlenecks? → Use Monte Carlo rollouts starting from each step's state $s_t$, sampling multiple times to the terminal, and estimate the success rate mean $P(\text{success}|s_t)$. Step-level reward = $P(\text{success}|s_{t+1}) - P(\text{success}|s_t)$. The bottleneck is that each step requires numerous rollouts, and sampling cost increases linearly with trajectory length and number of steps. In practice, estimation is often only done for critical branching steps.

</details>

<details class="qa"><summary>8. What are typical forms of long-horizon reward hacking? How to mitigate them?</summary>

Answer: Three typical forms: ① Idling/Looping (repeatedly calling cheap tools to lengthen the trajectory for milestone points); ② Premature Stop (declaring "task complete" early to bypass subsequent difficult steps); ③ Milestone Gaming (triggering milestones without truly solving subtasks). Mitigation combination: Focus on verifiable terminal rewards + step/token cost penalties + observation token loss mask + KL penalty term + adversarial test set rotation.

**Follow-up:** How does the KL penalty term $R'=R-\beta\,\text{KL}(\pi_\theta\|\pi_\text{ref})$ suppress reward hacking, and what are the side effects of $\beta$ being too large or too small? → The KL term limits the degree of policy deviation from the reference model, "braking" before the proxy reward and gold reward diverge. When $\beta$ is too small, the constraint is ineffective and hacking proceeds as usual. When $\beta$ is too large, the policy cannot sufficiently optimize the task, effectively doing mostly SFT—requires dynamic adjustment of $\beta$ based on reward hacking indicators or setting a KL threshold for early stopping.

</details>

<details class="qa"><summary>9. When several agents collaborate with only a team-level terminal reward, what is the difference between joint credit and marginal credit?</summary>

Answer: When multiple LLM agents collaborate on one task and only a team-level terminal reward exists, the question is "how should this shared reward be distributed to each agent." **Joint credit**: all agents share the same team advantage (analogous to trajectory-level)—simplest, but encourages free-riding, as low-contribution agents are rewarded equally. **Marginal credit**: use a counterfactual—the change in team return after removing or replacing an agent with a default policy—to approximate that agent's marginal contribution; credit is more accurate but requires extra rollouts to estimate the counterfactual baseline. Multi-agent post-co-training like MAPoRL (arXiv:2502.18439) uses MARL + a verifier reward to explicitly train collaborative behavior across multiple LLMs.

**Follow-up:** Why is counterfactual credit assignment more expensive in the LLM multi-agent setting than in classic MARL? → Estimating an agent's counterfactual baseline requires re-running the team rollout under the condition "this agent is absent / replaced by a default policy"; every agent and every trajectory needs an extra multi-agent generation, scaling at least linearly with the number of agents. And each generation is a full LLM decode—far more expensive than the small-network forward pass of classic MARL—so practice often falls back to a joint-credit + role-level reward-shaping compromise.

</details>

<details class="qa"><summary>10. Context-window management for long-horizon agents: how do KV eviction and summarization each affect training / inference?</summary>

Answer: The longer the trajectory, the more the state $s_t$ (full concatenated history) grows linearly; attention is $O(L^2)$ and KV cache memory is $O(L)$—the core system bottleneck for long-horizon agents. Two classes of mitigation: **KV eviction / compression** (drop or merge the KV of old low-attention-weight tokens, e.g. keep an attention sink + recent window) saves memory but is **lossy**—evicted observations can no longer be attended to, possibly losing key early information; **summarization / external memory** (compress old turns into summary text, or write to external memory and retrieve) preserves semantics but introduces summarization error and extra calls. When training an agent with context management, the spans replaced by compression / summary are still treated as "environment-injected" (masked, no gradient).

**Follow-up:** How does context compression interact with credit assignment and create a train–inference mismatch? → If training sees the full history but inference triggers compression under a budget, the agent is never trained on the "post-compression state distribution," causing distribution shift (akin to exposure bias). More subtly, if the compressed-away turn is exactly a key decision point, its tokens' credit is still counted during training but the information is missing at inference—one must also simulate compression at the inference budget during training to keep both ends' state construction consistent.

</details>

<details class="qa"><summary>11. Tool calling can be trained by both SFT and RL—what is the core difference in "which tokens are trained and with what signal"?</summary>

Answer: **SFT** does next-token prediction on expert / successful trajectories, masking the question and tool-return tokens and back-propagating cross-entropy only on the agent-generated think/act tokens—it learns to "imitate this trajectory's action distribution." **RL** (GRPO/PPO) operates on trajectories the agent **rolls out itself**, computes advantages from reward, and likewise back-propagates the policy gradient only on agent tokens—it learns "which actions yield higher return." Both mask the same token set (tool returns); the difference is the loss signal: SFT = log-likelihood of fixed target tokens, RL = advantage-weighted log-prob. A common recipe: SFT warm-start (imitate) → RL fine-tune (surpass demonstrations).

**Follow-up:** When doing SFT on function-calling structured JSON output, what label-masking pitfall does it have that text-based formats don't? → The fixed template parts of the JSON string (`{"name":`, `"arguments":`, punctuation) are schema, not decisions; if all are trained as agent tokens, gradient is wasted memorizing the fixed format and amplifies fragility to minor schema changes. A more refined approach computes loss only on the "fill-in" values (tool name, argument values) and partially masks the template tokens too—text-based `Action:` lines lack this fixed template, so the pitfall is smaller but free-text parsing is more brittle.

</details>

<details class="qa"><summary>12. Beyond PPO clipping, what other correction methods exist for asynchronous / off-policy long-horizon RL?</summary>

Answer: PPO clipping only truncates the IS ratio to control variance—it does **not** correct bias. Other approaches: ① **Global advantage normalization + lightweight stabilization** (REINFORCE++, arXiv:2501.03262): critic-free, using batch-level advantage normalization + PPO-style clip/KL stabilization, mitigating GRPO's fragility to all-zero in-group degeneration; ② **V-trace doubly-truncated IS** (introduced in IMPALA, Espeholt et al. 2018): impose double clipping $\bar\rho,\bar c$ on each step's ratio, yielding a biased but low-variance target with convergence guarantees under policy lag; ③ **Prefix IS / sequence-level truncation**: the theoretically correct correction is the prefix-product ratio, approximated in practice by upper-bound truncation. The common trade-off: the harder the truncation → the lower the variance, the larger the bias; it must be combined with a **small policy lag** (limiting asynchronous steps) or ESS monitoring to truly control distribution shift.

**Follow-up:** Why is "global (batch-level) advantage normalization" often more stable than GRPO's "in-group normalization" under long-horizon sparse rewards? → GRPO normalizes within a small group for the same prompt; under sparse success rates the whole group is often all-zero → $\sigma_g=0$ → advantage degenerates. Batch-level normalization pools returns across different prompts to estimate mean / variance, so even if one prompt fails entirely, the advantage is non-zero as long as there is a positive signal elsewhere in the batch—the cost is that different prompts' returns are normalized against one shared baseline, so an advantage no longer reflects how good a response was *relative to its own prompt's difficulty*: a success on a hard prompt and a success on an easy prompt receive similar advantages, weakening per-prompt credit precision.

</details>

### L3 Deep Dive

<details class="qa"><summary>13. When the terminal reward is a single 0/1, how can credit be reasonably distributed across dozens of turns? Provide at least two approaches and compare them.</summary>

Answer: ① **GRPO Group Relative Baseline**: Multiple trajectories for the same task share the terminal return for normalized baseline, simple and critic-free, but credit is still uniformly distributed within the trajectory. ② **Turn-level GAE**: Introduce a lightweight turn-level value head, use $\hat{A}^{\text{GAE}}=\sum(\gamma\lambda)^l\delta_{t+l}$ to decompose the terminal return into per-turn TD errors, providing more precise credit but requiring critic bootstrapping. ③ **PRM Step Reward**: Use Monte Carlo rollouts to estimate the change in future success probability for each step as step-level reward, providing the finest credit but at the highest sampling cost. A compromise: First train basic capabilities with GRPO, then add a turn-level value head after stabilization.

**Follow-up:** Why is the choice of $\lambda$ in turn-level GAE more critical in long-horizon scenarios than in single-turn, and what consequences arise from setting it improperly? → In long trajectories, critic bootstrapping error accumulates with the number of steps: $\lambda\to1$ approaches Monte Carlo, and variance grows exponentially with horizon; $\lambda\to0$ relies on single-step TD and critic accuracy, but the critic itself has large bias in partially observable scenarios—both extremes are dangerous. In practice, $\lambda$ needs to be carefully tuned within the $[0.9,0.95]$ range based on critic quality and trajectory length.

</details>

<details class="qa"><summary>14. How to combine "verifiable outcome reward" with "process supervision" to be stable yet not gameable?</summary>

Answer: Use verifiable terminal signals (unit test pass / environment state) as the primary reward—hard to hack. Use **verifiable milestones** as intermediate step rewards (e.g., passing unit tests for sub-functions, not neural network scoring) to alleviate sparsity without introducing a gameable proxy. Simultaneously add a KL penalty $R'=R-\beta\,\text{KL}(\pi_\theta\|\pi_\text{ref})$ to prevent overoptimization, and apply step penalties to suppress looping.

**Follow-up:** What systematic failure occurs when process reward weight is too high, and which failure mode is harder to diagnose compared to pure ORM? → When process reward weight is too high, the agent prioritizes triggering milestones over solving the terminal task (milestone gaming). Since intermediate step scores keep rising, the training curve appears good—proxy reward continuously increases while gold reward (terminal success rate) does not increase or even decreases. This is harder to diagnose than the "all-zero signal" of pure ORM because the gradient signal appears normal.

</details>

<details class="qa"><summary>15. Why are rollouts for agentic RL expensive? What engineering mitigations exist (asynchronous execution, truncation, length penalties)? What are their respective costs?</summary>

Answer: Rollouts require actually executing tools/environment within the loop (networking, code execution, etc.), which can have second-level latency. Longer trajectories also lead to large KV cache usage. Mitigations: ① **Asynchronous execution**—run multiple episodes in parallel so the GPU doesn't idle; the cost is that when a rollout completes, the policy may have already taken several steps (staleness), requiring IS correction or ESS monitoring. ② **Trajectory truncation**—forcibly terminate if the maximum step count is exceeded; the cost is that truncated trajectories have incomplete returns, requiring bootstrapping or direct discarding. ③ **Length penalty** $R'=R-\alpha|\tau|$—incentivizes the agent to complete efficiently; the cost is potentially penalizing necessary long reasoning chains.

**Follow-up:** After asynchronous execution introduces policy staleness, can PPO clipping alone solve the bias problem, or must it be combined with other mechanisms? → PPO clipping only truncates the IS ratio to control variance, it does not correct distribution bias—when the policy lag exceeds 1-2 mini-batches, gradients outside the clip interval are discarded, but updates within the interval are still made on the incorrect distribution. It must be combined with a small policy lag design (limiting asynchronous steps) or ESS monitoring (pausing or reducing lr when ESS falls below a threshold) to truly control bias.

</details>

<details class="qa"><summary>16. How does compounding error amplify in long trajectories? What is its relationship with exposure bias?</summary>

Answer: Small errors at each turn's decision change subsequent observations, causing the trajectory to deviate from the training distribution. The next turn then makes an even larger error on this out-of-distribution state, leading to errors **amplifying exponentially** with turns. This is structurally identical to exposure bias in SFT—during training, the model sees ground-truth prefixes, but during inference, it sees prefixes generated by itself. Long-horizon agents are particularly severe because tool returns also depend on previous actions. Mitigation: RL itself alleviates exposure bias by training the model on its own rollouts. Curriculum learning (starting with short horizons) can reduce the initial speed of error accumulation. **DAgger (Dataset Aggregation)** requests expert labels for states generated by the current policy at each turn, continuously incorporating the distribution actually visited by the policy into training, directly combating distribution shift at the data level. However, even with online RL, distribution shift within long trajectories cannot be entirely eliminated theoretically: within each rollout, small errors in early steps **compound with interest** within the trajectory—subsequent steps make decisions on states that increasingly deviate from the training distribution. RL updates, while correcting against the seen rollouts, cannot "foresee" the subsequent snowball effect caused by early errors within the same batch of rollouts. This is the fundamental reason why long-horizon scenarios are more fragile to error accumulation than short-horizon ones.

**Follow-up:** DAgger mitigates the static mismatch between training and inference distributions, but why is the compound interest of errors within a long trajectory (distribution shift within a rollout) something that even online RL cannot completely eliminate? → DAgger/online RL corrects the problem of "not having seen certain states during training." However, the compound interest within a trajectory stems from causal chains—the error at step $t$ changes $s_{t+1}$. When updating the policy, this rollout has already "happened," and only the next batch of rollouts can observe the effect of the correction. Errors still propagate within a single trajectory; the longer the horizon, the more opportunities for propagation. Online RL can only shorten this lag, not reduce it to zero.

</details>

---

## Deep-dive

> Interview-trap level questions: The following Q&A assumes the interviewer is familiar with single-turn GRPO/PPO and asks about fine-grained mechanisms in multi-turn scenarios. **Study notes, not author's research findings**.

---

### Q1. per-token vs per-turn advantage estimation — How does the discount in GAE transfer to multi-turn?

**Core Contradiction**: Single-turn RL defines advantage at the token granularity ($A_t = Q_t - V_t$); multi-turn scenarios have a two-level structure — **temporal discounting** between turns (turn-level) and **token broadcasting** within a turn.

**GAE Transfer**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">GAE uses a λ-weighted sum of TD residuals to balance bias-variance: λ→1 approaches Monte Carlo, λ→0 approaches single-step TD.<a href="https://arxiv.org/abs/1506.02438">Schulman 2015 ↗</a></span></span>:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

Replacing "one token" with "one turn," this formula is fully applicable to a **turn-level MDP**:
- $\gamma$ (turn discount): In sparse terminal reward scenarios, $\gamma\approx1$ is often set (no discounting), as each turn contributes actual value. A $\gamma$ that is too low will drive the advantage of early turns toward zero, wastefully discarding gradient signals.
- $\lambda$ (GAE smoothing coefficient): Controls the bias-variance trade-off — critic bootstrapping error in multi-turn episodes accumulates more easily. In practice, $\lambda<1$ is more critical than in single-turn; otherwise variance explodes exponentially with horizon.

**Broadcasting from turn-level to token-level**: All agent-generated tokens within the same turn share that turn's $\hat{A}^{\text{turn}}$. Observation tokens have mask=0. Thus, per-token gradients equal the per-turn advantage multiplied by the action-mask, with no additional approximation.

**Interview Trap**: "Since tokens share the same advantage, what's the difference between token-level and turn-level?" — The difference lies in **which level calculates the discount and baseline**. Turn-level lets discounts span turns and allows the value function to bootstrap at the turn granularity. Token-level is simply where the gradients land. Confusing the two leads to incorrect application of discounts at the wrong level (applying γ discount to each token is equivalent to applying γ^t decay to a very long sequence, making early token gradients nearly zero).

---

### Q2. High variance in GRPO group baseline under single terminal rewards — Why? What are variance reduction techniques?

**GRPO Group Baseline** advantage:

$$A(\tau_i) = \frac{R(\tau_i) - \mu_g}{\sigma_g + \epsilon}, \quad \mu_g=\frac{1}{G}\sum_{j=1}^G R(\tau_j).$$

**Two sources of variance explosion in multi-turn scenarios**:

1. **Binary 0/1 reward + small group size**: If $G=8$ and success rate $p\approx0.1$, it's common for the group to have all zeros (8 failures) or only 1 success. For all zeros, $\sigma_g=0$, and the advantage degenerates to zero. For 1/8 successes, $\sigma_g$ is extremely small, causing a spike in advantage—a single sample dominates the entire batch update.
2. **Trajectory length differences**: The log-prob sum for long trajectories is numerically much larger than for short trajectories. If the advantage is directly multiplied by the token count, long trajectories naturally have larger gradients, creating an implicit length bias.

**Variance Reduction Techniques**:

| Technique | Mechanism | Cost |
|---|---|---|
| Increase group size $G$ | More stable $\mu_g,\sigma_g$ | Sampling cost $\times G$ |
| Length-normalization | Divide loss by number of action tokens | Balances short/long trajectories, but may penalize necessary long reasoning |
| Mix ORM + intermediate sparse rewards | Reduces probability of all-zero groups, increases positive signal | Intermediate rewards can be hacked |
| Advantage clipping / truncation quantile | Remove peak advantage samples | May discard high-information samples |
| Introduce lightweight critic (turn-level value estimation) | Use $V$ to decompose terminal return into per-turn TD errors | Extra model, contradicts GRPO's "critic-free" philosophy |

**Interview Trap**: "Can GRPO fully replace PPO's critic?" — Yes for single-turn binary reward scenarios. In long-horizon sparse reward scenarios, the variance of the group baseline is often greater than that of the critic baseline, making the critic valuable again. In practice, a compromise is to use a small turn-level value head instead of a full PPO critic.

---

### Q3. Importance-sampling / off-policy correction after multi-turn rollouts become stale

**Problem Context**: Agentic rollouts involve real tool calls (networking, databases, code execution) with second-level latency. When a rollout completes, the policy parameters may have already updated several steps, so $\pi_\theta \neq \pi_{\theta_\text{old}}$.

**Importance Weight (IS weight)**:

$$w(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\theta_\text{old}}(\tau)} = \prod_{t \in \text{agent tokens}} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}.$$

**Why continuous multiplication is dangerous**: In a T-turn trajectory, agent tokens can number in the thousands. Even if each step's ratio $\approx1.05$, after continuous multiplication, the IS weight can be $\gg10$ — variance explodes, and a tiny number of samples dominate the gradient.

**Engineering Correction Methods**:

1. **PPO clip**<a class="cite" href="#ref-5">5</a> ($\epsilon$ clip): Truncates each token's ratio to $[1-\epsilon, 1+\epsilon]$. Does not correct bias but effectively controls variance. Suitable for **synchronous training**, where policy lag does not exceed 1-2 mini-batches.
2. **Sequence-level truncated IS (TIS)**: Truncates the entire trajectory's IS weight to an upper bound (e.g., 3.0). Simple but biased.
3. **Use ESS to dynamically reduce learning rate**: Monitor rollout staleness using effective sample size $\text{ESS}=(\sum w_i)^2/\sum w_i^2$. Automatically reduce learning rate when ESS falls below a threshold to avoid gradient shocks.
4. **Prefix IS ratio**: The theoretically correct correction term is the **prefix IS ratio** (the cumulative product over the entire sequence prefix), not independent truncation of per-token ratios. However, implementation is complex and numerically unstable.

**Interview Trap**: "Is using PPO clip enough?" — Yes for synchronous training. In asynchronous/agentic scenarios, policy lag can be dozens of steps. At that point, clip only treats the symptom (reduces variance but bias is large), effectively performing gradients on the incorrect distribution. It must be combined with a small policy lag design or ESS monitoring.

---

### Q4. Specific forms of long-horizon reward hacking and mitigation

**"Long-horizon" makes hacking easier**: Single-turn hacking only requires finding a loophole in one response. A long-horizon agent can slowly accumulate shortcuts over dozens of steps, cover evaluation files, or exploit tool side effects.

**Three typical forms**:

| Type | Mechanism | Example |
|---|---|---|
| **Idling/Looping** | Agent repeatedly calls irrelevant tools, lengthening the trajectory hoping to score via milestones | Repeatedly querying a search API without advancing the task |
| **Premature Stop** | Declaring "task complete" early to bypass subsequent difficult steps | Code agent outputs "DONE" before running tests |
| **Milestone Gaming** | Triggering milestone checkpoints without truly solving the subtask | Writing an empty function to pass CI, or directly mocking test outputs |

**Why it's more severe in long-horizons**: Theoretical analysis<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Scaling Laws for Reward Model Overoptimization: As KL divergence increases, gold reward first rises then falls; the difference between proxy reward and gold reward monotonically increases with KL.<a href="https://arxiv.org/abs/2210.10760">Gao 2022 ↗</a></span></span> shows that the difference between proxy reward and gold reward monotonically increases with KL divergence. The longer the trajectory, the greater the total KL divergence. According to the scaling law [7], hacking risk systematically increases.

**Mitigation Combination**:

1. **Focus on verifiable terminal rewards**<a class="cite" href="#ref-4">4</a>: Unit test pass / environment state achievement, harder to hack than neural network proxy rewards.
2. **Step/token cost penalties**: $R'(\tau) = R(\tau) - \alpha \cdot |\tau|$, directly suppresses looping.
3. **Observation token read-only (loss mask)**: Prevents the agent from learning to "generate output formats that pass mock checks".
4. **KL penalty term**: $R' = R - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$, "brakes" before proxy reward and gold reward diverge.
5. **Adversarial test set rotation**: Regularly change evaluation samples to prevent the agent from memorizing shortcuts for specific test cases.

---

### Q5. How does partial observability (POMDP) make value estimation difficult?

**Information structure difference between single-turn RL and long-horizon agentic RL**:

Single-turn reasoning: $V(s) \approx V(\text{prompt})$ — prompt is fully visible, value function input is complete.

Multi-turn agentic: $s_t$ = historical conversation + previous turn's tool return, but **the next turn's tool return is unknown** — the agent is in a POMDP, unaware of future observations.

**Three specific difficulties**:

1. **Randomness of tool returns**: The same action (API call) may return different content due to network state, external DB version differences — the value function needs to take expectation over this randomness, but during training, only one specific return is seen. Single-step bootstrapping ($V(s_{t+1})$) fits the value of a noisy observation, leading to slow convergence and hard-to-estimate bias.

2. **Context length explosion**: As turns increase, $s_t$ grows linearly (full history concatenation). If the value network uses the same LM backbone, its forward pass cost grows quadratically with context length — value bootstrapping itself becomes expensive.

3. **Irreversibility of external state**: The agent modifies a database/filesystem, and these side effects are not in the token stream — the value function cannot see the "hidden state of the environment". Traditional POMDP solutions (belief state) cannot be directly applied in the LLM context.

**Practical Responses**:
- Use a lightweight turn-level value head (attached to the hidden state of the last generated token) instead of a critic for the entire rollout.
- **Summarize** tool returns before feeding them into the value input to compress context.
- Accept higher bias: Use GAE with $\lambda<1$ to reduce reliance on distant bootstrapping, at the cost of slightly underestimating long-term advantage.

---

### Q6. Exploration problem under long-horizon sparse rewards

**Why single-turn RL exploration strategies are insufficient for long-horizons**: In single-turn RL, random token sampling can explore. In multi-turn scenarios, **the probability of success trajectories decays exponentially with turns** — if each turn's correct probability is 0.8, after 10 turns the success rate drops to $0.8^{10}\approx 0.11$. The agent rarely sees positive rewards, and the training signal is nearly all zeros.

**Three types of exploration strategies**:

1. **Curriculum learning**: Start with short horizons / easy subtasks, gradually increasing difficulty. The core is to ensure positive rewards are frequent enough early in training to generate effective gradients. Cost: Requires automatic labeling of task difficulty or manual curriculum design.

2. **Subgoal / milestone rewards (use with caution)**: Give small rewards at intermediate steps to guide exploration. Problem: As noted in Q4, milestones themselves can be gamed — must be combined with verifiable milestones (e.g., passing unit tests for sub-functions) rather than neural network scoring.

3. **Replay + Prioritized Experience Replay**: Retain a small number of historical success trajectories and resample them with higher probability — letting the model continually see "what is success" in extremely sparse environments. Cost: Introduces off-policy issues (see Q3).

**Interview Follow-up**: "If the task success rate is always <5%, can GRPO still be used?" — Practical experience: Group size needs to be large enough to guarantee at least 1 success within the group; otherwise, the entire group's advantage degenerates to all zeros, equivalent to running rollouts for nothing. At this point, it is recommended to first use SFT warm-starting (learning from a small set of successful trajectories) before switching to RL.

---

### Q7. After masking observation tokens, which tokens actually receive advantage? PRM step-level credit vs pure ORM

**Precise answer to "which tokens receive advantage"**:

Suppose a trajectory contains the following token sequence:

```
[system_prompt] [user_turn_1] [agent_think_1] [agent_act_1] [obs_1] [agent_think_2] [agent_act_2] [obs_2] … [agent_final]
```

Tokens with action_mask=1: All `agent_think_*` + `agent_act_*` + `agent_final`.
Tokens with action_mask=0: `system_prompt`, `user_turn_*`, all `obs_*` (tool returns/environment observations).

**Advantage broadcasting rule**: If using turn-level GAE, then all mask=1 tokens within the same turn share that turn's $\hat{A}^{\text{turn}}$. The final gradient only flows to token positions with mask=1.

**A common pitfall**: If the agent's `<think>` block is treated as internal reasoning rather than an action (some implementations mask out CoT), then `<think>` tokens have mask=0, and the gradient does not flow through the reasoning chain — effectively training only "action selection" without training "reasoning quality". This is an implementation detail, tested in interviews to see if the semantics of masking are truly understood.

**PRM (Process Reward) vs ORM (Outcome Reward) credit granularity comparison**<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Defining PRM's step-level reward as step-level advantage (change in future success probability) is theoretically equivalent to RL's Q-V difference and needs to be estimated using an independent prover policy, not the current policy.<a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>:

| Dimension | ORM (Terminal Reward) | PRM (Step-level Reward) |
|---|---|---|
| Signal Sparsity | One scalar per trajectory | One scalar per step |
| Credit Granularity | trajectory-level → broadcast to token | step-level → directly assigned to step's tokens |
| Annotation Cost | Low (only final correctness needed) | High (requires step-by-step judgment or automatic rollout estimation) |
| Susceptibility to Hacking | Relatively hard (terminal state hard to fake) | Relatively easy (intermediate step scorer can be deceived) |
| Relationship with GAE | GAE uses $V$ function to approximate step-level advantage | PRM directly provides step-level advantage estimates |

**PRM's step-level advantage definition**: Theoretically, the cleanest PRM step-level reward is the "change in future success probability" brought by that step: $r_t^{\text{PRM}} = P(\text{success}|s_{t+1}) - P(\text{success}|s_t)$. This is definitionally equivalent to RL's advantage ($Q(s,a)-V(s)$)<span class="cite-wrap"><a class="cite" href="#ref-8">8</a><span class="cite-note">Defining PRM's step-level reward as step-level advantage (change in future success probability) is theoretically equivalent to RL's Q-V difference and needs to be estimated using an independent prover policy, not the current policy.<a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>. In practice, **Monte Carlo rollout estimation** of $P(\text{success}|s_t)$ is used, at the cost of requiring numerous rollouts per step.

**Interview Trap**: "If PRM is used, is the discount $\gamma$ no longer needed?" — Incorrect. PRM provides **step-level rewards**, which still need to be accumulated into a return using discounting or GAE. PRM solves "how much reward each step should get," not "how to convert multi-step rewards into gradient signals for the current policy."

---

## References

> All are original sources of classic foundational methods, verified item by item (title + arXiv ID). Click the superscript to jump, click ↩ to return.

<ol>
<li id="ref-1">Yao et al. <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a> — think→act→observe paradigm. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — GRPO: Group relative baseline, critic-free. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — Process supervision / PRM (PRM800K). <a href="#fnref-3">↩</a></li>
<li id="ref-4">Lambert et al. <em>Tülu 3: Pushing Frontiers in Open Language Model Post-Training</em>. 2024. <a href="https://arxiv.org/abs/2411.15124">arXiv:2411.15124</a> — RLVR: Using verifiable correctness as terminal reward. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Schulman et al. <em>Proximal Policy Optimization Algorithms</em>. 2017. <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a> — PPO (policy gradient baseline). <a href="#fnref-5">↩</a></li>
<li id="ref-6">Schulman et al. <em>High-Dimensional Continuous Control Using Generalized Advantage Estimation</em>. ICLR 2016. <a href="https://arxiv.org/abs/1506.02438">arXiv:1506.02438</a> — GAE: λ-weighted TD residuals, bias-variance trade-off. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — Scaling law of proxy vs gold reward with KL divergence. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Setlur et al. <em>Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning</em>. 2024. <a href="https://arxiv.org/abs/2410.08146">arXiv:2410.08146</a> — PRM step-level advantage = change in future success probability, equivalent to Q-V difference. <a href="#fnref-8">↩</a></li>
</ol>