# Agentic RL Infrastructure

> Turn the algorithms from [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl-en.html) into a runnable **systems engineering** reality: how to set up an async rollout cluster, manage environment sandboxes, budget GPU memory, and choose a training stack. Read the agentic page first for the *why*; this page gives you the *how to build*.

> ⚠️ **Study notes, not the author's own research** (see README disclaimer). Framework versions and performance numbers change fast with releases; this page records only architectural principles and design tradeoffs, never benchmark scores.

## 0. TL;DR

- The essential difference between agent RL training and standard RLHF: **multi-turn trajectories** (not single-turn responses) + **interactive environments** (not static datasets) + **async rollout** (inference decoupled from training).
- **The three-pool architecture** = the core mental model: Rollout Pool (inference engines + environments) → Reward Pool (verification / judging) → Training Pool (policy updates); three pools are async and independently scalable.
- **Training stack selection along four axes**: native multi-turn support / async rollout support / environment-management approach / GPU memory strategy (hybrid engine vs separated deployment).
- **verl** (Volcano Engine): hybrid engine (inference + training on same GPUs, time-multiplexed), SPMD 3D parallelism, agent RL support in active development.
- **OpenRLHF** (Ray ecosystem): Ray actors are naturally async; mature PPO / DPO / Rejection Sampling pipelines; agent extension via custom `Environment` wrappers.
- **AReaL** (Ant Group): fully async agent-native design, generation decoupled from training; systems-level, not an algorithmic variant (see [agentic §5](cheatsheet-agentic-and-long-horizon-rl-en.html)).
- **The environment is the bottleneck, not the GPU**: a single SWE test takes 10–60s, a web page load takes seconds — environment latency >> inference latency; throughput ceiling is capped by `N_parallel_envs × per-env speed`.
- **Multi-turn KV cache memory explosion**: $T$ turns, $L$ tokens each → cache $\propto T \times L$; mitigations = KV eviction (lossy) / summarization (error-prone) / truncation-restart (loses history).
- **Trajectory storage = the new data engineering**: one agent trajectory can reach tens of thousands of tokens; training on thousands means GB-scale data; raw trajectories + rewards + metadata must be stored, and format design affects replay and filtering efficiency.
- **Fault tolerance is a hard requirement**: environments are flaky (network timeouts / sandbox crashes / non-deterministic tests), GPUs OOM, rollout workers die — the training framework must handle retries / degradation / checkpointing, or it won't survive one epoch.

## 1. Why agent RL infra ≠ RLHF infra

Every assumption of standard RLHF breaks in the agent setting:

| Dimension | Standard RLHF | Agentic RL |
|---|---|---|
| Trajectory length | single-turn response (~hundreds of tokens) | multi-turn think→act→observe (~thousands–tens of thousands of tokens) |
| Data source | static dataset (model generates once) | **interactive environment** (each action changes state, not replayable) |
| Inference–training relation | can be synchronous (generate batch → train → next batch) | **usually needs to be async** (for slow interactive envs, env latency >> GPU latency; synchronous = idle GPU; fast verifiers or generation-dominated scenarios may keep GPU as bottleneck — see Q9) |
| Reward computation | reward model forward pass (~milliseconds) | **environment execution** (run unit tests / web checks / scripts, seconds) |
| KV cache | short sequences, can keep all | long sequences, memory is a hard constraint |
| Fault tolerance | re-generate one response | environment can crash / timeout / be non-deterministic — systematic fault tolerance needed |

> 💡 **In one sentence:** RLHF infra is a batch pipeline that "scores what the model generated → trains"; agent RL infra is a **distributed async system** that "lets the model interact in a real/simulated environment over multiple steps → collects trajectories → trains."

This directly dictates the core design of an agent RL training stack:

1. **Inference and training usually need to be decoupled** (async): for slow interactive envs (code execution / web interaction, env step in seconds >> GPU inference in milliseconds), synchronous waiting = idle GPU; fast verifiers or generation-dominated scenarios may keep GPU as the bottleneck — the bottleneck shifts with task and scale (see Q9).
2. **Environment management is a first-class citizen**: you need environment pools (docker sandbox pool / browser pool), health checks, timeout kills, and automatic resets.
3. **Trajectories are long sequences**: KV cache management, truncation strategies, and storage formats all need rethinking from scratch.

## 2. The three-pool architecture

The **core mental model** for agent RL training: three independently-scalable async pools, each with different resource profiles.

```
                    ┌─────────────────────┐
                    │   Training Pool       │
                    │  (GPU: policy update) │
                    │  - consumes trajectory │
                    │    batches             │
                    │  - gradient accum +    │
                    │    update              │
                    │  - periodically pushes │
                    │    new weights         │
                    └──────────┬──────────┘
                               │ new weights
                               ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Rollout Pool       │  │   Reward Pool        │
│  (GPU: inference +   │  │  (CPU: verification) │
│   CPU: environment)  │  │                      │
│                      │  │  - run unit tests /   │
│  - inference engine   │  │    check terminal     │
│    (vLLM/SGLang)      │  │    state              │
│    generates actions  │  │  - rule-based checks  │
│  - environment executes│  │  - LLM-judge (opt.)  │
│    actions            │  │                      │
│  - collects full       │──▶  - produces reward    │
│    trajectories       │  │    vector             │
│                      │  │                      │
└─────────────────────┘  └─────────────────────┘
```

**Rollout Pool** (inference + environment, GPU+CPU hybrid):
- Inference engines (vLLM<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">PagedAttention + continuous batching. <a href="https://arxiv.org/abs/2309.06180">Kwon 2023 ↗</a></span></span>/SGLang<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">RadixAttention + structured generation. <a href="https://arxiv.org/abs/2312.07104">Zheng 2024 ↗</a></span></span>) deployed with TP/DP, serving token output for multi-turn generation
- Each rollout worker is bound to one environment instance (docker container / browser instance)
- Loop: inference engine outputs action → worker executes environment step → observation injected into context → continue inference until termination
- **Throughput key**: `N_parallel_workers × inference_speed_per_worker`; the GPU does not directly wait for the environment (during the environment step the GPU serves other workers)

**Reward Pool** (verification, CPU-intensive):
- Receives complete trajectories, executes reward computation: run unit tests / check environment terminal state / run verification scripts
- Can be physically co-located with the Rollout Pool but logically independent (for independent scaling)
- **Bottleneck**: SWE unit-test execution can take tens of seconds; WebArena terminal-state checks require browser interaction

**Training Pool** (policy update, GPU-intensive):
- Pulls batches from the trajectory buffer, computes PPO/GRPO loss + gradient update
- **Hybrid engine mode**: training GPUs also serve inference (GPU time-multiplexed: train one batch → switch to inference → generate new rollouts → switch back to training)
- **Separated mode**: inference GPUs and training GPUs physically separated; suitable for large-scale continuous training

> 💡 **Each pool has a different optimal hardware type**: the Rollout Pool cares about inference throughput (high GPU utilization + many CPU env workers); the Reward Pool is almost pure CPU; the Training Pool cares about gradient computation (high GPU memory). **Separated deployment** is more expensive than hybrid but each pool scales independently — suitable for production scale.

**From-scratch implementation** (async three-pool rollout skeleton, interview hand-tear standard):

```python
import threading, queue, time
from concurrent.futures import ThreadPoolExecutor

class AsyncRolloutPipeline:
    """Async three-pool skeleton: Rollout worker → Reward worker → Trajectory buffer → Training."""
    def __init__(self, env_factory, reward_fn, policy_model, buffer_size=1000):
        self.env = env_factory
        self.reward_fn = reward_fn
        self.model = policy_model                         # synced from Training pool periodically
        self.buffer = queue.Queue(maxsize=buffer_size)

    def rollout_worker(self, prompts, num_envs=8):
        """Rollout Pool: sample trajectories in parallel against interactive environments."""
        def run_one(prompt):
            env = self.env()
            obs, traj = env.reset(prompt), []
            for _ in range(max_steps := 50):
                action = self.model.generate(obs["history"])
                next_obs, done = env.step(action)
                traj.append({"obs": obs, "action": action, "done": done})
                if done: break
                obs = next_obs
            return traj
        with ThreadPoolExecutor(max_workers=num_envs) as ex:
            return list(ex.map(run_one, prompts))

    def reward_worker(self, trajectories):
        """Reward Pool: compute final reward per trajectory (env verifier / judge, pure CPU)."""
        for traj in trajectories:
            traj.append({"reward": self.reward_fn(traj[-1]["obs"])})
        return trajectories

    def run_async(self, prompts):
        """Launch async loop: rollout+reward produce → buffer → train consumes."""
        def producer():
            while True:
                for t in self.reward_worker(self.rollout_worker(prompts)):
                    self.buffer.put(t)
                time.sleep(0.1)
        threading.Thread(target=producer, daemon=True).start()
```
# Key: ① Three pools decoupled — no blocking ② Environment is bottleneck (env delay ≫ GPU)  
# ③ ThreadPoolExecutor = toy; production → Ray actors / k8s pods ④ Buffer decouples cadence

## 3. Training stack comparison

### 3.1 verl (Volcano Engine)

verl<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Initiated by ByteDance Seed, community-maintained RL training framework: hybrid engine time-multiplexes GPUs (inference ↔ training switch), SPMD 3D parallelism, multi-algorithm support. <a href="https://github.com/verl-project/verl">github.com/verl-project/verl</a> (formerly volcengine/verl)</span></span>'s core design is the **hybrid engine**: the same set of GPUs time-multiplexes between inference and training modes — train one batch → switch to inference mode to generate new rollouts → switch back to training. Saves GPUs (no need for two clusters), but switching incurs overhead (weight sync + memory reallocation).

- **Parallelism strategy**: SPMD (Single Program Multiple Data), supporting DP/TP/PP 3D combinations
- **Algorithm support**: PPO / GRPO / DPO / ReMax; agent RL support in active development
- **Agent adaptation**: community `verl-agent` extension embeds environment interaction steps into the RL loop
- **Suitable for**: medium scale (tens to hundreds of GPUs), teams already experienced with FSDP/Megatron

### 3.2 OpenRLHF

OpenRLHF<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Ray-native RLHF framework: naturally async actors, PPO/DPO/Rejection Sampling support. <a href="https://arxiv.org/abs/2405.11143">Hu 2024 ↗</a></span></span> is built on **Ray**'s distributed actor model:

- **Naturally async**: each Ray actor is independently scheduled; rollout workers / trainer / reward model communicate through Ray — no manual synchronization needed
- **Flexible scheduling**: can assign different GPUs to rollout and trainer, or go hybrid
- **Algorithms**: mature PPO / DPO / Rejection Sampling / Conditional SFT implementations
- **Agent extension**: any interactive environment can be plugged in via a custom `Environment` abstraction; Ray's actor model makes env-worker scaling and fault tolerance relatively natural
- **Suitable for**: rapid prototyping, medium scale, teams wanting scheduling flexibility

### 3.3 AReaL (Ant Group)

AReaL<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Ant Group's fully async RL system (for LLM reasoning): generation and training fully separated; paper experiments focus on math/code reasoning, but the async architecture is equally applicable to agentic long-horizon settings. <a href="https://arxiv.org/abs/2505.24298">arXiv:2505.24298</a></span></span> is **a systems-level design, not a GRPO algorithmic variant** (see [agentic §5](cheatsheet-agentic-and-long-horizon-rl-en.html)):

- **Fully async**: generation (rollout) and training fully decoupled, each on an independent GPU pool; the generation pool continuously produces trajectories, the training pool continuously consumes them
- **Agent applicability**: paper experiments are in the reasoning domain, but its fully async architecture imposes no single-turn assumption and is equally applicable to long-horizon, multi-turn agent RL
- **Suitable for**: large-scale continuous training, high-throughput long-horizon agent RL

### 3.4 SkyRL-Agent (NovaSky/Berkeley)

SkyRL-Agent<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">NovaSky/Berkeley's async training stack for long-horizon multi-turn tools: overlaps CPU-side runtime with GPU generation. <a href="https://arxiv.org/abs/2511.16108">arXiv:2511.16108</a></span></span>'s core contribution is an **async pipeline dispatcher**:

- Overlaps CPU-side environment init / reward computation with GPU generation in time
- Not stalled by slow trajectories when mixing short and long ones (via dynamic scheduling)
- Reports roughly **1.55×** throughput improvement over naive async batching
- Pluggable into VeRL / Tinker and other backends

| Framework | Async mode | Parallelism strategy | Agent support | Suitable scale |
|---|---|---|---|---|
| **verl** | hybrid engine (time-multiplexed) | SPMD 3D (DP+TP+PP) | community extension | medium–large |
| **OpenRLHF** | Ray actors (naturally async) | Ray-scheduled | custom Environment | small–medium |
| **AReaL** | fully async (gen/train separated) | independent GPU pools | **agent-native** | large |
| **SkyRL-Agent** | async pipeline (CPU/GPU overlap) | pluggable into multiple backends | **long-horizon tool specialized** | medium |

> 💡 **Selection heuristic (non-authoritative; frameworks iterate fast):** rapid prototyping + medium scale → OpenRLHF (if you know the Ray ecosystem); already have a large cluster + need high throughput → AReaL (agent-native); already on FSDP/Megatron stack → verl (hybrid saves GPUs). SkyRL-Agent's value is at the scheduling layer and can be combined with any backend. **Actual choice requires evaluating current-release maturity + team-stack fit; do not trust this table blindly.**

## 4. Environment management

This is the **most underestimated** layer of agent RL infra — environments are slower, less reliable, and harder to scale than GPUs.

### 4.1 Three environment types and their challenges

| Environment type | Example | Per-step latency | Reset cost | Typical failures |
|---|---|---|---|---|
| **Code execution** | SWE-bench (run tests in sandbox)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">SWE-RL: uses real GitHub issues and FAIL_TO_PASS tests as the RL environment. <a href="https://arxiv.org/abs/2502.18449">Duan 2025 ↗</a></span></span> | 10–60s (incl. docker startup) | medium (rebuild container/branch) | timeout, non-deterministic tests, missing deps |
| **Web interaction** | WebArena (self-hosted sites)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">WebRL: self-evolving curriculum + web-domain RL environment. <a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> | 1–10s (page load + render) | medium (reset DB + restart services) | page load timeout, selector breakage, inconsistent site state |
| **GUI/OS** | OSWorld (real OS interaction) | 1–5s (screenshot + action execute) | high (VM snapshot rollback) | screenshot failure, coordinate drift, non-deterministic UI |

### 4.2 Environment pool design pattern

```
          ┌──────────────────────────────┐
          │     Environment Pool           │
          │                                │
          │  ┌─────┐ ┌─────┐ ┌─────┐     │
          │  │ Env │ │ Env │ │ Env │ ...  │  ← N pre-warmed instances
          │  │ #1  │ │ #2  │ │ #3  │     │
          │  └─────┘ └─────┘ └─────┘     │
          │       ↑  acquire               │
          │       │  release               │
          └───────┼────────────────────────┘
                  │
          ┌───────┴────────┐
          │ Rollout Worker │ → acquire env → run one trajectory → reset → return
          └────────────────┘
```

Key mechanisms:

1. **Pre-warm**: environment instances are started ahead of time in the pool (docker image already pulled, web server already running); a worker can use one immediately. Cold-starting a docker container can take tens of seconds.
2. **Health check**: periodically or on-acquire, check that the environment is reachable (docker ps, HTTP ping); bad instances are automatically replaced.
3. **Auto-reset**: after a trajectory completes, the worker calls reset to restore the initial state (git checkout back to initial commit, DB rollback, VM snapshot restore); must reset before returning to pool.
4. **Timeout kill**: single-step or whole-trajectory timeout → forcefully terminate the env process → recreate a fresh instance (rather than reuse), preventing residual state contamination.
5. **Pool size = N_env**: determined by `N_concurrent_workers + env_reset_latency`; too small → workers idle; too large → wasted resources.

> ⚠️ **Environment non-determinism = the silent training killer.** The same action executed twice on the same environment state may yield different results (network jitter, disk I/O races, tests depending on random seeds). This contaminates the reward signal — a good trajectory may accidentally be judged as failed due to an environment glitch, or a bad one accidentally passes. Mitigations: ① run critical reward computation multiple times and take the majority; ② record env version/seed for reproducibility; ③ tolerate a small amount of reward noise (Robust RL) — do not demand that every trajectory's reward be perfect.

### 4.3 Matching environment throughput to inference

**Core formula**: to avoid the GPU waiting on environments, you need `N_env × throughput_per_env ≥ GPU_inference_throughput`.

Concretely: if one environment trajectory averages 20 steps, per-step env latency is 5s, and GPU inference per step is 0.2s, then the GPU spends 0.2s on one worker's step but waits 5s for the environment → GPU utilization is only `0.2/5.2 ≈ 3.8%`. So you need many parallel env workers: at least `5.2/0.2 ≈ 26` workers so the GPU always has a request to serve at any moment. **In practice you often need hundreds of env workers to saturate a single GPU.**

## 5. GPU memory & multi-turn KV cache

### 5.1 Multi-turn KV cache growth model

Single-turn inference: encode prompt once → decode $L$ tokens → KV cache $\propto \text{prompt\_len} + L$. Multi-turn agent: $T$ turns of dialogue, each turn's context includes all prior-turn history. At turn $t$ the context length is roughly $t \times L$, so total KV cache grows as **$O(T \times L)$**.

Concrete numbers: if each turn is 500 tokens (think 200 + act 100 + observe 200), 20 turns gives a 10k-token context. For a 7B model (FP16), per-layer KV cache ≈ 2×2×num_heads×head_dim×seq_len bytes; summed across all layers this can reach several GB — a single trajectory's KV cache could fill one GPU's memory.

### 5.2 Mitigation strategies

| Strategy | Approach | Advantage | Cost |
|---|---|---|---|
| **KV eviction** | discard KV of low-attention-weight tokens (keep attention sink + recent window) | controllable memory, no semantic change to inference | **lossy**: discarded early observations can no longer be attended to; key info may be lost |
| **Summarization** | compress old turns into summary text, replace raw tokens | preserves semantics, most memory-efficient | **summary error**: compression may lose info critical for later decisions |
| **Truncation-restart** | when max length is exceeded, truncate old history and re-encode | simple, hard memory cap | **loses historical context**: agent forgets early actions |
| **External memory** | write to external vector DB / knowledge graph; retrieve rather than keep in context | theoretically unlimited capacity | retrieval latency + precision loss; retrieved results still cost some tokens |

> 💡 **Practical combo:** recent turns (last 3–5) kept with full KV + distant turns embedded as summary text + global KV eviction threshold. Framework support for these varies in maturity — when evaluating a framework, pay special attention to its multi-turn KV management primitives.

### 5.3 Hybrid engine memory management

In hybrid engine mode, the same GPU serves inference and training in time slices, and memory must switch between two modes:

- **Inference phase**: memory holds model weights + KV cache (multi-turn! Large!)
- **Training phase**: memory holds model weights + optimizer states + gradients + activations (roughly 3–4× weight memory vs inference)
- **Switch overhead**: free KV cache → allocate optimizer states → load; seconds per switch. Frequent switching (once per trajectory) significantly reduces throughput.

> ⚠️ **Practical pitfall:** in multi-turn agent scenarios with hybrid engine, inference KV cache memory can exceed the memory available for training — making it impossible to switch to training mode (OOM). Fix: proactively evict distant-turn KV during inference / cap max context length / or abandon hybrid and go with separated deployment.

## 6. Trajectory data pipeline

The metadata structure of one agent trajectory:

```
trajectory = {
  task_id,              # task identifier
  trajectory_id,        # unique trajectory ID
  turns: [              # multi-turn list
    {think, act, obs, reward_step, done},
    ...
  ],
  total_reward,         # terminal reward
  metadata: {           # metadata
    env_version,        # environment version (for reproducibility)
    total_steps,        # step count
    total_tokens,       # token consumption
    wall_time,          # wall-clock time
    truncated,          # whether truncated
    env_errors,         # environment error count
  }
}
```

**Storage magnitude**: one 20-turn trajectory ≈ 10k–20k tokens (think + act + obs); one thousand trajectories ≈ 10M–20M tokens → roughly 50–100MB in raw format (with metadata + reward). Training at the ten-thousand-trajectory scale needs GB-level storage; **offline datasets of millions of trajectories can reach TB scale**.

**Filtering pipeline** (before training, which trajectories enter a batch):
1. **Completeness filter**: discard truncated / env-errored trajectories (optionally keep for diagnostics)
2. **Quality filter**: discard trajectories with zero or anomalously low/high reward (task-dependent; all-zero trajectories carry no signal in GRPO but keep a few as negatives)
3. **Diversity filter**: discard trajectories with too-high duplication vs already-collected ones (based on action n-gram or embedding similarity) — prevents Echo Trap (see [agentic Q14](cheatsheet-agentic-and-long-horizon-rl-en.html))
4. **Balanced sampling**: ensure a reasonable ratio of positive (reward > 0) to negative examples in each batch, avoiding one side dominating the gradient

> 💡 **Trajectory storage format recommendation:** prefer Parquet (Apache Arrow columnar) over JSONL — higher compression (~5–10× vs raw JSON), supports column pruning (read only the `reward` column without deserializing the full text), and has a mature ecosystem (pandas / polars / spark natively read it).

## 7. Async staleness & flow control

In a fully async architecture, **policy lag** is inherent between rollout and training:

```
Rollout pool generates trajectories with θ_t → Reward pool computes → Training pool has since updated to θ_{t+k} → trajectories were sampled with an old policy
```

**Impact of staleness**:
- Trajectories are sampled from an **old policy version**; under the current policy θ_{t+k}, their log-probs and advantages have changed → the variance of IS (importance sampling) correction grows with staleness
- **Off-policy degree** = `(trainer step at rollout completion) - (policy version at rollout start)`; if too large the trajectories come from a nearly different policy → training signal degrades

**Control mechanisms**:
1. **Weight versioning**: each trajectory is tagged with its generating `policy_version`; at training time, discard trajectories whose `version` is too old (set a `max_policy_lag`)
2. **IS correction** (V-trace / truncated IS): correct with $\frac{\pi_{\theta_{\text{new}}}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$, but variance grows with lag; need clipping ($\rho = \min(\frac{\pi_{\text{new}}}{\pi_{\text{old}}}, \bar{c})$) or ESS (effective sample size) monitoring
3. **Queue backpressure**: if Training Pool consumption rate < Rollout Pool production rate → trajectory queue backlog → staleness grows → need backpressure (throttle rollout or accelerate training), maintaining steady-state queue depth
4. **Stale trajectory reuse policy**: slightly stale trajectories are not entirely useless — they can be downweighted (weight = $e^{-\lambda \cdot \text{lag}}$) rather than discarded outright, trading off throughput against signal quality

> 💡 **Practical rule of thumb:** the first pitfall in a fully async system is usually "unbounded queue growth" — rollout too fast → trajectory backlog → staleness spikes → training degrades → slower convergence → even more rollout backlog, a positive feedback loop. Setting `max_queue_size` + monitoring the `policy_lag` distribution is mandatory before going live.

## 8. Fault tolerance

The agent RL training chain is long; every link can fail. Production-grade training **must** handle these failures:

| Failure type | Frequency | Impact | Handling |
|---|---|---|---|
| Environment timeout (test runs too long) | common | lose that trajectory's reward | mark truncated on timeout → optionally discard or give partial reward |
| Environment crash (docker segfault) | occasional | worker hangs | health check + auto-restart + discard trajectory |
| GPU OOM (multi-turn KV exceeds limit) | occasional | inference fails, batch incomplete | reduce max_seq_len / enable KV eviction / retry |
| Network jitter (weight sync fails) | occasional | learner receives no new rollouts | exponential-backoff retry + weight-version check |
| Entire rollout worker dies | rare | all its bound environments lost | Ray/cluster scheduler auto-restarts worker + re-allocates env pool |

**Checkpoint strategy**: at minimum save `(model_weights, optimizer_state, LR_scheduler_state, global_step)` → resume from the most recent checkpoint. Recommend checkpointing every N training steps or every M minutes. If using hybrid engine, ensure you switch to training mode before checkpointing to save the full optimizer state.

> ❌ **Misconception:** "agent RL training = run a script and wait for the result." RLHF might get away with this (batch processing, few fault points), but agent RL's interactive environments make the failure rate far higher than standard training — **without proper fault tolerance, training will very likely die halfway through with no recovery.** Your first agent RL training run should get fault tolerance right *first*, then scale up — not the other way around.

## 9. Cost estimation

Given the following parameters, you can roughly estimate the cost of one agent RL training run:

| Parameter | Meaning | Typical value |
|---|---|---|
| $N_{\text{task}}$ | number of training tasks | 1k–10k |
| $\bar{T}$ | average trajectory turns | 10–30 |
| $\bar{L}_{\text{turn}}$ | tokens per turn | 300–800 |
| $N_{\text{epoch}}$ | training epochs | 1–5 |
| $\text{GPU}$ | GPU type | A100-80G / H100-80G |
| $\text{GPU}_\text{inference}$ | inference GPU count | 8–64 |
| $\text{GPU}_\text{training}$ | training GPU count | 8–32 |

**Inference cost** (rollout):
$$\text{Tokens}_{\text{rollout}} = N_{\text{task}} \times N_{\text{rollout-per-task}} \times \bar{T} \times \bar{L}_{\text{turn}}$$

For 5k tasks, 4 rollouts per task, average 20 turns, 500 tokens per turn: roughly **200M tokens**. With vLLM on 8×A100 at roughly 5k tokens/s inference speed, this takes about **11 wall-clock hours (≈89 GPU-hours = 8 GPU × 11h)**.

**Training cost**:
$$\text{Tokens}_{\text{train}} = \text{Tokens}_{\text{rollout}} \times N_{\text{epoch}}$$

Training throughput depends on model size + parallelism strategy. A 7B model on 8×A100 doing PPO training achieves roughly 2k–5k tokens/s (including advantage computation + gradient update). Training 200M tokens for 1 epoch ≈ **11–28 wall-clock hours (≈89–222 GPU-hours)**.

**Environment cost** (often ignored): SWE docker sandboxes need CPU instances; 100 concurrent environments running for 1 week — the CPU instance cost can approach the GPU cost.

> 💡 **Order-of-magnitude conclusion (not a quote; pedagogical estimate only):** one medium-scale (5k tasks, 7B model, ~20k trajectories) agent RL training run costs on the order of hundreds to thousands of dollars in GPU + environment total (varies by cloud provider and GPU type), **with environment CPU cost potentially accounting for 20–50%** — good environment reuse and pooling has direct financial ROI.

---

## Stratified follow-ups

> ⚠️ The "would be asked" items below are **inferred from public JDs + technical reports**, not real interview questions.

### L1 Basics

<details class="qa"><summary>1. Why can't agent RL training directly use standard RLHF infra?</summary>

A: All three core RLHF assumptions break in the agent setting: ① trajectories go from single-turn to multi-turn (context length $O(T)$ rather than constant); ② data goes from a static dataset to an interactive environment (cannot be pre-generated; each action changes state); ③ inference and training go from synchronizable to usually-needs-async (for slow interactive envs, env latency in seconds >> GPU inference in milliseconds; synchronous = idle GPU). Beyond these, long-trajectory KV cache management, environment fault tolerance, and trajectory storage pipelines are all new problems RLHF infra never had to handle.

**Follow-up:** If you forced RLHF's synchronous batch pipeline onto agent RL, what would you observe? → Extremely low GPU utilization, because every step waits for the environment to return, and environment latency far exceeds GPU inference; across multi-step trajectories, the GPU idles between every step, and overall GPU utilization can fall below 10%.

</details>

<details class="qa"><summary>2. What does each of the three pools do? Why can't they be merged into one pool?</summary>

A: **Rollout Pool** (inference + environment) produces trajectories; **Reward Pool** (CPU verification) computes rewards; **Training Pool** (GPU training) updates the policy. They cannot be merged because: ① the three have **different resource profiles** — inference cares about throughput and low latency, training cares about memory and gradient computation, reward is almost pure CPU; merging causes GPU-waiting-for-CPU (training waits for reward computation) and CPU-waiting-for-GPU (environment waits for inference results), lowering overall utilization. ② The three have **different scaling needs** — reward computation scales with task complexity, training with model size, rollout with concurrent task count; after merging they cannot scale independently.

**Follow-up:** When is a hybrid engine (merged inference + training) more suitable than separated deployment? → When GPU resources are limited and training is at medium scale — hybrid saves GPUs (no need for two clusters); but the switching overhead (weight sync + memory reallocation) reduces throughput, and in multi-turn agent scenarios inference KV cache and training optimizer states compete for memory, requiring careful management.

</details>

<details class="qa"><summary>3. Why is "the environment the bottleneck, not the GPU"?</summary>

A: A single GPU inference step takes milliseconds, whereas an environment step — running a unit test (10–60s), loading a webpage (1–10s), taking a screenshot + executing a GUI action (1–5s) — takes seconds. One rollout worker's GPU utilization = `GPU_inference_time / (GPU_inference_time + env_wait_time)`, typically below 5% in synchronous mode. So the system throughput ceiling is capped by `N_parallel_envs × per-env speed`, not GPU compute.

**Follow-up:** How do you prevent the GPU from waiting on environments? → Many parallel env workers (tens to hundreds); at any moment some environment has already returned and is ready for inference; plus async dispatch — the rollout worker issues an inference request as soon as the environment returns, and the GPU always has work. This is exactly the problem the three-pool architecture and SkyRL-Agent's async pipeline solve.

</details>

### L2 Intermediate

<details class="qa"><summary>4. What are the core architectural differences among verl / OpenRLHF / AReaL? How do you choose?</summary>

A: verl is a **hybrid engine** (inference + training time-multiplexed on the same GPUs); OpenRLHF is **Ray actors** (naturally async scheduling); AReaL is **fully async separated** (independent GPU pools for generation and training). Selection: rapid prototyping → OpenRLHF (Ray ecosystem is flexible); medium scale + existing FSDP → verl (hybrid saves GPUs); large-scale continuous training → AReaL (agent-native, fully async, high throughput). But **evaluating current-release maturity** matters more than architectural philosophy — all frameworks are iterating fast.

**Follow-up:** When is a hybrid engine actually a disadvantage? → ① multi-turn long-trajectory scenarios where KV cache and training state compete for memory; ② when inference and training need different GPU precisions (inference can use INT8/FP8, training needs FP16/BF16); ③ when the inference-to-training throughput ratio is imbalanced, the hybrid switching overhead eats the gains.

</details>

<details class="qa"><summary>5. Why does multi-turn agent KV cache "explode"? What are the mitigation strategies and their respective costs?</summary>

A: At turn $t$ the context = all prior $t-1$ turns of history → sequence length $O(T \times L)$, KV cache grows proportionally. Mitigations: ① **KV eviction** — discard low-attention tokens, lossy (discarded info may later be needed); ② **summarization** — compress distant turns into text, saves memory but compression error affects decisions; ③ **truncation-restart** — drop old history beyond the limit, agent loses memory of early actions; ④ **external memory** — write to a vector DB, retrieval precision and latency are the cost. In practice: recent turns full + distant turns summarized + global eviction threshold.

**Follow-up:** Why isn't "add more GPUs" listed as a solution? → Simply adding DP (data-parallel) GPUs does not solve the single-trajectory KV cache ceiling — each trajectory's KV cache still must fit on a single GPU; TP (tensor parallelism) / PP (pipeline parallelism) / context parallelism can shard KV across GPUs to raise the single-trajectory ceiling, but at the cost of communication overhead and complexity.

</details>

<details class="qa"><summary>6. Why is Parquet recommended over JSONL for trajectory storage? What does the concrete pipeline look like?</summary>

A: ① **Columnar**: can read only the `reward` column for filtering without deserializing full text → filtering speed >10× over JSONL; ② **Compression**: columnar compression ratios are high for same-type data (~5–10× vs JSONL text); ③ **Ecosystem**: pandas/polars natively read it, spark can do distributed processing. Pipeline: rollout → store as Parquet (one file per N trajectories) → filtering stage reads reward + metadata columns for selection → training reads the filtered trajectories' token sequences. Caution: for tasks that modify trajectory text (summarization, rewrite), Parquet string columns are less diff-friendly than JSONL, but most RL scenarios are read-only.

**Follow-up:** How do you manage a Parquet data lake of millions of trajectories? → Partition by date / task type / model version; each trajectory carries a unique ID + model checkpoint version for full traceability; use parquet → arrow → dataloader zero-copy chain to load token columns, avoiding the Python-loop deserialization bottleneck.

</details>

### L3 Deep

<details class="qa"><summary>7. To design an agent RL training system at production SWE-agent scale (thousands of repos, tens of thousands of issues — beyond SWE-bench's original 12 repos / 2,294 tasks), what are the hardest engineering problems at the environment layer?</summary>

A: ① **Multi-repo heterogeneity**: each repo needs an independent docker image + dependencies; cannot share a single image → image build and caching pipeline (incremental builds, layer sharing). ② **Non-deterministic tests**: the same fix running the same test suite twice can yield different results (network / random seeds / timing dependencies) → reward computation needs multiple runs with majority voting, or CI-style rerun mechanisms. ③ **Test execution safety**: agent-generated code may include `rm -rf /` or data-exfiltrating malicious/erroneous operations → docker must enforce network isolation + filesystem write restrictions + resource quotas. ④ **Large-scale docker scheduling**: thousands of concurrent tests need an equivalent number of docker daemons; single-machine docker concurrency is capped by PID/memory → requires an environment cluster across multiple CPU machines + a unified scheduling layer. ⑤ **Repo state management**: each issue corresponds to a specific git commit of the repo; checkout and reset operations must complete in seconds (hot checkout pool).

**Follow-up:** How do you prevent the agent from modifying the environment itself during training (e.g. editing test files to "fool" the criterion)? → The files/directories containing the criteria (unit tests) are read-only or invisible to the agent; test execution runs in an independent non-interactive docker container where the agent cannot touch the test code; after environment reset, a docker diff checks for unexpected file changes.

</details>

<details class="qa"><summary>8. When building an agent RL training run from scratch, what are the three most underestimated things at the infra level?</summary>

A: ① **Environment reliability investment**: the first training script usually assumes environments are reliable; in reality docker timeouts, network jitter, flaky tests, and web page changes make 5–20% of rollouts produce unreliable rewards; fixing these costs far more than writing the training loop itself. ② **Fine-grained KV cache and memory management**: single-turn RLHF training never worries about KV cache, but multi-turn agent KV cache can instantly OOM an 80G A100; solving this requires deep configuration of the inference engine's KV eviction/compression, not simply adding GPUs. ③ **Trajectory quality beats quantity**: the first instinct is "run more rollouts," but unfiltered masses of low-quality trajectories (all-zero reward / env failures / Echo Trap) contaminate training batches and waste GPU compute; investment in filtering pipeline design pays off more than simply scaling up rollouts.

**Follow-up:** What are the dependencies among these three? → Environment and KV cache are **hardware prerequisites** (if they aren't solved, training won't run at all); trajectory filtering is an **efficiency lever** (it runs, but bad quality = wasted compute). The sane order: first ensure environment reliability (single trajectory reproducible) → then tune KV cache (batch and concurrency work) → then add filtering (improve training efficiency).

</details>

<details class="qa"><summary>9. Some say "most of the cost of agent RL training is in the environment, not the GPU." Quantitatively analyze when this claim holds and when it does not.</summary>

A: **When it holds**: environment step latency is high (e.g. SWE unit tests 30s) and concurrent env count is limited → few effective trajectories produced per GPU-hour → most wall-clock time is spent waiting on environments rather than computing gradients. Here adding GPUs does not accelerate (environment is the bottleneck first); adding env workers does. **When it does not hold**: environment is fast (e.g. rule-based checks <100ms) or already highly parallelized (thousands of workers) → GPU training becomes the bottleneck (gradient computation time > environment wait time) → here adding GPUs is effective. **Quantitative judgment**: compute `GPU_utilization = GPU_inference+training_time / (GPU_inference+training_time + GPU_waiting_on_env_time)`; if <50% → env bottleneck; >80% → GPU bottleneck. In practice, **small-scale experiments are usually GPU-bottlenecked; large-scale production is usually environment-bottlenecked** (the bottleneck shifts with scaling).

**Follow-up:** What does this bottleneck analysis imply for GPU selection? → If environment-bottlenecked, buying faster GPUs (H100 over A100) barely improves training speed — spend the money on more CPU env workers instead; if GPU-bottlenecked, faster GPUs directly improve throughput. Run the bottleneck analysis before ordering hardware.

</details>

---

## References

> All are primary system/framework sources, web-verified. Framework versions and performance numbers change fast; this page records only architectural design ideas.

<ol>
<li id="ref-1">Volcano Engine. <em>verl: Volcano Engine Reinforcement Learning for LLMs</em>. <a href="https://github.com/volcengine/verl">github.com/volcengine/verl</a> — hybrid engine, SPMD 3D parallelism, agent RL support in active development. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Hu et al. <em>OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework</em>. 2024. <a href="https://arxiv.org/abs/2405.11143">arXiv:2405.11143</a> — Ray-native async RLHF framework. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Ant Group. <em>AReaL: A Large-Scale Asynchronous RL System for Language Reasoning</em>. 2025. <a href="https://arxiv.org/abs/2505.24298">arXiv:2505.24298</a> — fully async RL system (systems-level, not an algorithmic variant; paper focused on reasoning, architecture applicable to agents). <a href="#fnref-3">↩</a></li>
<li id="ref-4">NovaSky/Berkeley. <em>SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent</em>. 2025. <a href="https://arxiv.org/abs/2511.16108">arXiv:2511.16108</a> — CPU/GPU pipeline overlap + long-horizon tool scheduling. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Kwon et al. <em>Efficient Memory Management for Large Language Model Serving with PagedAttention</em>. 2023. <a href="https://arxiv.org/abs/2309.06180">arXiv:2309.06180</a> — vLLM: PagedAttention + continuous batching. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Zheng et al. <em>SGLang: Efficient Execution of Structured Language Model Programs</em>. 2024. <a href="https://arxiv.org/abs/2312.07104">arXiv:2312.07104</a> — SGLang radix attention + structured generation. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Duan et al. <em>SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</em>. 2025. <a href="https://arxiv.org/abs/2502.18449">arXiv:2502.18449</a> — SWE-domain RL training data pipeline (this page cites its environment design). <a href="#fnref-7">↩</a></li>
<li id="ref-8">Qi et al. <em>WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning</em>. 2024. <a href="https://arxiv.org/abs/2411.02337">arXiv:2411.02337</a> — Web-domain RL training environment and self-evolving curriculum. <a href="#fnref-8">↩</a></li>
</ol>
