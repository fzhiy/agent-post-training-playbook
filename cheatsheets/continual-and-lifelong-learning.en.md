# Continual & Lifelong Learning

> When a model learns continuously on **sequential tasks**, how can it acquire new knowledge without forgetting old knowledge? This is a mandatory question beyond the "train-once-then-deploy" paradigm, and a latent risk throughout the LLM post-training pipeline (pretrain → SFT → DPO → RL).

> ⚠️ **Study notes, not the author's own research** (see README honesty statement). Numbers / conclusions follow the original papers; uncertain items are noted.

## 0. The evolution

`IID one-shot training` → `sequential multi-task (Task 1 → Task 2 → …)` → **`catastrophic forgetting appears`**: gradients directly overwrite old weights.

**Stability-plasticity dilemma**: a network must be **plastic** — accepting gradient updates for new tasks — and **stable** — preserving representations of old tasks. The two requirements are inherently in tension.

## 1. Why catastrophic forgetting happens

A neural network's parameters are **shared storage** for all tasks. When running SGD on task $\mathcal{T}_2$, the gradient of the loss with respect to the parameters has no knowledge that "these weights matter for $\mathcal{T}_1$", so it overwrites them — this is **catastrophic forgetting**.

Classic settings:

| Setting | Task ID known at test time | Clear task boundary |
|---|---|---|
| Task-IL | Yes | Yes |
| Domain-IL | No | Yes |
| Class-IL (hardest) | No | Yes |
| Continual pretraining | No explicit boundary | No |

## 2. Three method families

### 2.1 Regularization

**Core idea**: add a **penalty that protects old weights** to the new-task loss, so that weights important to old tasks change as little as possible.

**EWC (Elastic Weight Consolidation)**<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Uses the diagonal of the Fisher information matrix to measure weight importance; quadratic penalty prevents forgetting. <a href="https://arxiv.org/abs/1612.00796">Kirkpatrick 2017 ↗</a></span></span> total loss:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\mathcal{T}_2}(\theta) + \frac{\lambda}{2} \sum_i F_i \,(\theta_i - \theta_i^*)^2$$

- $\theta_i^*$: parameter values after completing the old task
- $F_i$: diagonal element of the Fisher information matrix (measures the "importance" of parameter $i$ to the old task)
- $\lambda$: hyperparameter controlling the stability vs. plasticity trade-off

$$F_i = \mathbb{E}_{\mathcal{D}_1}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**SI (Synaptic Intelligence)**: an online version of EWC that **accumulates** each parameter's contribution to the loss during training, without needing to recompute the Fisher after the task ends.

**MAS (Memory Aware Synapses)**: importance is estimated by the gradient norm of the output function with respect to parameters, requiring no labeled data.

| Method | Importance estimate | Needs old data | Compute cost |
|---|---|---|---|
| EWC | Fisher diagonal | No | Medium (one backward pass) |
| SI | Online trajectory integral | No | Low (computed during training) |
| MAS | Output gradient norm | No (only unlabeled input needed) | Low–medium |

### 2.2 Rehearsal & Replay

**Core idea**: **mix in old-task samples** during new-task training, so that gradients "remember" the past simultaneously.

**Experience Replay**: maintain an **episodic memory** buffer holding real samples from old tasks; randomly interleave them at a fixed ratio during new-task training.

**GEM (Gradient Episodic Memory)**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Projects gradient updates into the feasible region where old-task losses do not increase, while allowing forward transfer. <a href="https://arxiv.org/abs/1706.08840">Lopez-Paz 2017 ↗</a></span></span>:

For each old task $k$, requires the updated gradient to satisfy:
$$\langle g, g_k \rangle \geq 0$$
i.e., the new gradient must not point "opposite" to the old-task gradient. If violated, $g$ is projected onto the feasible region.

**A-GEM (Averaged GEM)**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Merges multiple old-task constraints into a single average-gradient constraint, drastically reducing computation while achieving performance comparable to GEM. <a href="https://arxiv.org/abs/1812.00420">Chaudhry 2019 ↗</a></span></span>: uses the **average gradient** $g_{\text{ref}}$ over all old-task buffers as the single constraint, reducing the per-step QP projection from $K$ constraints to 1:

$$\langle g, g_{\text{ref}} \rangle \geq 0, \quad g_{\text{ref}} = \frac{1}{K}\sum_k g_k$$

**Generative Replay**: use a **generative model** (e.g., VAE, GAN) to learn the old-task distribution and synthesize "pseudo-old data" at training time — no real old samples need to be stored, but generative quality bottlenecks accumulate error.

**DER (Dark Experience Replay)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Stores old-sample logits (soft targets) in the buffer and uses MSE to match them, fusing rehearsal with knowledge distillation. <a href="https://arxiv.org/abs/2004.07211">Buzzega 2020 ↗</a></span></span>: buffer stores $(x, y, z)$ where $z$ is the **logit** (dark knowledge) from the model at the past time step; at replay time the model must also match $z$:

$$\mathcal{L}_{\text{DER}} = \mathcal{L}_{\text{CE}}(x,y) + \alpha \cdot \text{MSE}(f_\theta(x),\, z)$$

### 2.3 Parameter Isolation & Architectural CL

**Core idea**: different tasks occupy **different sub-networks**; new tasks expand capacity without modifying old-task parameters — forgetting is structurally eliminated.

**Progressive Neural Networks**<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Adds a new network column for each new task; lateral connections exploit knowledge from frozen old columns; forgetting is structurally impossible. <a href="https://arxiv.org/abs/1606.04671">Rusu 2016 ↗</a></span></span>: each task gets an independent network column; old columns are frozen; new columns read old-column activations via **lateral connections**:

$$h_k^{(\ell)} = f\!\left(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)}\right)$$

- Advantage: zero forgetting, natural forward transfer
- Disadvantage: parameter count grows linearly with the number of tasks

**PackNet**: performs **iterative pruning + freezing** within a single network, assigning a parameter mask to each task, with no shared gradient paths between tasks.

**LoRA-based / Adapter-based CL**: incrementally adds a set of LoRA weights or adapter modules per task; the backbone is frozen; tasks are routed to the corresponding adapter via task ID — parameter overhead is manageable and LLM-friendly.

| Method family | Zero forgetting | Forward transfer | Parameter growth | Needs old data |
|---|---|---|---|---|
| Regularization (EWC/SI/MAS) | Approximate | Limited | None | No |
| Replay (ER/GEM/A-GEM/DER) | Approximate | Yes | Buffer | Yes (partial) |
| Parameter isolation (ProgNN/PackNet/LoRA-CL) | Yes | Limited–yes | Linear–lightweight | No |

## 3. Knowledge Distillation: LwF

**LwF (Learning without Forgetting)**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">When training on a new task, uses the soft outputs of the old model as distillation targets, mitigating forgetting without storing any old data. <a href="https://arxiv.org/abs/1606.09282">Li 2016 ↗</a></span></span>:

- During inference on new-task $\mathcal{T}_2$ data, first use the **old model** $f_{\theta^*}$ to produce soft output $\hat{y}^{\text{old}}$
- Joint optimization:

$$\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{new}}(f_\theta(x), y) + \lambda \cdot \mathcal{L}_{\text{KD}}(f_\theta(x), \hat{y}^{\text{old}})$$

- $\mathcal{L}_{\text{KD}}$ is typically temperature-scaled KL divergence or cross-entropy
- **No old data required**; downside: soft-target quality degrades when task drift is large

LwF is essentially **implicit replay of the old model's knowledge** rather than old data — attractive in privacy-sensitive or storage-constrained settings.

## 4. Evaluation Metrics

Let $a_{i,j}$ denote accuracy on task $j$ measured after completing task $i$, across $T$ tasks total.

**Average Accuracy (AA)**: average accuracy over all tasks after learning all of them:

$$\text{AA} = \frac{1}{T} \sum_{j=1}^{T} a_{T,j}$$

**Backward Transfer (BWT)** — more negative means more forgetting:

$$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} \bigl(a_{T,j} - a_{j,j}\bigr)$$

**Forward Transfer (FWT)** — positive values indicate old tasks benefited new ones:

$$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^{T} \bigl(a_{j-1,j} - b_j\bigr)$$

where $b_j$ is the baseline accuracy for task $j$ trained independently from random initialization.

| Metric | Measures | Ideal value |
|---|---|---|
| AA | Overall memory retention | Higher is better |
| BWT | Degree of forgetting | ≥ 0 (closer to 0 or > 0 is better) |
| FWT | Forward transfer | > 0 |

> **Forgetting** is sometimes defined directly as the mean accuracy drop for each task from "when learned" to "final", which is the negation of BWT.

## 5. The LLM Angle

### 5.1 Continual Pretraining

Continuing to train a language model on new-domain corpora after initial pretraining (e.g., medical text, code updates); core challenges:

- Old general capabilities (math reasoning, instruction following) may degrade
- Distribution shift from new corpora may be milder than task-level shift but persists longer
- Common mitigations: learning-rate warm-up restart, replaying a small amount of general data, reducing the learning rate

### 5.2 Continual Instruction-Tuning

Sequentially introducing new instruction types (e.g., coding → then math → then safety); each SFT round may overwrite behaviors learned in previous rounds. LoRA-based or adapter-per-task approaches are natural low-parameter solutions: the backbone is shared, and behaviors are separated by module routing.

### 5.3 Continual Alignment & Alignment Tax

**Sequential alignment pipeline**: `pretrain → SFT → DPO → RL (RLHF/RLVR)` — each step continues training from the previous step's checkpoint.

**Alignment tax**: alignment often trades away part of general capabilities (e.g., code generation, factuality). When steps are stacked sequentially, the tax accumulates:

- Excessive SFT may compress knowledge diversity
- Doing RLHF after DPO can lead to over-refusal or format degradation
- Every hop faces the CL problem of "new alignment objective vs. behavior learned in the previous hop"

**Mitigation strategies**:

1. **KL constraint** (PPO clip / DPO reference model) — limits each step's deviation from the reference; structurally analogous to implicit EWC
2. **Replay old preference data** — mix in data from earlier alignment stages
3. **LoRA with independent adapter per stage** — backbone unchanged, alignment behavior localized

### 5.4 Why CL Matters for LLM Post-Training

| Scenario | CL challenge |
|---|---|
| Incremental update of a new model version | Cannot retrain from scratch; must fine-tune incrementally without losing old capabilities |
| Multiple rounds of RLHF iteration | Each policy update may overwrite alignment from the previous round |
| Personalization / continuous user feedback | Adapt to new user preferences while preserving general capabilities |
| Knowledge update (time-sensitive information) | Inject new facts without disturbing old knowledge structure |

## 6. From-scratch EWC

```python
import torch
import torch.nn.functional as F

def compute_fisher_diagonal(model, dataloader, device="cpu"):
    """
    Estimate the Fisher information matrix diagonal using squared log-likelihood gradients.
    dataloader: old-task data; model: model after completing the old task (parameters fixed).
    Returns dict: param_name -> F_i (same shape as parameter)
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        log_prob = F.log_softmax(logits, dim=-1)
        # Use predicted label to approximate sampling — this is a simplified implementation;
        # the original EWC paper actually uses the true label y to estimate Fisher
        # (empirical Fisher = E_{(x,y)~D}[grad log p(y|x)^2])
        pred = log_prob.argmax(dim=-1)
        loss = F.nll_loss(log_prob, pred, reduction="sum")
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        n_samples += x.size(0)

    fisher = {n: f / n_samples for n, f in fisher.items()}
    return fisher


class EWCTrainer:
    """
    Adds an EWC quadratic penalty during new-task training to prevent old-task parameters from being overwritten.
    """
    def __init__(self, model, old_dataloader, ewc_lambda=5000.0, device="cpu"):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda

        # Save a snapshot of old-task parameters
        self.theta_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
        }
        # Compute Fisher diagonal
        self.fisher = compute_fisher_diagonal(model, old_dataloader, device)

    def ewc_penalty(self):
        """EWC penalty term: (lambda/2) sum_i F_i (theta_i - theta*_i)^2"""
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            penalty = penalty + (
                self.fisher[n] * (p - self.theta_star[n]).pow(2)
            ).sum()
        return 0.5 * self.ewc_lambda * penalty

    def train_step(self, x, y, optimizer, criterion):
        """One training step on the new task: task loss + EWC penalty."""
        x, y = x.to(self.device), y.to(self.device)
        optimizer.zero_grad()
        logits = self.model(x)
        task_loss = criterion(logits, y)
        loss = task_loss + self.ewc_penalty()
        loss.backward()
        optimizer.step()
        return task_loss.item(), loss.item()
```

> `ewc_lambda` is the core hyperparameter: too small and forgetting remains severe; too large and the new task cannot be learned. In practice one typically searches in $[100, 10^4]$ and accumulates multiple Fisher groups as tasks grow (online EWC merges them via a running average).

---

## Stratified follow-ups

### L1 Foundations

<details class="qa"><summary>1. What is catastrophic forgetting? Why are neural networks especially prone to it?</summary>

Answer: A neural network's parameters are **shared storage** for all tasks. When running SGD on new task $\mathcal{T}_2$, the gradient has no knowledge of which weights matter for old task $\mathcal{T}_1$, so it directly overwrites them and causes a sudden drop in old-task performance — this is catastrophic forgetting. The root cause is parameter sharing combined with independent optimization objectives: the new-task gradient is "noise" with respect to the old-task loss.

**Follow-up:** From an optimization-theory perspective, what is the essence of catastrophic forgetting? → The essence is that the gradient-descent direction for the new task is an ascent direction on the old-task loss surface — the optimal parameter regions for the two tasks do not overlap in parameter space. SGD moving along the new-task gradient simultaneously destroys the local minimum of the old task. This is the degenerate form of Pareto conflict in multi-objective optimization under single shared parameterization.

</details>

<details class="qa"><summary>2. What is the stability-plasticity dilemma? Give an intuitive analogy.</summary>

Answer: A network must simultaneously have **plasticity** — accepting gradient updates for new tasks — and **stability** — preserving old-task representations. The two are inherently in tension. Intuitive analogy: when learning a new language, the brain must remember the mother tongue (stability) while building new language circuits (plasticity); memorizing the new language by rote may crowd out the neural pathways of the mother tongue.

**Follow-up:** Along the stability-plasticity axis, where do the three method families (regularization / replay / parameter isolation) each fall? How do you choose based on task requirements? → Regularization (EWC/SI) leans toward stability: the penalty constrains parameter deviation, limiting plasticity. Replay sits in the middle: mixing in old data balances both ends, but buffer size determines the lean. Parameter isolation (ProgNN/LoRA-CL) leans most toward stability: old parameters are fully frozen and only new capacity is added. When tasks are highly correlated and forward transfer is valuable, choose replay; when tasks are independent and storage/privacy is constrained, choose parameter isolation.

</details>

<details class="qa"><summary>3. What is the core idea of EWC? What role does the Fisher information matrix play?</summary>

Answer: EWC (Elastic Weight Consolidation) adds a quadratic penalty $\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$ to the new-task loss, so that weights important to old tasks change as little as possible. The Fisher information matrix diagonal $F_i = \mathbb{E}[(\partial \log p_\theta / \partial \theta_i)^2]$ measures the **importance** of parameter $i$ to the old task — larger $F_i$ means a stronger penalty, and that parameter is "elastically" protected.

**Follow-up:** What are the main limitations of the Fisher information matrix diagonal approximation? What improved alternatives exist? → The diagonal approximation ignores inter-parameter covariance. When two parameters have strong coupled importance to the old-task loss (e.g., Q/K weights in attention), penalizing each independently underestimates the true curvature. Improvements include: block-diagonal approximation (K-FAC, retaining intra-layer covariance per block), Kronecker factorization, and SI's online trajectory integral (replaces Fisher with each parameter's actual contribution to loss reduction, avoiding post-hoc recomputation).

</details>

<details class="qa"><summary>4. Both LwF and EWC avoid storing old data — how do their anti-forgetting mechanisms differ?</summary>

Answer: EWC imposes constraints in **parameter space** — Fisher penalties keep important old-task weights close to their old values. LwF (Learning without Forgetting) imposes constraints in **output space** — it uses the old model's soft output on new-task data as a distillation target $\mathcal{L}_{\text{KD}}$, keeping the new model's output behavior close to the old model's. LwF requires no importance-score recomputation, but soft-target quality degrades as task drift accumulates.

**Follow-up:** Under what conditions does LwF's anti-forgetting effect severely degrade? How can it be mitigated? → When the input distributions of old and new tasks are extremely dissimilar (e.g., old task is image classification, new task is text), the old model's soft outputs on new-task data become nearly uniform or have very low confidence — the distillation signal quality approaches random, equivalent to having no constraint. Mitigation: combine with parameter-space constraints (EWC penalties on important weights) to supplement output-space distillation; or compute $\mathcal{L}_{\text{KD}}$ only on the subset where old and new task input spaces overlap, filtering out low-confidence soft targets.

</details>

### L2 Advanced

<details class="qa"><summary>5. What is the core difference between GEM and A-GEM in replay methods? Why is A-GEM more efficient?</summary>

Answer: GEM imposes a separate gradient constraint $\langle g, g_k\rangle \geq 0$ for each old task $k$, requiring a QP with $(K-1)$ inequalities at time complexity $\mathcal{O}(K^3)$. A-GEM merges all old-task constraints into a single mean constraint $\langle g, g_{\text{ref}}\rangle \geq 0$ with a closed-form projection; per-step computation drops from $\mathcal{O}(K \cdot d)$ to $\mathcal{O}(d)$, independent of the number of tasks.

**Follow-up:** How stable is A-GEM's mean constraint under buffer-sampling noise? What improvement directions exist? → A-GEM estimates $g_{\text{ref}}$ by randomly sampling from the buffer at each step; high-variance sampling makes the constraint direction unstable — when the estimated $g_{\text{ref}}$ deviates significantly from the true mean, the projection may go in the wrong direction and introduce additional forgetting noise. Improvement directions include: increasing the buffer sample size per step to reduce variance; smoothing historical $g_{\text{ref}}$ with momentum (similar to EMA); and follow-up work such as ER-ACE, which bypasses the gradient-projection framework entirely via asymmetric cross-entropy.

</details>

<details class="qa"><summary>6. What does each of BWT and FWT measure? What does it indicate if a model has very negative BWT but very positive FWT?</summary>

Answer: BWT (Backward Transfer) $= \frac{1}{T-1}\sum_{j=1}^{T-1}(a_{T,j}-a_{j,j})$ measures forgetting; more negative means greater performance drop on old tasks. FWT (Forward Transfer) $= \frac{1}{T-1}\sum_{j=2}^{T}(a_{j-1,j}-b_j)$ measures positive transfer from old tasks to new ones; more positive means better transfer. Very negative BWT but very positive FWT means the model leveraged old knowledge to accelerate learning on new tasks (good transfer) while severely overwriting old-task parameters (heavy forgetting) — a typical high-plasticity, low-stability model.

**Follow-up:** What are the limitations of directly using BWT and FWT to evaluate LLMs' continual-learning capability? → LLM task boundaries are blurry (SFT/DPO/RL are soft boundaries rather than discrete task sequences), making it hard to construct a clear $a_{i,j}$ performance matrix. LLM "capability" is multi-dimensional (reasoning / code / safety), and single-number accuracy cannot capture capability interference. Additionally, the stability gap (D3) means that end-of-task snapshots underestimate the worst-case forgetting during training — for deployment safety, peak forgetting magnitude is more critical than BWT.

</details>

<details class="qa"><summary>7. Why do Progressive Networks achieve "zero forgetting"? What is the cost?</summary>

Answer: Progressive Neural Networks add an independent network column for each new task; old-column parameters are fully frozen; new columns read old-column activations via lateral connections $h_k^{(\ell)} = f(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)})$ — the gradient path to old columns is cut off, so forgetting is structurally impossible. The cost is that parameter count grows linearly with the number of tasks.

**Follow-up:** How does LoRA-based CL mitigate the linear parameter growth compared to Progressive Networks? What is the fundamental difference in their zero-forgetting guarantees? → LoRA-CL adds only low-rank matrices $\Delta W = BA$ (rank $r \ll d$) per task; parameter increment is $\mathcal{O}(r \cdot d)$ rather than $\mathcal{O}(d^2)$, far more scalable than full-column expansion. However, the zero-forgetting guarantees differ in nature: ProgNN achieves a hard guarantee by structurally cutting gradients to frozen old columns; LoRA-CL relies on backbone freezing plus low-rank subspace allocation — if orthogonality is not enforced (e.g., without O-LoRA), different task adapters' activations may interfere, making it a soft guarantee that depends on subspace overlap.

</details>

<details class="qa"><summary>8. What advantages does LoRA-based CL have over EWC / replay? How is it used in the LLM post-training pipeline?</summary>

Answer: LoRA-based CL incrementally adds a set of low-rank adapters per task with the backbone frozen — parameter overhead is manageable, old tasks receive zero gradient interference, no old data needs to be stored, and no Fisher computation is required. In the LLM post-training pipeline (SFT → DPO → RL), each stage freezes the backbone and updates only one set of LoRA weights, confining the alignment tax to the adapter layer and leaving the backbone's general knowledge intact; at test time, routing by task-ID selects the corresponding adapter.

**Follow-up:** When using LoRA-based CL in LLMs, what engineering challenges arise from managing multiple adapter sets at deployment? Can O-LoRA's orthogonality constraint fundamentally eliminate this problem? → Engineering challenges include: needing to store and dynamically load adapters by task-ID (increasing inference latency and memory scheduling complexity), inability to merge adapter weights for mixed-task batches, and routing failure in Class-IL / continual pretraining settings where task-ID is unknown. O-LoRA's orthogonality constraint addresses only the adapter activation-interference problem; it does not solve routing failure — in settings without task-ID, additional mechanisms (such as a prototype classifier or task-inference module) are still needed to determine which adapter to use.

</details>

### L3 Deep-dive

<details class="qa"><summary>9. What are the main differences between LLM continual pretraining and the classic Task-IL setting? What unique challenges does it pose?</summary>

Answer: Classic Task-IL has explicit task boundaries and task IDs; LLM continual pretraining has no explicit boundaries — domain corpora flow in continuously, and task granularity is fuzzy (general capabilities and new domains overlap heavily). Unique challenges include: old general capabilities (math reasoning, instruction following) may degrade in hard-to-detect ways; distribution shift from new corpora persists for a long time; forgetting cannot be quantified with the standard $a_{i,j}$ matrix; and the ratios for learning-rate warm-up restarts and general-data replay are difficult to tune.

**Follow-up:** In LLM continual pretraining without explicit task boundaries, how can general-capability degradation be monitored and diagnosed in real time? → A "probe benchmark matrix" must be designed (covering dimensions such as reasoning / code / math / instruction following), evaluated at fixed step intervals during training — effectively implementing per-iteration continuous evaluation at LLM scale (see stability gap D3). This should be combined with gradient/activation drift detection to locate which layers are heavily rewritten by new corpora, so as to decide whether to trigger general-data replay or lower the learning rate. The cost is significant additional compute overhead, requiring a trade-off between monitoring frequency and training efficiency.

</details>

<details class="qa"><summary>10. What CL problem does the alignment tax in the sequential alignment pipeline (SFT → DPO → RL) fundamentally represent? What approaches mitigate it?</summary>

Answer: The alignment tax is fundamentally the accumulated forgetting tax per hop in sequential CL — each alignment step continues training from the previous step's checkpoint, and the gradient of the new alignment objective overwrites previously learned behavior. Mitigation approaches: (1) **KL constraint** (PPO clip / DPO reference model) limits the deviation magnitude per step; its second-order expansion is equivalent to EWC weighted by Fisher; (2) **Replay old preference data** mixed in from earlier alignment stages; (3) **LoRA with independent adapter per stage** localizes alignment behavior, leaving the backbone unchanged.

**Follow-up:** The KL penalty in PPO is mathematically an implicit EWC, but what key practical differences affect its anti-forgetting effectiveness? → Three key differences: (1) **Anchor dynamics**: the KL constraint anchors to a dynamically updated reference policy $\pi_{\text{ref}}$ that may be updated each RL round; the EWC anchor $\theta^*$ is a fixed snapshot after the old task — a dynamic anchor in long sequential alignment causes "anchor drift," continuously shifting the anti-forgetting center. (2) **Full matrix vs. diagonal approximation**: KL divergence uses the complete Fisher matrix; EWC uses a diagonal approximation — the former is a more accurate constraint but not computed explicitly. (3) **Coefficient tuning**: PPO's $\beta$ must be dynamically balanced between exploration and conservatism, and too large a value prevents policy convergence; EWC's $\lambda$ is typically set statically after the task is fixed — $\beta$ is harder to tune in dynamic alignment scenarios.

</details>

<details class="qa"><summary>11. EWC uses the empirical Fisher (true labels) rather than the true Fisher (labels sampled from the model) — when does this cause problems?</summary>

Answer: The empirical Fisher $F_i^{\text{emp}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}[(\partial \log p_\theta(y|x)/\partial \theta_i)^2]$ uses the dataset's true labels $y$, and equals the true Fisher only when the model perfectly fits the data ($p_\theta(y|x) \approx \delta_y$). Problematic scenarios: high estimation noise in $F^{\text{emp}}$ when the model is far from convergence on the old task; $F^{\text{emp}}$ direction contaminated when old-task labels are noisy; errors that propagate continuously during online multi-task accumulation (online EWC); and dual degradation from the diagonal approximation and empirical error when inter-task distribution difference is extreme — protecting the wrong directions.

**Follow-up:** Besides switching to the true Fisher, what structural alternatives exist to fundamentally bypass the limitations of diagonal empirical Fisher? → Two categories of fundamental alternatives: (1) **Replace the importance measure**: SI uses each parameter's actual contribution to loss reduction, integrated as $\Omega_i \propto \int \frac{\partial \mathcal{L}}{\partial \theta_i} \dot\theta_i\, dt$, rather than Fisher — label-free and online-accumulative, avoiding both empirical and diagonal errors. MAS uses output gradient norms, requiring no labels at all. (2) **Replace the quadratic-penalty framework entirely**: PackNet / LoRA-CL's parameter isolation schemes require no importance estimation — old-task parameters are structurally frozen, so the accuracy of importance estimation becomes irrelevant. This shows that Fisher estimation limitations are a systemic problem of the regularization paradigm, not an engineering problem that can be improved indefinitely.

</details>

<details class="qa"><summary>12. What are the advantages and limitations of Generative Replay compared to experience replay? Is it more feasible in the era of large models?</summary>

Answer: Generative Replay uses a generative model (VAE/GAN) to synthesize "pseudo-old data" for training. **Advantages**: no real old samples need to be stored, naturally resolving privacy and storage constraints. **Limitations**: generative-quality bottlenecks accumulate error along the task sequence — the generative model itself faces forgetting, and the deviation between synthesized data and the true distribution compounds over long task chains. In the era of large models, LLMs have extremely strong generative capabilities, and using an LLM to synthesize old-task data achieves quality far higher than a small GAN — feasibility is significantly improved, but generation cost is high and synthesized data may still have systematic distributional bias relative to the original.

**Follow-up:** In LLM Generative Replay, what are the core problems with using the model itself as the generator ("self-replay") versus using an independently frozen old model as the generator? → Self-replay (having the current model generate old-task data to train itself) suffers from **self-reinforcing bias**: the model has already partially forgotten the old task after new-task training, so the generated old-task samples are themselves lower quality; training on these samples reinforces forgetting — forming a negative feedback loop. Using an **independently frozen old model** as the generator (similar to LwF's teacher) ensures generation quality does not degrade with the current model, but requires storing a full additional model copy whose storage overhead may be higher than an experience-replay buffer; moreover, the frozen old model cannot be updated to track shifts in the new data distribution, accumulating distributional bias over long task chains.

</details>

---

## Deep-dive

> The following are interview-level advanced Q&A covering the 7 most frequently probed hard points from the notes above. All conclusions come from the cited papers and contain no original research by the author.

### D1. Empirical Fisher vs. True Fisher — Which does EWC use, and when does it break down?

**True Fisher** is defined as the expectation of the outer product of log-likelihood gradients under the **model's predictive distribution** $p_\theta(y|x)$:

$$F_i^{\text{true}} = \mathbb{E}_{x \sim \mathcal{D},\; y \sim p_\theta(\cdot|x)}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**Empirical Fisher** replaces the $y$ sampled from the model distribution with the **true labels** $y$ in the dataset:

$$F_i^{\text{emp}} = \mathbb{E}_{(x,y) \sim \mathcal{D}}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

The two are equal only when the model perfectly fits the data ($p_\theta(y|x) \approx \delta_y$). Research has shown that during optimization, empirical Fisher cannot generally capture second-order information, and its deviation from the true Hessian in practice can be large — pathological behavior can occur even on simple optimization problems.

**EWC uses empirical Fisher**: the implementation computes the expected squared gradient over $(x,y)$ pairs from the old-task dataset $\mathcal{D}_1$. The code comments also note "use true label $y$ to estimate Fisher."

**When does the approximation break down?**

| Scenario | Risk |
|---|---|
| Computing $F$ when the model is far from convergence on the old task | $F^{\text{emp}}$ estimate is noisy; protects wrong directions |
| Old-task labels are noisy | $F^{\text{emp}}$ direction is contaminated by label noise |
| Multi-task sequential accumulation (online EWC) | Early-task $F^{\text{emp}}$ errors propagate continuously in the running average |
| Extreme inter-task distribution gap (e.g., language → vision) | Diagonal approximation is already a strong assumption; empirical error compounds the degradation |

Intuition: EWC's quadratic penalty is essentially a local quadratic approximation of the parameter space — diagonal empirical Fisher is the coarsest layer of this approximation. When the old task is well-converged, labels are clean, and parameter correlations are weak, the approximation is acceptable; otherwise the "importance scores" decouple from the true loss-landscape curvature, and the penalty protects wrong directions.

---

### D2. Online / Streaming EWC — How to accumulate Fisher across tasks?

Standard EWC stores one $(F^{(k)}, \theta^{*(k)})$ pair for each new old-task encountered, with memory growing linearly in task count $K$. **Online EWC** (Progress & Compress, Schwarz et al. 2018)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Progress & Compress dual-network framework: active column learns new tasks; knowledge base consolidates with online EWC; Fisher is accumulated across tasks via exponential moving average. <a href="https://arxiv.org/abs/1805.06370">Schwarz 2018 ↗</a></span></span> merges all historical tasks' Fisher into a single $\tilde{F}$ via **exponential moving average (EMA)**:

$$\tilde{F}^{(k)} = \gamma \cdot \tilde{F}^{(k-1)} + (1-\gamma) \cdot F^{(k)}$$

The total penalty degenerates to a single penalty term:

$$\mathcal{L}_{\text{online-EWC}} = \mathcal{L}_{\mathcal{T}_k}(\theta) + \frac{\lambda}{2}\sum_i \tilde{F}_i^{(k-1)}\bigl(\theta_i - \theta_i^{*(k-1)}\bigr)^2$$

**Advantage**: constant memory (stores only one $\tilde{F}$ and one $\theta^*$ snapshot).

**Cost and risks**:

- EMA exponentially downweights Fisher from earlier tasks — the older the task, the weaker the protection; it inherently favors protecting "recent" tasks.
- $\gamma$ is a new hyperparameter: $\gamma \to 1$ retains history but forgetting rate is high; $\gamma \to 0$ degenerates to looking only at the previous task.
- The reference point $\theta^*$ is updated each round; after each compression the new $\theta^*$ is not the common optimum for all historical tasks — the penalty center drifts as tasks accumulate.

**SI (Synaptic Intelligence)** is another streaming approach: it **online-accumulates** each parameter's integral contribution to loss reduction during training as an importance measure:

$$\Omega_i \propto \int_{\text{trajectory}} \frac{\partial \mathcal{L}}{\partial \theta_i} \cdot \dot\theta_i \, dt$$

SI requires no additional backward pass after the task ends and is suited to **streaming settings without explicit task boundaries**, but the signal-to-noise ratio of the importance estimate is lower than EWC.

---

### D3. Stability Gap — Why does forgetting worsen before recovering?

De Lange et al. (arXiv 2022, ICLR 2023)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Proposes a per-iteration continuous evaluation framework; first systematically documents the stability gap: performance drops sharply after a task switch then recovers; evaluating only at task end misses this phenomenon. <a href="https://arxiv.org/abs/2205.13452">De Lange 2022 ↗</a></span></span> discovered through **per-iteration continuous evaluation** that almost all mainstream CL methods (including EWC, ER, A-GEM) experience a sharp drop in old-task performance in **the first few steps after switching to a new task** — followed by gradual recovery and even exceeding the pre-switch level as training progresses. This "drop then recover" phenomenon is called the **stability gap**.

**Mechanistic intuition:**

```
Task T1 training complete → switch to T2
     ↓ first few dozen steps
T2's large gradients hit shared feature layers → T1 representations temporarily disrupted → T1 performance drops sharply
     ↓ training continues
regularization / replay constraints begin to take effect → T1 representations gradually recover
     ↓ end of T2
T1 performance recovers (but may be below its peak at the end of T1 training)
```

**Why was this not found before?**

The standard evaluation protocol tests only once **after each task is fully learned**, which happens to skip the drop period — the stability gap is nearly invisible in the end-of-task "snapshots."

**Interview key points:**

- The stability gap means the BWT metric (end-of-task snapshot) **underestimates the true magnitude of forgetting** — in safety-critical settings (incremental updates of deployed LLMs), worst-case performance during training may be much more severe than BWT reflects.
- Mitigation directions: reduce new-task learning rate during the warm-up transition period; "probe" the new task with small batches before full-speed training; increase the old-task proportion in replay at the beginning of the task switch.

---

### D4. Replay Sample Selection Strategies and Buffer Size Effects

Buffer size $M$ is one of the most critical hyperparameters in replay methods. Three main selection strategies:

**① Random Reservoir Sampling**

For a data stream of unknown length, maintain a buffer of size $M$; each new sample $x_t$ is included in the buffer with probability $M/t$ (randomly replacing an existing sample). This guarantees that every sample in the buffer is a uniform subset of all historical samples — no class bias, simple to implement, and the default strategy for ER / A-GEM.

**② Herding (iCaRL)**

iCaRL's herding algorithm: greedily and iteratively selects samples so that the feature mean of the exemplar set best approximates the feature mean of the entire class:

$$p_t = \arg\min_{x \in \mathcal{D}_k} \left\| \mu_k - \frac{1}{t}\!\left(\phi(x) + \sum_{j=1}^{t-1}\phi(p_j)\right)\right\|$$

Herding outperforms random sampling for class exemplar selection, especially when $M$ is very small (only a few samples per class), preserving within-class diversity better. However, herding requires having all class data available and depends on the feature space — after feature drift, old-class exemplars may no longer represent their current features.

**③ Gradient-Based Selection**

Selects old samples that have the greatest influence on the new-task gradient update — typically samples whose gradient direction most "conflicts" with the new-task gradient. The intuition is to constrain gradients using the hardest-to-satisfy samples. Compute cost is high (requires extra backward passes to estimate gradient influence per candidate sample); rarely used in large-scale experiments.

**Buffer size effects summary:**

| Buffer size $M$ | Random/Reservoir | Herding | Gradient-based |
|---|---|---|---|
| Very small (1–5 samples per class) | Poor coverage, severe forgetting | Clearly better than random | Good effect but extremely high cost |
| Medium (20–50 per class) | Approaches herding | Gap narrows | Diminishing returns |
| Large (approaching unlimited) | All three converge, approaching joint training | Same | Same |

Core insight: as $M \to \infty$, all replay methods degenerate to joint training (the upper bound); when $M$ is small, exemplar representativeness matters more than randomness — herding wins in this regime.

---

### D5. GEM's Per-Task QP Cost vs. A-GEM's Single Mean Constraint

**GEM's QP problem:**

Current new-task gradient is $g$; for each old task $k$ there is a constraint $\langle g, g_k \rangle \geq 0$. If the constraint is violated, solve:

$$\tilde{g} = \arg\min_v \|v - g\|^2 \quad \text{s.t.} \quad \langle v, g_k\rangle \geq 0,\; \forall k \in \{1,\ldots,K-1\}$$

This is a quadratic program (QP) with $(K-1)$ inequality constraints. Standard QP solvers have time complexity $\mathcal{O}(K^3)$ and space complexity $\mathcal{O}(K^2)$ — quickly infeasible as task count grows. GEM's implementation approximates the solution with iterative methods such as Frank-Wolfe; each step still requires $K$ gradient inner-product computations, i.e., $\mathcal{O}(K \cdot d)$ ($d$ = parameter dimension).

**A-GEM's single constraint:**

A-GEM merges all old-task constraints into one average gradient:

$$g_{\text{ref}} = \frac{1}{K-1}\sum_{k=1}^{K-1} g_k, \quad \text{s.t.}\; \langle g, g_{\text{ref}} \rangle \geq 0$$

If violated, the projection has a closed-form solution:

$$\tilde{g} = g - \frac{\langle g, g_{\text{ref}} \rangle}{\|g_{\text{ref}}\|^2} g_{\text{ref}}$$

**Per-step computation drops from $\mathcal{O}(K \cdot d)$ to $\mathcal{O}(d)$** — independent of task count (computing $g_{\text{ref}}$ can be done once and reused).

**Cost:**

A-GEM satisfies an average-direction constraint and does not guarantee $\langle \tilde{g}, g_k\rangle \geq 0$ for every individual old task $k$ — the loss on some single old task may increase. Empirically, A-GEM's AA and BWT are comparable to GEM's, but when gradients across tasks are highly heterogeneous (some task directions deviate greatly from the mean), certain old tasks occasionally experience more forgetting than with GEM.

**Interview memory point:** GEM = per-task constraint + quadratic program, $\mathcal{O}(K^3)$ QP; A-GEM = single mean constraint + closed-form projection, $\mathcal{O}(d)$; the trade-off is sacrificing per-task constraint guarantees for linear time complexity.

---

### D6. Why Is Class-IL Hardest? The Role of the Output Head

van de Ven & Tolias 2019<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">Systematically compares difficulty across the three CL settings (Task-IL / Domain-IL / Class-IL) on Split MNIST and Split CIFAR-100; regularization methods almost completely fail on Class-IL. <a href="https://arxiv.org/abs/1904.07734">van de Ven 2019 ↗</a></span></span>'s three-scenario framework reveals the fundamental difficulty gap:

**Output head structure comparison across three scenarios:**

| Scenario | Task-ID at test time | Output head | What the model must do |
|---|---|---|---|
| Task-IL | Known | Independent head per task (only current task activated) | Classify within the task; candidate set is known |
| Domain-IL | Unknown | Shared head (fixed output dimensionality) | Classify in a fixed output space; no need to distinguish tasks |
| Class-IL | Unknown | Shared head (all task classes) | Classify among all classes across all historical tasks |

**Three obstacles that make Class-IL hardest:**

1. **Task-ID inference problem**: the model does not know which task the current input belongs to and cannot route to the corresponding sub-classifier — it must distinguish all classes on a single head.

2. **Output head historical bias (recency bias)**: when learning a new task, only the new task's classes generate large gradient updates, skewing the logit scale of the output layer toward the new task — old-class logits are suppressed even if the feature layer still remembers the old task. This is "classifier-layer forgetting" unique to Class-IL.

3. **Fundamental failure of regularization**: EWC protects feature-layer parameters, but the new-class nodes initialized in the output head interfere with the gradient flow of old-class nodes — the Fisher diagonal cannot capture this cross-class output-layer interference. van de Ven et al.'s experiments show that EWC's accuracy on Class-IL approaches chance level.

**Remedies:**

- **Replay + prototype classifier** (e.g., iCaRL): use exemplar feature means for nearest-neighbor classification, bypassing output head bias.
- **Task-agnostic feature learning**: freeze a pre-trained backbone and only update a lightweight classification head — reducing feature drift.
- **Empirical bias correction**: at Class-IL test time, apply temperature adjustment or weighting to old-class logits to counteract recency bias.

---

### D7. LLM Forgetting vs. Capability Interference — Measurement Methods and Task-Order Sensitivity

**Key differences between LLM "forgetting" and classic CL:**

| Dimension | Classic CL (small model / classification) | LLM post-training |
|---|---|---|
| Task granularity | Clear (Task 1/2/3…) | Blurry ("math reasoning" / "code" / "safety alignment" overlap heavily) |
| Manifestation of forgetting | Drop in old-task classification accuracy | Capability **interference**: new-capability activation paths overwrite old-capability paths; not simple "forgetting" |
| Measurement difficulty | Directly quantifiable with $a_{T,j}$ matrix | Requires dedicated benchmarks (MMLU/GSM8K/HumanEval…) tracking each capability |
| Boundary clarity | Explicit task boundaries | No explicit boundaries; SFT→DPO→RL are **soft boundaries** |

**How to measure LLM CL quality:**

1. **Capability matrix tracking**: after each alignment stage ends, evaluate on several "probe benchmarks" (covering general reasoning, code, math, safety) — equivalent to constructing an $a_{i,j}$ matrix at LLM scale.
2. **BWT analogy for LLMs**: measure "change in coding capability after SFT compared to pretrain baseline" — if negative, it is part of the alignment tax.
3. **Activation/gradient analysis**: detect which layers' activation distributions change most before and after new-task training — locating the knowledge-storage layers that are "overwritten."

**Task-order sensitivity:**

LLM CL is highly sensitive to task order, because:

- The gradient directions during sequential training depend on the loss landscape shaped by previous tasks — doing math SFT before safety RLHF produces a very different result from the reverse order.
- Harder tasks (math/code) require large learning rates and large gradients, causing greater damage to parameters used by subsequent smaller tasks.
- **Cumulative nature of alignment tax**: in the SFT → DPO → RL sequence, each step's forgetting tax accumulates on the next step's checkpoint.

**KL constraint is implicit EWC:**

The KL penalty in PPO-RLHF:

$$r_{\text{KL}}(\theta) = \mathbb{E}\!\left[\log\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}\right] \cdot (-\beta)$$

Expanding this with a Taylor expansion around reference policy $\pi_{\text{ref}}$, the second-order term of KL divergence is proportional to the Fisher information matrix:

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{1}{2}(\theta - \theta_{\text{ref}})^T F(\theta_{\text{ref}}) (\theta - \theta_{\text{ref}})$$

That is, **PPO's KL penalty ≈ a full-matrix EWC weighted by Fisher** — both are "quadratic penalties on parameter deviation from a reference point, weighted by information-geometric curvature." The difference: EWC uses a diagonal approximation and explicitly stores the old-task Fisher; the KL constraint uses the full distributional distance and dynamically anchors to the current reference policy. The KL term in DPO follows the same logic — the reference model is mathematically equivalent to EWC's $\theta^*$.

---

### D8. LoRA-based CL — Adapter Interference and Routing

**Sources of adapter interference:**

The naive approach is to train an independent set of LoRA weights $\Delta W_k = B_k A_k$ for each task and route by task-ID at test time. Two sources of interference exist:

1. **Parameter-space overlap**: different tasks' low-rank subspaces may overlap substantially — if $\text{span}(A_k) \cap \text{span}(A_j) \neq \emptyset$, one task's adapter can interfere with another task's activations after merging.
2. **Routing failure without task-ID**: in Class-IL or continual pretraining settings, task-ID is unavailable at test time, making it impossible to route to the correct adapter.

**O-LoRA's orthogonal subspace approach:**

O-LoRA enforces, during LoRA training for task $k$, that the new task's low-rank subspace is orthogonal to all previously seen tasks' subspaces:

$$A_k V_{\text{prev}} \approx 0, \quad V_{\text{prev}} = \text{span}\bigl(\{A_j\}_{j<k}\bigr)$$

This is achieved by projecting gradients onto the orthogonal complement of $V_{\text{prev}}$. Orthogonal subspaces guarantee that different task adapters' activations do not interfere with each other — in settings where task-ID is known this yields approximately zero forgetting without storing old-task data.

**Limitations:**

- Available orthogonal dimensions decrease as task count increases — in rank-$r$ LoRA, at most about $d/r$ fully orthogonal tasks can be supported ($d$ = weight matrix dimension).
- In Class-IL settings without task-ID, orthogonality does not solve the routing problem — additional mechanisms to infer task-ID or use prototype classification are still required.
- Enforcing orthogonality may limit **forward transfer** between related tasks — tasks that could share subspaces to accelerate learning lose that benefit.

**Practice in LLM post-training:**

In the sequential alignment pipeline (SFT → DPO → RL), freezing the backbone and updating only one set of LoRA per stage is equivalent to a parameter isolation scheme with known task-ID — the accumulation of alignment tax is largely confined to the LoRA layers and backbone knowledge is not overwritten. The cost is the need to manage multiple adapter sets and their merging/routing logic at deployment.

---

## References

> All are original sources for classic load-bearing methods, verified line by line (title + arXiv ID). Click the superscript to jump; click ↩ to return.

<ol>
<li id="ref-1">Kirkpatrick et al. <em>Overcoming catastrophic forgetting in neural networks</em>. PNAS 2017. <a href="https://arxiv.org/abs/1612.00796">arXiv:1612.00796</a> — EWC: Fisher diagonal quadratic penalty against forgetting. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Lopez-Paz et al. <em>Gradient Episodic Memory for Continual Learning</em>. NeurIPS 2017. <a href="https://arxiv.org/abs/1706.08840">arXiv:1706.08840</a> — GEM: gradient projection constraint + forward/backward transfer. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Chaudhry et al. <em>Efficient Lifelong Learning with A-GEM</em>. ICLR 2019. <a href="https://arxiv.org/abs/1812.00420">arXiv:1812.00420</a> — A-GEM: single mean gradient constraint, efficient GEM. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Buzzega et al. <em>Dark Experience for General Continual Learning: a Strong, Simple Baseline</em>. NeurIPS 2020. <a href="https://arxiv.org/abs/2004.07211">arXiv:2004.07211</a> — DER: store logits as soft distillation targets + rehearsal. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Rusu et al. <em>Progressive Neural Networks</em>. 2016. <a href="https://arxiv.org/abs/1606.04671">arXiv:1606.04671</a> — Add a new column per task + lateral connections; structurally zero forgetting. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Li and Hoiem. <em>Learning without Forgetting</em>. ECCV 2016. <a href="https://arxiv.org/abs/1606.09282">arXiv:1606.09282</a> — LwF: old model soft outputs as distillation targets; no old data needed. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Schwarz et al. <em>Progress &amp; Compress: A scalable framework for continual learning</em>. ICML 2018. <a href="https://arxiv.org/abs/1805.06370">arXiv:1805.06370</a> — Dual-network active column + knowledge base; online EWC accumulates Fisher across tasks via EMA, constant memory. <a href="#fnref-7">↩</a></li>
<li id="ref-8">De Lange, van de Ven, and Tuytelaars. <em>Continual evaluation for lifelong learning: Identifying the stability gap</em>. ICLR 2023. <a href="https://arxiv.org/abs/2205.13452">arXiv:2205.13452</a> — Per-iteration continuous evaluation framework; discovers the stability gap phenomenon of sharp performance drop then recovery after task switch. <a href="#fnref-8">↩</a></li>
<li id="ref-9">van de Ven and Tolias. <em>Three scenarios for continual learning</em>. 2019. <a href="https://arxiv.org/abs/1904.07734">arXiv:1904.07734</a> — Systematically defines the three scenarios Task-IL / Domain-IL / Class-IL; shows that regularization methods almost completely fail on Class-IL. <a href="#fnref-9">↩</a></li>
</ol>
