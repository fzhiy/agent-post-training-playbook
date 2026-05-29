# Continual & Lifelong Learning

> How can a model learn new knowledge while retaining old knowledge when trained on **sequential tasks**? This is a fundamental question beyond the "train once, deploy" paradigm, and a hidden risk in the LLM post-training chain (pretrain → SFT → DPO → RL).

> ⚠️ **Study notes, not the author's research** (see README integrity statement). Numbers and conclusions follow the original papers; uncertainties are annotated.

## 0. The evolution

`IID one-shot training` → `sequential multi-task (task 1 → task 2 → …)` → **`catastrophic forgetting emerges`**: gradients directly overwrite old weights.

**Stability-plasticity dilemma**: the network must be **plastic** — accepting gradient updates for new tasks — and **stable** — preserving representations for old tasks. The two are inherently in conflict.

## 1. Why catastrophic forgetting happens

A neural network's parameters are **shared storage** for all tasks. When performing SGD on task $\mathcal{T}_2$, the loss gradient with respect to parameters has no knowledge that "these weights are important for $\mathcal{T}_1$", so it overwrites them — this is **catastrophic forgetting**.

Classic settings:

| Setting | Task ID known? | Task boundaries clear? |
|---|---|---|
| Task-IL | Known at test time | Yes |
| Domain-IL | Unknown at test time | Yes |
| Class-IL (hardest) | Unknown at test time | Yes |
| Continual pretraining | No explicit boundaries | No |

## 2. Three method families

### 2.1 Regularization

**Core idea**: add a **penalty term protecting old weights** to the new task loss, keeping "weights important to old tasks" as unchanged as possible.

**EWC (Elastic Weight Consolidation)**<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">Measures weight importance via the Fisher information matrix diagonal; uses a quadratic penalty to prevent forgetting. <a href="https://arxiv.org/abs/1612.00796">Kirkpatrick 2017 ↗</a></span></span> total loss:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\mathcal{T}_2}(\theta) + \frac{\lambda}{2} \sum_i F_i \,(\theta_i - \theta_i^*)^2$$

- $\theta_i^*$: parameter value after completing old tasks
- $F_i$: diagonal element of the Fisher information matrix (measures "importance" of parameter $i$ for old tasks)
- $\lambda$: hyperparameter controlling the stability vs. plasticity trade-off

$$F_i = \mathbb{E}_{\mathcal{D}_1}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**SI (Synaptic Intelligence)**: online version of EWC that **accumulates** each parameter's contribution to the loss during training, eliminating the need to recompute Fisher.

**MAS (Memory Aware Synapses)**: importance is estimated by the gradient norm of the output function with respect to parameters, independent of labeled data.

| Method | Importance estimation | Requires old data? | Computational cost |
|---|---|---|---|
| EWC | Fisher diagonal | No | Medium (one backward pass) |
| SI | Online trajectory integration | No | Low (incidental during training) |
| MAS | Output gradient norm | No (only unlabeled inputs needed) | Low to medium |

### 2.2 Rehearsal & Replay

**Core idea**: **mix in old task samples** during new task training so that gradients "remember" the past simultaneously.

**Experience Replay**: maintain an **episodic memory** buffer storing real samples from old tasks; during new task training, randomly mix them in proportionally.

**GEM (Gradient Episodic Memory)**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Projects gradient updates into a feasible region that does not increase old task loss, while allowing forward transfer. <a href="https://arxiv.org/abs/1706.08840">Lopez-Paz 2017 ↗</a></span></span>:

For each old task $k$, require the updated gradient to satisfy:
$$\langle g, g_k \rangle \geq 0$$
i.e., the new gradient must not point "opposite" to the old task gradient. If violated, project $g$ onto the feasible region.

**A-GEM (Averaged GEM)**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Merges multiple old-task constraints into a single averaged gradient constraint, greatly reducing computation with performance comparable to GEM. <a href="https://arxiv.org/abs/1812.00420">Chaudhry 2019 ↗</a></span></span>: use the **average gradient** $g_{\text{ref}}$ across all old task buffers as the sole constraint, reducing per-step QP projections from $K$ to 1:

$$\langle g, g_{\text{ref}} \rangle \geq 0, \quad g_{\text{ref}} = \frac{1}{K}\sum_k g_k$$

**Generative Replay**: use a **generative model** (e.g., VAE, GAN) to learn the old task distribution and synthesize "pseudo-old data" to inject into training — no need to store real old samples, but generation quality bottlenecks cause accumulating errors.

**DER (Dark Experience Replay)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Stores old-sample logits (soft targets) in the buffer and matches them via MSE, fusing rehearsal with knowledge distillation. <a href="https://arxiv.org/abs/2004.07211">Buzzega 2020 ↗</a></span></span>: the buffer stores $(x, y, z)$, where $z$ is the model's **logit** (dark knowledge) from a past time step; during replay, match $z$ as well:

$$\mathcal{L}_{\text{DER}} = \mathcal{L}_{\text{CE}}(x,y) + \alpha \cdot \text{MSE}(f_\theta(x),\, z)$$

### 2.3 Parameter Isolation & Architectural CL

**Core idea**: different tasks occupy **different sub-networks**; new tasks extend without altering old task parameters — eliminating forgetting structurally.

**Progressive Neural Networks**<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">Adds a new column of network for each new task, utilizes old-column knowledge via lateral connections, and freezes old columns entirely — forgetting is structurally impossible. <a href="https://arxiv.org/abs/1606.04671">Rusu 2016 ↗</a></span></span>: each task gets an independent column of network; old columns are frozen, and new columns read old-column activations through **lateral connections**:

$$h_k^{(\ell)} = f\!\left(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)}\right)$$

- Pros: zero forgetting, natural forward transfer
- Cons: parameters grow linearly with the number of tasks

**PackNet**: within a single network, perform **iterative pruning + fixing**, allocating a set of parameter masks per task with no shared gradient paths between tasks.

**LoRA-based / Adapter-based CL**: incrementally add a set of LoRA weights or adapter modules per task; the backbone is frozen, and tasks are routed to their corresponding adapters via task ID — controllable parameter count and LLM-friendly.

| Method family | Zero forgetting | Forward transfer | Parameter growth | Stores old data |
|---|---|---|---|---|
| Regularization (EWC/SI/MAS) | Approximate | Limited | None | No |
| Replay (ER/GEM/A-GEM/DER) | Approximate | Yes | Buffer | Yes (partial) |
| Parameter isolation (ProgNN/PackNet/LoRA-CL) | Yes | Limited to yes | Linear to lightweight | No |

## 3. Knowledge Distillation route: LwF

**LwF (Learning without Forgetting)**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">During new-task training, uses the old model's soft output as a distillation target, alleviating forgetting without storing any old data. <a href="https://arxiv.org/abs/1606.09282">Li 2016 ↗</a></span></span>:

- When performing inference on new task $\mathcal{T}_2$ data, first use the **old model** $f_{\theta^*}$ to produce soft output $\hat{y}^{\text{old}}$
- Joint optimization:

$$\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{new}}(f_\theta(x), y) + \lambda \cdot \mathcal{L}_{\text{KD}}(f_\theta(x), \hat{y}^{\text{old}})$$

- $\mathcal{L}_{\text{KD}}$ is typically temperature-scaled KL divergence or cross-entropy
- **No old data needed**; drawback: soft target quality degrades under large task drift

LwF is essentially **implicit replay of old model knowledge**, not old data — attractive in privacy-sensitive or storage-constrained scenarios.

## 4. Evaluation Metrics

Let $T$ denote the total number of tasks completed, and $a_{i,j}$ denote the accuracy on task $j$ after completing task $i$.

**Average Accuracy (AA)**: average accuracy across all tasks after all tasks have been learned:

$$\text{AA} = \frac{1}{T} \sum_{j=1}^{T} a_{T,j}$$

**Backward Transfer (BWT)** — more negative values indicate more severe forgetting:

$$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} \bigl(a_{T,j} - a_{j,j}\bigr)$$

**Forward Transfer (FWT)** — positive values indicate old tasks helped with new tasks:

$$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^{T} \bigl(a_{j-1,j} - b_j\bigr)$$

where $b_j$ is the baseline accuracy of training task $j$ independently from random initialization.

| Metric | What it measures | Ideal value |
|---|---|---|
| AA | Overall memory retention | Higher is better |
| BWT | Degree of forgetting | ≥ 0 (closer to 0 or > 0 is better) |
| FWT | Forward transfer | > 0 |

> **Forgetting** is sometimes directly defined as the mean accuracy drop from "just learned" to "final" for each task, which is the negation of BWT.

## 5. The LLM Angle

### 5.1 Continual Pretraining

Continue training a language model on new domain corpora after initial pretraining (e.g., medical text, code updates). Core challenges:

- Old general capabilities (math reasoning, instruction following) may degrade
- Distribution shift of new corpora may be milder than task-level shifts but more sustained
- Common mitigations: learning rate warm-up restart, replay of small amounts of general data, appropriate learning rate reduction

### 5.2 Continual Instruction-Tuning

Add new instruction types sequentially (e.g., first coding → then math → then safety); each round of SFT may overwrite previously fine-tuned behaviors. LoRA-based or adapter-per-task approaches are natural low-parameter solutions: the backbone is shared, and behaviors are separated through module routing.

### 5.3 Continual Alignment & Alignment Tax

**Sequential alignment chain**: each step in `pretrain → SFT → DPO → RL (RLHF/RLVR)` continues training from the previous step's checkpoint.

**Alignment tax**: alignment often comes at the cost of some general capabilities (e.g., code generation, factuality). When stacked sequentially, taxes accumulate:

- Excessive SFT may compress knowledge diversity
- RLHF after DPO may lead to over-refusal or format degradation
- Each hop faces the CL problem of "new alignment target vs. behaviors already learned in the previous hop"

**Mitigation strategies**:

1. **KL constraints** (PPO clip / DPO reference model) — limit the magnitude of deviation from the reference at each step, essentially an implicit analogue of EWC
2. **Replay old preference data** — mix in data from prior alignment stages
3. **LoRA independent adapter per stage** — backbone stays unchanged; alignment behaviors are localized

### 5.4 Why CL matters for LLM post-training

| Scenario | CL challenge |
|---|---|
| Incremental model update for new versions | Cannot retrain from scratch; need incremental fine-tuning without forgetting old capabilities |
| Multi-round RLHF iteration | Each round's policy update may overwrite prior alignment |
| Personalization / continual user feedback | Adapt to new user preferences while preserving general capabilities |
| Knowledge update (time-sensitive information) | Inject new facts without disrupting old knowledge structure |

## 6. From-scratch: EWC quadratic penalty

```python
import torch
import torch.nn.functional as F

def compute_fisher_diagonal(model, dataloader, device="cpu"):
    """
    Estimate the Fisher information matrix diagonal using squared log-likelihood gradients.
    dataloader: old task data; model: model after completing old tasks (parameters fixed).
    Returns dict: param_name -> F_i (same shape as the parameter)
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        log_prob = F.log_softmax(logits, dim=-1)
        # Approximate sampling with predicted labels — a simplified implementation;
        # the original EWC paper actually uses true labels y to estimate Fisher
        # (empirical Fisher = E_{(x,y)~D}[∇log p(y|x)^2])
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
    Trains on new tasks with an added EWC quadratic penalty to prevent
    overwriting old task parameters.
    """
    def __init__(self, model, old_dataloader, ewc_lambda=5000.0, device="cpu"):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda

        # Save snapshot of old task parameters
        self.theta_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
        }
        # Compute Fisher diagonal
        self.fisher = compute_fisher_diagonal(model, old_dataloader, device)

    def ewc_penalty(self):
        """EWC penalty term: (λ/2) Σ_i F_i (θ_i - θ*_i)^2"""
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

> `ewc_lambda` is the key hyperparameter: too small and forgetting remains severe; too large and the new task cannot be learned. In practice, it is typically searched in the range $[100, 10^4]$, and multiple Fisher matrices are accumulated across tasks (online EWC merges them via exponential moving average).

---

## Stratified follow-ups

### L1 Basics

<details class="qa"><summary>1. What is catastrophic forgetting? Why are neural networks particularly susceptible?</summary>

Answer: A neural network's parameters are **shared storage** for all tasks. When performing SGD on a new task $\mathcal{T}_2$, the gradient has no knowledge of which weights are important for old task $\mathcal{T}_1$, directly overwriting them and causing a sharp drop in old task performance — this is catastrophic forgetting. The root cause is parameter sharing combined with independent optimization objectives: the new task gradient acts as "noise" with respect to the old task loss.

</details>

<details class="qa"><summary>2. What is the stability-plasticity dilemma? Give an intuitive analogy.</summary>

Answer: A network must simultaneously possess **plasticity** — accepting gradient updates for new tasks — and **stability** — preserving representations for old tasks; the two are inherently in conflict. Intuitive analogy: when learning a new language, the brain must remember the mother tongue (stability) while shaping new language circuits (plasticity); rote memorization of the new language may crowd out the neural pathways for the mother tongue.

</details>

<details class="qa"><summary>3. What is the core idea of EWC? What role does the Fisher information matrix play?</summary>

Answer: EWC (Elastic Weight Consolidation) adds a quadratic penalty term $\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$ to the new task loss, keeping "weights important to old tasks" as unchanged as possible. The Fisher information matrix diagonal $F_i = \mathbb{E}[(\partial \log p_\theta / \partial \theta_i)^2]$ measures the **importance** of parameter $i$ for old tasks — the larger $F_i$, the stronger the penalty, "elastically" protecting that parameter.

</details>

<details class="qa"><summary>4. Both LwF and EWC store no old data. How do their anti-forgetting mechanisms differ?</summary>

Answer: EWC applies constraints in **parameter space** — using a Fisher penalty to keep important old-task weights from deviating from their old values; LwF (Learning without Forgetting) applies constraints in **output space** — using the old model's soft output on new task data as a distillation target $\mathcal{L}_{\text{KD}}$, keeping the new model's output behavior consistent with the old model. LwF requires no recomputation of importance scores, but the old model's soft target quality degrades with task drift.

</details>

### L2 Intermediate

<details class="qa"><summary>5. What is the core difference between GEM and A-GEM in replay methods? Why is A-GEM more efficient?</summary>

Answer: GEM imposes a separate gradient constraint $\langle g, g_k\rangle \geq 0$ for each old task $k$, requiring the solution of a QP with $(K-1)$ inequalities at time complexity $\mathcal{O}(K^3)$. A-GEM merges all old task constraints into a single mean constraint $\langle g, g_{\text{ref}}\rangle \geq 0$, which has a closed-form projection, reducing per-step computation from $\mathcal{O}(K \cdot d)$ to $\mathcal{O}(d)$, independent of the number of tasks.

</details>

<details class="qa"><summary>6. What do BWT and FWT each measure? What does it indicate if a model has very negative BWT but very positive FWT?</summary>

Answer: BWT (Backward Transfer) $= \frac{1}{T-1}\sum_{j=1}^{T-1}(a_{T,j}-a_{j,j})$ measures the degree of forgetting; more negative means more severe old-task performance degradation. FWT (Forward Transfer) $= \frac{1}{T-1}\sum_{j=2}^{T}(a_{j-1,j}-b_j)$ measures positive transfer from old tasks to new tasks; more positive means better transfer. Very negative BWT with very positive FWT indicates that the model leveraged old knowledge to accelerate learning on new tasks (good transfer) while severely overwriting old task parameters (severe forgetting) — a classic high-plasticity, low-stability model.

</details>

<details class="qa"><summary>7. How does Progressive Networks achieve "zero forgetting"? What is the cost?</summary>

Answer: Progressive Neural Networks add a new independent column of network for each new task; old columns are entirely frozen, and new columns read old-column activations via lateral connections $h_k^{(\ell)} = f(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)})$ — old-column gradient paths are severed, making forgetting structurally impossible. The cost is that the number of parameters grows linearly with the number of tasks.

</details>

<details class="qa"><summary>8. What advantages does LoRA-based CL have over EWC / replay? How is it used in the LLM post-training chain?</summary>

Answer: LoRA-based CL incrementally adds a set of low-rank adapters per task with the backbone frozen — controllable parameter count and zero gradient interference with old tasks, requiring no old data storage or Fisher computation. In the LLM post-training chain (SFT → DPO → RL), each stage freezes the backbone and updates only one set of LoRA, limiting alignment tax to the adapter layers; backbone general knowledge is not overwritten. At test time, routing is done by task ID to the corresponding adapter.

</details>

### L3 Deep-dive

<details class="qa"><summary>9. What are the main differences between LLM continual pretraining and the classic Task-IL setting? What unique challenges arise?</summary>

Answer: Classic Task-IL has explicit task boundaries and task IDs; LLM continual pretraining has no explicit boundaries, domain corpora flow continuously, and task granularity is fuzzy (general capabilities and new domains have substantial overlap). Unique challenges include: old general capabilities (math reasoning, instruction following) may degrade in hard-to-detect ways; new-corpora distribution shifts are long-lasting; forgetting cannot be quantified with a standard $a_{i,j}$ matrix; and learning rate warm-up restart and the ratio of replayed general data are difficult to tune.

</details>

<details class="qa"><summary>10. What CL problem does alignment tax in the sequential alignment chain (SFT → DPO → RL) fundamentally represent? What mitigations exist?</summary>

Answer: Alignment tax is fundamentally the cumulative forgetting tax of sequential CL — each alignment step continues training from the previous checkpoint, and the new alignment target's gradient overwrites previously learned behaviors. Mitigations: (1) **KL constraints** (PPO clip / DPO reference model) limit per-step deviation magnitude; its second-order expansion is equivalent to Fisher-weighted EWC; (2) **Replay old preference data** by mixing in samples from prior alignment stages; (3) **LoRA independent adapter per stage** localizes alignment behavior while the backbone remains unchanged.

</details>

<details class="qa"><summary>11. EWC uses empirical Fisher (with true labels) rather than true Fisher (with model-sampled labels) — in what situations does this fail?</summary>

Answer: Empirical Fisher $F_i^{\text{emp}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}[(\partial \log p_\theta(y|x)/\partial \theta_i)^2]$ uses true dataset labels $y$, and is only equivalent to true Fisher when the model perfectly fits the data. Failure scenarios: when the model is far from converged on old tasks, $F^{\text{emp}}$ has high estimation noise and protects the wrong directions; when old-task label noise is high, directions are corrupted by label noise; during multi-task sequential accumulation (online EWC), early-task errors persist through the moving average; and when cross-task distribution differences are extreme, the diagonal approximation and empirical error compound — protecting the wrong directions.

</details>

<details class="qa"><summary>12. What are the advantages and limitations of generative replay compared to experience replay? Is it more feasible in the era of large models?</summary>

Answer: Generative Replay uses a generative model (VAE/GAN) to synthesize "pseudo-old data" for training injection. **Advantages**: no need to store any real old samples, naturally solving privacy and storage constraints. **Limitations**: generation quality bottlenecks cause accumulating errors across the task sequence — the generative model itself suffers from forgetting, and the deviation between synthetic and real data compounds over long task chains. In the era of large models, the LLM's own generative capability is extremely strong, making the quality of LLM-synthesized old-task data far superior to that of small GANs — feasibility is significantly improved, though generation cost is high and synthetic data may still have systematic deviations from the original distribution.

</details>

---

## Deep-dive

> The following are interview-level advanced Q&As covering the 7 most commonly questioned topics from the notes above. All conclusions are drawn from cited papers and contain no author's own research.

### D1. Empirical Fisher vs. True Fisher — which does EWC use, and when does it break down?

**True Fisher** is defined as the expectation of the outer product of log-likelihood gradients with respect to the **model's predictive distribution** $p_\theta(y|x)$:

$$F_i^{\text{true}} = \mathbb{E}_{x \sim \mathcal{D},\; y \sim p_\theta(\cdot|x)}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**Empirical Fisher** substitutes **true labels** $y$ from the dataset for the $y$ sampled from the model's distribution:

$$F_i^{\text{emp}} = \mathbb{E}_{(x,y) \sim \mathcal{D}}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

The two are equal only when the model perfectly fits the data ($p_\theta(y|x) \approx \delta_y$). Existing research has shown that empirical Fisher cannot generally capture second-order information during optimization, and its deviation from the true Hessian in practice can be substantial — even on simple optimization problems pathological behavior can arise.

**EWC uses empirical Fisher**: in implementation, it computes the expectation of squared gradients over $(x,y)$ pairs from old-task dataset $\mathcal{D}_1$. The code comments also note "estimates Fisher using true labels $y$."

**When does the approximation break down?**

| Scenario | Risk |
|---|---|
| Computing $F$ while the model is far from converged on old tasks | $F^{\text{emp}}$ has high estimation noise; protects the wrong directions |
| High label noise in old tasks | $F^{\text{emp}}$ directions are corrupted by label noise |
| Multi-task sequential accumulation (online EWC) | Early-task $F^{\text{emp}}$ errors persist through the moving average |
| Extreme cross-task distribution differences (e.g., language → vision) | Diagonal approximation is already a strong assumption; compounded with empirical error |

Intuition: EWC's quadratic penalty is essentially a local quadratic approximation of the parameter space — diagonal empirical Fisher is the coarsest layer of this approximation. The approximation works tolerably when old tasks have converged well, labels are clean, and parameter correlations are weak; otherwise, the "importance scores" decouple from the true loss landscape curvature, and the penalty protects the wrong directions.

---

### D2. Online/Streaming EWC — how to accumulate Fisher across tasks?

Standard EWC stores one set of $(F^{(k)}, \theta^{*(k)})$ per old-new task pair, with memory growing linearly in the number of tasks $K$. **Online EWC** (Progress & Compress, Schwarz et al. 2018)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Progress & Compress dual-network framework: active column learns new tasks, knowledge base uses online EWC for compression/consolidation; Fisher is accumulated across tasks via exponential moving average. <a href="https://arxiv.org/abs/1805.06370">Schwarz 2018 ↗</a></span></span> uses **exponential moving average (EMA)** to merge all historical tasks' Fisher into a single $\tilde{F}$:

$$\tilde{F}^{(k)} = \gamma \cdot \tilde{F}^{(k-1)} + (1-\gamma) \cdot F^{(k)}$$

The total penalty degenerates to a single set of penalties:

$$\mathcal{L}_{\text{online-EWC}} = \mathcal{L}_{\mathcal{T}_k}(\theta) + \frac{\lambda}{2}\sum_i \tilde{F}_i^{(k-1)}\bigl(\theta_i - \theta_i^{*(k-1)}\bigr)^2$$

**Pros**: constant memory (only one $\tilde{F}$ and one $\theta^*$ snapshot stored).

**Costs and risks**:

- EMA weights for early tasks' Fisher decay exponentially — the older the task, the weaker the protection, inherently favoring "recent" tasks.
- $\gamma$ is a new hyperparameter: $\gamma \to 1$ preserves history but has high forgetting rate; $\gamma \to 0$ degenerates to looking only at the previous task.
- The reference point $\theta^*$ is updated each round; the new $\theta^*$ after each compression is not the optimal common point for all historical tasks — the penalty center drifts under multi-task accumulation.

**SI (Synaptic Intelligence)** is another streaming approach: during training, it **online-accumulates** each parameter's contribution to loss reduction as an importance integral:

$$\Omega_i \propto \int_{\text{trajectory}} \frac{\partial \mathcal{L}}{\partial \theta_i} \cdot \dot\theta_i \, dt$$

SI requires no additional backward passes after task completion, making it suitable for **streaming scenarios without explicit task boundaries**, but its importance estimation has a lower signal-to-noise ratio than EWC.

---

### D3. Stability Gap — why does forgetting get worse before it recovers?

De Lange et al. (arXiv 2022, ICLR 2023)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Proposes a per-iteration continuous evaluation framework, first systematically documenting the stability gap: performance drops sharply after task switching then recovers; evaluating only at the end of a task would miss this phenomenon. <a href="https://arxiv.org/abs/2205.13452">De Lange 2022 ↗</a></span></span> discovered through **per-iteration continuous evaluation**: nearly all mainstream CL methods (including EWC, ER, A-GEM) experience a sharp drop in old task performance during **the first few steps after switching to a new task**, followed by gradual recovery and even exceeding pre-switch levels. This "drop-then-recover" phenomenon is called the **stability gap**.

**Mechanism intuition:**

```
Task T1 training complete → switch to T2
     ↓ first few dozen steps
T2's large gradients hit shared feature layers → T1 representations temporarily invalidate → T1 performance plummets
     ↓ continued training
Regularization / replay constraints start taking effect → T1 representations gradually repair
     ↓ end of T2
T1 performance recovers (but may be below peak at T1 training completion)
```

**Why wasn't this discovered earlier?**

Standard evaluation protocols only test **once after each task is completed**, exactly skipping the sharp-drop period — the stability gap is nearly invisible in end-of-task "snapshots."

**Interview key points:**

- The stability gap means BWT metrics (end-of-task snapshots) **underestimate the true magnitude of forgetting** — in safety-critical scenarios (incremental updates of deployed LLMs), the worst-case performance during training may be far more severe than BWT reflects.
- Mitigation directions: reduce the new task learning rate during a warm-up transition period; use small batches to "probe" the new task before full-speed training; increase the ratio of old task replay at the beginning of task switching.

---

### D4. Replay sample selection strategies and buffer size effects

Buffer size $M$ is one of the most critical hyperparameters for replay methods. Three mainstream selection strategies:

**① Reservoir Sampling**

For data streams of unknown length, maintain a buffer of size $M$; each new sample $x_t$ is admitted to the buffer with probability $M/t$ (randomly replacing an old sample). Guarantees that each sample in the buffer is a uniform subset of all historical samples — no class bias, simple to implement, and the default strategy for ER / A-GEM.

**② Herding (iCaRL)**

iCaRL's herding algorithm: greedily iterates to select samples so that the feature mean of the exemplar set is closest to the feature mean of the entire class:

$$p_t = \arg\min_{x \in \mathcal{D}_k} \left\| \mu_k - \frac{1}{t}\!\left(\phi(x) + \sum_{j=1}^{t-1}\phi(p_j)\right)\right\|$$

Herding outperforms random sampling for class exemplar selection, especially when $M$ is very small (only a few samples per class) by better preserving intra-class diversity. However, herding requires data from all existing classes and depends on the feature space — after feature drift, old-class exemplars may no longer represent their current features.

**③ Gradient-based selection**

Select old samples that have the greatest impact on the new task's gradient update — typically those most "conflicting" with the new task's gradient direction. The intuition is to use the hardest-to-constrain samples to constrain the gradient. High computational cost (requires extra backward passes to estimate each candidate's gradient impact), rarely used in large-scale experiments.

**Buffer size effect summary:**

| Buffer size $M$ | Random/Reservoir | Herding | Gradient-based |
|---|---|---|---|
| Very small (1–5 samples/class) | Poor coverage, severe forgetting | Significantly better than random | Good effect but extremely high cost |
| Medium (20–50/class) | Close to herding | Gap narrows | Diminishing returns |
| Large (near unlimited) | All three converge, close to joint training | Same as left | Same as left |

Core insight: as $M \to \infty$, all replay methods degenerate to joint training (upper bound); when $M$ is small, exemplar representativeness matters more than randomness — herding wins in this regime.

---

### D5. GEM's per-task QP cost vs. A-GEM's single mean constraint

**GEM's QP problem:**

The current gradient for the new task is $g$, and for each old task $k$ there is a constraint $\langle g, g_k \rangle \geq 0$. If constraints are violated, the following QP must be solved:

$$\tilde{g} = \arg\min_v \|v - g\|^2 \quad \text{s.t.} \quad \langle v, g_k\rangle \geq 0,\; \forall k \in \{1,\ldots,K-1\}$$

This is a quadratic program (QP) with $(K-1)$ inequality constraints. Standard QP solvers have time complexity $\mathcal{O}(K^3)$ and space complexity $\mathcal{O}(K^2)$ — quickly infeasible as the number of tasks grows. GEM's implementation uses iterative methods like Frank-Wolfe for approximate solutions, but each step still requires $K$ gradient inner product computations, i.e., $\mathcal{O}(K \cdot d)$ ($d$ being the parameter dimension).

**A-GEM's single constraint:**

A-GEM merges all old-task constraints into a single averaged gradient:

$$g_{\text{ref}} = \frac{1}{K-1}\sum_{k=1}^{K-1} g_k, \quad \text{s.t.}\; \langle g, g_{\text{ref}} \rangle \geq 0$$

If violated, the projection formula has a closed-form solution:

$$\tilde{g} = g - \frac{\langle g, g_{\text{ref}} \rangle}{\|g_{\text{ref}}\|^2} g_{\text{ref}}$$

**Per-step computation drops from $\mathcal{O}(K \cdot d)$ to $\mathcal{O}(d)$** — independent of the number of tasks (computing $g_{\text{ref}}$ can be done once and reused).

**Cost:**

A-GEM satisfies an averaged directional constraint and does not guarantee $\langle \tilde{g}, g_k\rangle \geq 0$ for each individual old task $k$ — some individual old tasks may see increased loss. Empirically, A-GEM's AA and BWT are comparable to GEM, but when gradients across tasks are highly heterogeneous (some tasks' directions deviate significantly from the mean), occasionally a specific old task's forgetting can be more severe than with GEM.

**Interview memory aid:** GEM = per-task constraints + quadratic programming, $\mathcal{O}(K^3)$ QP; A-GEM = single mean constraint + closed-form projection, $\mathcal{O}(d)$; what's sacrificed is per-task constraint guarantees, in exchange for linear time complexity.

---

### D6. Why is Class-IL the hardest? The role of the output head

The three-scenario framework of van de Ven & Tolias 2019<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">Systematically compares three CL settings (Task-IL / Domain-IL / Class-IL) for difficulty differences on Split MNIST and Split CIFAR-100; regularization methods nearly completely fail on Class-IL. <a href="https://arxiv.org/abs/1904.07734">van de Ven 2019 ↗</a></span></span> reveals fundamental difficulty differences:

**Output head structure comparison across three scenarios:**

| Scenario | Task ID at test time | Output head | What the model must do |
|---|---|---|---|
| Task-IL | Known | Independent head per task (only current task activated) | Classify within task, known candidate set |
| Domain-IL | Unknown | Shared head (fixed output dimension) | Classify within fixed output space, no task distinction needed |
| Class-IL | Unknown | Shared head (all task classes) | Classify among all classes from all historical tasks |

**Why Class-IL is the hardest — three barriers:**

1. **Task-ID inference problem**: the model does not know which task the current input belongs to, so it cannot route to the corresponding sub-classifier — it must distinguish all classes on a single head.

2. **Output head recency bias**: when learning a new task, only the new task's classes produce large gradient updates, tilting the output layer's logit scale toward the new task — old class logits are suppressed, even if the feature layer still remembers old tasks. This is Class-IL's unique "classifier-layer forgetting."

3. **Fundamental failure of regularization methods**: EWC protects feature-layer parameters, but the initialization of new class nodes in the output head interferes with old class node gradients — the Fisher diagonal cannot capture this cross-class output-layer interference. Van de Ven et al.'s experiments show that EWC accuracy on Class-IL approaches chance level.

**Remediation strategies:**

- **Replay + prototype classifier** (e.g., iCaRL): use exemplar means for nearest-neighbor classification, bypassing output head bias.
- **Task-agnostic feature learning**: pretrain and freeze backbone, only update lightweight classification heads — reducing feature drift.
- **Empirical bias correction**: at Class-IL test time, apply temperature scaling or reweighting to old-class logits to counteract recency bias.

---

### D7. LLM forgetting vs. capability interference — measurement methods and task-order sensitivity

**Key differences between LLM "forgetting" and classic CL:**

| Dimension | Classic CL (small models/classification tasks) | LLM post-training |
|---|---|---|
| Task granularity | Clear (Task 1/2/3…) | Fuzzy ("math reasoning," "code," "safety alignment" substantially overlap) |
| Manifestation of forgetting | Old-task classification accuracy drops | Capability **interference**: new capability activation paths overwrite old capability activation paths, not simple "forgetting" |
| Measurement difficulty | Directly quantifiable via $a_{T,j}$ matrix | Requires specialized benchmarks (MMLU/GSM8K/HumanEval…) to track each capability |
| Boundary clarity | Explicit task boundaries | No explicit boundaries; SFT→DPO→RL are **soft boundaries** |

**How to measure LLM CL quality:**

1. **Capability matrix tracking**: after each alignment stage, evaluate on several "probe benchmarks" (covering general reasoning, code, math, safety) — equivalent to constructing an $a_{i,j}$ matrix at LLM scale.
2. **LLM analogue of BWT**: measure "change in code capability after SFT compared to pretrain baseline" — if negative, this is part of the alignment tax.
3. **Activation/gradient analysis**: detect which layers' activation distributions change most before and after new task training — localize "overwritten" knowledge storage layers.

**Task-order sensitivity:**

LLM CL is highly sensitive to task order, for the following reasons:

- Sequential training's gradient direction depends on the loss landscape formed by prior tasks — doing math SFT then safety RLHF produces vastly different results than the reverse order.
- Harder tasks (math/code) require large learning rates and large gradients for optimization, causing more parameter damage to subsequent smaller tasks.
- **Cumulative nature of alignment tax**: in the SFT → DPO → RL sequence, each step's forgetting tax accumulates on the next step's checkpoint.

**KL constraints are implicit EWC:**

The KL penalty term in PPO-RLHF:

$$r_{\text{KL}}(\theta) = \mathbb{E}\!\left[\log\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}\right] \cdot (-\beta)$$

Taking a Taylor expansion around the reference policy $\pi_{\text{ref}}$, the second-order term of the KL divergence is proportional to the Fisher information matrix:

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{1}{2}(\theta - \theta_{\text{ref}})^T F(\theta_{\text{ref}}) (\theta - \theta_{\text{ref}})$$

That is, **PPO's KL penalty ≈ the full-matrix version of EWC with Fisher as the weight matrix** — both are "quadratic penalties on parameter deviation from a reference point, weighted by information-geometric curvature." The difference is: EWC uses a diagonal approximation and explicitly stores old-task Fisher; KL constraints use full distribution distance and dynamically anchor to the current reference policy. The KL term in DPO works the same way — the reference model is mathematically equivalent to $\theta^*$ in EWC.

---

### D8. Adapter interference and routing in LoRA-based CL

**Sources of adapter interference:**

The naïve approach trains one independent set of LoRA weights $\Delta W_k = B_k A_k$ per task and routes via task-ID at test time. There are two sources of interference:

1. **Parameter space overlap**: different tasks' low-rank subspaces may substantially overlap — if $\text{span}(A_k) \cap \text{span}(A_j) \neq \emptyset$, merging one task's adapter will interfere with another task's activations.
2. **Routing failure without task-ID**: in Class-IL or continual pretraining settings, there is no task-ID at test time, making it impossible to route to the correct adapter.

**O-LoRA's orthogonal subspace approach:**

O-LoRA, during task $k$'s LoRA training, forces the new task's low-rank subspace to be orthogonal to all historical tasks' subspaces:

$$A_k V_{\text{prev}} \approx 0, \quad V_{\text{prev}} = \text{span}\bigl(\{A_j\}_{j<k}\bigr)$$

This is achieved by projecting gradients onto the orthogonal complement of $V_{\text{prev}}$. Orthogonal subspaces ensure that different tasks' adapter activations do not interfere — approximately zero forgetting in settings with known task-IDs, and no need to store old task data.

**Limitations:**

- Available orthogonal dimensions decrease as the number of tasks increases — in LoRA with rank $r$, at most approximately $d/r$ fully orthogonal tasks are supported ($d$ being the weight matrix dimension).
- In Class-IL settings without task-ID, orthogonality does not solve the routing problem — an additional mechanism is still needed to infer task-ID or use prototype classification.
- Forcing orthogonality may limit **forward transfer** between tasks — related tasks could otherwise share subspaces to accelerate learning.

**Practice in LLM post-training:**

In the sequential alignment chain (SFT → DPO → RL), each stage freezes the backbone and updates only one set of LoRA, equivalent to a parameter isolation scheme with known task-IDs — alignment tax accumulation is substantially confined to the LoRA layers, and backbone knowledge is not overwritten. The cost is the need to manage multiple adapter sets and their merging/routing logic at deployment time.

---

## References

> All are original sources for classic foundational methods, verified one by one (title + arXiv ID). Click superscripts to navigate; click ↩ to return.

<ol>
<li id="ref-1">Kirkpatrick et al. <em>Overcoming catastrophic forgetting in neural networks</em>. PNAS 2017. <a href="https://arxiv.org/abs/1612.00796">arXiv:1612.00796</a> — EWC: Fisher diagonal quadratic penalty against forgetting. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Lopez-Paz et al. <em>Gradient Episodic Memory for Continual Learning</em>. NeurIPS 2017. <a href="https://arxiv.org/abs/1706.08840">arXiv:1706.08840</a> — GEM: gradient projection constraints + forward/backward transfer. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Chaudhry et al. <em>Efficient Lifelong Learning with A-GEM</em>. ICLR 2019. <a href="https://arxiv.org/abs/1812.00420">arXiv:1812.00420</a> — A-GEM: single averaged gradient constraint, efficient GEM. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Buzzega et al. <em>Dark Experience for General Continual Learning: a Strong, Simple Baseline</em>. NeurIPS 2020. <a href="https://arxiv.org/abs/2004.07211">arXiv:2004.07211</a> — DER: store logits for soft-target distillation + rehearsal. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Rusu et al. <em>Progressive Neural Networks</em>. 2016. <a href="https://arxiv.org/abs/1606.04671">arXiv:1606.04671</a> — Add new column per task + lateral connections; structural zero forgetting. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Li and Hoiem. <em>Learning without Forgetting</em>. ECCV 2016. <a href="https://arxiv.org/abs/1606.09282">arXiv:1606.09282</a> — LwF: old model soft output as distillation target, no old data needed. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Schwarz et al. <em>Progress &amp; Compress: A scalable framework for continual learning</em>. ICML 2018. <a href="https://arxiv.org/abs/1805.06370">arXiv:1805.06370</a> — Dual-network active column + knowledge base; online EWC uses Fisher EMA for cross-task accumulation with constant memory. <a href="#fnref-7">↩</a></li>
<li id="ref-8">De Lange, van de Ven, and Tuytelaars. <em>Continual evaluation for lifelong learning: Identifying the stability gap</em>. ICLR 2023. <a href="https://arxiv.org/abs/2205.13452">arXiv:2205.13452</a> — Per-iteration continuous evaluation framework; discovers the stability gap phenomenon of sharp performance drop followed by recovery after task switching. <a href="#fnref-8">↩</a></li>
<li id="ref-9">van de Ven and Tolias. <em>Three scenarios for continual learning</em>. 2019. <a href="https://arxiv.org/abs/1904.07734">arXiv:1904.07734</a> — Systematically defines Task-IL / Domain-IL / Class-IL three scenarios; reveals that regularization methods nearly completely fail on Class-IL. <a href="#fnref-9">↩</a></li>
</ol>