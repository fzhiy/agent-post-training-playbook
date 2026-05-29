# Drill: EWC + Experience Replay from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.

这是**学习笔记**,不是作者的研究工作。方法均来自引文中列出的经典论文。
These are study notes, not the author's research. All methods come from the cited papers.

---

## 背景 / Background

持续学习 (continual learning) 的核心难题是**灾难性遗忘** (catastrophic forgetting):
神经网络在学完 Task 2 后,为 Task 1 优化的权重被大幅覆写,Task 1 性能骤降
(McCloskey & Cohen 1989; Ratcliff 1990)。

The core challenge in continual learning is **catastrophic forgetting**: after
training on Task 2, the weights tuned for Task 1 are largely overwritten and
Task 1 performance collapses (McCloskey & Cohen 1989; Ratcliff 1990).

---

## 方法 A: EWC (Elastic Weight Consolidation) / Method A

**论文 / Paper**: Kirkpatrick et al. 2017, arXiv:1612.00796
"Overcoming catastrophic forgetting in neural networks"

### 核心思想 / Core idea

贝叶斯视角:Task 2 的后验 ∝ Task-2 似然 × Task-1 后验 (作为先验)。
用对角 Laplace 近似 Task-1 后验,得到 EWC 损失:

$$\mathcal{L}_{EWC} = \mathcal{L}_{T2}(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta^*_i)^2$$

Bayesian view: Task-2 posterior ∝ Task-2 likelihood × Task-1 posterior (as prior).
Approximating the Task-1 posterior with a diagonal Laplace gives EWC loss above.

各项含义 / Term meanings:
- $\theta^*_i$ — Task 1 结束时记录的权重锚点 (anchor)
- $F_i$ — 对角 Fisher 信息估计,度量 $\theta_i$ 对 Task 1 的重要性
- $\lambda$ — 正则化强度超参数

### Fisher 对角估计 / Diagonal Fisher estimate

$$F_i = \mathbb{E}\!\left[\left(\frac{\partial \log p(y \mid x, \theta)}{\partial \theta_i}\right)^{\!2}\right] \approx \frac{1}{N}\sum_{n=1}^N \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2$$

即**梯度平方的均值**。对回归任务等价于对 MSE 梯度求平方后平均。
This is the **mean squared gradient**: weights whose gradients were large on Task 1
data are the important ones and will be penalised most for drifting.

---

## 方法 B: Experience Replay / Method B

**经典参考 / Classic reference**: Robins 1995 — "Catastrophic Forgetting, Rehearsal and
Pseudorehearsal" (Connection Science); Lopez-Paz & Ranzato 2017,
arXiv:1706.08840 (GEM, a gradient-constrained replay variant).

### 核心思想 / Core idea

训练 Task 2 时,将一小批 Task-1 样本 (来自 ring buffer) 与 Task-2 样本混合:

$$\mathcal{L}_{replay} = (1-\alpha)\,\mathcal{L}_{T2} + \alpha\,\mathcal{L}_{buffer}$$

When training on Task 2, interleave a mini-batch of Task-1 samples from a
ring buffer with the current task batch.

- $\alpha$ = replay ratio (0.5 → equal weight to both tasks)
- Buffer stores Task-1 $(x, y)$ pairs; FIFO eviction when capacity is full
- No auxiliary parameters — memory cost is O(buffer size)

---

## 对比 / Comparison

| 方法 | 额外参数 | 原始数据需求 | 复杂度 |
|------|---------|------------|--------|
| 朴素微调 Naive | 0 | 不需要 | 基线 |
| EWC | 2× weights (F + θ*) | 少量 Task-1 (估 Fisher) | 低 |
| Replay | buffer size | 完整 Task-1 subset | 低 |

EWC 适合无法重用原始数据的场景 (如隐私限制);
Replay 更直接,但需要存储原始样本。

EWC is preferred when original data cannot be retained (e.g., privacy constraints);
Replay is simpler and often more effective when data storage is acceptable.

---

## 文件 / Files

- `from_scratch.py` — `SmallMLP`, `estimate_fisher`, `ewc_penalty`, `ReplayBuffer`,
  `train_naive` / `train_ewc` / `train_replay`
- `test_ewc_replay.py` — 6 assertion-based tests; fully deterministic (fixed seeds)

```bash
python test_ewc_replay.py        # 或 python -m pytest test_ewc_replay.py
```

测试断言 / Test assertions:
1. 朴素微调确实遗忘 (Task-1 loss > 0.3 after T2 training)
2. EWC 遗忘量显著低于朴素基线 (reduction ≥ 0.5 MSE units)
3. Replay 遗忘量显著低于朴素基线 (reduction ≥ 0.5 MSE units)
4. EWC 不阻止 Task-2 学习 (T2 loss drops ≥ 50% from random init)
5. Fisher 对角值全部 ≥ 0 (必然性质:梯度平方)
6. ReplayBuffer 容量上限和采样形状正确

---

## 追问分层 / Stratified follow-ups

**L1 (概念)**
- 为什么 Fisher 对角是梯度平方的期望?和权重的 Hessian 什么关系?
- EWC 与 L2 正则化的本质区别是什么?(提示:L2 锚定到 0,EWC 锚定到 θ*)
- Replay 为什么 FIFO 而不是按重要性保留?(引出 reservoir sampling / MIR)

**L2 (实现细节)**
- 多任务顺序下怎么扩展 EWC?(每个任务累积一组 F + θ*;online EWC)
- Fisher 估计的采样数 N 怎么影响估计质量?bias-variance tradeoff?
- Replay buffer 满了应该怎么选 eviction policy?iCaRL 用的是什么策略?

**L3 (研究层)**
- GEM (arXiv:1706.08840) 如何用 replay buffer 加梯度约束取代损失项?优势是什么?
- EWC 的对角 Fisher 近似忽略了什么?(参数间协方差 → Kronecker-factored 近似 → K-FAC/KFAC-EWC)
- 持续学习的三类范式 (regularisation / replay / architecture expansion) 各自的根本瓶颈是什么?

---

## 参考文献 / References

- Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks."
  *PNAS*. arXiv:1612.00796
- Lopez-Paz & Ranzato (2017). "Gradient Episodic Memory for Continual Learning."
  *NeurIPS*. arXiv:1706.08840
- Robins (1995). "Catastrophic Forgetting, Rehearsal and Pseudorehearsal."
  *Connection Science*, 7(2).
- McCloskey & Cohen (1989). "Catastrophic Interference in Connectionist Networks."
  *Psychology of Learning and Motivation*, 24.
- Ratcliff (1990). "Connectionist Models of Recognition Memory."
  *Psychological Review*, 97(2).
