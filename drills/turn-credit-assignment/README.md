# Drill: Turn credit assignment from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.

## 背景 / Background

训练 LLM agent 时，轨迹由多轮交替 token 构成：

```
[系统提示] [观察_1] [动作_1] [工具输出_1] [动作_2] [工具输出_2] …
```

Credit assignment 要解决三个问题：

1. **折扣回报** — 把稀疏的最终奖励分摊回每个时间步。
2. **基线消除** — 降低梯度方差，同时不引入额外偏差。
3. **掩码损失** — 梯度只流向 agent 动作 token；工具输出/观察 token 不参与训练。

Training an LLM agent requires solving three sub-problems:

1. **Discounted return** — propagate a sparse terminal reward back to every step.
2. **Baseline subtraction** — reduce gradient variance without adding bias.
3. **Masked loss** — gradient flows only through agent action tokens; tool-output and observation tokens are masked out.

---

## 数学 / The math

### 1. 折扣回报 Discounted return

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k\, r_{t+k}$$

逆向递推:  $G_{T-1} = r_{T-1}$,  $G_t = r_t + \gamma\, G_{t+1}$

- $\gamma \in [0,1]$：折扣因子。$\gamma=1$ 等权所有未来奖励；$\gamma \to 0$ 只看即时奖励。
- 逆向一次遍历即可，时间复杂度 $O(T)$，无需存储 $T\times T$ 矩阵。

### 2. GRPO 组相对优势 Group-relative advantage

对同一 prompt 采样 $G$ 条轨迹（一组），组内归一化代替价值网络：

$$A_i = \frac{G_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \varepsilon}$$

- $\mu_{\text{group}}$：蒙特卡洛估计的 $V(s)$，消除基线偏差。
- 每组优势**零均值**（数学上恒成立），方差约为 1。
- 不需要单独的 critic 网络 —— 节省约一倍的显存开销。

来源：DeepSeekMath, Shao et al., 2024 (arXiv:2402.03300).

### 3. 掩码策略梯度损失 Masked policy-gradient loss

$$\mathcal{L} = -\frac{\sum_{(b,t):\, m_{b,t}=1} A_{b,t}\,\log \pi_\theta(a_{b,t})}{\sum_{b,t} m_{b,t}}$$

- $m_{b,t} \in \{0,1\}$：动作掩码，1 = agent 动作 token。
- 分母是活跃 token 数，使不同长度序列贡献均等。
- 对环境 token 置零而非截断，保持形状不变，方便批量计算。

---

## 文件 / Files

| 文件 | 内容 |
|---|---|
| `from_scratch.py` | `discounted_return` + `group_relative_advantages` + `masked_pg_loss` |
| `test_turn_credit.py` | 17 个断言测试：零均值属性、掩码正确性、数值精度、端到端流水线 |

```bash
python test_turn_credit.py        # 或 python -m pytest test_turn_credit.py
```

---

## 追问分层 / Stratified follow-ups

### L1 — 概念

- 为什么要折扣 ($\gamma < 1$)？$\gamma = 0$ 和 $\gamma = 1$ 分别退化成什么行为？
- GRPO 用组均值做基线，这个估计是有偏的还是无偏的？为什么不用全局均值？
- 为什么掩码位置要填 0 而不是直接从序列里删除？

### L2 — 实现

- 折扣回报的逆向递推为什么比正向更简单？用矩阵乘如何实现，复杂度是多少？
- 组内只有 1 条轨迹时 $\sigma = 0$，如何处理？代码里的 `eps` 够不够？
- 如果 `action_mask` 全为 False（所有 token 都被遮掉），损失应该是什么？代码怎么处理？

### L3 — 算法

- GRPO 与 PPO 的核心区别是什么？去掉 critic 对 variance-bias 折中有什么影响？
- 多轮 agent 轨迹中每个动作 token 是否应该有不同的折扣步长？当前实现用的是哪种假设？
- 在 agentic 设置里，哪些 token 算"动作"（应训练）、哪些算"观察"（应掩码）？边界模糊时如何决策？
- REINFORCE 梯度的高方差问题除了组内基线之外还有哪些常见缓解方法（GAE、PPO clip、reward whitening）？

---

## 参考 / References

- Shao et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. arXiv:2402.03300. — GRPO 原始来源 / origin of GRPO.
- Williams (1992). *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. — REINFORCE 基础 / REINFORCE foundation.
- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. — PPO，GRPO 的直接前身 / PPO, direct predecessor of GRPO.
- Schulman et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438. — GAE，另一种低方差优势估计 / GAE, another low-variance advantage estimator.
