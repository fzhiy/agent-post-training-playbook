# Drill: Self-Refine Loop from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩证。
> Runnable from-scratch implementation with tests — derive and defend every line.

## 核心思想 / Core idea

Self-Refine (Madaan et al., 2023, arXiv:[2303.17651](https://arxiv.org/abs/2303.17651)) 的直觉:
用同一个模型既生成候选 *又* 对候选打分与反思 —— 把输出当草稿、反复改写,每轮保留最优。

The intuition from Self-Refine: use the same model to both produce a candidate
and critique it, then feed that critique back as the context for the next
generation, keeping the best result seen so far.

```
候选生成                 自评分                  反思/编辑
generate(state)  -->  score(candidate)  -->  reflect(candidate, score)
     ^                                              |
     |________ candidate for next round ____________|
                     (keep best)
```

本钻题用**玩具连续优化**代替 LLM:生成器输出一个实向量,评分器是已知二次型,
反思器是对评分做一步梯度上升。环路结构与论文完全一致,但无需网络或数据集。

This drill replaces the LLM with a **toy continuous objective**: a generator
outputs a real vector, the scorer is a known quadratic, and the reflector takes
one gradient-ascent step on the score.  The loop structure is identical to the
paper; no network calls or datasets needed.

## 数学 / The math

设目标向量为 $t$，候选为 $x$，评分为：

$$s(x) = -\|x - t\|^2 \in (-\infty,\, 0]$$

反思步（梯度上升，步长 $\alpha$）：

$$x' = x + \alpha\,\nabla_x\, s(x) = x + 2\alpha\,(t - x)$$

对凸二次型，每步严格减小 $\|x-t\|^2$，只要 $0 < \alpha < 1$。

**Best-keeping invariant:**
$$\text{best\_score}_k = \max_{i \le k}\, s(x_i)$$
该量单调不减，即循环的核心不变量 (loop invariant)。

## 与 Reflexion 的关系 / Relation to Reflexion

Reflexion (Shinn et al., 2023, arXiv:[2303.11366](https://arxiv.org/abs/2303.11366)) 把反思结果
存入外部记忆缓冲区 (episodic memory)，下次生成时当上下文读入。
Self-Refine 更简洁：直接把反思文本拼入 prompt 重新生成，无独立记忆模块。
两者的核心循环结构相同：**生成 → 评分 → 反思 → 再生成**。

Reflexion stores reflections in an episodic memory buffer and reads them back
at the next episode start. Self-Refine inlines the feedback directly into the
prompt. Both share the same loop skeleton: **generate → score → reflect → regenerate**.

## 文件 / Files

- `from_scratch.py` — `QuadraticScorer` + `Generator` + `reflect_and_edit` + `self_refine`
  loop; no `nn.MultiheadAttention`, no external data.
- `test_self_refine.py` — 7 assertion-based tests, deterministic (`torch.manual_seed`).

```bash
python test_self_refine.py        # plain run
python -m pytest test_self_refine.py
```

## 追问分层 / Stratified follow-ups

- **L1**: Self-Refine 的三个核心步骤是什么？为什么保留历史最优而不只看最新？
  为什么用 $-\infty$ 而不是 0 来 mask 不可选位置？
  *(What are the three steps? Why keep the historical best? Why mask with $-\infty$?)*
- **L2**: Self-Refine 与带 RLHF 奖励模型微调的本质区别？何时 Self-Refine 会失效（评分器与生成器使用同一模型的偏差问题）？
  梯度上升在凸目标上为何保证单调改善？
  *(Key difference from RLHF fine-tuning? When does Self-Refine fail? Why does gradient ascent guarantee monotone improvement on convex objectives?)*
- **L3**: Reflexion 的 episodic memory 解决了 Self-Refine 的哪个局限？
  如何把这个循环扩展到多智能体（一个生成，一个独立评分）以减少自我确认偏差？
  Best-of-N 采样与 Self-Refine 的根本区别是什么（推理时计算的利用方式）？
  *(What limitation does Reflexion's memory address? How to extend to multi-agent to reduce self-confirmation bias? Fundamental difference between Best-of-N sampling and Self-Refine?)*
