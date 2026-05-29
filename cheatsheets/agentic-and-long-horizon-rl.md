# Agentic & Long-horizon RL / 长程 Agent 强化学习

> 单轮推理 RL(GRPO / RLVR,见姊妹仓库 [reasoning-rl-frontier](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reasoning-rl-frontier.html))的**下一棒**:把奖励信号从"一问一答一奖励"推广到**多轮轨迹**(思考→调用工具→观察)\*。
>
> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。数字 / 结论以原论文为准,不确定处标注。

## 0. 一句话演化 / The evolution

`单轮 RLHF(prompt→response→reward)` → `单轮可验证 RLVR(对/错→reward)` → **`多轮 agentic RL(轨迹→稀疏终端 reward)`**。
变的不是损失函数,而是 **episode 的形状**:一条 episode 是 `(reasoning, tool_call, observation)` 反复多轮<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">让 LLM 把推理与工具调用交错(think→act→observe),边想边行动。<a href="https://arxiv.org/abs/2210.03629">Yao 2022 ↗</a></span></span>,reward 往往只在**最后**(任务成功与否)给一次。

## 1. 什么让任务"长程" / What makes it long-horizon

| 维度 | 单轮推理 RL | 长程 agentic RL |
|---|---|---|
| episode 长度 | 1 轮 | 数轮~数十轮 |
| reward | 每条 response 一个 | **稀疏 / 延迟**,常只在终端 |
| 观测 | 全可见(prompt) | **部分可见**(工具返回才知道) |
| 动作空间 | token | token + **工具调用** + 何时停止 |
| 主要难点 | 偏好 / 正确性 | **长程信用分配** + 误差累积 |

## 2. 形式化 / Formalization (POMDP)

把一条轨迹 $\tau=(s_0,a_0,\dots,s_T,a_T)$ 看作 POMDP。终端任务奖励 $R(\tau)\in\{0,1\}$(成功/失败)或标量分。轨迹回报

$$G_t=\sum_{k=0}^{T-t}\gamma^{k}\,r_{t+k}.$$

实务里常把"一轮(turn)"作为决策粒度(turn-level MDP),而梯度仍落在该轮**生成的 token** 上 —— 即 **turn-level 信用 + token-level 更新**。

## 3. 信用分配 / Credit assignment(核心难点)

终端只有一个 reward,要回答:**这几十轮里,哪一轮、哪个 token 该被奖励/惩罚?**

- **trajectory-level**:整条轨迹共享一个 advantage —— 最简单,但把"好轨迹里的坏步"也一起奖励了。
- **turn-level**:给每一轮一个 advantage(需要 step reward 或价值估计)。
- **token-level**:最细;通常由 turn-level advantage **广播**到该轮 token。

GRPO<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">同一任务采样一组 rollout,用组内相对回报当 baseline、免去 critic。<a href="https://arxiv.org/abs/2402.03300">Shao 2024 ↗</a></span></span> 的思路天然可搬过来:**同一任务采样一组轨迹,用组内相对回报当 baseline**(无需 critic):

$$A(\tau_i)=\frac{R(\tau_i)-\mathrm{mean}(R)}{\mathrm{std}(R)+\epsilon}.$$

> 过程奖励 **PRM**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">对每一步推理打分(过程监督),不只看最终对错。<a href="https://arxiv.org/abs/2305.20050">Lightman 2023 ↗</a></span></span>(给每步打分)对长程信用分配更友好,但**标注/训练更贵**且自身可能被 hack;**ORM**(只看结果)便宜但信用分配粗。长程场景常是两者折中。

## 4. 奖励设计 / Reward design

- **可验证结果奖励(RLVR→agentic)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">用可自动判定的对错(而非奖励模型)作 RL 信号。<a href="https://arxiv.org/abs/2411.15124">Lambert 2024 ↗</a></span></span>:能自动判定的终端信号最稳 —— 单元测试通过、环境状态达成、答案可校验。
- **过程 / step reward**:中间里程碑给分,缓解稀疏性,但**易被刷**(agent 学会触发里程碑而不解决任务)。
- **长程 reward hacking**:轨迹越长,捷径越多(空转凑步数、反复调用廉价工具)。缓解:终端可验证为主 + 步数/成本惩罚 + 对工具输出**只读不学**(见 §5 掩码)。

## 5. 算法要点 / Algorithm essentials

多轮 PPO<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">带 clip 的策略梯度,RLHF / agentic RL 的基线算法。<a href="https://arxiv.org/abs/1707.06347">Schulman 2017 ↗</a></span></span> / GRPO 的两个"长程专属"细节:

1. **观测 token 掩码**:工具返回 / 环境观测是**注入**进上下文的,不是 policy 生成的 —— 必须在损失里**掩掉**,否则等于让模型去"拟合"工具输出(类比 SFT 的 loss masking)。
2. **advantage 广播 + 掩码**:turn-level advantage 广播到该轮 agent 生成的 token,再乘以 action mask。

```python
import torch

def group_relative_advantages(returns, group_ids, eps=1e-6):
    """GRPO 式组内相对 advantage(无 critic)。
    returns:   (N,) 每条轨迹的终端回报(如成功={0,1} 或标量分)
    group_ids: (N,) 同一任务的多条 rollout 共享一个 group id
    """
    adv = torch.zeros_like(returns)
    for g in group_ids.unique():
        m = group_ids == g
        r = returns[m]
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + eps)
    return adv

def masked_pg_loss(logp, adv_per_token, action_mask):
    """策略梯度损失:只在 agent 自己生成的 token 上回传。
    logp:          (B,T) 当前策略下所取 token 的 log-prob
    adv_per_token: (B,T) 由 turn-level advantage 广播到 token
    action_mask:   (B,T) 1=agent 生成的 token, 0=工具输出/观测(掩掉)
    """
    pg = -(logp * adv_per_token) * action_mask
    return pg.sum() / action_mask.sum().clamp_min(1)
```

> rollout 阶段要**在循环里真实执行工具/环境**(常异步),轨迹更长 → 采样更贵 → 是 agentic RL 的主要系统瓶颈之一。

## 6. 与单轮的衔接 / Bridge from single-turn

姊妹仓库的 **GRPO / RLVR / 损失掩码** 是积木;agentic RL ≈ 把它们**作用在轨迹上** + 解决"稀疏终端 reward 的信用分配"。会单轮 → 抓住"轨迹化 + 掩码 + 组相对 baseline"三点即可迁移。

---

## 分层面试题 / Stratified follow-ups

### L1 基础
1. 什么样的任务算"长程(long-horizon)"?举两个 LLM agent 的例子。
2. 为什么单轮 RLHF/RLVR 不足以训练多轮工具使用 agent?
3. 稀疏 / 延迟奖励为什么难?
4. 训练时为什么要把**工具返回的 token** 掩掉?

### L2 进阶
5. trajectory-level / turn-level / token-level advantage 各是什么?权衡?
6. GRPO 的组内相对 baseline 如何搬到多轮?为什么能省掉 critic?
7. PRM 与 ORM 在长程信用分配上的取舍?
8. 长程 reward hacking 有哪些典型形态?怎么缓解?

### L3 深挖
9. 终端只有一个 0/1 reward 时,如何把信用合理分到几十轮?给出至少两种思路并比较。
10. 如何把"可验证结果奖励"与"过程监督"结合,既稳又不被刷?
11. agentic RL 的 rollout 为什么贵?有哪些工程缓解(异步执行、截断、长度惩罚)?各自代价?
12. 误差累积(compounding error)在长轨迹中如何放大?与 exposure bias 的关系?

---

## 参考文献 / References

> 均为经典承重方法的原始出处,已逐条核对(标题 + arXiv ID)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Yao et al. <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a> — think→act→observe 范式. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — GRPO:组内相对 baseline、去 critic. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — 过程监督 / PRM(PRM800K). <a href="#fnref-3">↩</a></li>
<li id="ref-4">Lambert et al. <em>Tülu 3: Pushing Frontiers in Open Language Model Post-Training</em>. 2024. <a href="https://arxiv.org/abs/2411.15124">arXiv:2411.15124</a> — RLVR:以可验证对错为终端奖励. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Schulman et al. <em>Proximal Policy Optimization Algorithms</em>. 2017. <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a> — PPO(策略梯度基线). <a href="#fnref-5">↩</a></li>
</ol>

