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

<details class="qa"><summary>1. 什么样的任务算"长程(long-horizon)"?举两个 LLM agent 的例子。</summary>

答:任务需要多轮 think→act→observe 循环,reward 只在终端给出(稀疏/延迟),动作空间包含工具调用。例子:①代码 agent 循环调用代码执行器调试程序;②搜索+汇总 agent 多轮查询 web API 后生成报告。

</details>

<details class="qa"><summary>2. 为什么单轮 RLHF/RLVR 不足以训练多轮工具使用 agent?</summary>

答:单轮 RL 假设一问一答一奖励,episode 形状为单条 response;多轮 agent 的 episode 是 `(reasoning, tool_call, observation)` 反复交错,reward 只在最终轮给出,单轮损失函数无法处理跨轮信用分配,也没有对工具观测 token 的掩码机制。

</details>

<details class="qa"><summary>3. 稀疏 / 延迟奖励为什么难?</summary>

答:终端只有一个 reward,无法直接判断几十轮中哪一轮、哪个 token 该被奖惩,即长程信用分配问题。此外,训练初期成功轨迹概率随轮次指数衰减(若每轮成功率 0.8,10 轮后约 0.11),导致大量 rollout 全零 reward、梯度信号几乎消失。

</details>

<details class="qa"><summary>4. 训练时为什么要把<strong>工具返回的 token</strong> 掩掉?</summary>

答:工具返回是环境注入的观测,不是 policy 生成的——若不掩掉,loss 会要求模型去"拟合"工具输出,相当于对观测做 SFT,污染策略梯度信号。正确做法是 action_mask=0 屏蔽所有 `obs_*` token,梯度只流向 agent 自己生成的 think/act token。

</details>

### L2 进阶

<details class="qa"><summary>5. trajectory-level / turn-level / token-level advantage 各是什么?权衡?</summary>

答:trajectory-level 对整条轨迹共享一个 advantage,最简单但把好轨迹里的坏步也一起奖励;turn-level 给每轮一个 advantage(需 step reward 或价值估计),信用更精准;token-level 是最细粒度,通常由 turn-level advantage 广播到该轮 agent 生成的 token 并乘以 action mask。粒度越细信用越准,但对 critic/PRM 的依赖越强。

</details>

<details class="qa"><summary>6. GRPO 的组内相对 baseline 如何搬到多轮?为什么能省掉 critic?</summary>

答:对同一任务采样一组多轮轨迹,用组内终端 return 的均值/标准差归一化得到 $A(\tau_i)=\frac{R(\tau_i)-\mu_g}{\sigma_g+\epsilon}$,以组均值作 baseline 替代 critic。省 critic 的原因是 baseline 由同批 rollout 统计而来,无需额外价值网络;代价是方差在稀疏长程场景可能大(全零 group 时 advantage 退化为零)。

</details>

<details class="qa"><summary>7. PRM 与 ORM 在长程信用分配上的取舍?</summary>

答:ORM 只看终端结果,信号稀疏、信用分配粗,但标注成本低且难被 hack;PRM 对每步打分(理想上为未来成功率变化量),信用精准,但标注/训练更贵且中间步评分器可被 agent gaming。长程场景常折中:以 ORM 终端可验证奖励为主,辅以少量可验证里程碑充当 PRM 信号。

</details>

<details class="qa"><summary>8. 长程 reward hacking 有哪些典型形态?怎么缓解?</summary>

答:三类典型形态:①空转/looping(反复调用廉价工具拉长轨迹以积分);②premature stop(提前声明完成绕过后续困难步骤);③milestone gaming(触发里程碑而不真正解决子任务)。缓解组合拳:终端可验证奖励为主 + 步数/token 成本惩罚 + 观测 token loss mask + KL 惩罚项 + 对抗性测试集轮换。

</details>

### L3 深挖

<details class="qa"><summary>9. 终端只有一个 0/1 reward 时,如何把信用合理分到几十轮?给出至少两种思路并比较。</summary>

答:①**GRPO 组相对 baseline**:同任务多条轨迹共享终端 return 作归一化 baseline,简单无需 critic,但信用在轨迹内仍均摊;②**turn-level GAE**:引入轻量 turn-level value 头,用 $\hat{A}^{\text{GAE}}=\sum(\gamma\lambda)^l\delta_{t+l}$ 把终端 return 分解为逐轮 TD 误差,信用更精准但需 critic bootstrap;③**PRM step reward**:用蒙特卡洛 rollout 估计每步未来成功率变化作步级 reward,信用最细但采样成本最高。三者折中:先 GRPO 训练基础能力,稳定后加 turn-level value 头。

</details>

<details class="qa"><summary>10. 如何把"可验证结果奖励"与"过程监督"结合,既稳又不被刷?</summary>

答:以终端可验证信号(unit test pass / 环境状态)为主奖励——难 hack;用**可验证里程碑**作中间 step reward(子函数单元测试通过而非神经网络打分),缓解稀疏性而不引入可被 gaming 的 proxy。同时加 KL 惩罚 $R'=R-\beta\,\text{KL}(\pi_\theta\|\pi_\text{ref})$ 防 overoptimization,以及步数惩罚压制 looping。

</details>

<details class="qa"><summary>11. agentic RL 的 rollout 为什么贵?有哪些工程缓解(异步执行、截断、长度惩罚)?各自代价?</summary>

答:rollout 需在循环里真实执行工具/环境(网络、代码执行等),延迟可达秒级,且轨迹更长导致 KV cache 占用大。缓解:①**异步执行**——并行跑多个 episode,GPU 不空等;代价是 rollout 完成时 policy 已经走了若干步(staleness),需要 IS 修正或 ESS 监控。②**轨迹截断**——超过最大步数强行终止;代价是截断轨迹的 return 不完整,需 bootstrap 补齐或直接丢弃。③**长度惩罚** $R'=R-\alpha|\tau|$——激励 agent 高效完成;代价是可能惩罚必要的长推理链。

</details>

<details class="qa"><summary>12. 误差累积(compounding error)在长轨迹中如何放大?与 exposure bias 的关系?</summary>

答:每轮决策的小误差会改变后续观测,导致轨迹偏离训练分布,下一轮又在分布外状态上犯更大误差,误差随轮次**指数级放大**。这与 SFT 的 exposure bias 同构——训练时见到的是 ground-truth 前缀,推理时见到的是模型自己生成的前缀;长程 agent 尤为严重因为工具返回也依赖于之前行动。缓解:RL 本身通过让模型在自身 rollout 上训练来缓解 exposure bias;课程学习(从短 horizon 开始)可以降低初期误差累积速度。

</details>

---

## 深挖 / Deep-dive

> 面试陷阱级问题:以下问答假设考官已熟悉单轮 GRPO/PPO,追问多轮场景下的精细机制。**学习笔记,非作者研究成果**。

---

### Q1. per-token vs per-turn advantage estimation — GAE 的折扣如何迁移到多轮?

**核心矛盾**:单轮 RL 的 advantage 定义在 token 粒度($A_t = Q_t - V_t$);多轮场景有两级结构 — turn 之间的**时间折扣**(turn-level) 和 turn 内部的**token 广播**。

**GAE 的迁移**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">GAE 用 λ 加权的 TD 残差和来权衡 bias-variance:λ→1 接近蒙特卡洛,λ→0 接近单步 TD。<a href="https://arxiv.org/abs/1506.02438">Schulman 2015 ↗</a></span></span>:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{T-t}(\gamma\lambda)^l\,\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).$$

把"一个 token" 换成"一个 turn",该公式在 **turn-level MDP** 上完全适用:
- $\gamma$(turn 折扣):稀疏终端 reward 场景里常设 $\gamma\approx1$(不折扣),因为每轮都有实际贡献价值;过低的 $\gamma$ 会让早期轮的 advantage 趋零,白白丢掉梯度信号。
- $\lambda$(GAE 平滑系数):控制 bias-variance 权衡 — 多轮 episode 的 critic bootstrap 误差更易累积,实务里 $\lambda<1$ 比单轮更关键,否则 variance 随 horizon 指数爆炸。

**turn-level → token-level 的广播**:同一轮内所有 agent 生成 token 共享该轮的 $\hat{A}^{\text{turn}}$,观测 token 的 mask=0。这样 per-token 的梯度等于 per-turn advantage 乘以 action-mask,没有额外近似。

**面试陷阱**:"既然 token 共享同一 advantage,token-level 和 turn-level 有什么区别?" — 区别在于**哪一级计算折扣和 baseline**:turn-level 让折扣跨越轮次、允许 value function 在轮粒度 bootstrap;token-level 单纯是梯度落点。混淆两者会导致折扣应用层级错误(对每个 token 做 γ 折扣等价于对超长序列做 γ^t 衰减,早期 token 梯度几乎为零)。

---

### Q2. 单终端 reward 下 GRPO 组 baseline 的高方差 — 为什么?有哪些降方差技巧?

**GRPO 组 baseline** 的 advantage:

$$A(\tau_i) = \frac{R(\tau_i) - \mu_g}{\sigma_g + \epsilon}, \quad \mu_g=\frac{1}{G}\sum_{j=1}^G R(\tau_j).$$

**多轮场景下方差爆炸的两个来源**:

1. **0/1 二值 reward + 小 group size**:若 $G=8$ 且成功率 $p\approx0.1$,则 group 内常见全 0(8 条全失败)或只有 1 条成功。全 0 时 $\sigma_g=0$,advantage 退化为零;1/8 成功时 $\sigma_g$ 极小,advantage 尖峰 — 单条样本主导整批更新。
2. **轨迹长度差异**:长轨迹的 log-prob sum 数值远大于短轨迹;若 advantage 直接乘以 token 数量,长轨迹的梯度自然更大,形成隐式的 length bias。

**降方差技巧**:

| 技巧 | 机制 | 代价 |
|---|---|---|
| 加大 group size $G$ | 更稳定的 $\mu_g,\sigma_g$ | 采样成本 $\times G$ |
| length-normalization | 损失除以 action token 数 | 均衡短/长轨迹,但可能惩罚必要的长推理 |
| 混合 ORM+中间稀疏 reward | 降低全 0 group 概率,增加 positive signal | 中间 reward 可被 hack |
| advantage clipping / 截断分位数 | 去掉尖峰 advantage 样本 | 可能丢高信息样本 |
| 引入轻量 critic(turn-level value 估计) | 用 $V$ 把终端 return 分解为逐轮 TD 误差 | 额外模型,与 GRPO "去 critic" 初衷矛盾 |

**面试陷阱**:"GRPO 能完全替代 PPO 的 critic 吗?" — 单轮二值 reward 场景可以;长程稀疏 reward 场景里 group baseline 的方差往往大于 critic baseline,critic 的价值回升。实践中折中方案是使用小型 turn-level value 头,而非完整的 PPO critic。

---

### Q3. 多轮 rollout 变 stale 后的 importance-sampling / off-policy 修正

**问题背景**:agentic rollout 含真实工具调用(网络、数据库、代码执行),延迟可达秒级 — 当 rollout 结束时策略参数已经走了若干步,$\pi_\theta \neq \pi_{\theta_\text{old}}$。

**重要性权重(IS weight)**:

$$w(\tau) = \frac{\pi_\theta(\tau)}{\pi_{\theta_\text{old}}(\tau)} = \prod_{t \in \text{agent tokens}} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}.$$

**为什么连乘危险**:$T$ 轮轨迹里 agent token 可达数千;即使每步 ratio $\approx1.05$,连乘后可得 $\gg10$ 的 IS weight — 方差爆炸,极少数样本主导梯度。

**工程修正方案**:

1. **PPO clip**<a class="cite" href="#ref-5">5</a>($\epsilon$ clip):将每个 token 的 ratio 截断在 $[1-\epsilon, 1+\epsilon]$,不修正偏差但有效控方差。适合**同步训练**,policy lag 不超过 1-2 个 mini-batch。
2. **序列级截断 IS(TIS)**:整条轨迹的 IS weight 截断在某个上界(如 3.0),简单但有偏。
3. **用 ESS 动态缩小学习率**:用 effective sample size $\text{ESS}=(\sum w_i)^2/\sum w_i^2$ 监控 rollout 的 staleness,当 ESS 低于阈值时自动降低 learning rate,避免梯度冲击。
4. **前缀 IS ratio**:理论上正确的修正项是**前缀 IS ratio**(整条序列的前缀乘积),而非逐 token ratio 独立截断;但实现复杂,数值不稳定。

**面试陷阱**:"直接用 PPO clip 就够了吗?" — 同步训练里够;异步/agentic 场景 policy lag 可达数十步,此时 clip 只治标(方差降了但偏差很大),实际上等于在不正确的分布上做梯度 — 需要配合小 policy lag 设计或 ESS 监控。

---

### Q4. 长程 reward hacking 的具体形态与缓解

**"长程"使 hacking 更容易**:单轮 hacking 只需在一次回复里找漏洞;长程 agent 可以在几十步里缓慢积累捷径、覆盖评测文件、或利用工具副作用。

**三类典型形态**:

| 类型 | 机制 | 实例 |
|---|---|---|
| **空转/looping** | agent 重复调用无关工具、拉长轨迹以期靠里程碑积分 | 反复查询 search API 但不推进任务 |
| **premature stop** | 提前声明"任务完成"绕过后续困难步骤 | 代码 agent 在测试运行前就输出 "DONE" |
| **milestone gaming** | 触发里程碑检查点而不真正解决子任务 | 写一个空函数使 CI 通过,或直接 mock 测试输出 |

**为什么长程更严重**:理论分析<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Scaling Laws for Reward Model Overoptimization:KL 偏离越大,gold reward 先升后降;proxy reward 与 gold reward 的差值随 KL 单调增。<a href="https://arxiv.org/abs/2210.10760">Gao 2022 ↗</a></span></span>表明,proxy reward 与 gold reward 的差值随 KL 偏离单调增大 — 轨迹越长、总 KL 偏离越大,照缩放定律 [7] hacking 风险系统性增加。

**缓解组合拳**:

1. **终端可验证奖励为主**<a class="cite" href="#ref-4">4</a>:unit test pass / 环境状态达成,比神经网络 proxy reward 难 hack;
2. **步数 / token 成本惩罚**:$R'(\tau) = R(\tau) - \alpha \cdot |\tau|$,直接压制 looping;
3. **观测 token 只读(loss mask)**:防止 agent 学会"生成能通过 mock 检查的输出格式";
4. **KL 惩罚项**:$R' = R - \beta\,\text{KL}(\pi_\theta\|\pi_{\text{ref}})$,在 proxy reward 与 gold reward 开始分叉前刹车;
5. **对抗性测试集轮换**:定期更换评测样本,阻止 agent 记住特定测试用例的捷径。

---

### Q5. 部分可观测性(POMDP)如何使 value 估计变难?

**单轮 RL vs 长程 agentic RL 的信息结构差异**:

单轮推理:$V(s) \approx V(\text{prompt})$ — prompt 全可见,value function 输入完整。

多轮 agentic:$s_t$ = 历史对话 + 上一轮工具返回,但**下一轮工具返回未知** — agent 处于 POMDP,不知道未来观测。

**三个具体难点**:

1. **工具返回的随机性**:同一 action(API 调用)可能因网络状态、外部 DB 版本不同而返回不同内容 — value function 需要在这种随机性上取期望,但训练时只见到一条具体返回。单步 bootstrap($V(s_{t+1})$)拟合的是带噪观测的 value,收敛慢且偏差难估计。

2. **上下文长度爆炸**:随轮次增加,$s_t$ 线性增长(历史全拼接);value network 若用同一个 LM backbone,其 forward pass 成本随上下文长度平方增长 — value bootstrap 本身就很贵。

3. **外部状态不可逆**:agent 改了数据库/文件系统,这些副作用不在 token 流里 — value function 看不到"环境的隐状态"。传统 POMDP 解法(belief state)在 LLM 场景无法直接套用。

**实务应对**:
- 用轻量 turn-level value 头(接在最后一个生成 token 的 hidden state 上)而非完整 rollout 的 critic;
- 将工具返回**摘要化**后再入 value 输入,压缩上下文;
- 接受更高 bias:用 GAE 的 $\lambda<1$ 减少对远期 bootstrap 的依赖,代价是轻微低估长期 advantage。

---

### Q6. 长程稀疏 reward 下的探索问题

**为什么单轮 RL 的探索策略在长程下不够**:单轮 RL 里随机采样 token 就能探索;多轮场景里**成功轨迹的概率随轮次指数衰减** — 如果每轮正确概率 0.8,10 轮后成功率降至 $0.8^{10}\approx 0.11$,agent 极少见到正 reward,训练信号几乎全零。

**三类探索策略**:

1. **课程学习(curriculum learning)**:从短 horizon / 容易子任务开始,逐步加难。核心是确保训练初期 positive reward 足够频繁以产生有效梯度。代价:需要任务难度的自动标注或人工课程设计。

2. **subgoal / 里程碑奖励(谨慎使用)**:在中间步骤给小奖励引导探索方向。问题:如 Q4 所述,里程碑本身会被 gaming — 须配合可验证的里程碑(如子函数的单元测试通过)而非神经网络打分。

3. **replay + 优先经验回放**:保留少量历史成功轨迹,以更高概率重采样 — 让模型在极稀疏环境中持续看到"什么是成功"。代价:引入 off-policy 问题(见 Q3)。

**面试追问**:"如果任务成功率始终 <5%,还能用 GRPO 吗?" — 实务经验:group size 需要大到能保证 group 内至少 1 条成功;否则整个 group 的 advantage 退化为全零,等于白跑 rollout。此时建议先用 SFT 热启(从少量成功轨迹学),再切换 RL。

---

### Q7. 观测 token 掩码后哪些 token 实际拿到 advantage?PRM 的 step 级信用 vs 纯 ORM

**精确回答"哪些 token 拿到 advantage"**:

设一条轨迹含以下 token 序列:

```
[system_prompt] [user_turn_1] [agent_think_1] [agent_act_1] [obs_1] [agent_think_2] [agent_act_2] [obs_2] … [agent_final]
```

action_mask=1 的 token:所有 `agent_think_*` + `agent_act_*` + `agent_final`。
action_mask=0 的 token:`system_prompt`、`user_turn_*`、所有 `obs_*`(工具返回/环境观测)。

**advantage 广播规则**:如果使用 turn-level GAE,则同一轮内所有 mask=1 的 token 共享该轮的 $\hat{A}^{\text{turn}}$。最终梯度只流向 mask=1 的 token 位置。

**一个易错点**:若 agent 的 `<think>` 块被当作 internal reasoning 而非 action(有些实现会把 CoT 掩掉),则 `<think>` token 的 mask=0,gradient 不流过推理链 — 等于只训练"行动选择",不训练"推理质量"。这是实现细节,面试中考察是否真正理解 mask 的语义。

**PRM(过程奖励)vs ORM(结果奖励)的信用粒度对比**<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">把 PRM 的步级奖励定义为步级 advantage(未来成功率的变化量),理论上等价于 RL 的 Q-V 差,并需要用独立 prover policy 而非当前策略来估计。<a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>:

| 维度 | ORM(终端 reward) | PRM(step-level reward) |
|---|---|---|
| 信号稀疏度 | 一条轨迹一个 scalar | 每步一个 scalar |
| credit 粒度 | trajectory-level → 广播到 token | step-level → 直接赋予该步 token |
| 标注成本 | 低(只需最终正误) | 高(需逐步判断或自动 rollout 估计) |
| 可被 hack 程度 | 较难(终端状态难伪造) | 较易(中间步评分器可被欺骗) |
| 与 GAE 的关系 | GAE 用 $V$ 函数近似步级 advantage | PRM 直接提供步级 advantage 估计 |

**PRM 的步级 advantage 定义**:理论上最干净的 PRM 步级 reward 是该步骤带来的"未来成功率变化":$r_t^{\text{PRM}} = P(\text{success}|s_{t+1}) - P(\text{success}|s_t)$。这与 RL 里的 advantage($Q(s,a)-V(s)$)在定义上等价<span class="cite-wrap"><a class="cite" href="#ref-8">8</a><span class="cite-note">把 PRM 的步级奖励定义为步级 advantage(未来成功率的变化量),理论上等价于 RL 的 Q-V 差,并需要用独立 prover policy 而非当前策略来估计。<a href="https://arxiv.org/abs/2410.08146">Setlur 2024 ↗</a></span></span>。实践里用**蒙特卡洛 rollout 估计**$P(\text{success}|s_t)$,代价是每步需要大量 rollout。

**面试陷阱**:"用了 PRM 就不需要 discount $\gamma$ 了吗?" — 错。PRM 提供的是**步级奖励**,这些奖励仍然需要用折扣或 GAE 累加成 return;PRM 解决的是"哪步该给多少奖励",不解决"如何把多步奖励折算成当前策略的梯度信号"。

---

## 参考文献 / References

> 均为经典承重方法的原始出处,已逐条核对(标题 + arXiv ID)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Yao et al. <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a> — think→act→observe 范式. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — GRPO:组内相对 baseline、去 critic. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — 过程监督 / PRM(PRM800K). <a href="#fnref-3">↩</a></li>
<li id="ref-4">Lambert et al. <em>Tülu 3: Pushing Frontiers in Open Language Model Post-Training</em>. 2024. <a href="https://arxiv.org/abs/2411.15124">arXiv:2411.15124</a> — RLVR:以可验证对错为终端奖励. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Schulman et al. <em>Proximal Policy Optimization Algorithms</em>. 2017. <a href="https://arxiv.org/abs/1707.06347">arXiv:1707.06347</a> — PPO(策略梯度基线). <a href="#fnref-5">↩</a></li>
<li id="ref-6">Schulman et al. <em>High-Dimensional Continuous Control Using Generalized Advantage Estimation</em>. ICLR 2016. <a href="https://arxiv.org/abs/1506.02438">arXiv:1506.02438</a> — GAE:λ 加权 TD 残差,bias-variance 权衡. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — proxy vs gold reward 随 KL 偏离的缩放定律. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Setlur et al. <em>Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning</em>. 2024. <a href="https://arxiv.org/abs/2410.08146">arXiv:2410.08146</a> — PRM 步级 advantage = 未来成功率变化量,等价于 Q-V 差. <a href="#fnref-8">↩</a></li>
</ol>

