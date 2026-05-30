# Drill: ReAct tool-call loop from scratch

> 可运行的 from-scratch 实现 + 测试。目标:每一行都能在面试里推导和辩护。
> Runnable from-scratch implementation with tests — derive and defend every line.
>
> ⚠️ **学习笔记,非作者研究成果**。追问为据公开论文/JD 推断的练习题,非真实面试原题。

## 背景 / Background

一个 agent 的最小骨架就是 **ReAct 循环**(Yao et al., ReAct):模型把**推理(Thought)**和**行动(Action)**交错输出,每次行动调用一个工具,把**观测(Observation)**注入回上下文,直到给出 **Final Answer**。

```
Question: …
Thought:  我需要先查一下 X。
Action:   search
Action Input: X
Observation: <工具返回——由环境注入,不是模型生成>
Thought:  现在我知道了。
Final Answer: …
```

这个 drill 实现三件最容易考的事:

1. **解析 + 控制流** — 把模型一段补全解析成 `(thought, action, action_input)` 或 Final Answer,路由到对应工具,注入观测,循环到终止。
2. **SFT label masking** — 训练 agent 时,**只有模型自己生成的 token(Thought/Action/Final Answer)该回传梯度**;问题与工具输出是注入的环境 token,label 置 `ignore_index`(-100)掩掉。这是 agentic SFT/RL 的承重细节。
3. **可数值验证的掩码** — 从零实现 `masked_cross_entropy`,与 `F.cross_entropy(ignore_index=-100)` 对拍,证明掩码精确丢掉的就是环境 token。

A minimal agent is just the **ReAct loop**: interleave **Thought** and **Action**, each action calls a tool, the **Observation** is injected back, until a **Final Answer**. This drill implements parsing + control flow, the SFT label mask (train only agent-emitted tokens), and a from-scratch masked cross-entropy so the mask is numerically checkable.

## 数学 / The math

### 1. 循环 / The loop

一条轨迹是 prompt、agent 段、observation 段交替拼接:

$$\tau = \underbrace{[\text{question}]}_{\text{prompt}}\; \underbrace{[\text{think}_1,\text{act}_1]}_{\text{agent}}\; \underbrace{[\text{obs}_1]}_{\text{env}}\; \underbrace{[\text{think}_2,\text{act}_2]}_{\text{agent}}\; \dots\; \underbrace{[\text{final}]}_{\text{agent}}$$

**为什么 ReAct 比纯 CoT 少幻觉**:纯 chain-of-thought 在自己的输出上滚动,中间事实无法被校正;ReAct 每一步把**真实工具返回**作为下一步的条件,推理被外部观测 grounding,错误事实在下一轮就能被观测纠偏。

### 2. 标签掩码 / Label masking

下一 token SFT 的损失只在 agent token 上计算:

$$\mathcal{L} = -\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}} \log p_\theta(x_t \mid x_{<t}), \qquad \mathcal{A}=\{t: \text{token } t \text{ 由 agent 生成}\}$$

实现上把环境 token 的 label 置 `-100`,`cross_entropy` 既不计入分子也不计入分母。这等价于 RL 里的 **action mask**(见姊妹 drill `turn-credit-assignment`):训练策略自己**能控制**的 token,而不是去拟合环境注入的文本。漏掉这一步,模型会去"背诵"工具输出的格式,污染信号。

> JSON function-calling 与 text-based 两种格式的差别仅在**哪些 span 算 agent**:function-calling 里被 train 的是结构化的 `{"name","arguments"}` JSON 串,text-based 里被 train 的是 `Action:/Action Input:` 行——两者都只 mask 工具返回那一段。

## 复杂度 / Complexity

- 循环:$O(\sum_t L_t)$,$L_t$ 为第 $t$ 轮上下文长度(每轮都重读全 transcript,故是步数 × 上下文,长程下与 KV cache/上下文管理强相关)。
- 掩码 CE:$O(N V)$,$N$ token 数、$V$ 词表——与普通 CE 同阶,掩码不增加渐进成本。

## 文件 / Files

| 文件 | 内容 |
|---|---|
| `from_scratch.py` | `parse_react_step` + `run_react_loop` + `build_sft_labels` + `masked_cross_entropy`(不依赖任何 agent/LLM 框架) |
| `test_react.py` | 13 个测试:解析、工具在正确轮次调用、观测精确拼接、Final Answer / max_steps 终止、未知工具 / 解析失败、掩码精确性、与 `F.cross_entropy` 数值对拍 |

```bash
python from_scratch.py        # 跑一个 2 步玩具 episode
python test_react.py          # 或 python -m pytest test_react.py
```

## 追问分层 / Stratified follow-ups

### L1 — 概念 / Concept

- ReAct 的 Thought/Action/Observation 三段各是什么?为什么交错 act 比纯 chain-of-thought 更少幻觉?
- 为什么 Observation 必须是**环境注入**而不是让模型自己"生成"工具返回?若让模型自己编 observation 会发生什么?
- 一个 ReAct episode 何时终止?除了 Final Answer,还需要哪个停止条件来保证不死循环,为什么?

### L2 — 实现 / Implementation

- 构造 agentic SFT 数据时,哪些 token 该做 label、哪些该 mask 成 `-100`?漏 mask 工具返回 token 会训练出什么坏行为?
- OpenAI function-calling(结构化 JSON)与 text-based(`Action:` 文本行)两种格式,在 SFT label masking 上的差别到底在哪一段?各自的工程权衡是什么?
- `masked_cross_entropy` 为什么要把分母也限制在未掩码 token 上(而非除以总长度)?除以总长度会引入什么偏差?

### L3 — 算法 / Algorithm

- 从 SFT 进到 RL:工具调用的奖励该给在 **trajectory-level**(整条对错)还是 **tool-call-level**(每次调用的格式/结果对错)?ToolRL 式的细粒度 reward shaping 解决了什么、又带来什么 reward-hacking 风险?
- AgentPRM 用蒙特卡洛 rollout 估计每步的过程奖励——它与 trajectory-level ORM 在长程信用分配上各自的代价是什么?(可联系姊妹 cheatsheet 的 PRM vs ORM)
- 解析模型自由文本输出(`Action:` 行)很脆弱:约束解码 / 强制 JSON schema 能消除解析错误吗?它对训练分布和模型表达力各有什么副作用?

## 参考 / References

> 均为承重方法的原始出处,已逐条 web 核对(标题 + arXiv ID)。

- Yao et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) — think→act→observe 范式 / the loop.
- Schick et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761) — 自监督学会"何时/如何"调用 API / self-supervised tool use.
- Zeng et al. (2023). *AgentTuning: Enabling Generalized Agent Abilities for LLMs*. THUDM/智谱. [arXiv:2310.12823](https://arxiv.org/abs/2310.12823) — 轨迹 SFT 提升 agent 能力而不损通用能力 / trajectory SFT.
- Qian et al. (2025). *ToolRL: Reward is All Tool Learning Needs*. [arXiv:2504.13958](https://arxiv.org/abs/2504.13958) — 工具学习的 RL 奖励设计 / reward design for tool-use RL.
- Choudhury (2025). *Process Reward Models for LLM Agents: Practical Framework and Directions*. [arXiv:2502.10325](https://arxiv.org/abs/2502.10325) — AgentPRM:MC rollout 估计 agent 过程奖励 / process rewards for agents.
