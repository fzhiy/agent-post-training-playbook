# Agent Foundations / Agent 基础

> 在上多轮 RL 之前的**前置课**:agent 是什么、ReAct 循环、工具调用、协议层(MCP/A2A)、生产工程模式、评测与失败模式。读懂这一篇,再去 [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) 学怎么用 RL 训它。

> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。数字 / 结论以原论文为准;benchmark 数字快变且易受污染,本页只记**原文 human baseline + 测什么**,不抄 model SOTA。

## 0. TL;DR 速查

- **Agent = LLM(policy) + 工具 I/O + 记忆 + 控制循环**;比 chatbot 多的是「能对外部世界采取改变状态的动作 + 自主多步」。
- 最小骨架 = **ReAct**:Thought→Action→Observation 循环到 Final Answer;**Observation 必须由环境注入,别让模型自己编**(stop-token footgun)。
- 工具使用两条路线:**文本协议**(ReAct / Toolformer)与**结构化 function calling**(JSON schema);训练时只在 agent 生成的 token 上回传(见 [react-tool-call-loop](drill-react-tool-call-loop.html) drill)。
- 规划:**Plan-and-Execute**(先全局计划)vs **ReAct**(逐步决策);生产常用「高层 plan + 每步 ReAct」混合 + plan repair。
- 协议层:**MCP**(Anthropic,连工具/数据,*纵向*)与 **A2A**(Google,agent 间互通,*横向*);标准化连接,**不解决安全**(注入由 host 防)。
- 长程三大系统瓶颈:上下文 $O(L^2)$ / 成本 $O(T^2)$、**lost-in-the-middle**、误差复利 $p^T$。
- 生产模式:**subagent 编排**(≠多 agent 辩论)、**工具检索**(100+ 工具)、记忆分层、**budget guard**。
- 评测:SWE-bench / GAIA / OSWorld / WebArena / τ-bench 各测不同能力;只记 human baseline 与「测什么」。
- 可靠性指标:**pass@k**(能不能做到)vs **pass^k**(稳不稳定);agent 部署看后者。
- 6 类失败模式有名字(见 §10),面试要能默写 + 配缓解。

## 1. 心智模型 / Mental model

把 LLM 看作**策略** $\pi_\theta(a \mid h)$:给定历史 $h$(对话 + 过往观测),输出下一个动作 $a$(一段文本,可能含工具调用)。**agent = policy + 工具 I/O + 记忆 + 控制循环**:

```
观测 obs ──▶ [LLM policy] ──▶ 动作 action ──▶ [环境 / 工具] ──┐
   ▲                                                          │
   └──────────────── 新 observation ◀───────────────────────┘
                  (循环到 Final Answer 或预算耗尽)
```

**三轴设计框架**(任何 agent 都可沿三轴定位):

| 轴 | 选项(由简到繁) |
|---|---|
| 推理结构 | 直接答 → CoT<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">给中间推理步当 few-shot 范例,显著提升推理任务。<a href="https://arxiv.org/abs/2201.11903">Wei 2022 ↗</a></span></span> → ReAct → ToT<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">把推理展开成树,自评中间「想法」做 deliberate 搜索。<a href="https://arxiv.org/abs/2305.10601">Yao 2023 ↗</a></span></span> / 搜索 |
| 工具接口 | 无 → 文本协议(ReAct)→ 结构化 function calling → computer-use(截图+坐标) |
| 学习信号 | 纯 prompt → 轨迹 SFT → RL(见姊妹篇 agentic-RL) |

**经典 RL vs LLM-agent**:

| 维度 | 经典 RL agent | LLM agent |
|---|---|---|
| 策略 | 小网络,从零训 | 预训练 LLM,少量后训练 |
| 动作空间 | 固定低维 | 开放文本 + 工具调用 |
| 先验 | 几乎无 | 海量世界知识 |
| 样本效率 | 低(百万步) | 高(prompt 即可零样本起步) |

> ❌ **误区:** 「会调工具的 chatbot 就是 agent」。关键不在工具,而在**闭环 + 自主多步 + 对外部状态的改变**:单轮检索增强(RAG)仍是一问一答,agent 要能根据观测决定下一步、反复行动直到目标达成。

## 2. ReAct — 最小 agent 骨架

ReAct<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">让 LLM 把推理与行动交错(think→act→observe),边想边做。<a href="https://arxiv.org/abs/2210.03629">Yao 2022 ↗</a></span></span> 把**推理(Thought)**与**行动(Action)**交错,每次行动调一个工具,把**观测(Observation)**注入回上下文:

```
Thought: 我需要先查 X。
Action: search
Action Input: X
Observation: <工具返回——由环境注入,不是模型生成>
Thought: 现在我知道了。
Final Answer: …
```

**为何比纯 CoT 少幻觉**:纯 chain-of-thought 在自己的输出上滚动,中间事实无法被校正;ReAct 每步把**真实工具返回**作为下一步条件,推理被外部观测 grounding,错误事实下一轮即可被纠偏。

> ❌ **误区:** 「ReAct 总优于 CoT」。在**纯推理**任务上 ReAct 不一定胜过 CoT-self-consistency(原文在 HotpotQA 上 ReAct 单用反而偏弱,需与 CoT-SC 结合);ReAct 的价值在**需要外部知识 / 动作**的任务。

> ⚠️ **stop-token footgun:** 推理时必须把 `Observation:` 设为 stop sequence。否则模型会**自己接着生成一段 `Observation: …`**(幻觉工具返回),而不是停下来等环境注入真实结果——这是 ReAct 落地最常见的 bug。手撕见 [react-tool-call-loop](drill-react-tool-call-loop.html)。

## 3. 规划 / Planning:Plan-and-Execute vs ReAct

- **ReAct**:逐步决策(reactive)——每步看当前观测再决定下一动作,灵活但无全局视野。
- **Plan-and-Execute / Plan-and-Solve**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">zero-shot 先让模型「制定计划→分解子任务」再执行,胜过 Zero-shot-CoT。<a href="https://arxiv.org/abs/2305.04091">Wang 2023 ↗</a></span></span>:先生成**完整计划**再逐步执行(执行器甚至可用不同模型),全局一致但计划可能过时。
- **生产常用混合**:高层 plan 切大步骤 + 每步内用 ReAct 反应;失败时 **plan repair**(Reflexion<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">用语言反思存进 episodic memory 来「口头强化」,不更新权重。<a href="https://arxiv.org/abs/2303.11366">Shinn 2023 ↗</a></span></span> 式反思 / ToT 式搜索 / 逐步 replan)。

**何时纯 plan-execute 失败**:环境不确定、中途状态大变(工具返回出乎预料)时,开局定死的静态计划会过时——此时 reactive 的 ReAct 或带 replan 的混合更稳。

## 4. 工具使用 / Tool use

**Toolformer**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">自监督学「何时/如何」调 API,用 utility filter 只保留有用的自标注调用。<a href="https://arxiv.org/abs/2302.04761">Schick 2023 ↗</a></span></span>:无人工标注地学会调 API。做法:在文本里随机插入候选 API 调用 → 执行得到返回 → **utility filter** 只保留「插入该调用 + 返回后,模型预测后续 token 的损失显著下降」的样本做 SFT。即用「调用是否真的帮到后续预测」自动筛掉无用 / 位置错的调用。

**结构化 function calling**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">2023-06 引入:用 JSON Schema 描述函数,模型输出结构化调用。<a href="https://openai.com/index/function-calling-and-other-api-updates/">OpenAI 2023 ↗</a></span></span>:用 **JSON Schema** 描述函数签名,模型(经微调)直接输出结构化的 `{name, arguments}` 调用。**parallel tool calls**(一次返回多个调用)要求这些调用**幂等 + 相互独立**(无数据依赖),否则不能并行。

> ❌ **误区:** 「function calling 和 ReAct 是两种对立的 agent」。两者都是工具使用,但**层级不同**:FC 是工具调用的**结构化格式**(模型 fine-tuned 输出 JSON schema),ReAct 是**推理-行动的循环模式**(prompt 范式);完全可以「用 ReAct 循环 + 每步用 function calling 发工具」。SFT label masking 差异见 [react drill](drill-react-tool-call-loop.html) 与 agentic 篇 Q11。

## 5. 协议层 / Protocols:MCP & A2A

| 协议 | 方向 | 标准化什么 |
|---|---|---|
| **MCP**(Model Context Protocol)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Anthropic 2024-11 开放协议:client-server + JSON-RPC 2.0,3 primitive(tools/resources/prompts)。<a href="https://modelcontextprotocol.io">Anthropic 2024 ↗</a></span></span> | **纵向**(agent ↔ 工具/数据) | 模型如何连外部工具与数据:client-server、JSON-RPC 2.0、三 primitive(**tools / resources / prompts**)、transport(stdio / Streamable HTTP) |
| **A2A**(Agent2Agent)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">Google 2025-04 提出、后归 Linux Foundation:agent 间互通,JSON-RPC over HTTP + agent card。<a href="https://a2a-protocol.org/latest/">Google 2025 ↗</a></span></span> | **横向**(agent ↔ agent) | 不同厂商 agent 如何互通:**agent card**(能力声明)+ task 状态机 + JSON-RPC over HTTP |

> ❌ **误区:** 「用了 MCP 就安全了」。协议**不防 prompt injection**——工具返回里藏的恶意指令能劫持 agent(见 §10 + Greshake<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a><span class="cite-note">indirect prompt injection:外部内容(网页/工具返回)里的指令劫持 LLM 应用。<a href="https://arxiv.org/abs/2302.12173">Greshake 2023 ↗</a></span></span>),防御是 **host/agent** 的责任(权限最小化、把工具输出当不可信)。

## 6. 生产工程模式 / Production patterns

- **Subagent 编排**:主 agent 把子任务派给**上下文隔离**的 subagent(各自有限工具集),并行或顺序分解 → 防主上下文爆炸、防工具表过长。
- **工具检索**:工具池 100+ 时,**不**把全部 schema 塞进 prompt,而是按当前子任务用 embedding **top-k 检索**相关工具再注入。
- **记忆分层**:working(上下文内)/ episodic(外部存历史轨迹)/ 检索回灌;长程必备。
- **Budget guard**:多维预算(token + 步数 + 工具调用数 + wall-time),任一超阈值就强制收敛 / 终止,防 looping 烧钱。

> ❌ **误区:** 「subagent = 多 agent 系统」。**Subagent 是层级分解**(一个主目标拆给下属,上下文隔离),**multi-agent debate / 协作**是多个**对等** agent 各持视角再聚合——两者目标、通信结构都不同(后者见 agentic 篇多智能体信用)。

## 7. Computer-use / GUI agent

动作空间从「文本工具」变成**截图 → 坐标点击 / 键盘输入**,直接操作真实 GUI。两大瓶颈:

1. **grounding**:把语义意图(「点登录按钮」)映射到**精确像素坐标**——视觉定位不准是主要错误源;实务常**优先用 accessibility tree**(结构化元素树)而非纯截图像素。
2. **long-horizon**:GUI 任务步数长(开 app→导航→填表→提交),误差复利严重(见 §9)。

评测场见 §8 的 **OSWorld**(桌面)与 **WebArena**(网页)。

## 8. 评测 / Benchmarks

> ⚠️ model SOTA 在这些 benchmark 上**快变且易受训练污染**,本页**只列原文 human baseline + 测什么**;当前 SOTA 请查各自官方 leaderboard,并注意污染与 scaffold 版本差异。

| Benchmark | 测什么 | human baseline(原文) |
|---|---|---|
| **SWE-bench**<span class="cite-wrap"><a class="cite" id="fnref-12" href="#ref-12">12</a><span class="cite-note">2294 个真实 GitHub issue,改代码库使测试通过。<a href="https://arxiv.org/abs/2310.06770">Jimenez 2023 ↗</a></span></span> | 真实 GitHub issue 修复(改代码库过单测),2294 任务 | 原文无 human 解题率;发布时最强模型仅约 2%(Claude 2)——显示其难度 |
| **SWE-bench Verified**<span class="cite-wrap"><a class="cite" id="fnref-13" href="#ref-13">13</a><span class="cite-note">OpenAI 2024-08 人工核验的 500 题子集,排除不可解/测试过严的题。<a href="https://openai.com/index/introducing-swe-bench-verified/">OpenAI 2024 ↗</a></span></span> | 同上的 500 题**人工核验**子集(更干净) | 无报告 human 解题率 |
| **GAIA**<span class="cite-wrap"><a class="cite" id="fnref-14" href="#ref-14">14</a><span class="cite-note">通用 AI 助手:需推理+多模态+web+工具,分三难度级。<a href="https://arxiv.org/abs/2311.12983">Mialon 2023 ↗</a></span></span> | 通用助手(推理 + 多模态 + web + 工具),三难度级 | **92%**(L1 93.9 / L2 91.8 / L3 87.3),原文标注者 |
| **OSWorld**<span class="cite-wrap"><a class="cite" id="fnref-15" href="#ref-15">15</a><span class="cite-note">369 个真实 OS 上的开放式 computer-use 任务(多 app/网页)。<a href="https://arxiv.org/abs/2404.07972">Xie 2024 ↗</a></span></span> | 真实 OS computer-use,369 任务 | **72.36%** |
| **WebArena**<span class="cite-wrap"><a class="cite" id="fnref-16" href="#ref-16">16</a><span class="cite-note">812 个真实网站上的长程任务(电商/论坛/代码库/CMS)。<a href="https://arxiv.org/abs/2307.13854">Zhou 2023 ↗</a></span></span> | 真实 web 长程任务,812 任务 | **78.24%** |
| **AgentBench**<span class="cite-wrap"><a class="cite" id="fnref-17" href="#ref-17">17</a><span class="cite-note">8 个交互环境(OS/DB/KG/游戏/web)综合评 LLM-as-agent。<a href="https://arxiv.org/abs/2308.03688">Liu 2023 ↗</a></span></span> | 8 环境综合评 LLM-as-agent | 无(设计为模型间对比) |
| **τ-bench**<span class="cite-wrap"><a class="cite" id="fnref-18" href="#ref-18">18</a><span class="cite-note">工具-agent-用户多轮、带策略约束的客服任务;引入 pass^k 可靠性指标。<a href="https://arxiv.org/abs/2406.12045">Yao 2024 ↗</a></span></span> | 多轮、带策略约束的工具-用户交互(客服) | 无;关键贡献是 **pass^k** 可靠性指标 |
| **MLE-bench**<span class="cite-wrap"><a class="cite" id="fnref-19" href="#ref-19">19</a><span class="cite-note">75 个 Kaggle ML 工程竞赛,按奖牌率(铜/银/金)评 agent。<a href="https://arxiv.org/abs/2410.07095">Chan 2024 ↗</a></span></span> | Kaggle ML 工程,75 竞赛,按奖牌率 | 按 Kaggle leaderboard 百分位 |

## 9. 成本与可靠性 / Cost & reliability

- **$O(T^2)$ 成本**:每步都重读全 context,而 context 随步数线性增长($L_t \propto t$),故总 token $\propto \sum_{t=1}^{T} t = O(T^2)$。这是长程 agent 的核心成本来源。缓解:上下文压缩 / 摘要、KV 驱逐、子任务分解到隔离 subagent、prompt 缓存。
- **串行延迟下界**:多步之间有数据依赖,**并行工具**能省**单步内**的等待,但打不破**跨步**的串行延迟下界——8 步任务再怎么并行也至少串 8 个 LLM 解码。
- **pass@k vs pass^k**:$\text{pass@}k$ = $k$ 次尝试**至少 1 次**成功(能力上界,偏乐观);$\text{pass}^k$ = $k$ 次**全部**成功(可靠性)。**agent 部署看 pass^k**——客服 / 代码 agent 跑 10 次有 1 次乱删库就不可用,τ-bench 正是用 pass^k 暴露这种不稳定。

## 10. 失败模式 taxonomy / Failure modes

| # | 失败模式 | 机制 | 缓解 |
|---|---|---|---|
| 1 | 幻觉工具调用 | 调不存在的工具/参数,或不设 stop 自己编 Observation | JSON schema 校验 + stop sequence + 工具白名单 |
| 2 | loop / 僵局 | 反复同一动作不前进 | 步数预算 + loop detection + 强制 final |
| 3 | lost-in-the-middle<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a><span class="cite-note">长上下文中段的关键信息易被忽略,呈 U 形。<a href="https://arxiv.org/abs/2307.03172">Liu 2024 ↗</a></span></span> | 长上下文中段信息被忽略(U 形) | 关键信息置首尾 + 摘要 + 检索 |
| 4 | 工具过用 / 欠用 | 该调不调、或不该调乱调 | reward / SFT shaping + 工具检索 |
| 5 | 工具输出注入 | 工具返回里藏指令劫持 agent | 工具输出当不可信 + 权限最小化 + host 防御 |
| 6 | benchmark reward hacking | 钻评测漏洞而非真解题 | 终端可验证 + 对抗测试集 + 防污染 |

---

## 分层面试题 / Stratified follow-ups

### L1 基础

<details class="qa"><summary>1. agent 与 chatbot 的本质区别是什么?给 LLM 接一个搜索 API 就算 agent 吗?</summary>

答:本质区别是 **闭环 + 自主多步 + 对外部状态的改变**——agent 能根据观测决定下一步、反复行动直到目标达成,且动作可改变外部世界状态;chatbot 是一问一答。单纯给 LLM 接搜索 API 做一次检索增强(RAG)**还不算 agent**(仍是单轮);只有当它能基于工具返回**自主决定**是否继续查、查什么、何时停止,才进入 agent 范畴。

**追问：** agent = policy + 什么? → policy(LLM)+ 工具 I/O + 记忆 + 控制循环;把 LLM 看作策略 $\pi_\theta(a\mid h)$,在「观测→动作→新观测」循环里运行。

</details>

<details class="qa"><summary>2. ReAct 的 Thought/Action/Observation 三段各是什么?为何比纯 CoT 少幻觉?</summary>

答:Thought = 推理,Action = 调工具,Observation = 工具返回(环境注入)。纯 CoT 在自己输出上滚动,中间事实无法被外部校正;ReAct 每步把**真实工具返回**作为下一步条件,推理被 grounding,错误事实下一轮即可纠偏。

**追问：** ReAct 落地最常见的 bug 是什么? → stop-token footgun:不把 `Observation:` 设为 stop sequence,模型会自己续写一段 `Observation: …` 幻觉工具返回,而非停下等环境注入真实结果。

</details>

<details class="qa"><summary>3. function calling 与 ReAct 是两种对立的 agent 吗?</summary>

答:不是,层级不同。function calling 是工具调用的**结构化格式**(模型微调后输出 JSON schema 的 `{name, arguments}`);ReAct 是**推理-行动的循环模式**(prompt 范式)。两者可组合:用 ReAct 循环,每步用 function calling 发工具调用。

**追问：** 训练时两种格式在 label masking 上有什么差别? → 集合相同(都掩工具返回 token),区别在 JSON 格式里固定模板部分(`{"name":`、标点)是 schema 而非决策,过度训练会浪费梯度在背模板上(见 react drill / agentic Q11)。

</details>

<details class="qa"><summary>4. MCP 解决什么问题?它能保证 agent 安全吗?</summary>

答:MCP(Model Context Protocol,Anthropic 2024-11)标准化**模型↔外部工具/数据**的连接(纵向):client-server + JSON-RPC 2.0 + 三 primitive(tools/resources/prompts)。它**不保证安全**——协议本身不防 prompt injection,工具返回里的恶意指令需由 host/agent 防御。

**追问：** MCP 与 A2A 的分工? → MCP 纵向(agent↔工具/数据),A2A(Google 2025)横向(agent↔agent,用 agent card + task 状态机互通)。

</details>

<details class="qa"><summary>5. pass@k 与 pass^k 有什么区别?agent 部署该看哪个?</summary>

答:$\text{pass@}k$ = $k$ 次尝试至少 1 次成功(衡量**能力上界**,偏乐观);$\text{pass}^k$ = $k$ 次全部成功(衡量**可靠性**)。agent 部署看 **pass^k**——一个客服/代码 agent 跑 10 次有 1 次闯祸就不可用。τ-bench 正是用 pass^k 暴露这种不稳定。

**追问：** 为什么长程 agent 的 pass^k 会远低于 pass@k? → 每步成功率 $p<1$,$k$ 次全成的概率随步数 / 次数指数衰减;长程任务步数多,任一步翻车就整条失败,故可靠性远低于「至少一次能做到」。

</details>

<details class="qa"><summary>6. 为什么 agent 的推理成本通常是 O(T²)?</summary>

答:每一步都要重读整个上下文,而上下文随步数线性增长($L_t \propto t$,历史全拼接),所以总 token 量 $\propto \sum_{t=1}^T t = O(T^2)$。这是长程 agent 成本的主来源,也是上下文管理(压缩/摘要/驱逐)的动机。

**追问：** 并行工具调用能把这个降到 O(T) 吗? → 不能。并行省的是**单步内**多个独立工具的等待,降的是延迟、不是总 token;跨步的串行依赖与 context 累积仍在,成本量级不变。

</details>

### L2 进阶

<details class="qa"><summary>7. Plan-and-Execute 与 ReAct 的取舍是什么?何时纯 plan-and-execute 会失败?</summary>

答:ReAct 逐步决策(reactive),灵活但无全局视野;Plan-and-Execute 先生成完整计划再执行(执行器可换模型),全局一致但计划可能过时。纯 plan-and-execute 在**环境不确定、中途状态大变**(工具返回出乎预料)时失败——开局定死的静态计划跟不上变化。生产常用「高层 plan + 每步 ReAct」混合 + plan repair。

**追问：** plan repair 有哪些做法? → Reflexion 式语言反思后 replan、ToT 式搜索备选计划、或逐步检测偏离后局部 replan。

</details>

<details class="qa"><summary>8. Toolformer 如何在没有人工标注的情况下学会调用 API?</summary>

答:自监督 + **utility filter**。在文本里随机插入候选 API 调用 → 执行得到返回 → 只保留「插入该调用及其返回后,模型预测**后续 token 的损失显著下降**」的样本做 SFT。即用「调用是否真的帮到后续预测」作为效用信号,自动筛掉无用或位置错误的调用,无需人工标谁该调、何时调。

**追问：** 这个 utility filter 的本质判据是什么? → 比较「有该 API 返回」vs「无 / 空返回」两种条件下后续 token 的加权损失,只有前者明显更低才保留——本质是「这个工具调用降低了多少后续困惑度」。

</details>

<details class="qa"><summary>9. 结构化 function calling 的 parallel tool calls 有什么前提约束?</summary>

答:一次返回多个工具调用要求这些调用**幂等 + 相互独立**(无数据依赖):若调用 B 需要调用 A 的结果,就不能并行,必须串行等 A 返回。并行只适用于「查天气 + 查汇率」这种彼此无关的调用;有依赖的链式调用要分轮。

**追问：** 为什么并行不能打破长程 agent 的串行延迟下界? → 并行省的是单步内独立调用的等待,跨步的数据依赖(下一步要用上一步结果)仍是串行的,T 步任务至少串 T 个 LLM 解码。

</details>

<details class="qa"><summary>10. subagent 编排与 multi-agent debate 有什么区别?</summary>

答:subagent 编排是**层级分解**——主 agent 把子任务派给上下文隔离、工具受限的下属,目标单一、通信是「派活↔交结果」;multi-agent debate/协作是多个**对等** agent 各持视角、互相质疑再聚合,目标是用多样性提升正确率。两者结构(层级 vs 对等)、通信、目的都不同。

**追问：** subagent 上下文隔离主要解决什么? → 防主 agent 上下文爆炸(子任务的中间 token 不回灌主线)+ 工具表过长(每个 subagent 只挂相关工具);代价是跨 subagent 的信息共享需显式传递。

</details>

<details class="qa"><summary>11. 工具池有 100+ 个工具时,怎么管理才不撑爆上下文?</summary>

答:不把全部工具 schema 塞进 prompt(既撑爆上下文又降低选择准确率),而是**工具检索**:把每个工具的描述向量化,按当前子任务 query 做 embedding **top-k 检索**,只注入最相关的几个工具 schema。本质是把「工具选择」从一次性全暴露改成检索召回。

**追问：** 工具检索的主要失败模式是什么? → 召回不全(该用的工具没被检索到 → agent 无法完成)与描述歧义(两个相似工具被混淆);缓解靠更好的工具描述 + 增大 k + 必要时分层检索。

</details>

<details class="qa"><summary>12. computer-use / GUI agent 的两大瓶颈是什么?为什么常优先用 accessibility tree?</summary>

答:① **grounding**——把语义意图映射到精确像素坐标,视觉定位不准是主要错误源;② **long-horizon**——GUI 任务步数长、误差复利严重。常优先用 **accessibility tree**(结构化元素树,带 role/label/坐标)而非纯截图,因为结构化元素比像素更可靠地定位「那个按钮」,绕开了部分 grounding 误差。

**追问：** 既然 accessibility tree 更可靠,为何还需要截图? → 很多界面(canvas、自定义渲染、游戏)无可用 a11y tree,或 tree 不完整;截图是通用兜底,实务常二者融合。

</details>

<details class="qa"><summary>13. 要评测一个 coding agent 和一个 web agent,各选什么 benchmark,为什么?</summary>

答:coding agent → **SWE-bench / SWE-bench Verified**(真实 GitHub issue 改代码过单测,Verified 是人工核验的干净子集);web agent → **WebArena**(真实网站长程任务,有 78.24% human baseline)或 **OSWorld**(若是桌面 computer-use)。选择依据是「动作空间与任务分布是否匹配目标场景」。

**追问：** 引用这些 benchmark 的 SOTA 数字时要带什么 caveat? → model SOTA 快变 + 训练污染 + scaffold 版本差异(SWE-bench 因此出了人工核验的 Verified 子集);只能引官方 leaderboard 当前值并标日期,不能把二手数字当稳定事实。

</details>

### L3 深入

<details class="qa"><summary>14. 设计一个长程 agent 的上下文管理:O(T²) 成本、lost-in-the-middle、误差复利三个问题如何协同缓解?</summary>

答:三者同源于「上下文随步数膨胀」,需组合拳:① 对 **O(T²) 成本**——旧轮**摘要化** + KV 驱逐(留 sink+近窗)+ 把独立子任务派给上下文隔离的 subagent,把单条长上下文拆成多条短的;② 对 **lost-in-the-middle**(中段信息被忽略,U 形)——关键信息(目标、约束)**置于首尾**、用检索把相关历史**即时召回**到近窗,而非靠模型在长中段里找;③ 对**误差复利** $p^T$——缩短有效 horizon(分解 + 每子任务可验证里程碑)、加 loop detection 与 budget guard 早停。三者协同:摘要 + 子任务隔离同时降成本与 horizon,检索 + 首尾置顶同时治 lost-in-the-middle 与 grounding。

**追问：** 摘要化本身会引入什么新风险,如何权衡? → 有损压缩可能丢掉日后才用得上的关键早期信息(且若训练时见全历史、推理时才压缩,会产生训练-推理状态分布不一致);权衡是对「可能被回看」的内容保留原文指针 / 可检索副本,只摘要低价值轮。

</details>

<details class="qa"><summary>15. 工具输出的 prompt injection 威胁模型是什么?给出纵深防御。</summary>

答:威胁模型(indirect prompt injection,Greshake 2023):攻击者把恶意指令藏在 agent 会读到的**外部内容**里(网页、检索结果、工具返回、文件),agent 把它当指令执行——可被诱导泄露上下文、滥用高权限工具、或对外发起请求。**纵深防御**:① 把所有工具/外部返回**标注为不可信数据**(与系统指令隔离,不当指令执行);② **权限最小化**(每个工具最小作用域,危险操作要二次确认);③ 输出侧约束(对外发送 / 删除等高危动作加 host 级策略门);④ 监控异常工具调用序列。关键认知:**协议(MCP)不负责防注入,是 host/agent 的责任**。

**追问：** 为什么「让模型自己判断指令是否可信」不是可靠防御? → 这把安全边界放回到可被同一注入攻破的模型内部;可靠防御应在**模型之外**用确定性的权限/策略层(白名单、作用域、人工确认)兜底,而非依赖模型自律。

</details>

<details class="qa"><summary>16. benchmark 上的 reward hacking 与数据污染,如何设计抗 hack 的 agent 评测?</summary>

答:两类问题——① **reward hacking**:agent 钻评测实现漏洞(改测试文件、mock 输出、空函数过 CI)而非真解题;② **污染**:测试样本进了训练数据,虚高。抗 hack 评测:用**终端可验证**且难伪造的成功判据(隐藏的额外单测、环境最终状态校验,而非 agent 可见的断言)、**对抗测试集轮换** / living benchmark(定期换题防记忆)、**人工核验子集**(如 SWE-bench Verified 排除可作弊/不可解题)、报告 **pass^k**(防靠多次抽样刷 pass@k)、并公开评测 harness 以便复现。

**追问：** 为什么「公开 leaderboard 数字越来越高」不能直接当成 agent 能力进步? → 高分可能来自污染、scaffold 工程或对该 benchmark 的过拟合;要看是否在**新发布、防污染**的 living benchmark 与 pass^k 可靠性指标上同步提升,并核实 harness 与日期。

</details>

---

## 参考文献 / References

> 均为承重方法的原始出处,已逐条 web 核对(标题 + arXiv ID / 官方 URL)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Yao et al. <em>ReAct: Synergizing Reasoning and Acting in Language Models</em>. ICLR 2023. <a href="https://arxiv.org/abs/2210.03629">arXiv:2210.03629</a> — think→act→observe 范式. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Wei et al. <em>Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</em>. NeurIPS 2022. <a href="https://arxiv.org/abs/2201.11903">arXiv:2201.11903</a> — CoT 推理. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Wang et al. <em>Plan-and-Solve Prompting</em>. ACL 2023. <a href="https://arxiv.org/abs/2305.04091">arXiv:2305.04091</a> — 先规划再执行. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Yao et al. <em>Tree of Thoughts: Deliberate Problem Solving with LLMs</em>. NeurIPS 2023. <a href="https://arxiv.org/abs/2305.10601">arXiv:2305.10601</a> — 思维树搜索. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Shinn et al. <em>Reflexion: Language Agents with Verbal Reinforcement Learning</em>. NeurIPS 2023. <a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a> — 语言反思 / episodic memory. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Schick et al. <em>Toolformer: Language Models Can Teach Themselves to Use Tools</em>. NeurIPS 2023. <a href="https://arxiv.org/abs/2302.04761">arXiv:2302.04761</a> — 自监督工具使用 + utility filter. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Anthropic. <em>Model Context Protocol (MCP)</em>. 2024-11. <a href="https://modelcontextprotocol.io">modelcontextprotocol.io</a> — 模型↔工具/数据标准(纵向). <a href="#fnref-7">↩</a></li>
<li id="ref-8">Google. <em>Agent2Agent Protocol (A2A)</em>. 2025-04(后归 Linux Foundation). <a href="https://a2a-protocol.org/latest/">a2a-protocol.org</a> — agent↔agent 互通(横向). <a href="#fnref-8">↩</a></li>
<li id="ref-9">OpenAI. <em>Function calling and other API updates</em>. 2023-06-13. <a href="https://openai.com/index/function-calling-and-other-api-updates/">openai.com</a> — 结构化 JSON Schema 工具调用. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Liu et al. <em>Lost in the Middle: How Language Models Use Long Contexts</em>. TACL 2024. <a href="https://arxiv.org/abs/2307.03172">arXiv:2307.03172</a> — 长上下文中段信息被忽略. <a href="#fnref-10">↩</a></li>
<li id="ref-11">Greshake et al. <em>Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection</em>. 2023. <a href="https://arxiv.org/abs/2302.12173">arXiv:2302.12173</a> — 间接提示注入. <a href="#fnref-11">↩</a></li>
<li id="ref-12">Jimenez et al. <em>SWE-bench: Can Language Models Resolve Real-World GitHub Issues?</em> ICLR 2024. <a href="https://arxiv.org/abs/2310.06770">arXiv:2310.06770</a> — 真实代码修复评测. <a href="#fnref-12">↩</a></li>
<li id="ref-13">OpenAI. <em>Introducing SWE-bench Verified</em>. 2024-08. <a href="https://openai.com/index/introducing-swe-bench-verified/">openai.com</a> — 500 题人工核验子集. <a href="#fnref-13">↩</a></li>
<li id="ref-14">Mialon et al. <em>GAIA: a benchmark for General AI Assistants</em>. ICLR 2024. <a href="https://arxiv.org/abs/2311.12983">arXiv:2311.12983</a> — 通用助手评测(human 92%). <a href="#fnref-14">↩</a></li>
<li id="ref-15">Xie et al. <em>OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments</em>. NeurIPS 2024. <a href="https://arxiv.org/abs/2404.07972">arXiv:2404.07972</a> — computer-use 评测(human 72.36%). <a href="#fnref-15">↩</a></li>
<li id="ref-16">Zhou et al. <em>WebArena: A Realistic Web Environment for Building Autonomous Agents</em>. ICLR 2024. <a href="https://arxiv.org/abs/2307.13854">arXiv:2307.13854</a> — web agent 评测(human 78.24%). <a href="#fnref-16">↩</a></li>
<li id="ref-17">Liu et al. <em>AgentBench: Evaluating LLMs as Agents</em>. ICLR 2024. <a href="https://arxiv.org/abs/2308.03688">arXiv:2308.03688</a> — 8 环境综合评测. <a href="#fnref-17">↩</a></li>
<li id="ref-18">Yao et al. <em>τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains</em>. 2024. <a href="https://arxiv.org/abs/2406.12045">arXiv:2406.12045</a> — 多轮工具-用户 + pass^k. <a href="#fnref-18">↩</a></li>
<li id="ref-19">Chan et al. <em>MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering</em>. ICLR 2025. <a href="https://arxiv.org/abs/2410.07095">arXiv:2410.07095</a> — Kaggle ML 工程评测. <a href="#fnref-19">↩</a></li>
</ol>
