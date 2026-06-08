# 📍 学习路径 / Roadmap

> 本手册是 [post-training-playbook](https://github.com/fzhiy/post-training-playbook) 的姊妹篇,聚焦**后训练在 agent 上的延伸**。建议先过完后训练主线(SFT→DPO→GRPO、reward modeling、PEFT)再来。
> 每个主题:先读 **cheatsheet** 理解 → 做对应 **drill** 手撕 → 用页内 L1/L2/L3 自测。

> ⚠️ **学习笔记,非作者研究成果**;详见 README 诚信声明。

## 总览 / Tracks

| # | 主题 / Track | Cheatsheet 题解 | 手撕 / Drill | 状态 |
|---|---|---|---|---|
| 0 | 前置 / Prereq | 见姊妹仓库 [post-training-playbook](https://ac.fzhiy.net/post-training-playbook/) | PPO · GRPO · RLVR · RM · PEFT | — |
| 1 | Agent Foundations | [题解](cheatsheet-agent-foundations.html) | [react-tool-call-loop](drill-react-tool-call-loop.html) | ✅ |
| 2 | Agent Evaluation | [题解](cheatsheet-agent-evaluation.html) | —(规划 benchmark-harness-audit) | ✅ |
| 3 | Agentic & Long-horizon RL | [题解](cheatsheet-agentic-and-long-horizon-rl.html) | [turn-credit-assignment](drill-turn-credit-assignment.html) | ✅ |
| 4 | Agentic RL Infrastructure | [题解](cheatsheet-agentic-rl-infra.html) | — | ✅ |
| 5 | Continual & Lifelong Learning | [题解](cheatsheet-continual-and-lifelong-learning.html) | [ewc-replay](drill-ewc-replay.html) | ✅ |
| 6 | Self-improving LLMs | [题解](cheatsheet-self-improving-llms.html) | [self-refine-loop](drill-self-refine-loop.html) | ✅ |
| 🚧 | 规划中 / Planned | agent-safety | — | 规划 |

## 0 · 前置 / Prereq
- 后训练主线(见姊妹仓库):PPO / GRPO / RLVR、reward modeling、PEFT。

## 1 · Agent Foundations
- [agent-foundations](cheatsheet-agent-foundations.html) — agent 心智模型、ReAct、规划(Plan-Execute)、工具使用(Toolformer / function calling)、协议层(MCP/A2A)、生产工程模式、评测与失败模式。
- 手撕:[react-tool-call-loop](drill-react-tool-call-loop.html) — ReAct 循环(parse→route→observe→loop)+ SFT label masking。

## 2 · Agent Evaluation
- [agent-evaluation](cheatsheet-agent-evaluation.html) — agent 评测度量学:benchmark 两正交轴分类、污染 / 饱和 / living benchmark(SWE-bench Verified 退役)、harness 与可复现 / 方差(pass@k 估计量)、轨迹 vs 结果评测、安全 / sabotage 评测诚信门。
- 衔接:承 [agent-foundations](cheatsheet-agent-foundations.html) §8/§9 的 benchmark 与 pass@k,启 [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) 的可验证奖励。

## 3 · Agentic & Long-horizon RL
- [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) — 多轮 / 工具使用 RL、长程信用分配(turn vs trajectory)、RLVR→agentic、PRM/ORM、观测 token 掩码。
- 手撕:[turn-credit-assignment](drill-turn-credit-assignment.html) — 组相对优势 + 掩码 PG。

## 4 · Agentic RL Infrastructure
- [agentic-rl-infra](cheatsheet-agentic-rl-infra.html) — agent RL 系统工程:三池架构(Rollout/Reward/Training)、训练栈对比(verl/OpenRLHF/AReaL/SkyRL-Agent)、环境管理(沙盒池/健康检查/容错)、多轮 KV cache 显存、轨迹数据管线(Parquet 列存)、成本估算。
- 衔接:承 [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) 的算法原理,启规划中的 agent-safety 篇。

## 5 · Continual & Lifelong Learning
- [continual-and-lifelong-learning](cheatsheet-continual-and-lifelong-learning.html) — 灾难性遗忘、正则(EWC/SI/MAS)/ replay(GEM/A-GEM/DER)/ 参数隔离、AA/BWT/FWT、continual alignment / alignment tax。
- 手撕:[ewc-replay](drill-ewc-replay.html) — Fisher + EWC 惩罚 + replay,验证抗遗忘。

## 6 · Self-improving LLMs
- [self-improving-llms](cheatsheet-self-improving-llms.html) — STaR / ReST / RFT、self-rewarding、self-play(SPIN)、RLAIF、反思(Reflexion/Self-Refine);自改进闭环与崩溃模式。
- 手撕:[self-refine-loop](drill-self-refine-loop.html) — 生成→批评→修订迭代,验证分数单调。

---

**复习法**:每题复习后标 ✅ 熟练 / ⚠️ 模糊 / ❌ 不会;之后只重刷 ⚠️/❌。
