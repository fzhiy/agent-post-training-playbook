# 📍 学习路径 / Roadmap

> 本手册是 [post-training-playbook](https://github.com/fzhiy/post-training-playbook) 的姊妹篇,聚焦**后训练在 agent 上的延伸**。建议先过完后训练主线(SFT→DPO→GRPO、reward modeling、PEFT)再来。
> 每个主题:先读 **cheatsheet** 理解 → 做对应 **drill** 手撕 → 用页内 L1/L2/L3 自测。

> ⚠️ **学习笔记,非作者研究成果**;详见 README 诚信声明。

## 总览 / Tracks

| # | 主题 / Track | Cheatsheet 题解 | 手撕 / Drill | 状态 |
|---|---|---|---|---|
| 0 | 前置 / Prereq | 见姊妹仓库 [post-training-playbook](https://ac.fzhiy.net/post-training-playbook/) | PPO · GRPO · RLVR · RM · PEFT | — |
| 1 | Agentic & Long-horizon RL | [题解](cheatsheet-agentic-and-long-horizon-rl.html) | [turn-credit-assignment](drill-turn-credit-assignment.html) | ✅ |
| 2 | Continual & Lifelong Learning | [题解](cheatsheet-continual-and-lifelong-learning.html) | [ewc-replay](drill-ewc-replay.html) | ✅ |
| 3 | Self-improving LLMs | [题解](cheatsheet-self-improving-llms.html) | [self-refine-loop](drill-self-refine-loop.html) | ✅ |
| 🚧 | 规划中 / Planned | agent-foundations · agent-evaluation · agentic-rl-infra · agent-safety | react-tool-call-loop | 规划 |

## 0 · 前置 / Prereq
- 后训练主线(见姊妹仓库):PPO / GRPO / RLVR、reward modeling、PEFT。

## 1 · Agentic & Long-horizon RL
- [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) — 多轮 / 工具使用 RL、长程信用分配(turn vs trajectory)、RLVR→agentic、PRM/ORM、观测 token 掩码。
- 手撕:[turn-credit-assignment](drill-turn-credit-assignment.html) — 组相对优势 + 掩码 PG。

## 2 · Continual & Lifelong Learning
- [continual-and-lifelong-learning](cheatsheet-continual-and-lifelong-learning.html) — 灾难性遗忘、正则(EWC/SI/MAS)/ replay(GEM/A-GEM/DER)/ 参数隔离、AA/BWT/FWT、continual alignment / alignment tax。
- 手撕:[ewc-replay](drill-ewc-replay.html) — Fisher + EWC 惩罚 + replay,验证抗遗忘。

## 3 · Self-improving LLMs
- [self-improving-llms](cheatsheet-self-improving-llms.html) — STaR / ReST / RFT、self-rewarding、self-play(SPIN)、RLAIF、反思(Reflexion/Self-Refine);自改进闭环与崩溃模式。
- 手撕:[self-refine-loop](drill-self-refine-loop.html) — 生成→批评→修订迭代,验证分数单调。

---

**复习法**:每题复习后标 ✅ 熟练 / ⚠️ 模糊 / ❌ 不会;之后只重刷 ⚠️/❌。
