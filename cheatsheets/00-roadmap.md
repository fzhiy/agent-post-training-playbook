# 📍 学习路径 / Roadmap

> 本手册是 [post-training-playbook](https://github.com/fzhiy/post-training-playbook) 的姊妹篇,聚焦**后训练在 agent 上的延伸**。建议先过完后训练主线(SFT→DPO→GRPO、reward modeling、PEFT)再来。
> 每个主题:先读 **cheatsheet** 理解 → 做对应 **drill** 手撕 → 用页内 L1/L2/L3 自测。
>
> ⚠️ **学习笔记,非作者研究成果**;详见 README 诚信声明。

## 0 · 前置 / Prereq
- 后训练主线(见姊妹仓库):PPO / GRPO / RLVR、reward modeling、PEFT。

## 1 · Agentic & Long-horizon RL
- 多轮 / 工具使用 RL、长程信用分配(turn-level vs trajectory-level advantage)、稀疏 / 延迟奖励、RLVR→agentic、环境与奖励设计。
- 手撕(规划):多轮信用分配 / trajectory advantage。
- *(WIP)*

## 2 · Continual & Lifelong Learning
- 灾难性遗忘、replay / 正则(EWC)/ 参数隔离、稳定-可塑权衡、保持率 / 遗忘度量、continual alignment / alignment tax。
- 手撕(规划):replay-buffer / EWC penalty toy。
- *(WIP)*

## 3 · Self-improving LLMs
- STaR / ReST / RFT(拒绝采样微调)、self-rewarding、self-play、reflection / 自我纠正;自造数据训练闭环;何时会崩(reward hacking / model collapse)。
- 手撕(规划):Reflexion / self-refine 循环。
- *(WIP)*

---

**复习法**:每题复习后标 ✅ 熟练 / ⚠️ 模糊 / ❌ 不会;之后只重刷 ⚠️/❌。
