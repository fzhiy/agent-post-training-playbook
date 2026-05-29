# 🤖 Agent Post-Training Playbook

> **把 LLM 后训练延伸到 agent —— agentic / long-horizon RL · 持续 / 终身学习 · 自我改进 的双语自学 & 面试手册。**
> 每个主题 = 公式推导 + from-scratch PyTorch + 分层面试题(L1/L2/L3),外加可运行的「手撕」drill。是 [post-training-playbook](https://github.com/fzhiy/post-training-playbook) 的**姊妹篇**,交付格式同源(借鉴 [ARIS-in-AI-Offer](https://github.com/wanshuiyin/ARIS-in-AI-Offer))。
>
> Agent post-training playbook: AI-assisted bilingual (中/EN) study & interview-prep notes and drills on RL, continual learning, and self-improvement for LLM agents. A sibling to **post-training-playbook**.

🔗 **在线阅读 / Read online:** <https://ac.fzhiy.net/agent-post-training-playbook/> · MIT · AI-assisted · WIP

## ⚠️ 诚信声明 / Honesty disclaimer

这是 **学习笔记**,**不是作者的研究成果**。本手册整理的是 agent 后训练前沿——其中**只有「持续 / 终身学习」**与作者已发表工作(**联邦持续微调** Fed-TaLoRA)真正相关;**agentic RL、self-evolving 等是作者正在学习 / 跟踪的前沿,并非已发表研究**。作者的一作论文见[学术主页](https://ac.fzhiy.net/)。

> These are **study notes, not the author's research**. Of the topics here, only *continual / lifelong learning* overlaps the author's published work (federated continual fine-tuning); *agentic RL* and *self-evolving agents* are frontiers being studied, not claimed research.

## 🚀 快速开始 / Quick start

打开 **<https://ac.fzhiy.net/agent-post-training-playbook/>**(可搜索、响应式、手机可读),先看 **[Roadmap](https://ac.fzhiy.net/agent-post-training-playbook/cheatsheet-00-roadmap.html)** 按序刷。

## 📂 内容 / Contents (MVP)

三条轴,各一篇 cheatsheet + 一个 drill:

1. **agentic & long-horizon RL** — 多轮 / 工具使用 RL、长程信用分配、稀疏 / 延迟奖励、RLVR→agentic、奖励 / 环境设计。
2. **continual & lifelong learning** — 灾难性遗忘、replay / EWC / 参数隔离、稳定-可塑、保持率、continual alignment。
3. **self-improving LLMs** — STaR / ReST / RFT、self-rewarding、self-play、reflection / 自我纠正;何时会崩(reward hacking / model collapse)。

- **`cheatsheets/`** — 上述三主题(公式 + from-scratch + L1/L2/L3 题 + 折叠简答 + 「深挖 / Deep-dive」高阶答疑)。
- **`drills/`** — 可运行 from-scratch + 测试:多轮信用分配 · replay/EWC toy · Reflexion 循环。
- **`docs/`** — 构建产物静态站点(marked + KaTeX + highlight.js **构建时**渲染,**零运行时 CDN**,国内直连 / 可离线)。

> **WIP**:章节陆续填充;数字 / 结论以原论文为准,不确定即标注(见 [CONTRIBUTING](CONTRIBUTING.md))。

## 🔧 构建 / Build

```bash
npm install      # marked + katex + highlight.js(仅构建时用)
node build.js    # 读 cheatsheets/ + drills/ → 生成 docs/
```

## 🔗 姊妹仓库 / Sibling

[**post-training-playbook**](https://github.com/fzhiy/post-training-playbook)(<https://ac.fzhiy.net/post-training-playbook/>)—— 后训练**主线**:SFT / RM / RLHF / PPO / DPO / GRPO / PEFT / 评测。本仓库是它在 **agent** 方向的延伸。

## License

[MIT](LICENSE) © 2026 Feng Yu
