<!--
  新 cheatsheet 骨架模板。复制内容 → 新建 cheatsheets/<slug>.md 填写。
  本模板放在仓库根(不在 cheatsheets/ 内),所以不会被 render-playbook 引擎当作一篇发布。

  深度标尺(见 interview/agent-playbook-depth-audit.md):
    • 折叠面试题 ≥ 25 题:L1 ~4 · L2 5–7 · L3 5–8;每题必须带「追问」(见下方 .qa 范式)。
    • L3 折叠答案 ≥ 6 句,含一个推导或量化对比;否则升级为「深挖」专属。
    • 深挖 / Deep-dive:6–8 条多段落答疑,可含伪码/表格;前沿话题至少 1 条带 arXiv 逐字引用。
    • 基础话题写到「面试官第二轮追问必答」;前沿话题写到「能在白板上推导核心 tradeoff + 一个 failure mode + mitigation」。
  诚信门:学习笔记非作者研究;不自引 BoHA/Fed-TaLoRA;不编 benchmark 数字;benchmark 数必带污染/复现 caveat;arXiv ID 先核实再写。
-->

# <emoji> 主题 / Topic

> 一句话定位:这个主题解决什么问题、在 post-training→agent 链条里的位置。
>
> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。数字 / 结论以原论文为准,不确定处标注。

<!-- ⚠️ 引导的 blockquote 会被 build.js 渲染成黄色 callout-warn。其它:💡/📝 蓝 info · ✅/🔒 绿 good · ❌/🚨 红 bad。 -->

## 0. TL;DR

- 5–7 条要点,每条一句话,先给结论。
- 覆盖:核心问题 · 主流方法 · 关键 tradeoff · 何时会崩 · 与姊妹主线的关系。

## 1. <核心概念> / <Concept>

正文:中文叙述 + 术语保留英文(LoRA、GRPO…)。公式用 `$...$` / `$$...$$`(build.js 用 KaTeX 构建时预渲染):

$$\text{示例公式} = \dots$$

对比表(slash 列表 ≤3 项;参数化对比优先用表):

| 维度 | 方法 A | 方法 B |
|---|---|---|
| … | … | … |

## 2 … N. <更多小节>

每节:动机 → 机制(公式/伪码)→ tradeoff / failure mode。代码块用 ```python(构建时 highlight.js 预渲染)。

## 面试题 / Interview Q (L1/L2/L3)

<!-- 每题一个 <details class="qa">;summary 是问题,body 是简答 + 追问。空行约定保证 body 内 markdown 正常渲染。 -->

### L1 基础
<details class="qa"><summary>1. <问题>?</summary>

<简答 3–5 句。>

**追问：** <更深一层的问题> → <1–2 句答案>
</details>

### L2 进阶
<!-- 5–7 题,答案含 1 个 inline formula 或 2 行对比;追问 escalate 到 L3 或 design tradeoff。 -->

### L3 深入
<!-- 5–8 题,答案 ≥6 句含推导或量化;追问 probe 实现细节或开放式 system-design。 -->

## 深挖 / Deep-dive

<!-- 6–8 条长答疑,每条多段落。前沿话题至少 1 条带 arXiv 逐字引用 + ID。 -->

### D1. <进阶问题>

多段落分析,可含伪码/表格/量化结论。

## References

<!-- 行内引用范式:正文里写 [N] 上标跳转 + 宽屏右页边侧边注;底部列书目 + ↩ 返回。 -->
<!-- 行内:<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">一句话注释。<a href="https://arxiv.org/abs/XXXX.XXXXX">作者 年份 ↗</a></span></span> -->

> 均为承重方法的原始出处,已逐条核对(标题 + arXiv ID)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">作者. <em>标题</em>. 年份. <a href="https://arxiv.org/abs/XXXX.XXXXX">arXiv:XXXX.XXXXX</a> — 一句话说明承重点. <a href="#fnref-1">↩</a></li>
</ol>
