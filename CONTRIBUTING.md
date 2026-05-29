# Contributing

谢谢帮忙改进这份学习手册!这是 **AI 辅助整理的学习笔记**,难免有错——**纠错、补充、更清楚的解释都欢迎**。
Thanks for improving this playbook — it's AI-assisted study notes, so corrections and clearer explanations are very welcome.

## 怎么改 / How to edit

- 内容在 **`cheatsheets/*.md`** 和 **`drills/<topic>/`**(`from_scratch.py` + `test_*.py` + `README.md`)。
- 改完重建站点:
  ```bash
  npm install && node build.js     # 生成 docs/
  ```
- 提 PR;没空跑构建的话,直接开 issue 指出问题也行。

## 原则 / Ground rules

1. **不编数字**:不要写没核实的 benchmark 分数;涉及论文的数字以原文为准,不确定就标注。
2. **drill 要能跑**:`test_*.py` 应在有 PyTorch 的环境 `python test_*.py` 通过。
3. **公式用 LaTeX**(`$...$` / `$$...$$`);代码用带语言的围栏(```python)以便高亮。
4. 保持「中文为主 + 英文术语」的双语风格。

## 站点 / Site

纯静态(marked + KaTeX + highlight.js **构建时**渲染),**零运行时 CDN**;GitHub Pages 从 `/docs` 部署。
