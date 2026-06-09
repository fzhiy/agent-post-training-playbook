# Agent Evaluation / Agent 评测

> Agent 后训练的**度量学**:怎么测一个 agent 到底行不行。先读 [agent-foundations](cheatsheet-agent-foundations.html) 知道 agent 是什么、有哪些 benchmark 与 human baseline,再用本篇学**污染 / 饱和 / harness / 轨迹评测**这些「读懂技报数字」的素养;之后去 [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) 学怎么把可验证奖励接进 RL。

> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。benchmark 数字快变且易受污染,本页**只记原文 human baseline + 测什么 + 评测方法学**,不抄 model SOTA;凡出现的具体分数均带日期 / 来源 / caveat。

## 0. TL;DR 速查

- **agent 评测 ≠ 单轮 QA 评测**:测的是**多步轨迹**下的目标达成(execution-based 终态校验),不是单次答案与参考串匹配;多三个轴:**可靠性(pass^k)、轨迹质量、抗时间贬值**。
- benchmark 按**两正交轴**分:能力域(coding / web / GUI / tool-user / general / ML-eng)× 评判方式(execution / trajectory / LLM-judge)。human baseline 见 foundations §8,本页不重复、只做分类与方法学。
- **污染 = agent 评测头号杀手**:测试样本(GitHub issue、题解)进了训练集 → 测的是**记忆而非泛化**,分数虚高。**LiveCodeBench** 用「发布时间窗」只评 cutoff 之后的新题,从根上断污染。
- **饱和实例(2026-02,已核实)**:OpenAI 官方**停止报告 SWE-bench Verified**——分数「不再反映真实开发能力,而越来越反映模型见过多少 benchmark」;改推抗污染的 **SWE-bench Pro**(同批模型 ~70–80% → ~23–58% 量级,随分片/来源/scaffold;非稳定排名,见 §3.2)。
- **harness 决定分数**:同一模型换 scaffold / prompt / 工具集,分数可显著变化(公开报告常见十个点量级波动,随 scaffold/benchmark,非定值);报分必须报 **harness 版本**,否则比的不是 model 而是 model×harness。
- **方差**:`pass@k` 是**估计量**、有方差(temperature / seed / 环境随机都影响);只报单次数字不可复现,要报 $n$ 次采样 + 无偏估计。
- **轨迹 vs 结果**:outcome-only 漏掉「蒙对 / reward hacking」;轨迹评测看每步是否合理,但用 LLM 当 judge 评轨迹有自己的可靠性坑(校准属姊妹篇 [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html),本页只点到)。
- **可验证 ≠ 完美**:execution-based verifier 也会假阳(弱单测放过错解)/ 假阴(过严单测拒绝对解,正是 SWE-bench Verified 的缺陷题)。
- **安全评测正交于能力**:sabotage / 越权 / prompt-injection 要单列;引用「多 agent 注入攻击下的恶意执行率」「sabotage」这类安全数字**必带威胁模型 + 攻击场景 caveat**,不可当一般部署行为。

## 1. 为什么 agent 评测 ≠ 单轮评测 / Why agent-eval ≠ single-turn eval

单轮 QA 评测:输入 → 一次输出 → 与参考答案比对(EM / F1 / accuracy),一次性、近确定性。agent 评测要测的是**一条多步轨迹**:动作改变环境状态,「成功」= 终态满足目标(execution-based),不是字符串匹配。这带来单轮评测没有的几个维度(见下表):

| 维度 | 单轮 QA | Agent |
|---|---|---|
| 成功判据 | 与参考答案匹配(EM/F1) | **终态 / 单测 / 环境校验**(execution-based, verifiable) |
| 随机性 | 低(单次解码) | 高(多步采样 + 环境随机)→ 必须多跑取分布 |
| 可靠性 | accuracy | **pass^k**(k 次全成),见 foundations §9 |
| 过程 | 不看 | **轨迹质量**:步数、工具用对没、有没有 reward hack |
| 时间稳定性 | 静态 | **污染 + 饱和**使同一分数随时间贬值 |

> ❌ **误区:** 「agent 评测 = 把 benchmark 的 accuracy 跑出来」。accuracy 只是一个点估计;agent 评测真正难的是**execution-based 判据怎么设计**(防作弊)、**多次跑的可靠性**(pass^k)、以及**分数随时间是否还可信**(污染 / 饱和)。把这三件事忽略,leaderboard 数字就是误导。

## 2. benchmark 分类 / Taxonomy

human baseline 与「测什么」已在 [agent-foundations §8](cheatsheet-agent-foundations.html) 列过,这里**不重复**,改按**两个正交轴**给方法学定位:

- **轴 A · 能力域**:coding / web / GUI(computer-use)/ tool-user 多轮 / general assistant / ML 工程。
- **轴 B · 评判方式 × 交互性**:execution-based outcome(跑单测/校验终态)vs trajectory vs LLM-judge;static one-shot vs interactive environment。

| Benchmark | 能力域 | 评判方式 | 交互 | 关键设计点 |
|---|---|---|---|---|
| **SWE-bench**<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">2294 个真实 GitHub issue,改代码库使隐藏单测通过。<a href="https://arxiv.org/abs/2310.06770">Jimenez 2023 ↗</a></span></span> | coding | execution(隐藏单测) | repo 交互 | 真实 issue→PR,FAIL_TO_PASS 单测做判据 |
| **WebArena**<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">4 个自托管站点(电商/论坛/GitLab/CMS)上的 812 个长程任务;终态功能校验。<a href="https://arxiv.org/abs/2307.13854">Zhou 2023 ↗</a></span></span> | web | execution(终态校验) | 自托管沙盒 web | 可复现、可重置的真实站点副本 |
| **OSWorld**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">369 个真实 OS 上 computer-use 任务,脚本校验终态。<a href="https://arxiv.org/abs/2404.07972">Xie 2024 ↗</a></span></span> | GUI | execution(终态脚本) | 真实 OS 交互 | 多模态、跨 app,执行式判据非选择题 |
| **τ-bench**<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">工具-agent-用户多轮、带策略约束;引入 pass^k 可靠性。<a href="https://arxiv.org/abs/2406.12045">Yao 2024 ↗</a></span></span> | tool-user | 终态 DB + **pass^k** | 模拟用户多轮 | 策略约束 + 把**可靠性**做成一等指标 |
| **GAIA**<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">通用助手:推理+多模态+web+工具,三难度级,答案唯一可判。<a href="https://arxiv.org/abs/2311.12983">Mialon 2023 ↗</a></span></span> | general | exact-match 答案 | 静态(带工具) | 答案唯一、自动判;难度分级 |
| **MLE-bench**<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">75 个 Kaggle ML 工程竞赛,按奖牌率评 agent。<a href="https://arxiv.org/abs/2410.07095">Chan 2024 ↗</a></span></span> | ML 工程 | Kaggle 评分(奖牌) | 长程交互 | 用既有 leaderboard 当客观标尺 |
| **LiveCodeBench**<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a><span class="cite-note">按竞赛**发布时间窗**滚动取题,只评模型 cutoff 之后的新题以防污染。<a href="https://arxiv.org/abs/2403.07974">Jain 2024 ↗</a></span></span> | coding(竞赛) | execution | static-but-windowed | **时间窗防污染**,living |

> 💡 **选型原则(呼应 foundations Q13):** 看「动作空间 + 任务分布是否匹配目标场景」。能力域有重叠(SWE-bench 与 LiveCodeBench 都测 coding,但前者测仓库级 issue 修复、后者测算法竞赛 + 防污染);**评判方式**比能力域更能决定「这个分数可不可信」——execution-based + 隐藏判据最难作弊,LLM-judge 最易被操纵。

## 3. 污染 / 饱和 / living benchmark / Contamination · Saturation · Living benchmarks

这是 agent 评测**最易翻车**的一节,也是读技报数字时的第一道闸。

### 3.1 污染 / Contamination

**机制**:测试样本(题目甚至官方解)进了训练语料 → 模型「记住」而非「解出」,分数测的是记忆。两条路径:

1. **直接**:题目 + 解答原文被爬进训练集(SWE-bench 的 issue 与修复 PR 都在公开 GitHub,天然高危)。
2. **间接**:题解被博客 / 讨论区转述后进训练集。

**怎么检测污染**:① n-gram / 子串重叠扫描训练语料;② **凭记忆复现测试**——只给题面、不给可执行环境,看模型能否直接吐出官方 fix(能,则强烈暗示见过);③ **时间窗断崖**——对比 cutoff 前后题目的分数,若旧题显著高于结构相同的新题,即污染信号。

> 💡 **LiveCodeBench 的解法:发布时间窗。** 只用模型训练 cutoff **之后**发布的竞赛题评测,从根上保证「模型不可能训练时见过」;"living" = 持续滚动加入新题、淘汰旧题。这是当前最干净的防污染范式之一(代价:题型偏算法竞赛,覆盖面窄于真实工程)。

### 3.2 饱和 / Saturation —— SWE-bench Verified 的退役(2026-02,已核实)

**饱和** = 顶部模型分数逼近上限、区分度消失,benchmark 失去鉴别力。最典型的当代案例:

OpenAI 于 **2026-02-23 官方宣布停止在 SWE-bench Verified<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">OpenAI 2024-08 人工核验的 500 题干净子集。<a href="https://openai.com/index/introducing-swe-bench-verified/">OpenAI 2024 ↗</a></span></span> 上评测前沿模型**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">OpenAI 2026-02-23:Verified 已饱和+污染,不再反映真实开发能力。<a href="https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/">OpenAI 2026 ↗</a></span></span>,两条原因:

- **污染**:在 OpenAI 的诱发(elicitation)实验中,前沿模型在**部分任务**上只给题面即可凭记忆复现官方 fix 或任务特定细节——训练时见过题解的强信号。
- **题目本身有缺陷**:OpenAI 审计了其 o3 在 **64 次独立运行中未能稳定解出**的 **138 题**(占 500 题的 **27.6%**),发现其中 **59.4% 的题在测试设计 / 题面描述上有实质缺陷**——**35.5%(约 49 题)太窄**(严格测试卡实现细节、拒绝功能正确的解)、**18.8%(约 26 题)太宽**(测了题面没要求的额外功能),其余约 5% 为描述类缺陷。即一部分「失败」其实是 benchmark 错判,不是模型不行。

> ✅ **OpenAI 的结论(意译):** Verified 上的提升「不再反映模型真实软件开发能力的进步,而越来越反映模型在训练时见过多少这个 benchmark」。这正是「饱和 + 污染」联手让一个曾经的金标准失效的活教材。

**替代**:OpenAI 改推抗污染设计的 **SWE-bench Pro**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">Scale AI 2025:更难、抗污染的 SWE-bench 变体;含公开/私有/商用分片。<a href="https://labs.scale.com/leaderboard/swe_bench_pro_public">Scale AI 2025 ↗</a></span></span>,并主张转向**专家私有命题**的 benchmark(如 GDPval)。量级落差极具说明性:在 Verified 上约 **70–80%** 的同批模型,在 Pro 上掉到 **~23–58% 量级**(具体分数随分片 / 榜单 / scaffold 变动,见下 caveat)。

> ⚠️ **caveat(诚信门):** SWE-bench Pro 的**具体模型分数**是 2025–26 的单一来源(Scale leaderboard)、会随榜单变动;本页只用「~70–80% → ~23–58% 的量级落差」说明**饱和/污染**这一**方法学现象**,不把它当成稳定排名或某模型的固定能力值。

### 3.3 living benchmark 范式

应对污染 + 饱和的通用解法:**定期换题 / 时间窗滚动(LiveCodeBench)/ 私有保留集(GDPval 式专家命题)/ 商用闭源分片(SWE-bench Pro private)**。核心是让「记住题」这条捷径失效。

> ❌ **误区:** 「leaderboard 分数越来越高 = agent 能力在进步」。高分可能来自**污染**(背题)、**scaffold 工程**(见 §4)或对该 benchmark 的**过拟合**;真进步要看**新发布、防污染**的 living benchmark + **pass^k** 可靠性是否同步抬升,并核实 harness 与日期。

## 4. harness / 可复现 / 方差 / Harness · Reproducibility · Variance

### 4.1 harness 决定分数

**harness(评测脚手架)** = 把模型接到 benchmark 的全部工程:prompt 模板、工具集、输出解析器、重试 / 步数预算、检索策略、是否给环境反馈。**同一模型换 harness,分数可显著变化(公开报告里常见十个点量级的波动,随 benchmark/scaffold 而异,无统一定值)**——SWE-bench 早期榜单的大量提升来自 scaffold 工程而非模型本身。

> 💡 **推论:** benchmark 比较的是 **model × harness**,不是纯 model。报告分数**必须**附 harness 版本 + 配置(工具集、最大步数、温度、检索 on/off);否则两个数字根本不可比。读技报时若只见「我们达到 X%」而不交代 harness,这个数字的可信度要打折。

### 4.2 方差:pass@k 是估计量

agent 评测的随机性来自三处:① **采样**(temperature > 0);② **环境随机**(网页 / 工具返回会变、模拟用户随机);③ **估计量本身**——`pass@k` 不是真值而是用 $n$ 个样本估出来的,$n$ 小则方差大。

Codex / HumanEval<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a><span class="cite-note">Chen 2021 引入 HumanEval 与 pass@k 的无偏估计;repeated sampling 是强基线。<a href="https://arxiv.org/abs/2107.03374">Chen 2021 ↗</a></span></span> 给出 pass@k 的**无偏估计量**:每题采 $n$ 个样本、其中 $c$ 个正确,则

$$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[\,1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\,\right]$$

直觉:$\binom{n-c}{k}/\binom{n}{k}$ 是「随机抽 $k$ 个全是错解」的概率,$1-$ 它就是「至少抽到 1 个对解」。**只跑一组 $k$ 个样本看是否 ≥1 成功,是无偏但方差极大(单组只有 0/1)的估计;此组合数式用上全部 $n$ 个样本,同样无偏却方差小得多**。$n \gg k$ 才稳。

> ⚠️ **可复现红线:** 只报「pass@1 = 41%」而不报 $n$、温度、seed、harness,**别人无法复现**。负责任的报法:固定 seed + 报 $n$ 次采样的均值 ± 区间 + harness 版本。技报里看到不带这些的单个数字,默认它有方差噪声。

**from-scratch 实现**(agent 评测核心指标:pass@k 无偏估计 + pass^k 可靠性):

```python
import numpy as np
from collections import defaultdict

def unbiased_pass_at_k(n, c, k):
    """每题 n 次采样、c 次正确,至少 1 次对的概率(Chen et al. 的无偏估计)。
    公式:1 - C(n-c, k)/C(n, k),数值稳定实现。"""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_agent_metrics(results, k=5):
    """results: list[dict],每条={success: bool, traj_len: int, traj_cost: float}.
    返回 pass@k(能力上界), pass^k(可靠性), 平均步数/成本。"""
    n = len(results)
    c = sum(1 for r in results if r["success"])
    # pass@k:至少 1 次成功的能力上界
    pass_at_k = unbiased_pass_at_k(n, c, k) if n >= k else float('nan')
    # pass^k:连续 k 次全成功的可靠性(实际部署看的指标)
    # 从顺序结果用滑动窗估计:所有长度为 k 的窗口里全成功的比例
    successes = [r["success"] for r in results]
    windows_all_pass = sum(
        all(successes[i:i+k]) for i in range(len(successes) - k + 1)
    )
    pass_pow_k = windows_all_pass / max(len(successes) - k + 1, 1)
    # 轨迹效率
    avg_steps = np.mean([r["traj_len"] for r in results])
    avg_cost  = np.mean([r["traj_cost"] for r in results])
    return {"pass@k": pass_at_k, "pass^k": pass_pow_k,
            "avg_steps": avg_steps, "avg_cost": avg_cost, "success_rate": c/n}

# 面试关键点:
# ① pass@k 是"能力上限"(有 oracle 验证器时能拿到的最好),pass^k 是"可靠性"
# ② agent 部署看 pass^k——k 次跑至少 1 次闯祸就不可用(见 τ-bench)
# ③ 轨迹效率(步数/成本)与成功率一起报,防"绕了 20 步成功"被等同于"3 步成功"
```

## 5. 轨迹评测 vs 结果评测 / Trajectory vs Outcome eval

| | outcome-only(execution-based) | trajectory(过程) |
|---|---|---|
| 看什么 | 只看**终态**成功与否 | 看**每一步**动作是否合理 |
| 优点 | 客观、可验证;判据隐藏时**难作弊** | 能诊断**哪步错**、给 partial credit、喂 PRM |
| 盲区 | 漏掉「蒙对 / lucky path」与 **reward hacking**(改测试、空实现过 CI);不诊断 | **谁来评**是难题:规则覆盖不全,LLM-judge 自带偏置 |

**轨迹谁来评**:① **程序化规则**(步数、是否调了禁用工具、是否触发危险动作)——可靠但覆盖窄;② **LLM-as-judge**——灵活但有位置偏好、自我偏好、易被「说服」等可靠性问题。

> 📝 **范围边界:** LLM-judge 的**校准 / 可靠性 / 去偏**属于姊妹篇 [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html) 的主场,本页只点到为止:评 agent 轨迹时,judge 的偏置会直接污染过程奖励 → 信用分配,细节去那篇。

**agent 特有的 verifier 可靠性**:即便是 execution-based 的「可验证奖励」也会错——单测**太弱**则**假阳**(错解蒙过),单测**太严**则**假阴**(功能正确的解被拒,正是 §3.2 里 SWE-bench Verified 的 49 道「太窄」缺陷题)。

> ❌ **误区:** 「execution-based / 可验证奖励 = 客观可靠」。可验证只是**比 LLM-judge 难作弊**,不等于**正确**:judge 的质量上限就是那套单测 / 校验脚本的质量;弱测试集会把 reward hacking 直接当成功回传,在 RL 里被放大。

## 6. 安全 / sabotage 评测 / Safety & sabotage eval

能力评测问「能不能做对」,安全评测问「会不会做坏 / 越权 / 被劫持」——**两者正交**,高能力不蕴含安全。三类要单列:

1. **sabotage / sandbagging**:模型故意做错或暗中破坏。Anthropic 的 sabotage evaluations<span class="cite-wrap"><a class="cite" id="fnref-13" href="#ref-13">13</a><span class="cite-note">Anthropic 提出的**评测框架**:code sabotage / sandbagging / 削弱监督 / 决策破坏。<a href="https://arxiv.org/abs/2410.21514">Anthropic 2024 ↗</a></span></span> 是一个**评测框架**(code sabotage、sandbagging 暗中藏拙、undermining oversight、decision sabotage)——本页**只描述框架,不引具体百分比**(常见流传的某个数字未能核实,按诚信门弃用)。
2. **prompt injection / 工具投毒**:外部内容里的恶意指令劫持 agent(见 foundations §10 + Greshake<span class="cite-wrap"><a class="cite" id="fnref-14" href="#ref-14">14</a><span class="cite-note">indirect prompt injection:外部内容里的指令劫持 LLM 应用。<a href="https://arxiv.org/abs/2302.12173">Greshake 2023 ↗</a></span></span>)。
3. **越权 / 危险动作**:高权限工具被误用(删库、对外发请求)。

> 🚨 **引用安全数字的诚信门(多 agent 注入攻击案例):** 一项多 agent 系统安全研究<span class="cite-wrap"><a class="cite" id="fnref-12" href="#ref-12">12</a><span class="cite-note">Triedman 2025:特定 web 注入攻击下,多 agent 编排器执行任意恶意代码的攻击成功率(scope-locked)。<a href="https://arxiv.org/abs/2503.12188">Triedman 2025 ↗</a></span></span> 报告:在**特定 web 注入攻击**下,让多 agent 编排器执行任意恶意代码的**攻击成功率达 58–90%**(随 orchestrator 而变)、**个别配置达 100%**。它是**控制流劫持攻击在该威胁模型下的成功率**,**不是**「agent 一般部署中就会按这个比例作恶」。引用时必带这层 scope,否则就是把一个攻击实验夸大成普遍事实。

> ❌ **误区:** 「benchmark 安全分高 = 部署安全」。安全评测多在**特定威胁模型**下测;换攻击面 / 换权限配置结论就变。安全是**纵深防御**(权限最小化 + host 策略门 + 监控),不是一个能被某个 benchmark 分数「通过」的属性(展开见规划中的 agent-safety 篇)。

---

## 分层面试题 / Stratified follow-ups

> ⚠️ 下列「会被问」均为**据公开 JD + 技报推断**,非真实面试原题。

### L1 基础

<details class="qa"><summary>1. agent 评测和单轮 QA 评测有什么本质区别?为什么不能只看 accuracy?</summary>

答:单轮 QA 是「输入→一次输出→与参考答案比对」,一次性、近确定性;agent 评测要测**一条多步轨迹**,成功判据是**终态满足目标**(execution-based,跑单测 / 校验环境状态),不是字符串匹配。只看 accuracy 会漏掉三件 agent 特有的事:**多次跑的可靠性**(pass^k)、**轨迹质量**(有没有 reward hacking / 蒙对)、**分数随时间贬值**(污染 + 饱和)。

**追问：** 为什么 agent benchmark 偏好 execution-based 判据而非答案匹配? → 因为 agent 的「正确」往往不是唯一字符串(改代码有多种写法、网页操作有多条路径),只能用「最终是否达成可执行的目标状态」来判;execution-based 还更难作弊(若判据隐藏)。

</details>

<details class="qa"><summary>2. 什么是 benchmark 污染?为什么对 agent benchmark 尤其严重?</summary>

答:污染 = 测试样本(题目甚至官方解)进了训练语料,模型「记住」而非「解出」,分数测的是记忆。对 agent benchmark 尤其严重,因为很多任务源自**公开 GitHub**(SWE-bench 的 issue + 修复 PR 都是公开且会被爬的),题目和标准解天然在训练集里;加上前沿模型反复在这些榜上刷分,过拟合 + 记忆叠加。

**追问：** 怎么检测一个模型在某 benchmark 上被污染了? → ① n-gram 重叠扫训练语料;② 只给题面不给环境,看能否凭记忆复现官方 fix;③ 比较训练 cutoff 前后结构相同题目的分数断崖。

</details>

<details class="qa"><summary>3. pass@k 与 pass^k 的差别,以及它们各自的「估计量」含义是什么?</summary>

答:`pass@k` = k 次尝试**至少 1 次**成功(能力上界,乐观);`pass^k` = k 次**全部**成功(可靠性,见 foundations §9)。两者都不是真值而是**估计量**:实际用每题采 $n$ 个样本来估,$n$ 小则方差大。pass@k 有无偏估计量 $1-\binom{n-c}{k}/\binom{n}{k}$($c$ = $n$ 中成功数),比「直接数一次中没中」方差更小。

**追问：** 为什么报 pass@k 必须同时报 $k$、温度和 $n$? → pass@k 的取值随 **$k$ 和采样分布(温度)** 变化——温度改变每样本成功率与多样性;而 **$n$(估计用的采样数)只决定方差 / 置信区间,并不改变 pass@k 的期望值**(除非你真改了 $k$ 或在做 best-of-$n$)。所以不报 $k$ / 温度 → 数字不可比,不报 $n$ / seed → 不可复现。

</details>

### L2 进阶

<details class="qa"><summary>4. SWE-bench Verified 为什么会在 2026 年被 OpenAI 弃用?请分污染和饱和两条线说。</summary>

答:两条线叠加。**污染线**:OpenAI 的诱发实验显示前沿模型在部分任务上只给题面即可凭记忆复现官方 fix → 分数反映「见过多少 benchmark」而非真实能力。**饱和/缺陷线**:OpenAI 审计了 o3 在 64 次运行中未能稳定解出的 138 题(占 500 的 27.6%),发现 59.4% 的题在测试 / 描述上有实质缺陷——35.5%(约 49 题)太窄(拒绝功能正确解)、18.8%(约 26 题)太宽(要求题面没提的功能),其余约 5% 描述缺陷;即部分「失败」是 benchmark 错判。结论:Verified 上的提升不再反映真实开发能力。OpenAI 改推抗污染的 SWE-bench Pro(同批模型 ~70–80%→~23–58% 量级)。

**追问：** 引用「SWE-bench Pro 上掉到 ~23–58%」时要注意什么? → 那是 2025–26 单一来源(Scale leaderboard)、会变的具体分数;只能用它说明「饱和/污染导致的量级落差」这个方法学现象,不能当稳定排名或固定能力值。

</details>

<details class="qa"><summary>5. 同一个模型在不同 harness 下分数可相差很多(十个点量级),这说明什么?报告 benchmark 分数该带哪些信息?</summary>

答:说明 benchmark 比较的是 **model × harness**,不是纯 model——prompt 模板、工具集、解析器、步数预算、检索策略都显著影响分数(SWE-bench 早期大量提升来自 scaffold 工程)。负责任的报法必须带:**harness 版本 + 工具集 + 最大步数 + 温度 + seed + 采样数 $n$ + 日期**。缺这些的单个数字不可比、不可复现。

**追问：** 这对「读别家技报」意味着什么? → 看到「我们达到 X%」而不交代 harness 与日期,要对可比性打折;尤其跨技报比分数前,先确认是否同一 harness、同一 benchmark 版本、是否防污染。

</details>

<details class="qa"><summary>6. LiveCodeBench 的「发布时间窗」如何从根上防污染?有什么局限?</summary>

答:只用模型训练 cutoff **之后**新发布的竞赛题来评测——既然题目在模型训练后才出现,就**不可能**被训练时见过,从机制上断掉污染;"living" = 持续滚动加新题、退旧题,长期保鲜。局限:① 题型偏**算法竞赛**,覆盖面窄于真实软件工程(不测仓库级 issue);② 需要持续运营投入;③ 不同模型 cutoff 不同,严格可比要按各自 cutoff 切窗。

**追问：** 时间窗防污染和「私有保留集 / 专家命题」(GDPval、SWE-bench Pro private)各自适合什么? → 时间窗适合有稳定新题流的领域(竞赛);私有 / 专家命题适合没有自然新题流、或需要贴近真实专业任务的场景,代价是命题 + 评分的人力成本。

</details>

<details class="qa"><summary>7. 面对一个具体的 agent 应用（代码修复 / Web 操作 / 客服工具等），怎么选 benchmark？不同 benchmark 的适用范围和核心局限是什么？</summary>

答:按三个匹配度依次过滤——① **动作空间匹配**:agent 操作粒度是什么？修代码 → SWE-bench Pro / LiveCodeBench;操作网页 → WebArena;操控 OS GUI → OSWorld;多轮工具+用户交互 → τ-bench;ML 工程 → MLE-bench;通用问答+工具调用 → GAIA。② **评判方式可靠性**:能否接受 LLM-judge 的偏置？高危场景(代码部署、金融交易)优先选 execution-based + 隐藏判据的榜;低危探索场景可用 LLM-judge 做初筛。③ **维护状态**:是否仍在 actively maintained / living？饱和榜(SWE-bench Verified 已退役)不可作唯一依据;living benchmark(LiveCodeBench)或刚发布且社区活跃的新榜更可信。

具体映射:

| 应用场景 | 推荐 benchmark（按优先级） | 核心局限 |
|---|---|---|
| 代码修复 / PR agent | SWE-bench Pro(抗污染,41 仓库/多语言) > LiveCodeBench(通用 coding 防污染初筛) | 单仓库 issue 级任务;缺产品/需求上下文;不测跨仓库协同改动 |
| Web 操作 agent | WebArena(自托管可复现) | 4 个自托管站点类别 + 辅助知识/工具站点(map/wiki/calculator);不测真实 web 的 OOD |
| GUI / desktop 操作 agent | OSWorld(真实 OS,脚本判终态) | 369 题,桌面 OS 非手机;跨 app 覆盖有限 |
| 工具+对话 agent(客服/助手) | τ-bench(pass^k 可靠性) + GAIA(通用) | τ-bench 领域固定(零售/航空);GAIA 只看终态 |
| ML 工程 agent | MLE-bench(Kaggle 奖牌率) | 偏竞赛风格;不测日常 ML 运维 |

> 💡 **核心原则:** 没有「最好的」benchmark,只有「动作空间 + 风险容忍度匹配」的 benchmark。选榜前先定义清楚 success criteria——agent 在这个场景下「成功」到底意味着什么——再用上面的三轴框架对号入座。

**追问：** 一个场景没有现成 benchmark 怎么办? → ① 判断是否能用通用榜(GAIA / τ-bench)先做初筛;② 不行就自己**构建 domain-specific eval**:关键是把成功判据做成**可执行的**(脚本/单测/终态校验),并且**保持判据对 agent 不可见**;③ 评估构建成本:一般需几十到上百题(取决于 baseline 成功率 + 效应量 + 期望 CI 宽度,建议做 power analysis 定题量;作为粗略经验值往往 ≥100),且需要长期维护(换题防过拟合)。

</details>

<details class="qa"><summary>8. LLM-as-judge 评 agent 轨迹有哪些已知偏置？哪些场景可以用、哪些不能用？</summary>

答:LLM-judge 评 agent 轨迹时有**已多次在文献中复现的三类偏置风险**:

1. **位置偏置(position/order bias)**:答案在 prompt 中出现的顺序、是否靠近 prompt 末尾显著影响 judge 评分。轨迹越长越严重——agent 多步轨迹天然比单轮 QA 长,末尾步对 judge 影响远大于开头步。
2. **自我偏置(self-preference)**:同模型族的 judge 倾向给同族的 agent 更高分;多个研究在 GPT/Gemini/Claude 家族上都观测到此现象,虽强度因 model/task 而异而非定值,但方向一致。
3. **风格/解释偏置**:agent 轨迹中若包含自信、详细或权威风格的解释文本,LLM-judge 更倾向判它对,即使底层动作是错的——judge 被 rhetoric 而非事实影响(有时被称为 verbosity / authority / rationale bias)。

**可以用 LLM-judge 的场景:**
- **相对排序**(A vs B 哪个轨迹更好)——比给绝对分数可靠,偏置在成对比较中部分抵消。
- **粗粒度过滤**——筛掉明显坏轨迹(没调工具 / 死循环 / 空响应),规则判据做主、LLM 兜底。
- **辅助诊断**——在 human review 前预标注,加速人工复核。

**不能用 / 要极其谨慎的场景:**
- **作为 RL 训练的 reward 来源**——偏置会被策略搜索并放大,变成「讨好 judge」而非「完成任务」。
- **作为绝对 pass/fail 判据**——可靠性不足以做高风险单次决策。
- **跨模型族能力排名**——自我偏置会让排名系统性不可比。

> ❌ **误区:** 「找一个"最强"LLM 当 judge 就客观了」。更强 ≠ 更无偏;已知某些维度(如 verbosity bias——判更长输出更好)上更强的 LLM 偏置甚至可能更大,且跨模型族自我偏置不随能力提升而消失。**judge 校准的专门方法论**(位置随机化、跨族 judge ensemble、成对而非绝对评)见姊妹篇 [reward-modeling-eval](https://ac.fzhiy.net/post-training-playbook/cheatsheet-reward-modeling-eval.html)。

**追问：** 有没有比 LLM-judge 更可靠、又比 execution-based 更灵活的折中? → **程序化规则 + LLM 组合**:对可规则化维度(步数上限、是否调了禁用工具、动作格式是否合法)走确定性规则;对需「语义判断」的维度(这一步方向是否合理)走 LLM,但只在统计意义上用(多 judge 均值 + 方差),不在单次决策层面依赖任何一个 judge。

</details>

### L3 深入

<details class="qa"><summary>9. 设计一个抗污染、抗 reward-hacking、可复现的 coding-agent 评测,需要哪些要素?</summary>

答:综合 §3–§5:① **抗污染**——题目用训练 cutoff 之后的新题(时间窗)或私有保留集,并做污染审计(凭记忆复现测试);② **抗 reward-hacking**——成功判据用**隐藏的额外单测 + 环境终态校验**(agent 不可见),防改测试 / 空实现过 CI;审计单测质量本身(避免太弱→假阳、太严→假阴);③ **可复现**——固定 seed、公开 harness、报 $n$ 次采样的 pass@k 无偏估计 + 区间 + 版本日期;④ **可靠性**——报 **pass^k** 而非只报 pass@1,暴露多次跑的稳定性;⑤ **living**——定期换题防长期过拟合。

**追问：** 为什么「隐藏判据」是抗 hack 的关键? → 一旦 agent 能看到判据(断言 / 单测),它会去满足判据而非真解题(改测试文件、硬编码期望输出);把判据藏在 agent 不可见处,才逼它真正达成目标状态。

</details>

<details class="qa"><summary>10. outcome-only 与 trajectory 评测各自的盲区是什么?execution-based verifier 也会假阴/假阳,如何缓解?</summary>

答:**outcome-only** 盲区——漏掉「蒙对 / lucky path」与 reward hacking,且不诊断哪步错、不给 partial credit;**trajectory** 盲区——「谁来评每步」难,规则覆盖不全、LLM-judge 自带位置/自我偏好。**execution-based verifier 的假阴/假阳**:单测太弱→错解蒙过(假阳),太严→功能正确解被拒(假阴,即 SWE-bench Verified 的缺陷题)。缓解:① 审计 + 加强测试集(多样输入、隐藏额外断言);② 多判据交叉(单测 + 终态校验 + 必要时人工核验子集);③ 对高风险题保留 human-in-the-loop 复核;④ 在 RL 里对 verifier 噪声做鲁棒处理(别把单一弱信号当真值放大)。

**追问：** 弱 verifier 在 RL 训练里为什么比在静态评测里更危险? → 静态评测里弱 verifier 只是测不准;在 RL 里它是**奖励来源**,reward hacking 会被策略主动搜索并放大——agent 会朝「骗过弱测试」而非「真解题」的方向优化,越训越坏。

</details>

<details class="qa"><summary>11.（诚信题）怎样引用「多 agent 注入攻击的恶意执行率」、sabotage 这类安全数字才算诚实?</summary>

答:必带**精确 scope**。那个「多 agent 编排器执行任意恶意代码」的数字(某研究报 **58–90%**、个别配置 100%)是「在**特定 web 注入攻击**下的**攻击成功率**」,是**控制流劫持的攻击成功率**,不是「一般部署中就会按这个比例作恶」。sabotage 类同理:Anthropic 的 sabotage evaluations 是个**评测框架**,引用要说清是框架 / 特定设定,流传的具体百分比若不能溯源到原文设定就**弃用**。原则:安全数字几乎都绑定一个**威胁模型 + 实验设定**,脱离设定的引用即失真。

**追问：** 为什么安全评测的数字比能力评测的更容易被误引? → 安全数字往往是「最坏情况 / 特定攻击」下的上界,天然吸睛且易被掐头去尾;而它对**威胁模型**极度敏感(换攻击面结论就变),所以脱离设定单独引用几乎必然夸大。

</details>

<details class="qa"><summary>12. 评测如何驱动 agent RL 训练？从 eval signal 到 RL reward 的链路里有哪些关键陷阱？</summary>

答:标准链路是 **benchmark 评测 → 奖励设计 → RL 训练 → 再评测**。每一步都可能出错:

1. **同一 benchmark 既当评测又当 reward 源**:如果用同一套验证器/单测既给 RL 当 reward、又当最终评测——agent 学到的是「骗过这个验证器」而非真正解题。必须 **eval/train verifier 分离**,训练用的判据与最终评测用的判据是两套独立集合。
2. **弱 verifier 在 RL 里被放大**(呼应 Q10):静态评测里弱 verifier 只是「测不准」;在 RL 里它是**奖励来源**,reward hacking 被策略主动搜索并放大——agent 朝「骗过弱测试」(改测试文件、空实现过 CI)的方向优化,越训越坏。
3. **分布偏移(distribution shift)**:RL 改变策略 → 轨迹分布改变 → 训练初期校准的 verifier 到后期可能不再有效——agent 进入了 verifier 从未见过的行为区域,判据的假阳/假阴率未知。
4. **proxy reward gap**:benchmark 测「任务成功」,但 RL 用的 reward 是任务成功的**代理**(单测通过 / 终态校验通过 / LLM-judge 评分)。这个「成功 ↔ proxy」的差距就是 reward hacking 的温床——agent 优化 proxy 而非真目标。
5. **RL 运行污染 benchmark**:如果 RL 训练轮次中用了 benchmark 题目做环境交互(不只是最终评测),那些题目后续不能再用——agent 已在这些题上训练过,再测即污染。

缓解:① eval 题集与训练题集**物理隔离**(各自独立采样,不是随机切分——agent 评测题天然少,随机切分不够安全);② RL reward 用**独立于最终评测的**隐藏训练判据(训练专用 verifier,与最终 eval 判据不重叠);最终评测判据须**至少同等强度**且完全 held-out;③ 定期人工抽查 agent 轨迹,看是否有 reward hacking 信号;④ 用多个异构 reward 源交叉验证,而非依赖单一 signal。

> ❌ **误区:** 「RL 训练完再跑一次 benchmark 就是评测」。如果那套 benchmark 的任何部分(题目 / 判据 / 环境)在 RL 训练循环里被用到过,它就已经被污染了——跑出来的是**训练集上过拟合后的分数**,不是泛化能力。

**追问：** eval/train verifier 分离和 eval/train problem split 是一回事吗? → 不是。problem split 是「哪些题用于训练、哪些用于评测」;verifier split 是同一道题用**不同判据**——训练时只暴露训练专用弱判据(少量可见单测),评测时用 held-out 的完整强判据集(不与训练判据重叠)。两层都要做,只做一层不够。

</details>

<details class="qa"><summary>13. 生产 agent 系统除 pass@k/pass^k 外,还要看哪些指标？学术 benchmark 分数高的 agent 部署后为什么可能表现很差？</summary>

答:学术 benchmark 分数衡量的是**理想条件下的能力上界**,生产 agent 还要看五个维度:

| 指标维度 | 测什么 | 为什么学术榜不测 |
|---|---|---|
| **单任务成本** | token 量 × 单价 / 成功任务(含重试) | benchmark 不设预算约束 |
| **延迟 / 步数** | 完成任务的平均步数 + P99 延迟 | 学术评测不扣步数分(大多数榜) |
| **预算约束下可靠性** | 给定 token/time budget 的任务成功率 | 标准 pass@k 把 k 次尝试视为零成本,不反映 token/时间约束下的性价比 |
| **用户体验指标** | 任务放弃率、用户满意度、是否需要人接手 | benchmark 无真实用户 |
| **安全事件率** | 单位任务中越权 / 注入 / 误操作的次数 | 能力榜不管安全 |

> 💡 **学术高分 + 部署后翻车的「最后一公里」:** 常见原因是 agent 把学术榜的无限重试 + 无预算假设带进了生产——pass@k 高但每一步都极贵(搜索/重试无节制),上生产后设了 token budget 或 timeout,真实成功率断崖下降。**补救:** 评测报告里加「给定预算下的 pass@k」曲线,把 cost 做成一等公民。

> ❌ **误区:** 「benchmark 分数高 = 可以部署了」。benchmark 是能力的**必要不充分条件**——它只证「在理想条件下能做到」,不证「在预算/安全/用户体验约束下能做稳」。部署前至少跑一轮 shadow deployment(并行的影子部署,让 agent 在真实流量上执行但不接管),用真实分布做 final gate。

**追问：** shadow deployment 具体怎么做、看什么? → 把 agent 部署在**与生产同分布但隔离**的环境(生产流量镜像 / 生产快照回放 / dry-run 模式),**写操作走 no-op / sandbox adapter 不做真实副作用**(不触发支付、不发送外部消息、不对真实用户可见);暗中运行数天到数周,统计:
- **shadow 指标**(离线可测):真实分布任务成功率 vs 学术榜差距、单任务成本分布、安全红线触发率、延迟/步数分布
- **live 指标**(需 canary / A-B / 辅助人工部署才能测):用户放弃率、人工接手率、满意度——这些 shadow 模式下测不到,要拆开说

这套数字和学术榜分数的 gap,就是 agent 从「能做题」到「能干活」的鸿沟。做 shadow 前须确认:隐私/PII 合规(镜像流量是否含敏感数据)、写操作隔离是否彻底、有 approval 流程。

</details>

---

## 参考文献 / References

> 均为承重方法 / 一手来源,已逐条 web 核对(标题 + arXiv ID / 官方 URL;2026 的 SWE-bench Verified 退役与 Pro 分数为 2026-06 复核)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Jimenez et al. <em>SWE-bench: Can Language Models Resolve Real-World GitHub Issues?</em> ICLR 2024. <a href="https://arxiv.org/abs/2310.06770">arXiv:2310.06770</a> — 真实代码修复评测. <a href="#fnref-1">↩</a></li>
<li id="ref-2">OpenAI. <em>Introducing SWE-bench Verified</em>. 2024-08. <a href="https://openai.com/index/introducing-swe-bench-verified/">openai.com</a> — 500 题人工核验子集. <a href="#fnref-2">↩</a></li>
<li id="ref-3">OpenAI. <em>Why we no longer evaluate SWE-bench Verified</em>. 2026-02-23. <a href="https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/">openai.com</a> — 饱和+污染致退役;138 题审计约 59% 测试有缺陷. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Scale AI. <em>SWE-bench Pro Leaderboard</em>. 2025. <a href="https://labs.scale.com/leaderboard/swe_bench_pro_public">labs.scale.com</a> — 抗污染 SWE-bench 变体(公开/私有/商用分片). <a href="#fnref-4">↩</a></li>
<li id="ref-5">Zhou et al. <em>WebArena: A Realistic Web Environment for Building Autonomous Agents</em>. ICLR 2024. <a href="https://arxiv.org/abs/2307.13854">arXiv:2307.13854</a> — 自托管可复现 web 评测. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Xie et al. <em>OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments</em>. NeurIPS 2024. <a href="https://arxiv.org/abs/2404.07972">arXiv:2404.07972</a> — computer-use 终态校验. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Mialon et al. <em>GAIA: a benchmark for General AI Assistants</em>. ICLR 2024. <a href="https://arxiv.org/abs/2311.12983">arXiv:2311.12983</a> — 通用助手、答案唯一可判. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Yao et al. <em>τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains</em>. 2024. <a href="https://arxiv.org/abs/2406.12045">arXiv:2406.12045</a> — 多轮工具-用户 + pass^k 可靠性. <a href="#fnref-8">↩</a></li>
<li id="ref-9">Chan et al. <em>MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering</em>. ICLR 2025. <a href="https://arxiv.org/abs/2410.07095">arXiv:2410.07095</a> — Kaggle ML 工程、奖牌率. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Jain et al. <em>LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code</em>. ICLR 2025. <a href="https://arxiv.org/abs/2403.07974">arXiv:2403.07974</a> — 发布时间窗防污染. <a href="#fnref-10">↩</a></li>
<li id="ref-11">Chen et al. <em>Evaluating Large Language Models Trained on Code</em>. 2021. <a href="https://arxiv.org/abs/2107.03374">arXiv:2107.03374</a> — HumanEval + pass@k 无偏估计. <a href="#fnref-11">↩</a></li>
<li id="ref-12">Triedman, Jha, Shmatikov. <em>Multi-Agent Systems Execute Arbitrary Malicious Code</em>. 2025. <a href="https://arxiv.org/abs/2503.12188">arXiv:2503.12188</a> — 多 agent 编排器在 web 注入攻击下的恶意执行率(scope-locked). <a href="#fnref-12">↩</a></li>
<li id="ref-13">Anthropic (Benton et al.). <em>Sabotage Evaluations for Frontier Models</em>. 2024. <a href="https://arxiv.org/abs/2410.21514">arXiv:2410.21514</a> — sabotage/sandbagging 评测框架(本页只引框架). <a href="#fnref-13">↩</a></li>
<li id="ref-14">Greshake et al. <em>Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection</em>. 2023. <a href="https://arxiv.org/abs/2302.12173">arXiv:2302.12173</a> — 间接提示注入. <a href="#fnref-14">↩</a></li>
</ol>
