# Self-improving LLMs / 自我改进

> LLM 如何用**自己生成的信号**给自己"打分→过滤→训练",从而在无大量人工标注的情况下持续迭代。
>
> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。数字 / 结论以原论文为准,不确定处标注。

## 0. 一句话框架 / The core loop

```
生成(Generate) → 过滤/打分(Filter / Score) → 训练(Train) → 重复(Repeat)
```

每一轮,**当前策略** 产出候选答案或 preference 对;某种过滤机制(规则、另一个模型、自身打分)淘汰低质输出;剩余高质样本用来更新权重;下一轮拿新模型重跑。这个 **自举闭环(self-improvement loop)** 是所有方法的共同骨架。

---

## 1. Bootstrap-then-Train:从正确迹自举

### 1.1 STaR — 拒绝采样 + 迭代微调

STaR<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">迭代微调在"生成了正确答案"的 chain-of-thought 上,无需大规模 rationale 数据集。<a href="https://arxiv.org/abs/2203.14465">Zelikman 2022 ↗</a></span></span>(Self-Taught Reasoner)是 LLM chain-of-thought 自举微调的奠基性方案:

1. **Rollout**:对每道题采样 $K$ 条 chain-of-thought rationale。
2. **过滤**:保留最终答案正确的那些 rationale(rejection sampling)。
3. **微调**:在保留集上 SFT,更新模型。
4. **Hint-retry**:对答案全部错误的题,给出正确答案后让模型"重新解释",再混入训练(防止简单题统治训练集)。

$T$ 轮迭代后,模型既是数据生成器,又是数据过滤器。

### 1.2 RFT — 拒绝采样微调(Rejection Sampling Fine-tuning)

RFT 是 STaR 的简化变体:省掉 hint-retry,直接从同一道题的 $K$ 条采样中保留答案正确的,汇总成更丰富的微调集。核心发现:**同一题的多条正确解法**比一条解法多样性更高,有助于泛化。

### 1.3 ReST — Grow-Improve 离线 RL 循环

ReST<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">先用当前策略生成大规模数据集(Grow),再按奖励阈值过滤后微调(Improve),比在线 RLHF 更样本高效。<a href="https://arxiv.org/abs/2308.08998">Gulcehre 2023 ↗</a></span></span> 将循环拆成两阶段:

- **Grow**:从当前策略 $\pi_\theta$ 采样,构建离线数据集 $\mathcal{D}$,用奖励函数 $r(\cdot)$ 打分。
- **Improve**:在奖励超过阈值 $\tau$ 的子集 $\mathcal{D}_{\ge\tau}$ 上微调 $\pi_\theta$。

关键点:**Improve 阶段可多次重复**(提高 $\tau$ 逐步筛严),但 Grow 只需偶尔刷新一次 —— 相比在线 RLHF 的每步采样,计算更集中。

| 方法 | 过滤依据 | 是否在线 | 训练方式 |
|---|---|---|---|
| STaR / RFT | 答案对错(规则) | 准在线(迭代) | SFT |
| ReST | 奖励函数阈值 | 离线批次 | SFT / best-of-N 蒸馏 |

---

## 2. Self-Rewarding:模型自己当裁判

Self-Rewarding Language Models<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">同一模型既生成回答、又用 LLM-as-a-Judge 打分;用迭代 DPO 同步提升生成和评判能力。<a href="https://arxiv.org/abs/2401.10020">Yuan 2024 ↗</a></span></span> 打破了"需要外部 reward model"的假设:

1. 对同一 prompt 采样多条回答。
2. **同一模型**用 LLM-as-a-Judge 格式(评分+理由)给每条回答打分。
3. 按分数构造 preference 对 $(y_w, y_l)$,用 DPO 更新。
4. 下一轮,打分能力也随之提升 —— **两个能力共享同一参数,协同进化**。

这条路的前提:模型的**生成能力**要与**判断能力**相互促进而不相互污染。实验表明在若干迭代内确实如此,但长期是否退化仍是开放问题(见 §6 失效模式)。

---

## 3. Self-Play:用"前一轮自己"当对手

SPIN<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">当前模型 vs 上一轮模型:后者生成负样本,前者学会区分,仅用 SFT 数据即可自我改进。<a href="https://arxiv.org/abs/2401.01335">Chen 2024 ↗</a></span></span>(Self-Play Fine-Tuning)的灵感来自博弈论:

- **正样本**:原始 SFT 数据集里的 human response $y^*$。
- **负样本**:上一轮模型 $\pi_{\theta_{t-1}}$ 对相同 prompt 的输出 $\tilde{y}$。
- **目标**:当前模型 $\pi_{\theta_t}$ 学会**区分**真实 human response 与"旧自我"的输出,用类 DPO loss 更新。

$$\mathcal{L}_{\text{SPIN}}(\theta_t) = -\mathbb{E}\left[\log\sigma\!\left(\lambda\log\frac{\pi_{\theta_t}(y^*|x)}{\pi_{\theta_{t-1}}(y^*|x)} - \lambda\log\frac{\pi_{\theta_t}(\tilde{y}|x)}{\pi_{\theta_{t-1}}(\tilde{y}|x)}\right)\right].$$

关键点:无需额外人工 preference 标注 —— 负样本完全由**自身历史版本**提供。随着每轮迭代,$\pi_{\theta_t}$ 不断逼近人类分布,直到两者无法区分时收敛。

---

## 4. AI Feedback:让 AI 替代人类打 preference 标签

Constitutional AI<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">用一套"宪法"原则让模型自我批评并修订输出;AI 生成的 preference 数据替代人工无害性标注(RLAIF)。<a href="https://arxiv.org/abs/2212.08073">Bai 2022 ↗</a></span></span>(CAI / RLAIF)是目前最有影响力的"以 AI 替代人工 preference"方案:

**SL-CAI(监督阶段)**:
1. 模型生成有害回答草稿。
2. 给出一条宪法原则(如"避免歧视性内容"),让模型**自我批评**。
3. 让模型根据批评**修订**回答。
4. 用修订后的回答做 SFT。

**RL-CAI(强化阶段)**:
5. 让模型对一对回答用 AI 评分(哪个更符合宪法),构造 preference 数据。
6. 用 AI-labeled preference 训练 reward model,再用 RL 迭代。

与 STaR/ReST 的区别:**过滤信号来自宪法准则**,而非任务答案对错 —— 面向 alignment 而非推理能力。

---

## 5. 推理时自我纠错(Training-free)

以下两种方法不更新权重,属于**推理时(inference-time)自我改进**,与上述训练循环不同,但概念上同根:

### 5.1 Reflexion — 语言强化学习

Reflexion<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">Agent 把任务反馈转换为自然语言反思,存入 episodic memory,下次尝试时引用 —— 无需梯度更新。<a href="https://arxiv.org/abs/2303.11366">Shinn 2023 ↗</a></span></span> 让 agent 在多次**试错循环**中:

- 执行任务 → 拿到环境反馈(成功/失败/错误信息)。
- 生成**verbal reflection**:用自然语言总结"哪里错了、下次怎么改"。
- 把 reflection 存入**episodic memory**,下一轮注入 context。

迭代几次后,成功率显著提升 —— 但改进**仅存在于当前会话的 context**,重启即失效。

### 5.2 Self-Refine — 生成-批评-修订循环

Self-Refine<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">同一冻结 LLM 循环:生成输出 → 自我批评 → 根据批评修订,无需训练或额外监督,跨任务一致涨点。<a href="https://arxiv.org/abs/2303.17651">Madaan 2023 ↗</a></span></span> 的三步固定循环:

$$\text{output}_0 \xrightarrow{\text{critique}} \text{feedback}_0 \xrightarrow{\text{refine}} \text{output}_1 \xrightarrow{\cdots}$$

无需训练、无需额外监督 —— 直接利用**预训练模型的 self-critique 能力**。实验跨多个任务(代码、摘要、对话、数学)均有收益,但收益上限受限于模型的初始评判能力。

| 方法 | 改进发生在 | 是否更新权重 | 能否持久化 |
|---|---|---|---|
| Reflexion | inference-time,多次试错 | 否 | 否(context 内) |
| Self-Refine | inference-time,单次循环 | 否 | 否 |
| STaR / ReST / SPIN / CAI | training-time | 是 | 是 |

---

## 6. 失效模式 / Failure modes

自我改进循环看似美好,但有三个结构性风险:

### 6.1 奖励 Hacking(Reward Hacking)

当过滤信号(奖励模型、LLM 打分、规则过滤)不完美时,模型会学到**得高分却不真正正确**的策略:捷径答案、表面流畅但内容错误的 rationale、专门迎合打分模板的输出。

- 根因:优化目标(代理奖励)与真实目标(任务质量)之间的 gap —— **Goodhart's Law**。
- 缓解:用多样化、独立的评估信号;限制单次 RL 更新幅度(KL 约束)。

### 6.2 模型坍塌 / 分布收窄(Model Collapse / Distribution Narrowing)

每轮只保留"高分"样本,低分多样性被淘汰。多轮后训练集趋于单一,模型输出多样性下降,泛化变差。在 Self-Rewarding 等"模型给自己打分"方案里尤甚:模型的盲点在偏好标注中**被系统性继承**。

$$\text{Diversity}(\pi_{\theta_t}) \le \text{Diversity}(\pi_{\theta_{t-1}}) \quad \text{(若每轮只保留 top-}k\text{)}$$

### 6.3 Reward Model 过优化(RM Over-optimization)

RL 阶段的 reward model 本身是**近似**;当策略被持续优化时,分数曲线最终会与真实质量脱钩(reward model 的 out-of-distribution 区域被利用)。KL 散度惩罚项是标准缓解手段:

$$\mathcal{J}(\theta) = \mathbb{E}[r(y)] - \beta\,\mathrm{KL}[\pi_\theta \,\|\, \pi_{\text{ref}}].$$

$\beta$ 越大,离参考策略越近,但改进幅度也越保守。

---

## 7. From-scratch 代码:STaR 风格拒绝采样微调循环

```python
"""
STaR-style rejection-sampling fine-tuning loop (illustrative).
依赖:transformers, torch — 用 GPT-2 作教学示范,真实训练换成更大模型即可。
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ---------- 假设的问答数据 ----------
PROBLEMS = [
    {"question": "What is 3 + 5?",  "answer": "8"},
    {"question": "What is 7 * 6?",  "answer": "42"},
    {"question": "What is 12 - 4?", "answer": "8"},
]

# ---------- 辅助:简单答案抽取 ----------
def extract_answer(text: str) -> str:
    """从生成文本中抽取最后一个数字(演示用)."""
    import re
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else ""

# ---------- 1. Rollout:每题采样 K 条 rationale ----------
def rollout(model, tokenizer, problems, K=4, max_new=64, device="cpu"):
    """返回 list of (question, rationale, is_correct)."""
    results = []
    model.eval()
    for prob in problems:
        prompt = f"Question: {prob['question']}\nLet's think step by step:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=True, temperature=0.8,
                num_return_sequences=K, pad_token_id=tokenizer.eos_token_id,
            )
        for seq in outputs:
            text = tokenizer.decode(seq, skip_special_tokens=True)
            rationale = text[len(prompt):]
            correct = extract_answer(rationale) == prob["answer"]
            results.append({"prompt": prompt, "rationale": rationale, "correct": correct})
    return results

# ---------- 2. Filter:只保留答案正确的 rationale ----------
def filter_correct(results):
    return [r for r in results if r["correct"]]

# ---------- 3. Dataset wrapper ----------
class RationaleDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.data = []
        for s in samples:
            text = s["prompt"] + s["rationale"]
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            input_ids = enc["input_ids"].squeeze()
            self.data.append({"input_ids": input_ids, "labels": input_ids.clone()})

    def __len__(self):  return len(self.data)
    def __getitem__(self, i): return self.data[i]

# ---------- 4. Train:在正确 rationale 上 SFT ----------
def finetune(model, tokenizer, samples, output_dir="./star-ckpt"):
    ds = RationaleDataset(samples, tokenizer)
    if len(ds) == 0:
        print("No correct samples — skip this iteration.")
        return
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=1,
        per_device_train_batch_size=2, logging_steps=5,
        save_strategy="no", report_to="none",
    )
    Trainer(model=model, args=args, train_dataset=ds).train()

# ---------- 5. STaR 主循环 ----------
def star_loop(model_name="gpt2", n_iters=3, K=4):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    for t in range(n_iters):
        print(f"\n=== Iteration {t+1}/{n_iters} ===")
        all_results = rollout(model, tokenizer, PROBLEMS, K=K, device=device)
        correct = filter_correct(all_results)
        print(f"  Correct rationales: {len(correct)} / {len(all_results)}")
        finetune(model, tokenizer, correct)

    return model

if __name__ == "__main__":
    star_loop(n_iters=2, K=4)
```

> 以上代码仅作原理示意:真实 STaR 用更大模型、更长 rationale、hint-retry 兜底。核心流程(sample → filter → finetune → repeat)与论文一致。

---

## 分层面试题 / Stratified follow-ups

### L1 基础
1. 自我改进的"生成-过滤-训练"循环是什么?为什么需要循环而不是一次性?
2. STaR 如何在没有 rationale 标注的情况下训练 chain-of-thought?hint-retry 解决什么问题?
3. Reflexion 和 Self-Refine 为什么被称为"training-free"?它们的改进能持久化吗?
4. Constitutional AI 里"宪法"起什么作用?AI feedback 如何替代人工 preference 标注?

### L2 进阶
5. ReST 的 Grow 和 Improve 两阶段如何分工?为什么比在线 RLHF 更样本高效?
6. SPIN 用"前一轮自己"做负样本,和 DPO 用人工 preference 对相比有什么优劣?
7. Self-Rewarding 里"生成"和"评判"共享同一参数会带来什么问题?
8. 奖励 hacking 和 RM 过优化是同一回事吗?如何用 KL 约束缓解?

### L3 深挖
9. 模型坍塌 / 分布收窄在数学上如何刻画?有哪些缓解手段(温度采样、多样性约束、数据混合)?
10. STaR 每轮只保留正确样本会引入什么 selection bias?如何缓解?
11. 把 Self-Rewarding 的 LLM-as-Judge 与外部 reward model 结合,各自的信息贡献是什么?如何防止两者互相"共谋"?
12. 如果自我改进循环收敛到某个局部最优(模型无法产出比自己更好的数据),有哪些破局思路?

---

---

## 深挖 Q&A / Deep-dive

> 进阶面试题的详细解析。⚠️ 学习笔记,非作者研究成果。数字以原论文为准。

---

### Q1. STaR 只保留"答案正确"的样本:这会引入什么 selection bias?对学到的分布有什么形式化影响?

**核心偏差**:STaR<a class="cite" href="#ref-1">1</a> 在每轮迭代中,只把最终答案正确的 chain-of-thought 纳入训练集。这在形式上等价于:

$$p_{\text{train}}(r \mid x) \propto p_\theta(r \mid x) \cdot \mathbf{1}[\text{answer}(r) = a^*]$$

其中 $r$ 是 rationale,$x$ 是问题,$a^*$ 是参考答案。

三个结构性后果:

1. **正确性 ≠ 推理质量**:一条 rationale 可能通过侥幸、捷径或"答案倒推"得到正确答案,但推理步骤本身是错误的。由于过滤只看最终答案,**错误推理路径被系统性地混入训练集**。这与 Lightman et al.<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a></span> 提出 process reward model(PRM)的动机一致:outcome supervision 无法区分"对的理由答对"和"错的理由答对"。

2. **分布偏移积累**:第 $t$ 轮的训练集取自 $\pi_{\theta_{t-1}}$ 的条件分布,而不是真实推理分布 $p^*(r \mid x)$。每轮迭代后,$\pi_{\theta_t}$ 进一步偏离 $p^*$,过滤器的"正确率"信号变得越来越自我参照。

3. **难题盲区**:对模型已经完全答不对的难题,过滤后训练集为空(hint-retry 兜一部分,但不能完全补救)。模型在这些题上既无梯度更新,也无法自举改进,造成"强者愈强、难题停滞"的马太效应。

**缓解方向**:PRM 对每一步打分(而非只看最终答案)可减少步骤级错误;数据混合(保留原始 SFT 数据)防止分布完全漂移。

---

### Q2. 迭代自训练为什么会收窄分布(模型坍塌)?直觉 + 何时咬人?

**直觉**:每轮"只保留高分样本"在统计上等价于截断抽样——每次只取分布的高密度区域。多轮下来,尾部低概率(但高多样性)的输出被系统性淘汰。

Shumailov et al.<span class="cite-wrap"><a class="cite" id="fnref-10" href="#ref-10">10</a></span> 从理论和实验两个层面分析了**递归训练于自产数据**的后果:

- **统计近似误差**:每次采样产出的数据集是有限的,尾部事件被低估甚至缺失。
- **函数近似误差**:模型容量有限,进一步压缩了对低频模式的表达。

两种误差在迭代中**叠加**,导致分布持续收窄。用不等式直觉刻画:

$$H(\pi_{\theta_t}) \le H(\pi_{\theta_{t-1}}) \quad \text{(若每轮只保留 top-}k\text{ 样本)}$$

熵单调下降,输出趋向重复和单一。

**何时真正咬人**:

| 场景 | 为何严重 |
|---|---|
| Self-Rewarding(模型给自己打分) | 评判者本身就在漂移,偏好数据与真实偏好的 gap 越来越大 |
| 长链 chain-of-thought 任务 | 每一步的采样方差大;尾部解法(非常规推理路径)在过滤中首先消失 |
| 多轮对话 / agent loop | context 中的历史也是自产数据,递归污染效应更强 |
| 仅用自产数据、不混合人类数据 | 没有锚点,分布漂移无约束 |

**缓解**:定期混入原始人类数据(防止漂移的锚点);提高采样温度保留多样性;使用 diversity reward 项明确奖励输出多样性。

---

### Q3. Self-Rewarding 的 judge-generator 耦合为什么会失效?

在 Self-Rewarding<a class="cite" href="#ref-3">3</a> 里,**同一组参数**既是生成器(产出答案),又是评判者(给答案打分)。这带来一个结构性问题:**生成器的盲点会被评判者继承**。

具体机制:

1. **共同盲点**:若生成器不擅长某类推理(如反事实推理),它给这类推理打高分的概率也低于真实水平——因为评判能力与生成能力共享同一知识基础。偏好数据因此系统性地低估了这类能力。

2. **自我确认偏差(Self-Confirmation Bias)**:模型倾向于给"听起来像自己风格"的答案打高分。这不是 hallucination 的随机错误,而是一种**系统性偏差**——偏好信号本身就在把模型拉向自己的已有风格,形成正反馈闭环。

3. **错误共同漂移**:每轮 DPO 更新后,生成器和评判者同步朝偏好数据的方向移动。若偏好数据本身有误(来自有盲点的评判),下一轮的评判者也会在相同方向上加剧偏差——误差不是均值回归,而是**累积放大**。

形式化地,设 $J_\theta$ 为评判分数函数,$G_\theta$ 为生成函数,两者共享 $\theta$。真实质量函数为 $q^*$。则:

$$\mathbb{E}[J_\theta(y) - q^*(y)] \ne 0 \quad \text{且与 } G_\theta \text{ 的偏差方向相关}$$

**对比**:外部 reward model(参数独立)至少在初始时刻与生成器偏差方向无关。但它有另一个问题:分布外泛化失效(见 Q5)。

---

### Q4. SPIN 收敛到 SFT 数据分布——为什么这是一个上界?实践中意味着什么?

SPIN<a class="cite" href="#ref-4">4</a> 的理论收敛条件是:当且仅当 $\pi_{\theta_t} = p_\text{data}$(当前模型与人类 SFT 数据分布完全相同)时,损失梯度消失,训练停止。

这在数学上给出了一个**严格的能力上界**:

$$\text{SPIN 极限策略} = p_\text{data} \quad \text{(SFT 数据分布)}$$

推论:

1. **无法超越 SFT 数据质量**:若 SFT 数据中存在错误、偏见或能力盲区,SPIN 收敛后的模型也会继承这些缺陷。SPIN 只是让模型"更像人类 SFT 数据",不能发现超越该数据的新能力。

2. **负样本质量随迭代下降**:第 $t$ 轮的负样本由 $\pi_{\theta_{t-1}}$ 生成;随着 $\pi_{\theta_t} \to p_\text{data}$,负样本质量越来越接近正样本质量,**对比信号越来越弱**。这在实践中表现为:早期迭代效果显著,后期迭代边际收益递减甚至归零。

3. **与 STaR/ReST 的根本差异**:STaR 类方法以**任务正确性**为过滤信号,理论上可以超越 SFT 数据(只要存在正确 rationale 就能学到)。SPIN 以**与人类数据的区分能力**为目标,天花板由人类数据质量决定。

**实践含义**:SPIN 适合作为"补齐 SFT 缺口"的工具(消除模型与人类数据分布之间的 gap),但不适合用作持续自我改进的无限循环——到了后期迭代,应切换为有外部验证信号的方法。

---

### Q5. Reward Model 过优化:reward 上升/真实质量下降的 scaling-law 形状是什么?

Gao et al.<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a></span> 系统研究了**对 RM 优化程度**（用 KL 散度 $d$ 衡量）与**真实质量**的关系,发现两者的曲线形状取决于优化方式:

- **Best-of-$N$ 采样**:proxy reward 大致随 $\sqrt{\log N}$ 增长（论文指出拟合困难,此为近似描述）,真实质量先升后平——过优化效应相对温和。
- **RL(策略梯度)**:proxy reward 可以持续上升,但真实质量在某个 $d^*$ 之后**单调下降**。

大致函数形式(论文中拟合):

$$\text{gold reward} \approx d\,(\alpha - \beta \ln d)$$

其中 $\alpha, \beta > 0$，$d = \sqrt{D_{\mathrm{KL}}}$，最优点 $d^* = \exp\!\left(\dfrac{\alpha - \beta}{\beta}\right)$（令 $\mathrm{d}R/\mathrm{d}d = 0$ 得到）。超过 $d^*$ 后,真实质量随 KL 增大而下降。

**关键 scaling 结论**:$\alpha, \beta$ 的系数随 RM 参数量**平滑变化**——更大的 RM 有更高的 $d^*$,即过优化"临界点"更晚到来,但临界点一定存在。这说明**增大 RM 不能消除过优化风险,只能推迟**。

**直觉**:RM 是在有限数据上拟合的近似,策略在 RM 的分布外区域(高 KL 处)找到了"钻空子"的捷径。这是 Goodhart's Law 在 RL 中的定量表述。

**实践防线**:设置 KL 惩罚系数 $\beta$;定期用 held-out gold reward(如人工评估或可验证答案)检查;避免在单一 RM 上迭代轮次过多。

---

### Q6. 为什么 Ground-truth 可验证器(RLVR)比学习型 judge 更安全?

RLVR(Reinforcement Learning with Verifiable Rewards)由 DeepSeekMath<span class="cite-wrap"><a class="cite" id="fnref-11" href="#ref-11">11</a></span> 系统化地应用于数学推理:对于有确定性答案的任务(数学题、代码单元测试),直接用程序化检查结果正确性作为奖励信号,而非训练一个 reward model。

安全性对比:

| 维度 | 学习型 Judge / RM | Ground-truth 可验证器 |
|---|---|---|
| 信号真实性 | 近似(有拟合误差) | 精确(规则/符号执行) |
| 过优化风险 | 高(可被策略"钻空子") | 极低(答案对错是二值事实) |
| 分布外泛化 | 在训练分布外不可靠 | 与策略分布无关,始终可信 |
| 盲点继承 | 可能与生成器共享盲点 | 无参数,无盲点 |
| 适用范围 | 广泛(但不精确) | 仅限可机械验证的任务 |

**为什么"钻空子"对 RM 容易而对可验证器难**:RM 的分布外行为未被约束,策略可以找到 RM 未见过的"高分但低质"输出。而程序化验证器只看最终结果是否符合规范——策略无法在"规范本身"上作弊(规范是外生的)。

**局限**:RLVR 的适用范围取决于任务的可验证性。自然语言生成、摘要、创意写作等任务没有唯一正确答案,无法直接套用。因此 RLVR 与学习型奖励并非替代关系,而是互补——在可验证任务上优先用 RLVR,在开放生成任务上必须依赖 RM + KL 约束。

---

### Q7. 自我改进何时停滞?探索/多样性的角色是什么?如何经验性区分真实改进与 reward hacking?

**停滞的三种根源**:

1. **过滤信号饱和**:当模型对训练集中的所有问题都能答对,rejection sampling 产出的正确样本与已有训练数据几乎重复——梯度信号趋近于零。
2. **分布收窄导致探索不足**:如 Q2 所述,分布熵下降后,模型不再采样到足够多样的 rationale,难以从错误路径中恢复或发现新策略。
3. **任务难度超出 bootstrapping 能力**:对模型完全无能的问题,无法通过拒绝采样产出任何正确样本;需要外部课程(更简单的子问题、更强的教师模型)。

**探索/多样性的角色**:自我改进本质上是一个**exploitation-exploration 权衡**。每轮只保留正确样本是纯 exploitation;为维持改进,需要:
- **温度调高**:采样更多多样路径,以更大方差换取更高概率命中新的正确路径。
- **多样性 reward**:在优化目标里加入熵正则或多样性项,防止模式坍塌。
- **课程学习**:逐步引入更难问题,而不是在固定集上反复迭代。

**区分真实改进与 reward hacking 的经验性方法**:

| 指标 | 真实改进的信号 | Reward hacking 的信号 |
|---|---|---|
| Proxy reward vs Gold reward | 两者同步上升 | Proxy 涨但 Gold reward 持平或下降 |
| Held-out 评估集 | 在**未见题型**上也涨点 | 只在训练分布内涨,分布外下降 |
| 人工抽查输出质量 | 推理步骤质量可见提升 | 表面流畅,步骤逻辑漏洞增多 |
| 输出多样性 | 分布熵维持或小幅下降 | 分布熵快速崩溃,输出高度重复 |
| KL 散度趋势 | 缓慢增长且与 Gold 正相关 | KL 快速增大,超过 Gao et al. 的 $d^*$ |

黄金标准始终是:**保留一个完全未被自训练过程接触的 held-out 评估集,定期用可信 oracle(人工或可验证答案)打分。** 若这个分数持续上升,才能认定为真实改进。

---

## 参考文献 / References

> 均为经典承重方法的原始出处,已逐条核对(标题 + arXiv ID)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Zelikman et al. <em>STaR: Bootstrapping Reasoning With Reasoning</em>. 2022. <a href="https://arxiv.org/abs/2203.14465">arXiv:2203.14465</a> — 迭代微调在正确 chain-of-thought 上,无需大规模 rationale 标注. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Gulcehre et al. <em>Reinforced Self-Training (ReST) for Language Modeling</em>. 2023. <a href="https://arxiv.org/abs/2308.08998">arXiv:2308.08998</a> — Grow-Improve 离线 RL 循环,比在线 RLHF 样本高效. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Yuan et al. <em>Self-Rewarding Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.10020">arXiv:2401.10020</a> — 同一模型兼任生成器与 LLM-as-Judge,用迭代 DPO 协同提升两者. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Chen et al. <em>Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models</em>. 2024. <a href="https://arxiv.org/abs/2401.01335">arXiv:2401.01335</a> — SPIN:用前一轮自身作对手,仅需 SFT 数据即可自我改进. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Bai et al. <em>Constitutional AI: Harmlessness from AI Feedback</em>. 2022. <a href="https://arxiv.org/abs/2212.08073">arXiv:2212.08073</a> — 宪法引导自我批评与修订;RLAIF 用 AI preference 替代人工无害性标注. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Shinn et al. <em>Reflexion: Language Agents with Verbal Reinforcement Learning</em>. 2023. <a href="https://arxiv.org/abs/2303.11366">arXiv:2303.11366</a> — 语言反思存入 episodic memory,无权重更新的多轮自我纠错. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Madaan et al. <em>Self-Refine: Iterative Refinement with Self-Feedback</em>. 2023. <a href="https://arxiv.org/abs/2303.17651">arXiv:2303.17651</a> — 冻结模型自循环:生成→批评→修订,无训练跨任务涨点. <a href="#fnref-7">↩</a></li>
<li id="ref-8">Gao et al. <em>Scaling Laws for Reward Model Overoptimization</em>. 2022. <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a> — proxy reward 与 gold reward 随 KL 增大的分离曲线;RM 规模缩放律. <a href="#fnref-8">↩</a></li>
<li id="ref-9">Lightman et al. <em>Let's Verify Step by Step</em>. 2023. <a href="https://arxiv.org/abs/2305.20050">arXiv:2305.20050</a> — 过程监督(PRM)优于结果监督(ORM);PRM800K 数据集. <a href="#fnref-9">↩</a></li>
<li id="ref-10">Shumailov et al. <em>The Curse of Recursion: Training on Generated Data Makes Models Forget</em>. 2023. <a href="https://arxiv.org/abs/2305.17493">arXiv:2305.17493</a> — 递归训练于自产数据导致分布尾部消失(model collapse). <a href="#fnref-10">↩</a></li>
<li id="ref-11">Shao et al. <em>DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models</em>. 2024. <a href="https://arxiv.org/abs/2402.03300">arXiv:2402.03300</a> — RLVR(可验证奖励 RL)+ GRPO;程序化验证替代学习型 RM. <a href="#fnref-11">↩</a></li>
</ol>
