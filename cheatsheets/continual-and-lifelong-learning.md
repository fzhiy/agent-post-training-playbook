# Continual & Lifelong Learning / 持续与终身学习

> 模型在**序列任务**上持续学习时,如何既习得新知识又不遗忘旧知识?这是"一次性训练后部署"范式以外的必答题,也是 LLM post-training 链条(pretrain → SFT → DPO → RL)的隐患所在。
>
> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。数字 / 结论以原论文为准,不确定处标注。

## 0. 一句话演化 / The evolution

`IID 一次性训练` → `序列多任务(任务 1 → 任务 2 → …)` → **`灾难性遗忘出现`**:梯度直接覆盖旧权重。

**stability-plasticity dilemma**:网络既要**可塑(plasticity)**——接受新任务的梯度更新;又要**稳定(stability)**——保住旧任务的表征。两者天然冲突。

## 1. 为什么会遗忘 / Why catastrophic forgetting happens

神经网络的参数是所有任务的**共享存储**。在任务 $\mathcal{T}_2$ 上做 SGD 时,损失对参数的梯度不知道"这些权重对 $\mathcal{T}_1$ 很重要",于是把它们覆盖掉——这就是 **catastrophic forgetting**。

经典设定:

| 设定 | 任务 ID 是否已知 | 任务边界是否清晰 |
|---|---|---|
| Task-IL | 测试时已知 | 是 |
| Domain-IL | 测试时未知 | 是 |
| Class-IL(最难) | 测试时未知 | 是 |
| Continual pretraining | 无明确边界 | 否 |

## 2. 三大方法族 / Three method families

### 2.1 正则化方法 / Regularization

**核心思想**:在新任务的损失上加一个**保护旧权重**的惩罚项,让"对旧任务重要的权重"尽量不动。

**EWC(Elastic Weight Consolidation)**<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">用 Fisher 信息矩阵对角线衡量权重重要性,以二次惩罚防遗忘。<a href="https://arxiv.org/abs/1612.00796">Kirkpatrick 2017 ↗</a></span></span> 的总损失:

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\mathcal{T}_2}(\theta) + \frac{\lambda}{2} \sum_i F_i \,(\theta_i - \theta_i^*)^2$$

- $\theta_i^*$:完成旧任务后的参数值
- $F_i$:Fisher 信息矩阵的对角元素(衡量参数 $i$ 对旧任务的"重要性")
- $\lambda$:超参,控制稳定性 vs. 可塑性的权衡

$$F_i = \mathbb{E}_{\mathcal{D}_1}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**SI(Synaptic Intelligence)**:在线版 EWC,训练过程中**累积**每个参数对损失的贡献,不需要重新计算 Fisher。

**MAS(Memory Aware Synapses)**:重要性由输出函数对参数的梯度范数估计,不依赖标注数据。

| 方法 | 重要性估计 | 是否需要旧数据 | 计算开销 |
|---|---|---|---|
| EWC | Fisher 对角线 | 否 | 中(一次后向) |
| SI | 在线轨迹积分 | 否 | 低(训练时顺带) |
| MAS | 输出梯度范数 | 否(只需无标注输入) | 低~中 |

### 2.2 回放 / Rehearsal & Replay

**核心思想**:在新任务训练时**混入旧任务样本**,让梯度"同时记得"过去。

**Experience Replay**:维护一个**episodic memory** buffer,存放旧任务的真实样本;新任务训练时按比例随机混入。

**GEM(Gradient Episodic Memory)**<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">把梯度更新投影到"不增加旧任务损失"的约束可行域内,同时允许向前迁移。<a href="https://arxiv.org/abs/1706.08840">Lopez-Paz 2017 ↗</a></span></span>:

对每个旧任务 $k$,要求更新后的梯度满足:
$$\langle g, g_k \rangle \geq 0$$
即新梯度与旧任务梯度不"反向"。如果违反,则将 $g$ 投影到约束可行域。

**A-GEM(Averaged GEM)**<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">把多个旧任务约束合并为一个平均梯度约束,大幅降低计算量,效果与 GEM 相当。<a href="https://arxiv.org/abs/1812.00420">Chaudhry 2019 ↗</a></span></span>:用所有旧任务 buffer 的**平均梯度** $g_{\text{ref}}$ 作为唯一约束,将每步的 QP 投影从 $K$ 个降为 1 个:

$$\langle g, g_{\text{ref}} \rangle \geq 0, \quad g_{\text{ref}} = \frac{1}{K}\sum_k g_k$$

**Generative Replay**:用一个**生成模型**(如 VAE、GAN)学旧任务的分布,需要时合成"伪旧数据"注入训练——无需真实保存旧样本,但生成质量瓶颈会积累误差。

**DER(Dark Experience Replay)**<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">在 buffer 中存储旧样本的 logit(软目标),以 MSE 匹配软目标,将 rehearsal 与知识蒸馏融合。<a href="https://arxiv.org/abs/2004.07211">Buzzega 2020 ↗</a></span></span>:buffer 同时存 $(x, y, z)$,其中 $z$ 是过去时刻模型的 **logit**(dark knowledge),回放时同时匹配 $z$:

$$\mathcal{L}_{\text{DER}} = \mathcal{L}_{\text{CE}}(x,y) + \alpha \cdot \text{MSE}(f_\theta(x),\, z)$$

### 2.3 参数隔离 / Parameter Isolation & Architectural CL

**核心思想**:不同任务占据**不同子网络**,新任务扩展而不改变旧任务参数——从结构上消除遗忘。

**Progressive Neural Networks**<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">每来一个新任务就新增一列网络,通过侧向连接利用旧列知识,旧列完全冻结,遗忘在结构上不可能发生。<a href="https://arxiv.org/abs/1606.04671">Rusu 2016 ↗</a></span></span>:每个任务一列独立网络,旧列冻结,新列通过 **lateral connection** 读取旧列的激活:

$$h_k^{(\ell)} = f\!\left(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)}\right)$$

- 优点:零遗忘,天然正向迁移
- 缺点:参数随任务数线性增长

**PackNet**:在单个网络内做**迭代剪枝+固定**,为每个任务分配一组参数掩码,任务间完全不共享梯度路径。

**LoRA-based / Adapter-based CL**:每个任务增量添加一套 LoRA 权重或 adapter 模块,主干冻结,任务间通过 task ID 路由到对应 adapter——参数量可控,且对 LLM 友好。

| 方法族 | 零遗忘 | 正向迁移 | 参数增长 | 需存旧数据 |
|---|---|---|---|---|
| 正则化(EWC/SI/MAS) | 近似 | 有限 | 无 | 否 |
| Replay(ER/GEM/A-GEM/DER) | 近似 | 有 | buffer | 是(部分) |
| 参数隔离(ProgNN/PackNet/LoRA-CL) | 是 | 有限~有 | 线性~轻量 | 否 |

## 3. 知识蒸馏路线 / Knowledge Distillation: LwF

**LwF(Learning without Forgetting)**<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">训练新任务时,把旧模型的软输出作为蒸馏目标,不需要存储任何旧数据就能缓解遗忘。<a href="https://arxiv.org/abs/1606.09282">Li 2016 ↗</a></span></span>:

- 在新任务 $\mathcal{T}_2$ 的数据上推理时,先用**旧模型** $f_{\theta^*}$ 产生 soft output $\hat{y}^{\text{old}}$
- 联合优化:

$$\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{new}}(f_\theta(x), y) + \lambda \cdot \mathcal{L}_{\text{KD}}(f_\theta(x), \hat{y}^{\text{old}})$$

- $\mathcal{L}_{\text{KD}}$ 通常是温度缩放的 KL 散度或交叉熵
- **无需旧数据**;缺点:任务漂移大时软目标质量下降

LwF 本质是**隐式回放旧模型的知识**,而非旧数据——在隐私敏感或存储受限场景很吸引人。

## 4. 评估指标 / Evaluation Metrics

设完成 $T$ 个任务,完成第 $i$ 个任务后对任务 $j$ 的准确率记为 $a_{i,j}$。

**Average Accuracy(AA)**:学完所有任务后对所有任务的平均准确率:

$$\text{AA} = \frac{1}{T} \sum_{j=1}^{T} a_{T,j}$$

**Backward Transfer(BWT)** — 负值越大表示遗忘越严重:

$$\text{BWT} = \frac{1}{T-1} \sum_{j=1}^{T-1} \bigl(a_{T,j} - a_{j,j}\bigr)$$

**Forward Transfer(FWT)** — 正值表示旧任务帮助了新任务:

$$\text{FWT} = \frac{1}{T-1} \sum_{j=2}^{T} \bigl(a_{j-1,j} - b_j\bigr)$$

其中 $b_j$ 是从随机初始化独立训练任务 $j$ 的基线准确率。

| 指标 | 衡量什么 | 理想值 |
|---|---|---|
| AA | 整体记忆保留 | 越高越好 |
| BWT | 遗忘程度 | ≥ 0(越接近 0 或 > 0 越好) |
| FWT | 正向迁移 | > 0 |

> **Forgetting** 有时直接定义为每个任务"学完时 vs. 最终"的准确率下降均值,与 BWT 互为正负。

## 5. LLM 视角 / The LLM Angle

### 5.1 Continual Pretraining

在初始预训练后用新领域语料继续训练语言模型(如医疗文本、代码更新),核心挑战:

- 旧通用能力(数学推理、指令遵循)可能退化
- 新语料的分布偏移可能比任务级偏移更温和,但持续时间更长
- 常见缓解:学习率 warm-up restart、replay 少量通用数据、适当减小学习率

### 5.2 Continual Instruction-Tuning

按序加入新指令类型(如先 coding → 再 math → 再 safety),每轮 SFT 都可能覆盖之前微调的行为。LoRA-based 或 adapter-per-task 是低参数量的天然解法:主干共享,行为通过模块路由分离。

### 5.3 Continual Alignment & Alignment Tax

**序列对齐链条**:`pretrain → SFT → DPO → RL(RLHF/RLVR)` 中每一步都是在前一步的 checkpoint 上继续训练。

**Alignment tax**:对齐往往以牺牲部分通用能力为代价(如代码生成、factuality)。序列叠加时税率累积:

- SFT 过度可能压缩知识多样性
- DPO 后再做 RLHF 可能导致 over-refusal 或格式退化
- 每一跳都面临"新对齐目标 vs. 前一跳已学的行为"的 CL 问题

**缓解策略**:

1. **KL 约束**(PPO clip / DPO reference model)——限制每步偏离参考的幅度,本质是 EWC 的隐式类比
2. **Replay 旧偏好数据**——混入前序对齐阶段的数据
3. **LoRA 每阶段独立 adapter**——主干不动,对齐行为局部化

### 5.4 为什么 CL 对 LLM post-training 重要

| 场景 | CL 挑战 |
|---|---|
| 新版本模型增量更新 | 不能全量重训,需增量微调不遗忘旧能力 |
| 多轮 RLHF 迭代 | 每轮 policy 更新可能覆盖前轮对齐 |
| 个性化 / 持续用户反馈 | 适应新用户偏好同时保住通用能力 |
| 知识更新(时效信息) | 注入新事实同时不干扰旧知识结构 |

## 6. 从零实现:EWC 二次惩罚 / From-scratch EWC

```python
import torch
import torch.nn.functional as F

def compute_fisher_diagonal(model, dataloader, device="cpu"):
    """
    用对数似然梯度的平方估计 Fisher 信息矩阵对角线。
    dataloader: 旧任务数据; model: 完成旧任务后的模型(参数已固定)。
    返回 dict: param_name -> F_i (shape 同参数)
    """
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    n_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        logits = model(x)
        log_prob = F.log_softmax(logits, dim=-1)
        # 用预测标签近似采样——这是一种简化实现；EWC 原文实际用真实标签 y 估计 Fisher (empirical Fisher = E_{(x,y)~D}[∇log p(y|x)^2])
        pred = log_prob.argmax(dim=-1)
        loss = F.nll_loss(log_prob, pred, reduction="sum")
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach().pow(2)
        n_samples += x.size(0)

    fisher = {n: f / n_samples for n, f in fisher.items()}
    return fisher


class EWCTrainer:
    """
    在新任务上训练时叠加 EWC 二次惩罚,防止旧任务参数被覆盖。
    """
    def __init__(self, model, old_dataloader, ewc_lambda=5000.0, device="cpu"):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda

        # 保存旧任务参数快照
        self.theta_star = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
        }
        # 计算 Fisher 对角线
        self.fisher = compute_fisher_diagonal(model, old_dataloader, device)

    def ewc_penalty(self):
        """EWC 惩罚项: (λ/2) Σ_i F_i (θ_i - θ*_i)^2"""
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            penalty = penalty + (
                self.fisher[n] * (p - self.theta_star[n]).pow(2)
            ).sum()
        return 0.5 * self.ewc_lambda * penalty

    def train_step(self, x, y, optimizer, criterion):
        """新任务上的一步训练:任务损失 + EWC 惩罚。"""
        x, y = x.to(self.device), y.to(self.device)
        optimizer.zero_grad()
        logits = self.model(x)
        task_loss = criterion(logits, y)
        loss = task_loss + self.ewc_penalty()
        loss.backward()
        optimizer.step()
        return task_loss.item(), loss.item()
```

> `ewc_lambda` 是核心超参:太小遗忘依旧严重;太大新任务无法学习。实践中通常在 $[100, 10^4]$ 范围内搜索,并随任务数累积多组 Fisher(online EWC 用滑动平均合并)。

---

## 分层面试题 / Stratified follow-ups

### L1 基础
1. 什么是 catastrophic forgetting?为什么神经网络特别容易出现?
2. stability-plasticity dilemma 是什么?给出一个直觉类比。
3. EWC 的核心思想是什么?Fisher 信息矩阵在里面扮演什么角色?
4. LwF 和 EWC 都不存旧数据,它们防遗忘的机制有何不同?

### L2 进阶
5. Replay 方法中 GEM 和 A-GEM 的核心区别是什么?A-GEM 为什么更高效?
6. BWT 和 FWT 各衡量什么?一个模型 BWT 很负但 FWT 很正说明什么?
7. Progressive Networks 为什么做到了"零遗忘"?代价是什么?
8. LoRA-based CL 相比 EWC / replay 有什么优势?在 LLM post-training 链条里怎么用?

### L3 深挖
9. LLM 的 continual pretraining 和经典 Task-IL 设定的主要差异在哪里?有哪些特有的挑战?
10. 序列对齐链条(SFT → DPO → RL)中的 alignment tax 本质上是什么 CL 问题?有哪些方案缓解?
11. EWC 用的是 empirical Fisher(用预测标签)而非 true Fisher(用真实标签)——这在什么情况下会出问题?
12. Generative Replay 相比 experience replay 的优点和局限各是什么?在大模型时代是否更可行?

---

## 参考文献 / References

> 均为经典承重方法的原始出处,已逐条核对(标题 + arXiv ID)。点上标跳转、点 ↩ 返回。

<ol>
<li id="ref-1">Kirkpatrick et al. <em>Overcoming catastrophic forgetting in neural networks</em>. PNAS 2017. <a href="https://arxiv.org/abs/1612.00796">arXiv:1612.00796</a> — EWC:Fisher 对角线二次惩罚防遗忘. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Lopez-Paz et al. <em>Gradient Episodic Memory for Continual Learning</em>. NeurIPS 2017. <a href="https://arxiv.org/abs/1706.08840">arXiv:1706.08840</a> — GEM:梯度投影约束 + forward/backward transfer. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Chaudhry et al. <em>Efficient Lifelong Learning with A-GEM</em>. ICLR 2019. <a href="https://arxiv.org/abs/1812.00420">arXiv:1812.00420</a> — A-GEM:单一平均梯度约束,高效 GEM. <a href="#fnref-3">↩</a></li>
<li id="ref-4">Buzzega et al. <em>Dark Experience for General Continual Learning: a Strong, Simple Baseline</em>. NeurIPS 2020. <a href="https://arxiv.org/abs/2004.07211">arXiv:2004.07211</a> — DER:存 logit 做软目标蒸馏 + rehearsal. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Rusu et al. <em>Progressive Neural Networks</em>. 2016. <a href="https://arxiv.org/abs/1606.04671">arXiv:1606.04671</a> — 每任务新增列 + 侧向连接,结构性零遗忘. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Li and Hoiem. <em>Learning without Forgetting</em>. ECCV 2016. <a href="https://arxiv.org/abs/1606.09282">arXiv:1606.09282</a> — LwF:旧模型软输出作蒸馏目标,无需旧数据. <a href="#fnref-6">↩</a></li>
</ol>
