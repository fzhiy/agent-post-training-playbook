# Continual & Lifelong Learning / 持续与终身学习

> 模型在**序列任务**上持续学习时,如何既习得新知识又不遗忘旧知识?这是"一次性训练后部署"范式以外的必答题,也是 LLM post-training 链条(pretrain → SFT → DPO → RL)的隐患所在。

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

<details class="qa"><summary>1. 什么是 catastrophic forgetting?为什么神经网络特别容易出现?</summary>

答:神经网络的参数是所有任务的**共享存储**,在新任务 $\mathcal{T}_2$ 上 SGD 时梯度不知道哪些权重对旧任务 $\mathcal{T}_1$ 重要,直接覆盖它们,导致旧任务性能骤降——这就是 catastrophic forgetting。关键原因是参数共享加上独立优化目标:新任务梯度对旧任务损失来说是"噪声"。

**追问：** 从优化理论的角度看,catastrophic forgetting 的本质是什么? → 本质是新任务的梯度下降方向在旧任务的损失曲面上是上升方向——两个任务的最优参数区域在参数空间中不重叠,SGD 沿新任务梯度移动时同时破坏旧任务的局部极值,即多目标优化中的 Pareto 冲突在单一参数共享下的退化形式。

</details>

<details class="qa"><summary>2. stability-plasticity dilemma 是什么?给出一个直觉类比。</summary>

答:网络需要同时具备 **plasticity(可塑性)**——接受新任务梯度更新,和 **stability(稳定性)**——保住旧任务表征;两者天然冲突。直觉类比:学一门新语言时大脑既要记住母语(stability)又要塑造新语言回路(plasticity),死记硬背新语言可能挤压母语的神经路径。

**追问：** 三大方法族(正则化 / Replay / 参数隔离)在 stability-plasticity 轴上各偏向哪端?如何根据任务需求选择? → 正则化(EWC/SI)偏 stability:惩罚项约束参数偏离,plasticity 受限;Replay 居中:通过混入旧数据平衡两端,但 buffer 大小决定偏向;参数隔离(ProgNN/LoRA-CL)最偏 stability:旧参数完全冻结,新任务只扩展新容量。当任务间相关性高、正向迁移有价值时选 Replay;当任务独立且存储/隐私受限时选参数隔离。

</details>

<details class="qa"><summary>3. EWC 的核心思想是什么?Fisher 信息矩阵在里面扮演什么角色?</summary>

答:EWC(Elastic Weight Consolidation)在新任务损失上加一个二次惩罚项 $\frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$,让"对旧任务重要的权重"尽量不变。Fisher 信息矩阵对角线 $F_i = \mathbb{E}[(\partial \log p_\theta / \partial \theta_i)^2]$ 衡量参数 $i$ 对旧任务的**重要性**——$F_i$ 越大则惩罚越强,该参数被"弹性"保护。

**追问：** Fisher 信息矩阵对角线近似的主要局限是什么?有哪些改进的替代方案? → 对角近似忽略参数间协方差,当两个参数对旧任务损失有强耦合重要性时(如 attention 的 Q/K 权重),单独惩罚每个参数会低估真实曲率。改进方向包括:块对角近似(K-FAC,按层分块保留同层内协方差)、Kronecker 分解近似,以及 SI 的在线轨迹积分(用参数对损失下降的实际贡献替代 Fisher,避免事后重算)。

</details>

<details class="qa"><summary>4. LwF 和 EWC 都不存旧数据,它们防遗忘的机制有何不同?</summary>

答:EWC 在**参数空间**施加约束——用 Fisher 惩罚让旧任务重要权重不偏离旧值;LwF(Learning without Forgetting)在**输出空间**施加约束——用旧模型在新任务数据上的 soft output 作蒸馏目标 $\mathcal{L}_{\text{KD}}$,让新模型保持旧模型的输出行为。LwF 无需重新计算重要性分数,但旧模型软目标质量随任务漂移而退化。

**追问：** LwF 在什么情况下防遗忘效果会严重退化?如何缓解? → 当新旧任务输入分布差异极大时(如旧任务是图像分类、新任务是文本),旧模型对新任务数据的软输出趋于均匀分布或置信度极低,蒸馏信号质量接近随机——等同于没有约束。缓解方法:混合使用参数空间约束(EWC 惩罚重要权重)补充输出空间蒸馏;或仅在新旧任务输入空间有重叠的子集上计算 $\mathcal{L}_{\text{KD}}$,过滤低置信软目标。

</details>

### L2 进阶

<details class="qa"><summary>5. Replay 方法中 GEM 和 A-GEM 的核心区别是什么?A-GEM 为什么更高效?</summary>

答:GEM 对每个旧任务 $k$ 单独施加梯度约束 $\langle g, g_k\rangle \geq 0$,需求解含 $(K-1)$ 个不等式的 QP,时间复杂度 $\mathcal{O}(K^3)$。A-GEM 将所有旧任务约束合并为单一均值约束 $\langle g, g_{\text{ref}}\rangle \geq 0$,投影有闭合解,每步计算量从 $\mathcal{O}(K \cdot d)$ 降为 $\mathcal{O}(d)$,与任务数无关。

**追问：** A-GEM 的均值约束在 buffer 采样噪声下稳定性如何?有哪些改进方向? → A-GEM 每步从 buffer 中随机采样估计 $g_{\text{ref}}$,高方差采样会使约束方向不稳定——某步估计的 $g_{\text{ref}}$ 与真实均值偏差大时,投影可能错误方向,引入额外遗忘噪声。改进方向包括:增大每步 buffer 采样量以降低方差;用动量平滑历史 $g_{\text{ref}}$(类似 EMA);以及 ER-ACE 等后续工作通过 asymmetric cross-entropy 绕开梯度投影整体框架。

</details>

<details class="qa"><summary>6. BWT 和 FWT 各衡量什么?一个模型 BWT 很负但 FWT 很正说明什么?</summary>

答:BWT(Backward Transfer) $= \frac{1}{T-1}\sum_{j=1}^{T-1}(a_{T,j}-a_{j,j})$ 衡量遗忘程度,越负说明旧任务性能下降越严重。FWT(Forward Transfer) $= \frac{1}{T-1}\sum_{j=2}^{T}(a_{j-1,j}-b_j)$ 衡量旧任务对新任务的正向迁移,越正说明迁移越好。BWT 很负但 FWT 很正,说明该模型在新任务上利用了旧知识加速学习(迁移好),但同时严重覆盖了旧任务参数(遗忘严重)——典型的高可塑、低稳定模型。

**追问：** 在评估 LLM 的持续学习能力时,直接使用 BWT 和 FWT 指标有哪些局限性? → LLM 任务边界模糊(SFT/DPO/RL 是软边界而非离散任务序列),难以构建明确的 $a_{i,j}$ 性能矩阵;LLM 的"能力"是多维的(推理/代码/安全),单一准确率无法捕捉能力干扰;此外 stability gap(D3)意味着任务末尾的快照低估了训练过程中的最坏遗忘情况——对部署安全性而言,峰值遗忘幅度比 BWT 更关键。

</details>

<details class="qa"><summary>7. Progressive Networks 为什么做到了"零遗忘"?代价是什么?</summary>

答:Progressive Neural Networks 每来一个新任务就新增一列独立网络,旧列参数完全冻结,新列通过 lateral connection 读取旧列激活 $h_k^{(\ell)} = f(W_k^{(\ell)} h_k^{(\ell-1)} + \sum_{j<k} U_{k,j}^{(\ell)} h_j^{(\ell-1)})$——旧列梯度路径被切断,遗忘在结构上不可能发生。代价是参数量随任务数线性增长。

**追问：** LoRA-based CL 相比 Progressive Networks 如何缓解参数线性增长?两者零遗忘保证有何本质差异? → LoRA-CL 每任务只增加低秩矩阵 $\Delta W = BA$(秩 $r \ll d$),参数增量为 $\mathcal{O}(r \cdot d)$ 而非 $\mathcal{O}(d^2)$,可扩展性远优于整列扩展。但零遗忘保证的性质不同:ProgNN 靠冻结旧列在结构上隔绝梯度,是硬保证;LoRA-CL 靠主干冻结+低秩子空间分配,若不强制正交(如 O-LoRA)则不同任务 adapter 激活可能干扰——是软保证,依赖子空间重叠程度。

</details>

<details class="qa"><summary>8. LoRA-based CL 相比 EWC / replay 有什么优势?在 LLM post-training 链条里怎么用?</summary>

答:LoRA-based CL 为每个任务增量添加一套低秩 adapter,主干冻结——参数量可控且对旧任务零梯度干扰,无需存旧数据也无需计算 Fisher。在 LLM post-training 链条(SFT → DPO → RL)中,每阶段冻结主干只更新一套 LoRA,将 alignment tax 限制在 adapter 层内,主干通用知识不被覆盖;测试时按 task-ID 路由到对应 adapter。

**追问：** 在 LLM 中使用 LoRA-based CL,部署时管理多套 adapter 会带来哪些工程挑战?O-LoRA 的正交约束能否从根本上消除这一问题? → 工程挑战包括:需存储并按 task-ID 动态加载 adapter(增加推理延迟和内存调度复杂度)、batch 内混合任务时无法合并 adapter 权重、以及 Class-IL / continual pretraining 设定下 task-ID 未知时路由失效。O-LoRA 的正交约束仅解决 adapter 间激活干扰问题,不解决路由失效——在无 task-ID 场景下仍需额外机制(如 prototype 分类器或任务推断模块)才能确定使用哪套 adapter。

</details>

### L3 深挖

<details class="qa"><summary>9. LLM 的 continual pretraining 和经典 Task-IL 设定的主要差异在哪里?有哪些特有的挑战?</summary>

答:经典 Task-IL 有明确任务边界和任务 ID;LLM continual pretraining 无明确边界,领域语料持续流入,任务粒度模糊(通用能力与新领域大量重叠)。特有挑战包括:旧通用能力(数学推理、指令遵循)可能以难以检测的方式退化、新语料的分布偏移持续时间长、无法用标准 $a_{i,j}$ 矩阵量化遗忘,以及学习率 warm-up restart 和 replay 通用数据的比例难以调优。

**追问：** 在无明确任务边界的 LLM continual pretraining 中,如何实时监控并诊断通用能力的退化? → 需设计一套"探针 benchmark 矩阵"(覆盖推理/代码/数学/指令遵循等维度),在训练过程中按固定步数间隔评测——相当于在 LLM 尺度实现 per-iteration 连续评估(参见 stability gap D3);同时结合梯度/激活漂移检测定位哪些层被新语料大幅改写,以便决定是否触发 replay 通用数据或调低学习率。代价是显著的额外计算开销,需在监控频率与训练效率间权衡。

</details>

<details class="qa"><summary>10. 序列对齐链条(SFT → DPO → RL)中的 alignment tax 本质上是什么 CL 问题?有哪些方案缓解?</summary>

答:alignment tax 本质是序列 CL 中每一跳的遗忘税累积——每步对齐目标都是在前一步 checkpoint 上继续训练,新对齐目标的梯度覆盖前序已学行为。缓解方案：(1) **KL 约束**(PPO clip / DPO reference model)限制每步偏离幅度,其二阶展开等价于以 Fisher 为权重的 EWC；(2) **Replay 旧偏好数据**混入前序阶段样本；(3) **LoRA 每阶段独立 adapter** 将对齐行为局部化,主干不变。

**追问：** PPO 中的 KL 惩罚项在数学上是隐式 EWC,但在实践中两者有哪些关键差异会影响防遗忘效果? → 三点关键差异:(1) **锚点动态性**:KL 约束锚定动态更新的 reference 策略 $\pi_{\text{ref}}$,每轮 RL 可能更新 reference;EWC 锚点 $\theta^*$ 是旧任务完成后的固定快照——动态锚点在长序列对齐中会导致"锚点漂移",防遗忘中心持续移动。(2) **约束全矩阵 vs. 对角近似**:KL 散度用完整 Fisher 矩阵,EWC 用对角近似——前者约束更精确但计算不显式。(3) **惩罚系数调节**:PPO 的 $\beta$ 需在探索与保守间动态平衡,过大导致策略不收敛;EWC 的 $\lambda$ 在任务固定后通常静态设置——动态对齐场景下 $\beta$ 的调优难度更高。

</details>

<details class="qa"><summary>11. EWC 用的是 empirical Fisher(用真实标签)而非 true Fisher(用模型采样标签)——这在什么情况下会出问题?</summary>

答:Empirical Fisher $F_i^{\text{emp}} = \mathbb{E}_{(x,y)\sim\mathcal{D}}[(\partial \log p_\theta(y|x)/\partial \theta_i)^2]$ 用数据集真实标签 $y$,只在模型完美拟合时与 true Fisher 等价。出问题的场景：模型对旧任务远未收敛时 $F^{\text{emp}}$ 估计噪声大、旧任务标签噪声高时方向被污染、多任务在线累积(online EWC)时早期误差持续传播、以及任务间分布差异极大时 diagonal 近似与 empirical 误差双重劣化——保护了错误方向。

**追问：** 除了切换到 true Fisher,还有哪些结构性替代方案可以从根本上绕开 diagonal empirical Fisher 的局限? → 两类根本性替代:(1) **换掉重要性度量**:SI 用参数对损失下降的实际贡献积分 $\Omega_i \propto \int \frac{\partial \mathcal{L}}{\partial \theta_i} \dot\theta_i\, dt$ 替代 Fisher——不依赖标签且在线累积,规避 empirical/diagonal 双重误差;MAS 用输出梯度范数替代,无需任何标注。(2) **换掉二次惩罚整体框架**:PackNet / LoRA-CL 的参数隔离方案完全不需要估计重要性——旧任务参数被结构性冻结,重要性估计的精度问题从根本上消失。这说明 Fisher 估计的局限性是正则化范式的系统性问题,而非可无限改进的工程问题。

</details>

<details class="qa"><summary>12. Generative Replay 相比 experience replay 的优点和局限各是什么?在大模型时代是否更可行?</summary>

答:Generative Replay 用生成模型(VAE/GAN)合成"伪旧数据"注入训练，**优点**是无需存储任何真实旧样本，天然解决隐私和存储约束。**局限**是生成质量瓶颈会随任务序列积累误差——生成模型本身也面临遗忘，合成数据与真实分布的偏差在长任务链上叠加。在大模型时代，LLM 本身的生成能力极强，用 LLM 合成旧任务数据的质量远高于小型 GAN——可行性显著提升，但生成成本高且合成数据仍可能与原始分布有系统偏差。

**追问：** 在 LLM 的 Generative Replay 中，用模型自身作为生成器("self-replay")与用独立冻结旧模型作为生成器相比，各有什么核心问题? → Self-replay(让当前模型生成旧任务数据再训练自身)存在**自我强化偏差**:模型在新任务训练后已有一定遗忘,生成的旧任务样本本身质量下降,再用这些样本训练会强化遗忘——形成负反馈循环。用**独立冻结旧模型**作为生成器(类似 LwF 的 teacher)可保证生成质量不随当前模型退化,但需额外存储一份完整模型副本,存储开销与 experience replay 的 buffer 相比可能更高;且冻结旧模型自身也不能随新数据分布更新,在长任务链上累积分布偏差。

</details>

---

## 深挖 / Deep-dive

> 以下为面试级进阶问答,涵盖上方笔记中最常被追问的 7 个难点。所有结论均来自所引论文,不含作者本人研究成果。

### D1. Empirical Fisher vs. True Fisher——EWC 用的是哪个,何时会崩?

**True Fisher** 定义为对数似然梯度外积关于**模型预测分布** $p_\theta(y|x)$ 的期望:

$$F_i^{\text{true}} = \mathbb{E}_{x \sim \mathcal{D},\; y \sim p_\theta(\cdot|x)}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

**Empirical Fisher** 用**数据集中的真实标签** $y$ 替代从模型分布采样的 $y$:

$$F_i^{\text{emp}} = \mathbb{E}_{(x,y) \sim \mathcal{D}}\!\left[\left(\frac{\partial \log p_\theta(y|x)}{\partial \theta_i}\right)^{\!2}\right]$$

两者只在模型完美拟合数据($p_\theta(y|x) \approx \delta_y$)时相等。已有研究指出,在优化过程中 empirical Fisher 不能一般性地捕获二阶信息,在实践中与真实 Hessian 的差异可能很大——即使在简单优化问题上也会出现病态。

**EWC 用的是 empirical Fisher**:实现时对旧任务数据集 $\mathcal{D}_1$ 中的 $(x,y)$ 对求梯度平方期望。代码注释中也已注明"用真实标签 $y$ 估计 Fisher"。

**何时近似会崩?**

| 场景 | 风险 |
|---|---|
| 模型对旧任务远未收敛时就计算 $F$ | $F^{\text{emp}}$ 估计噪声大,保护错了方向 |
| 旧任务标签噪声高 | $F^{\text{emp}}$ 方向被标签噪声污染 |
| 多任务序列累积(online EWC) | 早期任务的 $F^{\text{emp}}$ 误差在滑动平均中持续传播 |
| 任务间分布差异极大(如语言→视觉) | diagonal 近似本身已是强假设,加上 empirical 误差双重劣化 |

直觉:EWC 的二次惩罚本质是对参数空间作局部二次近似——diagonal empirical Fisher 是这个近似中最粗糙的一层。在旧任务已良好收敛、标签干净、参数相关性弱的情形下近似尚可;否则"重要性分数"与真实 loss landscape 曲率脱钩,惩罚保护了错误的方向。

---

### D2. Online/Streaming EWC——如何跨任务累积 Fisher?

标准 EWC 每遇到一个新旧任务对就存一组 $(F^{(k)}, \theta^{*(k)})$,内存随任务数 $K$ 线性增长。**Online EWC**(Progress & Compress,Schwarz et al. 2018)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">Progress & Compress 双网络框架:active column 学新任务,knowledge base 用 online EWC 压缩固化;Fisher 用指数移动平均跨任务累积。<a href="https://arxiv.org/abs/1805.06370">Schwarz 2018 ↗</a></span></span> 用**指数移动平均(EMA)**将所有历史任务的 Fisher 合并为单一 $\tilde{F}$:

$$\tilde{F}^{(k)} = \gamma \cdot \tilde{F}^{(k-1)} + (1-\gamma) \cdot F^{(k)}$$

总惩罚退化为单组惩罚:

$$\mathcal{L}_{\text{online-EWC}} = \mathcal{L}_{\mathcal{T}_k}(\theta) + \frac{\lambda}{2}\sum_i \tilde{F}_i^{(k-1)}\bigl(\theta_i - \theta_i^{*(k-1)}\bigr)^2$$

**优点**:内存恒定(只存一个 $\tilde{F}$ 和一个 $\theta^*$ 快照)。

**代价与风险**:

- EMA 对早期任务的 Fisher 权重呈指数衰减——任务越久远保护越弱,本质上倾向保护"最近"的任务。
- $\gamma$ 是新超参:$\gamma \to 1$ 保留历史但遗忘率高;$\gamma \to 0$ 退化为只看上一任务。
- 参考点 $\theta^*$ 每轮更新,每次压缩后新的 $\theta^*$ 并非所有历史任务的最优公共点——多任务积累下惩罚中心漂移。

**SI(Synaptic Intelligence)** 则是另一种流式方案:在训练过程中**在线累积**每个参数对损失下降的贡献积分作为重要性:

$$\Omega_i \propto \int_{\text{trajectory}} \frac{\partial \mathcal{L}}{\partial \theta_i} \cdot \dot\theta_i \, dt$$

SI 无需任务结束后的额外后向传播,适合**无明确任务边界**的 streaming 场景,但重要性估计的信噪比比 EWC 更低。

---

### D3. Stability Gap——为什么遗忘先变严重再恢复?

De Lange et al. (arXiv 2022, ICLR 2023)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">提出 per-iteration 连续评估框架,首次系统记录 stability gap:任务切换后性能骤降随后恢复,仅在任务末尾评估会错过这一现象。<a href="https://arxiv.org/abs/2205.13452">De Lange 2022 ↗</a></span></span> 通过 **per-iteration 连续评估**发现:几乎所有主流 CL 方法(包括 EWC、ER、A-GEM)在**切换到新任务的最初若干步**,旧任务性能会骤降——随后随训练进展逐渐恢复甚至超过切换前水平。这个"先跌后回"的现象被称为 **stability gap**。

**机制直觉:**

```
任务 T1 训练完毕 → 切换 T2
     ↓ 前几十步
T2 的大梯度冲击共享特征层 → T1 表征暂时失效 → T1 性能骤降
     ↓ 继续训练
正则化 / replay 约束开始发挥作用 → T1 表征逐渐修复
     ↓ T2 末尾
T1 性能恢复(但可能低于 T1 训练结束时的峰值)
```

**为什么之前没被发现?**

标准评估协议只在**每个任务学完后**测一次,正好跳过了骤降期——stability gap 在任务末尾的"快照"中几乎不可见。

**面试要点:**

- Stability gap 意味着 BWT 指标(任务末尾快照)**低估了真实的遗忘幅度**——在安全关键场景(部署中的 LLM 增量更新)中,训练过程的最差情况性能可能比 BWT 反映的严重得多。
- 缓解方向:warm-up 过渡期内降低新任务学习率;先用小 batch 对新任务"探测"再开全速训练;replay 在切换初期加大旧任务比例。

---

### D4. Replay 样本选择策略与 Buffer 大小效应

Buffer 大小 $M$ 是 replay 方法最关键的超参之一。三类主流选择策略:

**① 随机水库采样(Reservoir Sampling)**

对长度未知的数据流,维护大小为 $M$ 的 buffer,每个新样本 $x_t$ 以概率 $M/t$ 被纳入 buffer(随机替换一个旧样本)。保证 buffer 中每个样本是所有历史样本的均匀子集——无类别偏置,实现简单,是 ER / A-GEM 的默认策略。

**② Herding(iCaRL)**

iCaRL 的 herding 算法:贪心迭代选取样本,使 exemplar 集的特征均值最接近整个类的特征均值:

$$p_t = \arg\min_{x \in \mathcal{D}_k} \left\| \mu_k - \frac{1}{t}\!\left(\phi(x) + \sum_{j=1}^{t-1}\phi(p_j)\right)\right\|$$

Herding 在类别 exemplar 选择上优于随机采样,尤其在 $M$ 极小时(每类只存几个样本)能更好保留类内多样性。但 herding 需要已有所有类别数据、且依赖特征空间——特征漂移后旧类的 exemplar 可能不再代表其当前特征。

**③ 梯度驱动选择(Gradient-Based)**

选择对新任务梯度更新影响最大的旧样本——通常是那些与新任务梯度方向最"冲突"的样本。直觉是用最难约束的样本来约束梯度。计算开销高(需要额外反向传播估计每个候选样本的梯度影响),在大规模实验中少用。

**Buffer 大小效应总结:**

| Buffer 大小 $M$ | 随机/Reservoir | Herding | 梯度驱动 |
|---|---|---|---|
| 极小(每类 1-5 样本) | 覆盖差,遗忘严重 | 明显优于随机 | 效果好但代价极高 |
| 中等(每类 20-50) | 接近 herding | 差距缩小 | 收益递减 |
| 大(接近无限) | 三者趋同,接近 joint training | 同左 | 同左 |

核心洞察:$M \to \infty$ 时所有 replay 方法退化为 joint training(上界);$M$ 小时 exemplar 的代表性比随机性更重要——herding 在此区间胜出。

---

### D5. GEM 的逐任务 QP 代价 vs. A-GEM 的单一均值约束

**GEM 的 QP 问题:**

新任务当前梯度为 $g$,对每个旧任务 $k$ 有约束 $\langle g, g_k \rangle \geq 0$。若约束被违反,需求解:

$$\tilde{g} = \arg\min_v \|v - g\|^2 \quad \text{s.t.} \quad \langle v, g_k\rangle \geq 0,\; \forall k \in \{1,\ldots,K-1\}$$

这是一个含 $(K-1)$ 个不等式约束的二次规划(QP)。标准 QP 求解器的时间复杂度为 $\mathcal{O}(K^3)$,空间复杂度为 $\mathcal{O}(K^2)$——任务数增大时迅速不可行。GEM 的实现用 Frank-Wolfe 等迭代方法近似求解,每步仍需 $K$ 次梯度内积计算,即 $\mathcal{O}(K \cdot d)$($d$ 为参数维度)。

**A-GEM 的单一约束:**

A-GEM 将所有旧任务约束合并为一个平均梯度:

$$g_{\text{ref}} = \frac{1}{K-1}\sum_{k=1}^{K-1} g_k, \quad \text{约束:}\; \langle g, g_{\text{ref}} \rangle \geq 0$$

若违反,投影公式有闭合解:

$$\tilde{g} = g - \frac{\langle g, g_{\text{ref}} \rangle}{\|g_{\text{ref}}\|^2} g_{\text{ref}}$$

**每步计算量从 $\mathcal{O}(K \cdot d)$ 降为 $\mathcal{O}(d)$**——与任务数无关(计算 $g_{\text{ref}}$ 可一次性完成并复用)。

**代价:**

A-GEM 满足的是平均方向约束,不保证对每个旧任务 $k$ 都有 $\langle \tilde{g}, g_k\rangle \geq 0$——某些单个旧任务的损失可能上升。实证上 A-GEM 的 AA 和 BWT 与 GEM 相当,但在任务间梯度高度异质时(某些任务方向与均值偏差大)偶尔会出现某个旧任务的遗忘比 GEM 更严重。

**面试记忆点:**GEM = 逐任务约束 + 二次规划,$\mathcal{O}(K^3)$ QP;A-GEM = 单一均值约束 + 闭合解投影,$\mathcal{O}(d)$;牺牲的是per-task 约束保证,换来了线性时间复杂度。

---

### D6. 为什么 Class-IL 最难?Output Head 的角色

van de Ven & Tolias 2019<span class="cite-wrap"><a class="cite" id="fnref-9" href="#ref-9">9</a><span class="cite-note">系统对比三种 CL 设定(Task-IL / Domain-IL / Class-IL)在 Split MNIST 和 Split CIFAR-100 上的难度差异;正则化方法在 Class-IL 上几乎完全失效。<a href="https://arxiv.org/abs/1904.07734">van de Ven 2019 ↗</a></span></span> 的三场景框架揭示了根本难度差异:

**三场景的 output head 结构对比:**

| 场景 | 测试时 task-ID | Output head | 模型需要做什么 |
|---|---|---|---|
| Task-IL | 已知 | 每任务独立 head(仅激活当前任务) | 在任务内分类,已知候选集 |
| Domain-IL | 未知 | 共享 head(输出维度固定) | 在固定输出空间内分类,不需区分任务 |
| Class-IL | 未知 | 共享 head(所有任务类别) | 在所有历史任务的所有类别中分类 |

**Class-IL 为什么最难——三重障碍:**

1. **Task-ID 推断问题**:模型不知道当前输入属于哪个任务,无法路由到对应的子分类器——必须在单一 head 上区分所有类别。

2. **Output head 的历史偏差(recency bias)**:学新任务时只有新任务的类别产生大梯度更新,输出层的 logit 尺度向新任务倾斜——旧类别的 logit 被压制,即使特征层还记得旧任务。这是 Class-IL 独有的"分类器层遗忘"。

3. **正则化方法的根本失效**:EWC 保护了特征层参数,但 output head 的新类别节点初始化会干扰旧类别节点的梯度——Fisher 对角线无法捕捉这种跨类别的输出层干扰。van de Ven et al. 实验显示,EWC 在 Class-IL 上准确率接近 chance level。

**补救策略:**

- **Replay + prototype classifier**(如 iCaRL):用 exemplar 均值做最近邻分类,绕开 output head 偏差。
- **任务无关特征学习**:预训练冻结 backbone,只更新轻量分类头——减少特征漂移。
- **经验修正(bias correction)**:在 Class-IL 测试时对旧类别 logit 做温度调整或加权,抵消 recency bias。

---

### D7. LLM 的遗忘 vs. 能力干扰——测量方法与任务顺序敏感性

**LLM 的"遗忘"与经典 CL 的关键区别:**

| 维度 | 经典 CL(小模型/分类任务) | LLM post-training |
|---|---|---|
| 任务粒度 | 清晰(Task 1/2/3…) | 模糊("数学推理""代码""安全对齐"大量重叠) |
| 遗忘的表现 | 旧任务分类准确率下降 | 能力**干扰**(interference):新能力激活路径覆盖旧能力激活路径,非简单"忘记" |
| 测量难度 | 用 $a_{T,j}$ 矩阵直接量化 | 需要专项 benchmark(MMLU/GSM8K/HumanEval…)追踪各能力 |
| 边界清晰度 | 显式任务边界 | 无明确边界;SFT→DPO→RL 是**软边界** |

**如何测量 LLM 的 CL 质量:**

1. **能力矩阵追踪**:在每个对齐阶段结束后,在若干"探针 benchmark"(覆盖通用推理、代码、数学、安全)上评测——相当于在 LLM 尺度上构建 $a_{i,j}$ 矩阵。
2. **BWT 的 LLM 类比**:测量"SFT 后代码能力相比 pretrain baseline 的变化"——若为负即为 alignment tax 的一部分。
3. **激活/梯度分析**:检测哪些层的激活分布在新任务训练前后变化最大——定位"被覆盖"的知识存储层。

**任务顺序敏感性:**

LLM 的 CL 对任务顺序高度敏感,原因是:

- 顺序训练的梯度方向依赖于前序任务形成的损失 landscape——先做 math SFT 再做 safety RLHF 与反序结果截然不同。
- 较难任务(数学/代码)的优化需要大学习率大梯度,对后续小任务的参数损害更大。
- **Alignment tax 的累积性**:SFT → DPO → RL 的序列中,每一步的遗忘税在下一步的 checkpoint 上累加。

**KL 约束是隐式 EWC:**

PPO-RLHF 中的 KL 惩罚项:

$$r_{\text{KL}}(\theta) = \mathbb{E}\!\left[\log\frac{\pi_\theta(a|s)}{\pi_{\text{ref}}(a|s)}\right] \cdot (-\beta)$$

将其在参考策略 $\pi_{\text{ref}}$ 附近做 Taylor 展开,KL 散度的二阶项正比于 Fisher 信息矩阵:

$$\text{KL}(\pi_\theta \| \pi_{\text{ref}}) \approx \frac{1}{2}(\theta - \theta_{\text{ref}})^T F(\theta_{\text{ref}}) (\theta - \theta_{\text{ref}})$$

即 **PPO 的 KL 惩罚 ≈ 以 Fisher 为权重矩阵的 EWC 全矩阵版本**——两者都是"以信息几何曲率为权重,对参数偏离参考点的二次惩罚"。区别在于:EWC 用 diagonal 近似且显式存储旧任务 Fisher;KL 约束用完整分布距离且动态锚定当前 reference 策略。DPO 中的 KL 项同理——reference model 在数学结构上等同于 EWC 中的 $\theta^*$。

---

### D8. LoRA-based CL 的 Adapter 干扰与路由

**Adapter 干扰的来源:**

朴素方案是为每个任务训练一套独立 LoRA 权重 $\Delta W_k = B_k A_k$,测试时通过 task-ID 路由。但有两个干扰来源:

1. **参数空间重叠**:不同任务的低秩子空间可能大量重叠——若 $\text{span}(A_k) \cap \text{span}(A_j) \neq \emptyset$,合并后一个任务的 adapter 会干扰另一任务的激活。
2. **无 task-ID 时的路由失败**:在 Class-IL 或 continual pretraining 设定下,测试时无 task-ID,无法路由到正确 adapter。

**O-LoRA 的正交子空间方案:**

O-LoRA 在任务 $k$ 的 LoRA 训练时,强制新任务的低秩子空间与历史所有任务的子空间正交:

$$A_k V_{\text{prev}} \approx 0, \quad V_{\text{prev}} = \text{span}\bigl(\{A_j\}_{j<k}\bigr)$$

通过梯度投影到 $V_{\text{prev}}$ 的正交补来实现。正交子空间保证不同任务 adapter 激活互不干扰——在已知 task-ID 的设定下近似零遗忘,且不需要存储旧任务数据。

**局限:**

- 可用正交维度随任务数增加而减少——在秩 $r$ 的 LoRA 中,最多支持约 $d/r$ 个完全正交任务($d$ 为权重矩阵维度)。
- 无 task-ID 的 Class-IL 设定下,正交性不解决路由问题——仍需另外机制推断 task-ID 或用 prototype 分类。
- 强制正交可能限制任务间的**正向迁移**——相关任务本可共享子空间以加速学习。

**LLM post-training 中的实践:**

序列对齐链条(SFT → DPO → RL)中,每阶段冻结主干、单独更新一套 LoRA,等价于 task-ID 已知的参数隔离方案——alignment tax 的累积被大幅限制在 LoRA 层内,主干知识不被覆盖。代价是需要在部署时管理多套 adapter 及其合并/路由逻辑。

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
<li id="ref-7">Schwarz et al. <em>Progress &amp; Compress: A scalable framework for continual learning</em>. ICML 2018. <a href="https://arxiv.org/abs/1805.06370">arXiv:1805.06370</a> — 双网络 active column + knowledge base;online EWC 用 Fisher EMA 跨任务累积,内存恒定. <a href="#fnref-7">↩</a></li>
<li id="ref-8">De Lange, van de Ven, and Tuytelaars. <em>Continual evaluation for lifelong learning: Identifying the stability gap</em>. ICLR 2023. <a href="https://arxiv.org/abs/2205.13452">arXiv:2205.13452</a> — per-iteration 连续评估框架;发现任务切换后性能骤降后恢复的 stability gap 现象. <a href="#fnref-8">↩</a></li>
<li id="ref-9">van de Ven and Tolias. <em>Three scenarios for continual learning</em>. 2019. <a href="https://arxiv.org/abs/1904.07734">arXiv:1904.07734</a> — 系统定义 Task-IL / Domain-IL / Class-IL 三场景;揭示正则化方法在 Class-IL 上近乎完全失效. <a href="#fnref-9">↩</a></li>
</ol>
