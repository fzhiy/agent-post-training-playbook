# Agentic RL Infrastructure / Agentic RL 基础设施

> 把 [agentic-and-long-horizon-rl](cheatsheet-agentic-and-long-horizon-rl.html) 里的算法变成可跑的**系统工程**:怎么搭异步 rollout 集群、怎么管环境沙盒、怎么排 GPU 显存、怎么选训练栈。先读 agentic 篇懂「为什么」,再用本篇懂「怎么搭」。

> ⚠️ **学习笔记,非作者研究成果**(见 README 诚信声明)。具体框架版本 / 性能数字随 release 快变,本页只记架构原理与设计 tradeoff,不抄 benchmark。

## 0. TL;DR 速查

- agent RL 训练与标准 RLHF 的本质差异:**多轮轨迹**(非单轮 response)+ **交互式环境**(非静态数据集)+ **异步 rollout**(推理与训练解耦)。
- **三池架构** = 核心心智模型:Rollout 池(推理引擎+环境)→ Reward 池(验证/判题)→ Training 池(策略更新);三池异步、各可独立扩缩。
- **训练栈选型四维度**:是否原生支持多轮 / 是否支持异步 rollout / 环境管理方式 / GPU 显存策略(hybrid engine vs 分离部署)。
- **verl**(Volcano Engine):hybrid engine(推理+训练同 GPU,分时复用),SPMD 3D 并行,agent RL 支持在快速迭代中。
- **OpenRLHF**(Ray 生态):Ray actor 天然异步,PPO/DPO/Ray 强化学习管线成熟;agent 扩展通过自定义 environment wrapper。
- **AReaL**(Ant Group):全异步 agent-native 设计,解耦生成与训练;系统级而非算法级(见 [agentic §5](cheatsheet-agentic-and-long-horizon-rl.html))。
- **环境是瓶颈,不是 GPU**:SWE 单次测试 10-60s,web 页面加载数秒——环境延迟 >> 推理延迟;并行环境数 × 单环境速度决定吞吐上界。
- **多轮 KV cache 显存爆炸**:$T$ 轮、每轮 $L$ token → cache $\propto T \times L$;缓解 = KV 驱逐(有损) / 摘要压缩(有误差) / 截断重启(丢历史)。
- **轨迹存储 = 新的数据工程**:一条 agent 轨迹可达数万 token;训练千条即 GB 级;需存原始轨迹 + reward + 元数据,格式设计影响回放和过滤效率。
- **容错是硬需求**:环境 flaky(网络超时/沙盒崩溃/测试非确定性)、GPU OOM、rollout worker 挂——训练框架必须处理重试/降级/checkpoint,否则跑不完一个 epoch。

## 1. 为什么 agent RL 的 infra 不同于标准 RLHF / Why agent RL infra ≠ RLHF infra

标准 RLHF 的假设在 agent 场景全部失效:

| 维度 | 标准 RLHF | Agentic RL |
|---|---|---|
| 轨迹长度 | 单轮 response(~几百 token) | 多轮 think→act→observe(~数千至数万 token) |
| 数据来源 | 静态数据集(模型生成一次即可) | **交互式环境**(每步动作改变状态,不可回放) |
| 推理与训练关系 | 可同步(生成 batch → 训完 → 下一批) | **通常需异步**(慢交互环境下环境延迟 >> GPU 延迟,同步 = GPU 空等;快验证器或长生成场景下 GPU 可能仍是瓶颈,见 Q9) |
| 奖励计算 | 奖励模型前向(~毫秒) | **环境执行**(跑单测 / 网页校验 / 脚本,秒级) |
| KV cache | 短序列,可全部保留 | 长序列,显存是硬约束 |
| 容错 | 重试一次生成即可 | 环境可能崩溃/超时/非确定性,需系统性容错 |

> 💡 **一句话:** RLHF infra 是「把模型生成的内容打分→训练」的批处理流水线;agent RL infra 是「让模型在真实/模拟环境里多步交互→收集轨迹→训练」的**分布式异步系统**。

这直接决定了 agent RL 训练栈的核心设计:

1. **推理与训练通常需解耦**(异步):对慢交互环境(代码执行/web 交互,环境步秒级 >> GPU 推理毫秒级),同步等待 = GPU 空转;快验证器或长生成主导场景下 GPU 可能才是瓶颈——瓶颈位置随任务与规模转移(见 Q9)。
2. **环境管理是第一等公民**:需要环境池(docker sandbox pool / browser pool)、健康检查、超时杀、自动重置。
3. **轨迹是长序列**:KV cache 管理、截断策略、存储格式全得重新考虑。

## 2. 三池架构 / The three-pool architecture

agent RL 训练的**核心心智模型**:三个可独立扩缩的异步池,各自有不同资源特征。

```
                    ┌─────────────────────┐
                    │   Training Pool      │
                    │  (GPU: policy update)│
                    │  - 消费轨迹 batch    │
                    │  - 梯度累积 + 更新   │
                    │  - 定期推送新权重    │
                    └──────────┬──────────┘
                               │ new weights
                               ▼
┌─────────────────────┐  ┌─────────────────────┐
│   Rollout Pool       │  │   Reward Pool        │
│  (GPU: inference +   │  │  (CPU: verification) │
│   CPU: environment)  │  │                      │
│                      │  │  - 跑单测 / 判终态   │
│  - 推理引擎(vLLM/    │  │  - 规则校验          │
│    SGLang)生成动作   │  │  - LLM-judge(可选)   │
│  - 环境执行动作       │  │                      │
│  - 收集完整轨迹       │──▶  - 产出 reward 向量  │
│                      │  │                      │
└─────────────────────┘  └─────────────────────┘
```

**Rollout Pool** (推理 + 环境,GPU+CPU 混合):
- 推理引擎(vLLM<span class="cite-wrap"><a class="cite" id="fnref-5" href="#ref-5">5</a><span class="cite-note">PagedAttention + continuous batching。<a href="https://arxiv.org/abs/2309.06180">Kwon 2023 ↗</a></span></span>/SGLang<span class="cite-wrap"><a class="cite" id="fnref-6" href="#ref-6">6</a><span class="cite-note">RadixAttention + 结构化生成。<a href="https://arxiv.org/abs/2312.07104">Zheng 2024 ↗</a></span></span>)以 TP/DP 方式部署,承载多轮生成的 token 产出
- 每个 rollout worker 绑定一个环境实例(docker container / browser instance)
- 循环:推理引擎产出动作 → worker 执行环境步 → 观测注入上下文 → 继续推理,直到终止
- **吞吐关键**:并行 worker 数 × 每 worker 的推理速度;GPU 不直接等环境(环境步期间 GPU 服务其他 worker)

**Reward Pool** (验证,CPU 密集型):
- 接收完整轨迹,执行奖励计算:跑单元测试 / 校验环境终态 / 运行验证脚本
- 可与 Rollout Pool 物理同机但逻辑独立(方便独立扩缩)
- **瓶颈**:SWE 单测执行可达数十秒;WebArena 终态校验需要浏览器交互

**Training Pool** (策略更新,GPU 密集型):
- 从轨迹缓冲区取 batch,计算 PPO/GRPO 损失 + 梯度更新
- **hybrid engine 模式**:训练 GPU 同时承载推理(GPU 分时复用,训完一批→切推理→生成新轨迹→切回训练)
- **分离模式**:推理 GPU 和训练 GPU 物理分离,适合大规模持续训练

> 💡 **三池各自最适硬件不同**:Rollout 池重推理吞吐(高 GPU 利用率 + 大量 CPU 环境 worker);Reward 池几乎纯 CPU;Training 池重梯度计算(高 GPU 显存)。**分离部署**比 hybrid 更贵但各自可独立扩缩,适合生产级规模。

**from-scratch 实现**(三池异步 rollout 骨架,面试手撕标准):

```python
import threading, queue, time
from concurrent.futures import ThreadPoolExecutor

class AsyncRolloutPipeline:
    """异步三池骨架:Rollout worker → Reward worker → Trajectory buffer → Training."""
    def __init__(self, env_factory, reward_fn, policy_model, buffer_size=1000):
        self.env = env_factory                            # 环境工厂(每个 worker 独立实例)
        self.reward_fn = reward_fn                        # reward 计算(环境执行/判题/校验)
        self.model = policy_model                         # 当前策略(定期从 Training 池同步)
        self.buffer = queue.Queue(maxsize=buffer_size)    # 轨迹缓冲区(解耦推理与训练)

    def rollout_worker(self, prompts, num_envs=8):
        """Rollout 池:并行与交互式环境采样生成轨迹。GPU 推理 + CPU 环境执行交叉。"""
        def run_one(prompt):
            env = self.env()                              # 每 worker 独立环境实例
            obs, traj = env.reset(prompt), []
            for _ in range(max_steps := 50):
                # 1. 推理(在 GPU 上):当前策略给定历史→下一动作
                action = self.model.generate(obs["history"])
                # 2. 执行(在 CPU 上):动作送入环境,环境返回新观测与局部奖励
                next_obs, done = env.step(action)
                traj.append({"obs": obs, "action": action, "done": done})
                if done: break
                obs = next_obs
            return traj

        with ThreadPoolExecutor(max_workers=num_envs) as ex:
            trajectories = list(ex.map(run_one, prompts))
        return trajectories

    def reward_worker(self, trajectories):
        """Reward 池:对每条轨迹计算最终 reward(环境终态校验/判题)。"""
        for traj in trajectories:
            final_state = traj[-1]["obs"]                 # 轨迹终态
            traj_reward = self.reward_fn(final_state)     # 纯 CPU 计算(跑单测/查网页)
            traj.append({"reward": traj_reward})
        return trajectories

    def train_step(self, batch):
        """Training 池:从 buffer 取轨迹→算 loss→梯度更新(在这里调用,与 rollout 异步)。"""
        # 实际实现:从 self.buffer 取 batch → compute PPO/GRPO loss → optimizer.step()
        pass                                              # 训练细节见 agentic-RL 篇代码

    def run_async(self, prompts, sync_interval=10):
        """启动异步循环:rollout+reward 持续产出→喂 buffer→train 从 buffer 消费。"""
        def producer():
            while True:
                trajs = self.rollout_worker(prompts)
                trajs = self.reward_worker(trajs)
                for t in trajs:
                    self.buffer.put(t)                    # 解耦:rollout 产出速度 ≠ train 消费速度
                time.sleep(0.1)
        threading.Thread(target=producer, daemon=True).start()
        # Training 端(另一线程/进程)从 buffer 取 batch → train_step
# 面试关键点:
# ① 三池异步解耦: rollout/reward/training 各自独立,不互相阻塞
# ② 环境是瓶颈(source):环境步数×单步延迟 ≫ 推理延迟时,同步=GPU 空等
# ③ ThreadPoolExecutor 模拟多环境并行;生产会用 Ray actor / k8s pod
# ④ Buffer 解耦: rollout 产出速度波动不影响 training 消费节奏
```

## 3. 训练栈对比 / Training stack comparison

### 3.1 verl (Volcano Engine)

verl<span class="cite-wrap"><a class="cite" id="fnref-1" href="#ref-1">1</a><span class="cite-note">ByteDance Seed 发起、社区维护的 RL 训练框架:hybrid engine 分时复用 GPU(推理↔训练切换)、SPMD 3D 并行、多算法支持。<a href="https://github.com/verl-project/verl">github.com/verl-project/verl</a>(原 volcengine/verl)</span></span> 的核心设计是 **hybrid engine**:同一组 GPU 在推理和训练模式间分时切换——训完一批 → 切推理模式生成新 rollout → 切回训练。省 GPU(不需两套集群),但切换有开销(权重同步 + 显存重分配)。

- **并行策略**:SPMD(单程序多数据),支持 DP/TP/PP 3D 组合
- **算法支持**:PPO / GRPO / DPO / ReMax;agent RL 支持在活跃开发中
- **agent 适配**:社区有 `verl-agent` 扩展,在 RL 循环里嵌入环境交互步
- **适用场景**:中等规模(几十到几百 GPU)、团队已有 FSDP/Megatron 经验

### 3.2 OpenRLHF

OpenRLHF<span class="cite-wrap"><a class="cite" id="fnref-2" href="#ref-2">2</a><span class="cite-note">Ray 原生的 RLHF 框架:天然异步 actor、PPO/DPO/Rejection Sampling 支持。<a href="https://arxiv.org/abs/2405.11143">Hu 2024 ↗</a></span></span> 基于 **Ray** 的分布式 actor 模型:

- **天然异步**:每个 Ray actor 独立调度,rollout worker / trainer / reward model 通过 Ray 通信,无需手动同步
- **调度灵活**:可以给 rollout 和 trainer 分配不同 GPU,也可以 hybrid
- **算法**:成熟的 PPO / DPO / Rejection Sampling / Conditional SFT 实现
- **agent 扩展**:通过自定义 `Environment` 抽象接入任意交互式环境;Ray 的 actor 模型让环境 worker 的扩缩和容错较自然
- **适用场景**:快速原型、中等规模、需要灵活调度的团队

### 3.3 AReaL (Ant Group)

AReaL<span class="cite-wrap"><a class="cite" id="fnref-3" href="#ref-3">3</a><span class="cite-note">Ant Group 的全异步 RL 系统(面向 LLM 推理):生成与训练完全分离;paper 实验聚焦数学/代码推理,但其异步架构同样适用于 agentic 长程场景。<a href="https://arxiv.org/abs/2505.24298">arXiv:2505.24298</a></span></span> **系统级设计,非 GRPO 算法变体**(见 [agentic §5](cheatsheet-agentic-and-long-horizon-rl.html)):

- **全异步**:生成(rollout)与训练完全解耦,各自独立 GPU 池;生成池不停产轨迹,训练池不停消费
- **agent 适用性**:paper 实验在推理领域,但其全异步架构对长程、多轮 agent RL 场景同样适用(架构层面不依赖单轮 assumption)
- **适用场景**:大规模持续训练、需要高吞吐的长程 agent RL

### 3.4 SkyRL-Agent (NovaSky/Berkeley)

SkyRL-Agent<span class="cite-wrap"><a class="cite" id="fnref-4" href="#ref-4">4</a><span class="cite-note">NovaSky/Berkeley 的长程多轮工具异步训练栈:CPU 侧运行时与 GPU 生成重叠。<a href="https://arxiv.org/abs/2511.16108">arXiv:2511.16108</a></span></span> 的核心贡献是**异步流水线派发器**:

- 把 CPU 侧的环境初始化 / 奖励计算与 GPU 生成**时间重叠**
- 长短轨迹混批时不被慢轨迹拖住(通过动态调度)
- 报告较 naive 异步批处理约 **1.55×** 吞吐提升
- 可对接 VeRL / Tinker 等后端

| 框架 | 异步模式 | 并行策略 | Agent 支持 | 适合规模 |
|---|---|---|---|---|
| **verl** | hybrid engine(分时复用) | SPMD 3D(DP+TP+PP) | 社区扩展中 | 中–大 |
| **OpenRLHF** | Ray actor(天然异步) | Ray 调度 | 自定义 Environment | 小–中 |
| **AReaL** | 全异步(生成/训练分离) | 独立 GPU 池 | **agent-native** | 大 |
| **SkyRL-Agent** | 异步流水线(CPU/GPU 重叠) | 可对接多种后端 | **长程工具专用** | 中 |

> 💡 **选型建议(非权威,随框架迭代快变):** 快速原型 + 中等规模 → OpenRLHF(Ray 生态熟悉的话);已有大规模集群 + 需要高吞吐 → AReaL(agent-native);已有 FSDP/Megatron 技术栈 → verl(hybrid 省 GPU)。SkyRL-Agent 的价值在调度层,可与任一后端组合。**实际选择需评估当前 release 的成熟度 + 团队技术栈匹配度**,勿盲信本表。

## 4. 环境管理 / Environment management

这是 agent RL infra **最容易被低估**的一层——环境比 GPU 更慢、更不可靠、更难扩缩。

### 4.1 三类环境及其挑战

| 环境类型 | 实例 | 单步延迟 | 重置成本 | 典型故障 |
|---|---|---|---|---|
| **代码执行** | SWE-bench(sandbox 里跑单测)<span class="cite-wrap"><a class="cite" id="fnref-7" href="#ref-7">7</a><span class="cite-note">SWE-RL:用真实 GitHub issue 与 FAIL_TO_PASS 单测作 RL 环境。<a href="https://arxiv.org/abs/2502.18449">Duan 2025 ↗</a></span></span> | 10–60s(含 docker 启动) | 中(重建容器/分支) | 超时、非确定性测试、依赖缺失 |
| **Web 交互** | WebArena(自托管站点)<span class="cite-wrap"><a class="cite" id="fnref-8" href="#ref-8">8</a><span class="cite-note">WebRL:自进化课程 + web 领域 RL 环境。<a href="https://arxiv.org/abs/2411.02337">Qi 2024 ↗</a></span></span> | 1–10s(页面加载+渲染) | 中(重置数据库+重启服务) | 页面加载超时、选择器失效、站点状态不一致 |
| **GUI/OS** | OSWorld(真实 OS 交互) | 1–5s(截图+动作执行) | 高(VM 快照回滚) | 截图失败、坐标漂移、非确定性 UI |

### 4.2 环境池设计模式

```
          ┌──────────────────────────────┐
          │     Environment Pool           │
          │                                │
          │  ┌─────┐ ┌─────┐ ┌─────┐     │
          │  │ Env │ │ Env │ │ Env │ ...  │  ← N 个预热好的环境实例
          │  │ #1  │ │ #2  │ │ #3  │     │
          │  └─────┘ └─────┘ └─────┘     │
          │       ↑  acquire               │
          │       │  release               │
          └───────┼────────────────────────┘
                  │
          ┌───────┴────────┐
          │ Rollout Worker │ → 取环境 → 跑一条轨迹 → 重置 → 归还
          └────────────────┘
```

关键机制:

1. **预热(pre-warm)**:环境实例在池中预先启动好(docker 已拉镜像、web server 已启动),worker 取到即可用;冷启动 docker 容器可达数十秒。
2. **健康检查(health check)**:定期或取用前检查环境是否可达(docker ps、HTTP ping);坏实例自动替换。
3. **自动重置(auto-reset)**:轨迹完成后 worker 调用 reset 恢复初始状态(git checkout 回初始 commit、数据库回滚、VM 快照恢复);归还前必须重置。
4. **超时杀(timeout kill)**:单步或整条轨迹超时 → 强制终止环境进程 → 重新创建实例(而非复用),防止残留状态污染。
5. **池大小 = N_env**:由并发 worker 数 + 环境重置延迟决定;太小 → worker 空等;太大 → 资源浪费。

> ⚠️ **环境非确定性 = 训练安静杀手。** 同一个 action 在同一个环境状态上执行两次,结果可能不同(网络抖动、磁盘 IO 竞态、测试依赖随机种子)。这会污染 reward 信号——一条好轨迹偶然因环境故障被判失败,或坏轨迹偶然蒙对。缓解:① 关键 reward 计算跑多次取多数;② 记录环境版本/种子以复现;③ 容忍少量 reward 噪声(Robust RL),不追求每一条轨迹的 reward 都完美。

### 4.3 环境与推理的吞吐匹配

**核心公式**:要使 GPU 不等环境,需要 `N_env × throughput_per_env ≥ GPU_inference_throughput`。

具体:若一个环境轨迹平均 20 步、每步环境延迟 5s、GPU 推理每步 0.2s,那么 GPU 处理一个 worker 的每步仅需 0.2s 但要等 5s 环境 → GPU 利用率仅 `0.2/5.2 ≈ 3.8%`。所以需要大量环境 worker 并行:至少 `5.2/0.2 ≈ 26` 个 worker 才能让 GPU 在任意时刻都有可服务的请求。**实际通常需要百级环境 worker 来喂饱一块 GPU。**

## 5. GPU 显存与多轮 KV cache / GPU memory & multi-turn KV cache

### 5.1 多轮 KV cache 增长模型

单轮推理:prompt 编码一次 → decode $L$ token → KV cache $\propto \text{prompt\_len} + L$。多轮 agent:$T$ 轮对话,每轮上下文包含此前所有轮的历史。第 $t$ 轮时上下文长度约 $t \times L$,总 KV cache 呈 **$O(T \times L)$ 增长**。

具体数字:若每轮 500 token(think 200 + act 100 + observe 200),20 轮后上下文长 10k token。对 7B 模型(FP16)单层 KV cache ≈ 2×2×num_heads×head_dim×seq_len bytes,全层累计可达数 GB——一条轨迹的 KV cache 就可能占满一张 GPU 的显存。

### 5.2 缓解策略

| 策略 | 做法 | 优点 | 代价 |
|---|---|---|---|
| **KV 驱逐** | 丢弃低注意力权重 token 的 KV(保留 attention sink + 近窗) | 显存可控,不改变推理语义 | **有损**:丢弃的早期观测无法再被 attend,可能丢关键信息 |
| **摘要压缩** | 将旧轮压成摘要文本,替换原始 token | 保留语义,最省显存 | **摘要误差**:压缩可能丢失对后续决策关键的信息 |
| **截断重启** | 超最大长度后截断旧历史,重新编码 | 简单,显存有硬上界 | **丢历史上下文**:agent 丢失对早期行为的记忆 |
| **外置记忆** | 写入外部向量库/知识图谱,检索而非保留在上下文 | 理论上无容量上限 | 检索延迟 + 精度损失;检索结果仍需一些 token 表示 |

> 💡 **实务组合:** 近轮(最近 3-5 轮)完整 KV 保留 + 远轮摘要文本嵌入 + 全局 KV 驱逐阈值。各框架对此的支持成熟度不同,评估框架时重点看其对多轮 KV 管理的原语。

### 5.3 Hybrid engine 的显存管理

Hybrid engine 模式下,同一 GPU 分时用于推理和训练,显存要在两种模式间切换:

- **推理阶段**:显存放模型权重 + KV cache(多轮! 大!)
- **训练阶段**:显存放模型权重 + 优化器状态 + 梯度 + 激活(比推理多 ~3-4× 权重显存)
- **切换开销**:释放 KV cache → 分配优化器状态 → 加载;约数秒,频繁切换(每条轨迹切一次)会显著降低吞吐

> ⚠️ **实务坑:** hybrid engine 在多轮 agent 场景下,推理的 KV cache 占显存可能超过训练可用显存——导致切不到训练模式(OOM)。解决办法:推理时主动驱逐远轮 KV / 限制最大上下文长度 / 或放弃 hybrid,走分离部署。

## 6. 轨迹数据管线 / Trajectory data pipeline

一条 agent 轨迹的元数据结构:

```
trajectory = {
  task_id,              # 任务标识
  trajectory_id,        # 轨迹唯一 ID
  turns: [              # 多轮列表
    {think, act, obs, reward_step, done},
    ...
  ],
  total_reward,         # 终端 reward
  metadata: {           # 元数据
    env_version,        # 环境版本(可复现)
    total_steps,        # 步数
    total_tokens,       # token 消耗
    wall_time,          # 墙钟时间
    truncated,          # 是否被截断
    env_errors,         # 环境错误次数
  }
}
```

**存储量级**:一条 20 轮轨迹 ≈ 10k-20k token(含 think + act + obs);千条轨迹 ≈ 10M-20M token → 原始格式约 50-100MB(含 metadata + reward)。万条级训练需 GB 级存储;**百万条轨迹的离线数据集可达 TB 级**。

**过滤管线**(训练前,选哪些轨迹进 batch):
1. **完整性过滤**:丢弃被截断 / 环境报错的轨迹(可选保留用于诊断)
2. **质量过滤**:丢弃 reward 为零或异常低/高的轨迹(视任务而定;全零轨迹在 GRPO 中无信号但仍需保留少数作为负例)
3. **多样性过滤**:丢弃与已有轨迹重复度过高的轨迹(基于 action n-gram 或 embedding 相似度)——防 Echo Trap(见 [agentic Q14](cheatsheet-agentic-and-long-horizon-rl.html))
4. **平衡采样**:确保 batch 中正例(reward > 0)和负例的比例合理,避免一方主导梯度

> 💡 **轨迹存储格式建议:** 推荐用 Parquet(Apache Arrow 列存)而非 JSONL——压缩率高(~5-10× vs 原始 JSON)、支持列裁剪(只读 reward 不读全文)、生态成熟(pandas/polars/spark 原生读)。

## 7. 异步 staleness 与流量控制 / Async staleness & flow control

全异步架构下,rollout 和训练之间天然存在**策略滞后(policy lag)**:

```
Rollout 池用 θ_t 生成轨迹 → Reward 池计算 → Training 池此时已更新到 θ_{t+k} → 轨迹是用旧策略生成的
```

**staleness 的影响**:
- 轨迹是用**旧版本策略**采样的,当前策略 θ_{t+k} 下这些轨迹的 log-prob 和 advantage 都变了 → IS(importance sampling)修正的方差随 staleness 增大
- **离线程度** = `(rollout 完成时 trainer 的 step) - (rollout 开始时 policy 的 version)`;过大则轨迹几乎来自不同策略 → 训练信号退化

**控制手段**:
1. **权重版本号**:每条轨迹标记生成时的 `policy_version`,训练时丢弃 `version` 过旧的轨迹(设置 `max_policy_lag`)
2. **IS 修正**(V-trace / 截断 IS):用 $\frac{\pi_{\theta_{\text{new}}}(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}$ 修正,但方差随 lag 增长;需截断($\rho = \min(\frac{\pi_{\text{new}}}{\pi_{\text{old}}}, \bar{c})$)或 ESS(effective sample size)监控
3. **队列反压**:若 Training Pool 消费速度 < Rollout Pool 生产速度 → 轨迹队列积压 → staleness 增大 → 需要反压(限速 rollout 或提速训练),保持稳态队列深度
4. **stale 轨迹复用策略**:略旧轨迹并非完全无用——可降权使用(weight = $e^{-\lambda \cdot \text{lag}}$)而非直接丢弃,在高吞吐与信号质量之间折中

> 💡 **实务经验:** 全异步系统的第一个坑通常是「队列无限增长」——rollout 太快 → 轨迹堆积 → staleness 飙升 → 训练退化 → 更慢收敛 → 更多 rollout 堆积,正反馈循环。设置 `max_queue_size` + 监控 `policy_lag` 分布是上线前必做的事。

## 8. 容错 / Fault tolerance

agent RL 训练链路长,每一步都可能挂。生产级训练**必须**处理以下故障:

| 故障类型 | 出现概率 | 影响 | 处理方式 |
|---|---|---|---|
| 环境超时(单测跑太久) | 常见 | 丢失该轨迹的 reward | 超时后标记 truncated → 可选丢弃或给部分 reward |
| 环境崩溃(docker segfault) | 偶发 | worker 卡死 | 健康检查 + 自动重启 + 轨迹丢弃 |
| GPU OOM(多轮 KV 超限) | 偶发 | 推理失败,batch 不完整 | 减小 max_seq_len / 启用 KV 驱逐 / 重试 |
| 网络抖动(权重同步失败) | 偶发 | learner 收不到新 rollout | 指数退避重试 + 权重版本号校验 |
| 整个 rollout worker 挂 | 罕见 | 与其绑定的环境全部丢失 | Ray/集群调度器自动重启 worker + 重新分配环境池 |

**checkpoint 策略**:至少保存 `(model_weights, optimizer_state, LR_scheduler_state, global_step)`→ 从最近 checkpoint 续训。推荐每 N 个 training step 或每 M 分钟 checkpoint 一次。若用 hybrid engine,checkpoint 前确保切换到 training mode 以保存完整优化器状态。

> ❌ **误区:** 「agent RL 训练 = 跑一个脚本等结果就行」。RLHF 也许可以(批处理、容错点少),agent RL 的交互式环境让故障率远高于标准训练——**没有完善容错的训练,大概率跑一半就挂且无法恢复**。实际第一版 agent RL 训练应该先把容错做齐、再上规模,而不是反过来。

## 9. 成本估算 / Cost estimation

给定以下参数,可粗估一次 agent RL 训练的成本:

| 参数 | 含义 | 典型值 |
|---|---|---|
| $N_{\text{task}}$ | 训练任务数 | 1k–10k |
| $\bar{T}$ | 平均轨迹轮数 | 10–30 |
| $\bar{L}_{\text{turn}}$ | 单轮 token 数 | 300–800 |
| $N_{\text{epoch}}$ | 训练 epoch 数 | 1–5 |
| $\text{GPU}$ | GPU 类型 | A100-80G / H100-80G |
| $\text{GPU}_\text{推理}$ | 推理 GPU 数 | 8–64 |
| $\text{GPU}_\text{训练}$ | 训练 GPU 数 | 8–32 |

**推理成本**(rollout):
$$\text{Tokens}_{\text{rollout}} = N_{\text{task}} \times N_{\text{rollout-per-task}} \times \bar{T} \times \bar{L}_{\text{turn}}$$

对 5k 任务、每任务 4 条 rollout、平均 20 轮、每轮 500 token:约 **200M token**。以 vLLM 在 8×A100 上推理速度 ~5k token/s 估计,约需 **11 wall-clock hours(≈89 GPU-hours = 8 GPU × 11h)**。

**训练成本**:
$$\text{Tokens}_{\text{train}} = \text{Tokens}_{\text{rollout}} \times N_{\text{epoch}}$$

训练吞吐取决于模型大小 + 并行策略。7B 模型在 8×A100 上 PPO 训练约 2k-5k token/s(含 advantage 计算 + 梯度更新),200M token 训练 1 epoch ≈ **11-28 wall-clock hours(≈89-222 GPU-hours)**。

**环境成本**(常被忽略):SWE docker sandbox 需要 CPU 实例;若 100 并发环境跑 1 周,CPU 实例成本可接近 GPU 成本。

> 💡 **量级结论(非报价,仅教学估算):** 一次中等规模(5k 任务、7B 模型、~2 万条轨迹)的 agent RL 训练,GPU + 环境总成本在数百到数千美元量级(视云厂商与 GPU 类型),**其中环境 CPU 成本可占 20-50%**——做好环境复用和池化有直接的财务回报。

---

## 分层面试题 / Stratified follow-ups

> ⚠️ 下列「会被问」均为**据公开 JD + 技报推断**,非真实面试原题。

### L1 基础

<details class="qa"><summary>1. agent RL 训练为什么不能直接用标准 RLHF 的 infra？</summary>

答:RLHF 的三条核心假设在 agent 场景全部失效:① 轨迹从单轮变多轮(上下文长度 $O(T)$ 而非常数);② 数据从静态数据集变交互式环境(不可预生成,每步动作改变状态);③ 推理与训练从可同步变通常需异步(慢交互环境下环境延迟秒级 >> GPU 推理毫秒级,同步 = GPU 空转)。此外,长轨迹的 KV cache 管理、环境容错、轨迹存储管线都是 RLHF infra 不用操心的新问题。

**追问：** 如果硬把 RLHF 的同步 batch 流水线用于 agent RL,会看到什么现象? → GPU 利用率极低(因为每步都等环境返回,而环境延迟远大于 GPU 推理);对于多步轨迹,GPU 在每一步之间空等,整体 GPU 利用率可能不到 10%。

</details>

<details class="qa"><summary>2. 三池架构的三个池各自做什么?为什么不能合并成一个池?</summary>

答:**Rollout Pool**(推理+环境)产轨迹,**Reward Pool**(CPU 验证)算奖励,**Training Pool**(GPU 训练)更新策略。不能合并的原因:① 三者的**资源特征**不同——推理重吞吐与低延迟、训练重显存与梯度计算、reward 几乎纯 CPU;合并会导致 GPU 等 CPU(训完等 reward 算完)、CPU 等 GPU(环境等推理结果),整体利用率低。② 三者的**扩缩需求**不同——reward 计算量与任务复杂度挂钩,训练量与模型大小挂钩,rollout 与并发任务数挂钩,合并后无法独立扩缩。

**追问：** 什么情况下 hybrid engine(训推合并)比分离部署更合适? → GPU 资源有限、中等规模训练时,hybrid 省 GPU(不用买两套);但切换开销(权重同步+显存重分配)会降低吞吐,多轮 agent 场景下推理 KV cache 与训练优化器状态争抢显存,需精细管理。

</details>

<details class="qa"><summary>3. 为什么说「环境是瓶颈,不是 GPU」?</summary>

答:单次 GPU 推理 step 耗时毫秒级,而环境步——跑一次单测(10-60s)、加载网页(1-10s)、截图+执行 GUI 动作(1-5s)——耗时秒级,等于每步 GPU 在等环境。一个 rollout worker 的 GPU 利用率 = `GPU推理时间/(GPU推理时间+环境等待时间)`,在同步模式下通常 <5%。所以系统吞吐的上界被「并行环境数 × 单环境速度」卡死,而非 GPU 算力。

**追问：** 怎么做到 GPU 不等环境? → 大量环境 worker 并行(几十到百级),任一时点总有环境已返回、可推理;同时异步派发:rollout worker 收到环境返回后立即发起推理请求,GPU 始终有活可干。这正是三池架构和 SkyRL-Agent 的异步流水线要解决的问题。

</details>

### L2 进阶

<details class="qa"><summary>4. verl / OpenRLHF / AReaL 三者在架构上的核心差异是什么?选型时怎么看?</summary>

答:verl 是 **hybrid engine**(训推同 GPU 分时复用),OpenRLHF 是 **Ray actor**(天然异步调度),AReaL 是**全异步分离**(生成/训练独立 GPU 池)。选型:快速原型 → OpenRLHF(Ray 生态灵活);中等规模+已有 FSDP → verl(hybrid 省 GPU);大规模持续训练 → AReaL(agent-native 全异步高吞吐)。但评估框架的**当前 release 成熟度**比架构理念更重要——框架都在快速迭代中。

**追问：** hybrid engine 在哪些情况下反而是劣势? → ① 多轮长轨迹场景,KV cache 与训练状态争抢显存;② 推理和训练需要不同精度的 GPU(推理可用 INT8/FP8,训练需 FP16/BF16);③ 推理与训练的吞吐比不均衡时,hybrid 切换开销吃掉收益。

</details>

<details class="qa"><summary>5. 多轮 agent 的 KV cache 为什么会「爆炸」?有哪些缓解策略及各自的代价?</summary>

答:第 $t$ 轮上下文 = 此前 $t-1$ 轮全部历史 → 序列长度 $O(T \times L)$,KV cache 同比例增长。缓解:① **KV 驱逐**——丢弃低注意力 token,有损(丢的信息可能后来需要);② **摘要压缩**——远轮压成文本,省显存但压缩误差影响决策;③ **截断重启**——超限后丢旧历史,agent 失去对早期行为的记忆;④ **外置记忆**——写入向量库,检索精度和延迟是代价。实务:近轮完整 + 远轮摘要 + 全局驱逐阈值。

**追问：** 为什么没提「增多 GPU」这个方案? → 单纯加 DP(数据并行)GPU 不能解决单条轨迹的 KV cache 上限——每条轨迹的 KV cache 仍需装载在单个 GPU 上;TP(张量并行)/PP(流水线并行)/context parallelism(长上下文并行)可以将 KV 分片到多 GPU 上提升单轨迹上限,但引入通信开销与复杂度。

</details>

<details class="qa"><summary>6. 轨迹数据的存储格式为什么推荐 Parquet 而非 JSONL?具体管线怎么做?</summary>

答:① **列存**:可以只读 `reward` 列做过滤,无需反序列化全文 → 过滤速度 >10× JSONL;② **压缩**:列存对同类型数据压缩率高(~5-10× vs JSONL 文本);③ **生态**:pandas/polars 原生读,spark 可做分布式处理。管线:rollout → 存为 Parquet(每 N 条一个文件)→ 过滤阶段读 reward + metadata 列做筛选 → 训练时读筛选后的轨迹 token 序列。需注意:对于需要改轨迹文本的任务(摘要、rewrite),Parquet 的字符串列不如 JSONL 好 diff,但大多数 RL 场景只读不改。

**追问：** 百万条轨迹的 Parquet 数据湖怎么管理? → 按日期/任务类型/模型版本分区存储;每条轨迹带唯一 ID + model checkpoint version,保证可追溯;用 parquet → arrow → dataloader 零拷贝链加载体 token 列,避免 Python 循环解序列化的瓶颈。

</details>

### L3 深入

<details class="qa"><summary>7. 设计一个面向生产级 SWE-agent 训练规模(千级 repo、万级 issue,超过 SWE-bench 原版的 12 repo/2294 题)的 agent RL 训练系统,环境层需要处理哪些最棘手的工程问题?</summary>

答:① **多仓库异构**:每个 repo 需要独立的 docker 镜像 + 依赖,不能共用一个 image → 镜像构建与缓存管线(增量构建、layer sharing);② **非确定性测试**:同一 fix 跑同一套测试两次可能结果不同(网络/随机种子/时序依赖)→ reward 计算需多次运行取多数或引入 CI 的 rerun 机制;③ **测试执行安全性**:agent 生成的代码可能包含 `rm -rf /` 或对外发送数据的恶意/错误操作 → docker 必须做网络隔离 + filesystem write 限制 + 资源配额;④ **大规模 docker 调度**:千级并发测试需要等效数量的 docker daemon,单机 docker 并发受限于 pid/memory → 需要跨多台 CPU 机器的环境集群 + 统一调度层;⑤ **repo 状态管理**:每个 issue 对应 git repo 的特定 commit,checkout 和 reset 操作需在秒级完成(hot checkout pool)。

**追问：** 如何防止 agent 在训练中修改环境本身(如修改测试文件来"骗过"判据)? → 判据(单测)所在的文件/目录对 agent 只读或不可见;测试执行在独立的非交互式 docker 容器中运行,agent 无法接触测试代码;环境重置后 docker diff 检查是否产生了预期外的文件变更。

</details>

<details class="qa"><summary>8. 从零搭建一次 agent RL 训练,在 infra 层面最常被低估的三件事是什么?</summary>

答:① **环境可靠性投入**:第一版训练脚本往往假设环境是可靠的,实际上 docker 超时、网络抖动、测试 flaky、web 页面改版会让 5-20% 的 rollout 产出不可靠 reward;修复这些的成本远高于写训练循环本身。② **KV cache 与显存的精细管理**:单轮 RLHF 训练从不操心 KV cache,但多轮 agent 的 KV cache 可能让 80G A100 瞬间 OOM;解决这个问题需要深入推理引擎的 KV 驱逐/压缩配置,而非简单加 GPU。③ **轨迹质量比数量重要**:第一条直觉是「多跑 rollout」,但无过滤的大量低质轨迹(全零 reward / 环境故障 / Echo Trap)会污染训练 batch、浪费 GPU 计算;过滤管线的设计投入比单纯扩 rollout 更划算。

**追问：** 这三件事之间有什么依赖关系? → 环境和 KV cache 是**硬件前提**(它们不解决,训练跑不起来);轨迹过滤是**效率杠杆**(跑起来了,但质量差等于白跑)。合理的推进顺序:先确保环境可靠(单轨迹可复现)→ 再调 KV cache(批量和并发跑通)→ 再加过滤(提升训练效率)。

</details>

<details class="qa"><summary>9. 有人说「agent RL 训练的成本大部分在环境而不在 GPU」,请定量分析这个说法在什么条件下成立、什么条件下不成立?</summary>

答:**成立条件**:环境步延迟高(如 SWE 单测 30s)且并发环境数有限 → 每 GPU-hour 产出的有效轨迹少 → 大部分墙钟时间花在等环境而非算梯度。此时加 GPU 不加速(环境先瓶颈),加环境 worker 才加速。**不成立条件**:环境很快(如规则校验 <100ms)或环境已高度并行(千级 worker) → GPU 训练成为瓶颈(梯度计算时间 > 环境等待时间)→ 此时加 GPU 有效。**定量判断**:算 `GPU利用率 = GPU推理+训练时间 / (GPU推理+训练时间 + GPU等环境时间)`,若此值 < 50% → 环境瓶颈;> 80% → GPU 瓶颈。实务中**小规模实验通常是 GPU 瓶颈、大规模生产通常是环境瓶颈**(瓶颈位置会随扩缩而转移)。

**追问：** 这个 bottleneck 分析对 GPU 选型有什么影响? → 如果是环境瓶颈,买更快的 GPU(H100 over A100)对训练速度几乎无提升——应该把钱花在更多 CPU 环境 worker 上;如果是 GPU 瓶颈,更快的 GPU 直接提升吞吐。先跑瓶颈分析再下单硬件。

</details>

---

## 参考文献 / References

> 均为系统/框架的一手来源,已 web 核实。框架版本与性能数字快变,本页只引架构设计思想。

<ol>
<li id="ref-1">Volcano Engine. <em>verl: Volcano Engine Reinforcement Learning for LLMs</em>. <a href="https://github.com/volcengine/verl">github.com/volcengine/verl</a> — hybrid engine, SPMD 3D 并行, agent RL 支持活跃开发中. <a href="#fnref-1">↩</a></li>
<li id="ref-2">Hu et al. <em>OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework</em>. 2024. <a href="https://arxiv.org/abs/2405.11143">arXiv:2405.11143</a> — Ray 原生异步 RLHF 框架. <a href="#fnref-2">↩</a></li>
<li id="ref-3">Ant Group. <em>AReaL: A Large-Scale Asynchronous RL System for Language Reasoning</em>. 2025. <a href="https://arxiv.org/abs/2505.24298">arXiv:2505.24298</a> — 全异步 RL 系统(系统级,非算法变体;paper 聚焦推理,架构适用于 agent). <a href="#fnref-3">↩</a></li>
<li id="ref-4">NovaSky/Berkeley. <em>SkyRL-Agent: Efficient RL Training for Multi-turn LLM Agent</em>. 2025. <a href="https://arxiv.org/abs/2511.16108">arXiv:2511.16108</a> — CPU/GPU 流水线重叠 + 长程工具调度. <a href="#fnref-4">↩</a></li>
<li id="ref-5">Kwon et al. <em>Efficient Memory Management for Large Language Model Serving with PagedAttention</em>. 2023. <a href="https://arxiv.org/abs/2309.06180">arXiv:2309.06180</a> — vLLM: PagedAttention + continuous batching. <a href="#fnref-5">↩</a></li>
<li id="ref-6">Zheng et al. <em>SGLang: Efficient Execution of Structured Language Model Programs</em>. 2024. <a href="https://arxiv.org/abs/2312.07104">arXiv:2312.07104</a> — SGLang radix attention + 结构化生成. <a href="#fnref-6">↩</a></li>
<li id="ref-7">Duan et al. <em>SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</em>. 2025. <a href="https://arxiv.org/abs/2502.18449">arXiv:2502.18449</a> — SWE 领域 RL 训练数据管线(本页引其环境设计). <a href="#fnref-7">↩</a></li>
<li id="ref-8">Qi et al. <em>WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning</em>. 2024. <a href="https://arxiv.org/abs/2411.02337">arXiv:2411.02337</a> — Web 领域 RL 训练的环境与自进化课程. <a href="#fnref-8">↩</a></li>
</ol>
