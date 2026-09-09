---
title: "从一个 Token 出发：硬件架构师如何读懂 TPU 推理优化"
description: "先补齐 Attention、KV Cache、MoE 和 GDN 的最少模型知识，再把 TPU 推理优化还原为计算、内存、互连和状态管理问题。"
date: 2026-09-08
updated: 2026-09-09
slug: tpu-inference-hardware-software-codesign
status: published
github: true
public: true
wechat: draft
wechat_url:
cover: /assets/tpu-inference-hardware-software-codesign/figure-15.png
series: "Y26W37"
content_type: analysis
---

# 从一个 Token 出发：硬件架构师如何读懂 TPU 推理优化[Y26W37][解析]

***这篇文章不假设读者熟悉大模型。我们先回答三个基础问题：模型生成一个 Token 时在做什么，为什么需要保存 KV Cache，模型太大时怎样拆到多颗芯片。建立这张地图后，再解释 DP Attention、MoE、GDN、Paged Attention 和 PD 分离。***

***先记住一个结论：大模型推理不是“把一个大矩阵乘完”。每生成一个新 Token，系统都要读取模型权重、读取历史状态、执行矩阵计算、选择专家、在设备之间交换数据，再写回新状态。芯片算得快只是第一步，数据能否及时到达才决定实际性能。***

***Qwen3.5-397B 同时使用 Attention、MoE 和 GDN，恰好覆盖三类硬件压力：读取不断增长的历史、搬运不规则的 Token，以及更新每个请求的固定状态。本文用它解释软件术语背后的硬件因果链，并把单芯片优化连接到 ICI/Boardfly 网络、KV 分层和 Agentic AI。***

## 原始材料

**[TPU Inference Externalization Full Steam Ahead - InferenceX](https://newsletter.semianalysis.com/p/tpu-inferencex-full-steam)**

本文围绕原始材料第三至第七章进行硬件架构解读。原始性能数字来自 SemiAnalysis 对 TPUv7 Ironwood 与 Qwen3.5-397B 的公开分析及其引用的 vLLM TPU PR。不同数字来自不同模型、并发、输入输出长度和测量层级，不能直接累加。

## 关键结论

- **模型可以先理解成重复执行的 Layer。** 每层读取当前 Token 表示和历史状态，做矩阵与向量运算，再把结果送到下一层；最后输出下一个 Token 的概率。
- **Attention 是“回看历史”，KV Cache 是“保存已经算过的历史”。** 没有 KV Cache，每生成一个 Token 都要重新计算全部旧 Token；有了它，计算减少，但内存容量和带宽压力持续增长。
- **MoE 是“每次只调用部分专家”。** 它减少实际计算量，却引入 Token 路由、排序、跨设备搬运和负载不均衡。
- **GDN 是另一种历史压缩方法。** 它不保存随上下文线性增长的完整 KV，而是维护固定大小的 recurrent state；代价是每个 Token 都存在严格的状态更新依赖。
- **第三章的本质不是 kernel optimization，而是资源分工。** MXU/TensorCore 负责规则矩阵计算，VPU 负责向量与标量操作，SparseCore 负责不规则 gather、sort、permutation 和部分 collective，VMEM/HBM 保存不同生命周期的数据，ICI 负责设备间交换。
- **DP Attention 与 EP 组合是在复制成本和状态容量之间做选择。** 复制较小的 attention weight，按请求分片不断增长的 KV cache，同时继续分布巨大的 expert weight，避免把最昂贵的数据复制错地方。
- **MoE 优化首先是数据整形。** Expert GEMM 并不一定慢，真正的前置成本是 token 按 expert 分桶、排序、搬运、padding 和跨设备交换。SparseCore 的作用类似独立 data-movement engine，但它有启动、同步和 VMEM 占用成本。
- **GDN 优化首先是依赖图重写。** 代数展开把 MXU 从 VPU state update 的串行依赖中解放出来；double buffering 与 kernel fusion 则减少等待和 HBM round trip。
- **Paged Attention 优化首先是物理布局和容量。** Lane width、page size、VMEM 双缓冲和 KV-head count 决定实际可用容量。容量不足会通过排队放大成 TTFT，而不是只表现为 OOM。
- **第四、五章解释第三章为什么有效。** 双 logical device、SparseCore、256×256 MXU、die-to-die link 和 ICI torus 决定了优化的硬件边界；TPUv8i Boardfly 则试图降低大规模 collective 的 hop 与尾延迟。
- **第六、七章把优化边界推到集群。** PD 分离、TPU-Sync、DRAM/NVMe KV 池和 prefix caching 把状态从芯片内部扩展到服务级生命周期。Agentic workload 的核心指标因此是合格任务成本与端到端尾延迟，而非单卡 tokens/s。

## 目录

0. 阅读前只需掌握的模型知识
1. 再建立 TPU 硬件资源地图
2. 一个 Token 在 Qwen3.5 中经历什么
3. DP Attention 与 EP：如何放置权重和状态
4. MoE：为什么数据整形比 GEMM 更难
5. GDN：如何从依赖链中释放 MXU
6. Paged Attention：容量、布局和排队的闭环
7. Ironwood：这些优化依赖哪些芯片机制
8. ICI Torus 与 Boardfly：节点内优化如何扩展到 Pod
9. PD 分离与 KV 分层：状态离开 HBM 以后
10. Agentic AI：为什么优化目标必须改变
11. 架构决策表与验证路径

---

## 0. 阅读前只需掌握的模型知识

这一章不追求完整讲解 Transformer，只建立阅读后文所需的最小模型。对于硬件架构师，可以先把大模型看成一个**多层、反复读取权重和状态的流式计算程序**。

### 0.1 Token 是什么

模型不直接处理汉字或单词，而是先把文本切成 Token。Token 可以是一个字、词的一部分、标点或代码片段。每个 Token 被映射成一组数字，也就是向量。

```text
“TPU 很快”
  -> Token ID 序列
  -> 每个 ID 查表得到一个向量
  -> 向量依次经过几十层或上百层模型
  -> 输出下一个 Token 的概率
```

模型生成一句话时，并不是一次输出完整句子，而是循环执行：

```text
已有 Token -> 计算下一个 Token -> 把新 Token 加入历史 -> 再计算
```

因此，生成 1,000 个 Token，主模型通常要执行约 1,000 次 decode step。每一步的小延迟都会累积。

### 0.2 一层模型做哪几件事

对本文而言，一层可以简化成三类工作：

1. **Attention：从历史 Token 中查找当前需要的信息。**
2. **FFN 或 MoE：对当前 Token 做主要的特征变换。**
3. **状态更新：保存下一步仍需使用的信息。**

```text
当前 Token 向量
  -> Attention：读取历史
  -> FFN/MoE：执行主要矩阵计算
  -> GDN/其他状态更新
  -> 下一层
```

模型有很多层，所以这个过程会重复很多次。跨设备通信若每层发生一次，其固定延迟也会被层数放大。

### 0.3 Attention、Q、K、V 到底是什么

Attention 可以理解为一次“当前问题与历史记录的匹配”。对当前 Token，模型生成三组向量：

- **Q，Query：** 当前 Token 想从历史中找什么；
- **K，Key：** 每个历史 Token 可被匹配的索引；
- **V，Value：** 匹配后真正取回的内容。

计算过程可以简化为：

$$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

先用 $QK^T$ 计算当前 Token 与每个历史 Token 的相关性，再按相关性对 $V$ 加权求和。硬件上主要包含矩阵乘、Softmax、向量操作和历史状态读取。

### 0.4 KV Cache 为什么会不断增长

历史 Token 的 K 和 V 一旦算出，后续每一步都还要使用。把它们保存在内存中，就得到 KV Cache。

```text
第 1 步：保存 Token 1 的 K/V
第 2 步：读取 Token 1，保存 Token 2
第 3 步：读取 Token 1..2，保存 Token 3
...
第 n 步：读取 Token 1..n-1，保存 Token n
```

KV Cache 用容量换计算：避免反复重算历史，但其容量大致随以下变量增长：

$$
C_{KV}
\propto
N_{requests}\times
L_{context}\times
N_{layers}\times
N_{KV\ heads}\times
d_{head}\times
\mathrm{bytes/element}
$$

其中：

- $N_{requests}$：并发请求数；
- $L_{context}$：每个请求的上下文长度；
- $N_{layers}$：模型层数；
- $N_{KV\ heads}$：KV head 数量；
- $d_{head}$：每个 head 的向量宽度。

所以 KV Cache 既是性能优化，也是内存容量问题。长上下文和高并发会先把 HBM 占满，之后请求只能排队或把 KV 搬到 DRAM/NVMe。

### 0.5 Head、GQA 和 32Q/2KV 是什么意思

模型不会只做一组 Attention，而是使用多组 head，让不同 head 学习不同关系。普通 Multi-Head Attention 可以让每个 Query head 都有自己的 K/V head，但这会产生很大的 KV Cache。

**Grouped Query Attention（GQA）**让多个 Query heads 共享较少的 KV heads。Qwen3.5 的例子是：

- 32 个 Query heads；
- 2 个 KV heads；
- 每个 KV head 被 16 个 Query heads 共享。

这样可显著减少 KV Cache，但也产生设备映射问题：32 个 Query heads 很容易均分到 8 个设备，每设备 4 个；2 个 KV heads 却无法均分到 8 个设备。后文的 DP Attention 正是在解决这个问题。

### 0.6 FFN 与 MoE 的关系

普通 Transformer 的 FFN 可以理解为每层中的大型“特征加工厂”。所有 Token 都通过同一组 FFN 权重。

Mixture of Experts（MoE）把一个大 FFN 换成很多 Expert：

```text
Token
  -> Router 计算每个 Expert 的分数
  -> 选择 Top-k Experts
  -> 只执行被选中的 Expert
  -> 合并结果
```

Qwen3.5 有 512 个 routed experts，但每个 Token 只激活少数专家。收益是模型总参数可以很大，而每个 Token 的实际计算量较小。代价是 Token 会被路由到不同专家，产生排序、分桶、跨设备 All-to-All/AllGather 和负载不均衡。

对硬件而言，MoE 的矛盾是：**Expert 内部是规则 GEMM，Expert 之间的 Token 分配却高度不规则。**

### 0.7 GDN 为什么又多出一种 State

Gated DeltaNet（GDN）是一类 recurrent/linear-attention 模块。这里无需先理解其完整数学，只要知道它不像普通 Attention 那样保留全部历史 KV，而是把历史压缩进固定大小状态 $S_t$：

$$
S_t=f(S_{t-1},x_t)
$$

每来一个新 Token，就读取旧状态 $S_{t-1}$、更新为 $S_t$，再用于当前输出。

它与 KV Cache 的差异是：

| 状态 | 随上下文增长吗 | 每个 Token 更新吗 | 主要压力 |
|---|---|---|---|
| KV Cache | 是 | 追加新的 K/V | 容量、读取带宽 |
| GDN recurrent state | 否，大小固定 | 是，原状态被更新 | 依赖链、读写带宽、checkpoint |

Qwen3.5 同时使用 GQA 和 GDN，所以系统必须同时管理“不断增长的 KV”与“固定但不断更新的状态”。

### 0.8 Prefill 与 Decode 为什么是两种工作负载

收到一段输入时，模型先并行处理全部输入 Token，这叫 **Prefill**。之后逐个生成输出 Token，这叫 **Decode**。

| 阶段 | 输入方式 | 典型特征 | 主要约束 |
|---|---|---|---|
| Prefill | 一次处理很多输入 Token | 大矩阵、并行度高 | 计算能力、HBM 带宽 |
| Decode | 每次生成一个或少量 Token | 小矩阵、反复读权重和状态 | 内存带宽、通信、尾延迟 |

`8k1k` 表示约 8,000 个输入 Token 加 1,000 个输出 Token；`1k8k` 则相反。前者 prefill 占比更高，后者 decode 和状态管理压力更大。因此同一优化在 8k1k 与 1k8k 上可能表现完全不同。

### 0.9 TP、DP、EP 是三种“拆法”

模型或请求放不进一颗芯片时，需要并行化：

| 缩写 | 中文 | 拆分对象 | 主要代价 |
|---|---|---|---|
| TP | Tensor Parallelism，张量并行 | 一个矩阵/一层拆到多设备 | 每层频繁 collective |
| DP | Data Parallelism，数据并行 | 不同设备处理不同请求 | 复制模型权重、负载不均 |
| EP | Expert Parallelism，专家并行 | 不同设备保存不同 Experts | Token 路由与 All-to-All |

`TP8`、`DP8`、`EP8` 中的 8 表示使用 8 个逻辑设备。它们可以组合，例如 Attention 用 DP8，而 MoE 用 EP8。

### 0.10 Collective 是什么

Collective 是多设备共同参加的通信操作。后文主要出现：

- **AllGather：** 每个设备贡献一部分，最后每个设备都拿到完整结果；
- **ReduceScatter：** 先把各设备数据求和/归约，再把结果切片分回各设备；
- **AllReduce：** 所有设备的数据归约后，每个设备都拿到完整结果；
- **All-to-All：** 每个设备分别向所有其他设备发送不同数据，MoE 路由常用。

它们的时间不仅取决于字节数，还包括启动、同步、hop、拥塞和最慢参与者。小消息往往不是带宽受限，而是固定延迟受限。

### 0.11 先记住这些缩写

| 缩写 | 含义 | 阅读时可先理解为 |
|---|---|---|
| Token | 模型处理和生成的离散单位 | 一个文本片段 |
| Attention | 基于当前 Q 检索历史 K/V | 回看历史 |
| KV Cache | 保存历史 Token 的 K/V | 已计算历史 |
| GQA | 多个 Q heads 共享少量 KV heads | 用共享减少 KV 容量 |
| MoE | 每个 Token 只调用部分 Experts | 稀疏 FFN |
| GDN | 用固定 recurrent state 压缩历史 | 固定大小记忆 |
| GEMM | General Matrix Multiply | 大矩阵乘 |
| MXU | TPU Matrix Multiply Unit | 矩阵阵列 |
| HBM | High Bandwidth Memory | 大容量高带宽显存 |
| VMEM | TPU 片上向量存储 | Kernel 工作区 |
| TTFT | Time To First Token | 首 Token 等待时间 |
| TPOT | Time Per Output Token | 后续每 Token 时间 |
| Paged Attention | 把 KV Cache 按页管理 | KV 的虚拟内存 |
| Prefix Cache | 复用相同输入前缀的状态 | 避免重算共同历史 |
| PD 分离 | Prefill 与 Decode 使用不同资源池 | 计算池与生成池分开 |

读完这一章，后文只需要持续问四个问题：**计算在哪里，数据在哪里，为什么等待，优化后代价去了哪里。**

---

## I. 再建立 TPU 硬件资源地图

### 1.1 不要把 TPU 看成一块“大矩阵乘芯片”

对本文讨论的推理路径，可以把 Ironwood 简化为六类资源：

| 资源 | 在大模型中具体承担的工作 | 擅长工作 | 不擅长工作 | 主要瓶颈 |
|---|---|---|---|---|
| MXU / TensorCore | 执行 Q/K/V projection；计算 $QK^T$ 与 attention output；执行普通 FFN 和 MoE Expert 的线性层；完成 GDN 中的大矩阵投影 | 大而规则的矩阵乘、Expert GEMM、Attention Matmul | Ragged gather、sort、指针追踪、小消息同步 | 模型维度与 256×256 阵列不匹配造成的 padding；operand 供给不足；等待 VPU、SparseCore 或通信结果 |
| VPU | 执行 Softmax、归一化、激活函数、门控、逐元素运算；计算 GDN 的 decay、rank-one update 和小规模向量修正 | Vector update、elementwise、标量/向量运算 | 大矩阵乘、跨设备通信 | Register pressure；Q/K 等临时值存活过久导致 spill；与 MXU 形成串行数据依赖 |
| SparseCore | 执行 Embedding 查表；处理 MoE Top-k 路由、Token 分桶、gather/permutation、routing metadata；承担部分 AllGather/ReduceScatter 数据搬运 | 不规则索引访问、稀疏操作、Token 重排、部分 collective | 过小任务；已经位于 VMEM、可由 TensorCore 低成本完成的简单 collective | Offload 的启动与同步开销；局部存储容量；与 TensorCore/ICI 的生产者消费者节拍不匹配 |
| VMEM | 保存当前 Kernel 的 Q/K/V tile、Attention 中间块、GDN FP32 临时状态、MoE 重排缓冲区以及 double-buffering 的当前块和预取块 | 低延迟片上工作集、Kernel scratch、tile、双缓冲 | 完整模型权重、长上下文 KV Cache、跨请求长期状态 | 容量不足会限制 tile/block 大小和流水深度；bank/port 冲突；buffer lifetime 过长导致其他 Kernel 回退 |
| HBM | 常驻模型与 Expert 权重；保存活跃请求的 KV Cache、GDN recurrent state、block table 关联数据和较大中间张量 | 大容量、高带宽、规则连续的数据流 | 高频小粒度随机访问、跨节点共享、容量无限增长的长上下文 | Decode 反复读取权重与 KV 形成带宽瓶颈；KV 容量限制并发；Kernel 间中间结果回写产生 round trip |
| ICI / die-to-die | 在 TP/EP 设备间交换 Activation、Expert routing metadata 和 partial sum；执行 AllGather、ReduceScatter、AllReduce、All-to-All；在 PD 分离场景搬运 KV | 芯片内 logical device 和芯片间的大规模 collective、并行状态交换 | 不能与计算重叠的小消息；远距离多跳通信；热点和不均衡流量 | Message size、固定启动延迟、hop 数、拓扑映射、拥塞与最慢参与者决定的尾延迟 |

Ironwood 每颗芯片暴露两个独立 logical devices，而不是一个统一共享内存的 MegaCore。每颗芯片有 2 个 TensorCores 和 4 个第三代 SparseCores。两个 logical devices 先通过更快的 die-to-die link 交互，再通过 ICI 与其他芯片通信。

![Ironwood 芯片内部组织](/assets/tpu-inference-hardware-software-codesign/figure-28.png)

> **图 1：Ironwood 芯片内部组织。** 来源：[Google](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)。

这个组织直接产生一个优化原则：

> **先在芯片内聚合，再跨芯片交换；把规则计算留给 MXU，把不规则搬运交给 SparseCore；让 VMEM 容量决定流水深度。**

### 1.2 用五个时间项看任何优化

一个推理 kernel 或 layer 的时间可以粗略拆成：

$$
T_{layer}
=
\max(T_{matrix},T_{vector},T_{memory},T_{fabric})
+T_{serial}
+T_{queue}
$$

- $T_{matrix}$：MXU/TensorCore 的规则计算时间；
- $T_{vector}$：VPU/SparseCore 上的更新、路由和重排；
- $T_{memory}$：HBM 与 VMEM 之间的数据供给；
- $T_{fabric}$：die-to-die 和 ICI collective；
- $T_{serial}$：不能重叠的依赖、launch 和同步；
- $T_{queue}$：容量不足或调度冲突造成的等待。

前四项可以通过流水重叠，所以取最大值；后两项往往落在关键路径上，必须额外相加。第三章的大部分优化，不是在减少 FLOPs，而是在减少 $T_{serial}$、$T_{queue}$，或把某项搬到与 MXU 并行的资源上。

### 1.3 三种收益必须分开

- **少做工作**：减少 padding、copy、sort、collective 次数；
- **让工作并行**：MXU 与 VPU/SparseCore、DMA 与 compute、芯片内与芯片间流水重叠；
- **允许更多请求进入系统**：释放 HBM、增加 KV page，降低因容量不足产生的排队。

第三种最容易被误读。它可能只增加几个百分点的 kernel latency，却让 median TTFT 降低 95%，因为系统跨过了容量导致的 admission cliff。

---

## II. 一个 Token 在 Qwen3.5 中经历什么

前一章已经解释了基本组件。现在把它们放回 Qwen3.5-397B。这个模型同时包含：

- GQA attention：32 个 query heads、2 个 KV heads；
- MoE：512 个 routed experts；
- GDN：每请求固定大小 recurrent state；
- KV cache：随上下文长度增长；
- Serving metadata：request slot、block table、page map、routing result。

一次 decode step，也就是生成一个新 Token 的简化数据流是：

```text
请求调度 / slot 选择
  -> 读取本请求的 KV page 与 GDN state
  -> Attention: Q 计算、KV 读取、attention matmul
  -> Router: 为每个 token 选择 top-k experts
  -> Permute: 按 expert 重排 token
  -> AllGather: 汇集 activation 与 routing metadata
  -> Expert GroupedGEMM
  -> ReduceScatter / Combine
  -> GDN state update 与 output projection
  -> 写回 KV / recurrent state
  -> 生成下一个 token
```

这里至少存在三种数据形态：

| 数据 | 形态 | 生命周期 | 正确放置原则 |
|---|---|---|---|
| Model/expert weight | 大、只读、规则 | 模型常驻 | 分布后尽量不复制 |
| KV cache | 按请求增长、读写 | 多轮会话 | 按请求局部化，容量优先 |
| Routing metadata/token group | 小、不规则、每层变化 | 单 layer/step | 减少 collective 次数与整形开销 |
| GDN recurrent state | 每请求固定大小、每 token 更新 | 会话级 | 紧凑分配，避免覆盖 prefix checkpoint |

这张表是理解后续所有优化的钥匙。优化不是抽象的“换并行策略”，而是在重新决定这四类数据由谁拥有、在哪里复制、何时搬运。

---

## III. DP Attention 与 EP：如何放置权重和状态

> **先说人话：** 8 颗芯片合作时，不是什么都平均切八份。Attention 的历史状态适合跟着请求放在本地，巨大的 Expert 权重适合分散保存。DP Attention + EP 就是在为两类数据选择不同拆法。

### 3.1 为什么 TP8 对 2 个 KV Heads 很别扭

Qwen3.5 有 32 个 query heads。TP8 时，每个 device 分 4 个 query heads，完全整除。但只有 2 个 KV heads，无法自然分给 8 个 devices。

若强行把一个请求的 attention 横跨 8 个设备：

- 多个设备需要同一 KV head；
- KV 数据可能发生额外 All-to-All 或复制；
- 每个请求的 KV history 随 token 增长，通信量也随上下文增长。

更合理的做法是复制很小的 KV-head 逻辑或 attention weight，让不同 device 各自处理不同请求，把每个请求的 KV cache 留在本地。vLLM 原本已有“TP degree 大于 KV-head count 时复制 KV heads”的机制，TPU backend 的[兼容性修复](https://github.com/vllm-project/tpu-inference/pull/2661)避免了不必要的 All-to-All。

### 3.2 DEP8 的真实含义

高并发配置使用 **DP8 Attention + EP8 Experts**：

- Attention 采用 data parallel：8 个设备处理不同 request subset；
- 每个设备本地拥有这些请求需要的两个 KV heads 和各自 KV history；
- Attention weight 被复制；
- 512 个 expert 继续用 EP8 分布，避免复制庞大的 expert weight；
- 每层 attention 与 MoE 之间仍需交换 activation 和 routing metadata。

![DP8 Attention 与 EP8 组合](/assets/tpu-inference-hardware-software-codesign/figure-15.png)

> **图 2：DEP8 的关键不是“DP 加 EP”这个名字，而是小权重复制、大权重分布、请求状态局部化。** 来源：SemiAnalysis。

可以用一个简单决策式理解：

$$
C_{placement}
=
C_{weight\ replication}
+C_{state\ replication}
+C_{activation\ communication}
+C_{imbalance}
$$

DEP8 选择增加较小的 attention weight replication，换取 KV state locality；expert weight 仍分布，代价转移为 attention/MoE 边界的 activation collective。

### 3.3 为什么这不是静态拓扑问题

Serving engine 必须同步管理：

- 请求分配到哪个 attention rank；
- GDN recurrent-state slot 在哪里；
- KV block table 如何映射；
- 请求结束或迁移时如何回收；
- EP combine 后结果如何回到原 attention rank。

因此，硬件提供 DP/EP 能力不等于系统自动高效。Placement metadata 必须成为 serving scheduler 的一等状态。对应实现见 [DP8 支持](https://github.com/vllm-project/tpu-inference/pull/2187)与[request/state/block-table 协调](https://github.com/vllm-project/tpu-inference/pull/2577)。

### 3.4 成立边界

DEP8 更适合：

- 并发足够高，可把请求均匀分布到 8 个 attention ranks；
- KV state 相对 weight 更值得局部化；
- Expert weight 很大，复制成本不可接受；
- Attention/MoE 边界 collective 可以与计算重叠。

低并发或请求长度高度不均时，DP rank 可能空闲；此时 TP attention 可能更合适。原文的低并发 tuning 在部分 workload 上切回 TP8，这说明并行策略必须随 concurrency 和 shape 改变。

---

## IV. MoE：为什么数据整形比 GEMM 更难

> **先说人话：** MoE 像 512 个加工单元，但每个 Token 只去其中几个。矩阵乘本身很规则，难点是先把散落的 Token 送到正确 Expert，算完后再按原顺序拼回来。

### 4.1 Expert GEMM 之前发生了什么

MoE router 为每个 token 选出 top-k experts。假设一批 token 的选择结果是：

```text
Token 0 -> Expert 7, 19
Token 1 -> Expert 3, 7
Token 2 -> Expert 211, 19
...
```

MXU 不能直接高效消费这种散乱列表。系统必须：

1. 读取 expert ID 与 routing weight；
2. 按 expert 对 token 排序或分桶；
3. 把属于同一 expert 的 token 搬到连续区域；
4. 对不同长度 group 做 padding 或 ragged layout；
5. 跨 EP device 交换 activation/metadata；
6. 执行 GroupedGEMM；
7. 按原 token 顺序 combine 并回送结果。

真正的性能问题是，规则 GEMM 被不规则前后处理包围。

### 4.2 合并 metadata collective：少一次同步比少几个字节更重要

原实现分别 AllGather selected expert IDs 和 routing weights。两者都是小数组，传输本身不大，但每次 collective 都包含 launch、同步和 fabric latency。合并成一次 AllGather 后，DeepSeek-V3 测量每层约节省 **80 µs**。

![合并 expert ID 与 routing weight collective](/assets/tpu-inference-hardware-software-codesign/figure-16.jpg)

> **图 3：把两个小 collective 合成一个。** 来源：SemiAnalysis。

DeepSeek-V3/R1 有 58 个相关层，隔离估算为：

$$
80\ \mu s/layer\times58\ layers=4.64\ ms/forward
$$

这个数字说明小消息 latency 会沿层数串行累积。但它不是端到端实测增益，因为不同层可能还有 overlap、queue 和其他瓶颈。

### 4.3 为什么把 routing 搬到 SparseCore

MXU 的利用率要求 operand 规则、连续、尺寸足够大。Routing、gather、sort、permutation 的特征正相反：

- Index-driven；
- 数据不连续；
- Group 长度变化；
- 控制与 metadata 比例高；
- 算术强度低。

把 token rearrangement 放到 SparseCore，等于引入一个面向 irregular data movement 的旁路引擎：SparseCore 把每个 expert 的 token 整形成连续 group，TensorCore 同时准备或执行 expert GEMM。

![MoE 的 SparseCore 与 TensorCore 分工](/assets/tpu-inference-hardware-software-codesign/figure-19.jpg)

> **图 4：不规则数据整形与规则矩阵计算解耦。** 来源：SemiAnalysis。

相关收益：

- Grouped matmul v2 删除冗余 tile，只按 valid rows 搬运，并 triple-buffer expert weight；
- SparseCore rewrite 改善 memory-read pipeline，并沿 token/hidden dimension 拆分 combine；
- 相对原 SparseCore kernel，8k1k serving throughput 报告提高 **12%**；
- Top-k gather 移到 SparseCore 后，TensorCore overhead 从 **29 µs 降到 14 µs**，整个操作从 **146 µs 降到 137 µs**，测量边界为 DeepSeek-V3、batch 2k、EP16 microbenchmark。

### 4.4 Small batch 为什么反而用普通 Matmul

通用 ragged path 有建表、索引、排序和 launch 开销。Batch 很小时，这些固定成本可能大于实际搬运工作。专用 small-batch path 构造 one-hot matrix，再用普通 matmul 完成 token permutation/unpermutation。

这看似“用重计算代替聪明算法”，但符合硬件事实：小矩阵规则计算可能比启动一套不规则 pipeline 更便宜。该路径在 8k1k 下使并发 64 和 128 的吞吐分别提高 **7.3%**、**5.1%**。

### 4.5 Sort key 为什么能从 106.6 µs 降到 21.7 µs

原 routing sort 需要围绕 expert ID 与 token index 维护更复杂的 key/ordering。把两者编码进单一 sort key 后，XLA 看到更简单的排序问题，同时保持目标顺序。WIP PR 报告 sort latency 从 **106.6 µs 降至 21.7 µs**，8k1k serving 增益为 **0.6%–8.5%**，并为 FP8 AllGather 铺路。

硬件结论是：编译器生成的操作复杂度有时由 metadata representation 决定。改变 key encoding 可以比优化 comparator 更有效。

### 4.6 SparseCore Offload 不是永远正确

若 collective 很小、数据已在 VMEM、TensorCore 可快速完成，那么：

$$
T_{offload}
=T_{dispatch}+T_{transfer}+T_{sync}+T_{SparseCore}
$$

可能大于：

$$
T_{local}=T_{TensorCore/VMEM}
$$

因此 Qwen3.5 用基于 VMEM 容量的 threshold 决定 AllReduce/AllGather 是否 offload。该优化在 8k1k 并发 64 和 128 下分别提高 **2.7%**、**5.7%**。这也是硬件设计的重要提醒：异构执行单元需要动态 crossover policy，不能只提供静态 capability。

---

## V. GDN：如何从依赖链中释放 MXU

> **先说人话：** GDN 为每个请求维护一份固定大小的“运行状态”。旧执行顺序要求先更新状态，再做矩阵乘；优化通过数学等价变换，让状态更新和矩阵乘同时进行。

### 5.1 GDN 与 Attention 的状态不同

Attention 的 KV cache 随序列长度增长；Gated DeltaNet 的 recurrent state 对每个请求大小固定，但每个 token 都会更新。简化表示：

$$
S_t=\mathrm{decay}(S_{t-1})+\Delta_t k_t^T
$$

$$
y_t=S_tq_t
$$

若严格按公式执行：

```text
VPU: decay + rank-one state update
  -> 写出 S_t
  -> MXU: S_t × q_t
  -> output
```

MXU 必须等待 VPU 完成 state update。这是一条真正的数据依赖，不是简单调度问题。

### 5.2 代数重排如何创造并行

把 $S_t$ 展开：

$$
y_t
=(\mathrm{decay}(S_{t-1})+\Delta_t k_t^T)q_t
$$

$$
y_t
=\mathrm{decay}(S_{t-1})q_t
+\Delta_t(k_t^Tq_t)
$$

于是可以并行执行：

```text
MXU: decay(S_{t-1}) × q_t

VPU: 计算 k_t^T q_t
     构造 Δ_t(k_t^T q_t)
     更新供下一 token 使用的 S_t

最后做小规模 vector add
```

![GDN 代数重排](/assets/tpu-inference-hardware-software-codesign/figure-21.png)

> **图 5：代数等价变换删除了 VPU→MXU 的串行等待。** 来源：SemiAnalysis。

这类优化的价值不是减少乘加总数，而是改变 dependency graph。报告的 8k1k 吞吐增益在并发 64 和 512 下分别为 **2.79%** 和 **4.48%**。

### 5.3 Register Spill：片上容量也会制造“内存墙”

Q/K 值若在 decode loop 中存活时间过长，会占据 vector register；超过容量后 spill 到更低层存储，再 reload。通过在 loop 内切分 Q/K，缩短 live range：

- Decode-64 kernel 约快 20%；
- 端到端 8k1k、并发 512 只提升 0.8%；
- 端到端 1k8k、并发 512 提升 3.8%。

局部与端到端差异来自 Amdahl 定律：

$$
S_{total}=\frac{1}{(1-f)+\frac{f}{S_{kernel}}}
$$

只有 kernel 占比 $f$ 足够高，20% kernel speedup 才能显著影响系统。

### 5.4 Double Buffering 为什么可能先变慢

异步 state transfer 使用两套 buffer：计算当前 state 时，DMA 预取下一 state。理想情况下：

$$
T_{stage}\approx\max(T_{DMA},T_{compute})
$$

而不是二者相加。但第二套 buffer 消耗 VMEM，初版曾挤压 DP Attention 所需容量，导致整体回退。通过复用 scratch buffer 和缩短 temporary lifetime 才消除回退，最终 8k1k、并发 512 提升 **11.3%**。

这说明流水不是免费性能。每增加一级 pipeline，都要支付 buffer capacity、控制和 warm-up/drain 成本。

### 5.5 Kernel Fusion 删除 HBM Round Trip

GDN v3 融合 Conv1D 与 GDN：

```text
未融合：Conv1D -> HBM/中间张量 -> GDN
融合后：Conv1D -> 片上中间结果 -> GDN
```

同时改进 prefill layout，并把 mixed prefill/decode 统一到一条路径。Kernel-level speedup 为：

- Decode：1.41×；
- Prefill：1.60×；
- Mixed batch：2.14×。

![GDN v3 流水与融合](/assets/tpu-inference-hardware-software-codesign/figure-22.jpg)

> **图 6：GDN v3 的主要收益来自 HBM round trip 减少和阶段重叠。** 来源：SemiAnalysis。

这些是 kernel-level 数据，不是 serving-level 数据。系统收益仍受 attention、MoE、collective 和调度占比限制。

---

## VI. Paged Attention：容量、布局和排队的闭环

> **先说人话：** Paged Attention 类似操作系统用页管理内存。目标不是改变 Attention 算法，而是让不同长度请求的 KV Cache 能被分配、回收和复用，减少碎片与等待。

### 6.1 Hybrid Model 有两套状态分配器

Qwen3.5 同时维护：

- 随 token 数增长的 KV history；
- 每请求固定的 GDN recurrent state。

初始实现若为每个 layer group 按 `num_blocks` 分配 recurrent-state slot，会把“随 context 增长”的 KV 分配规则错误套在固定状态上。改为约每个 active request 一个 recurrent-state slot 后：

- 回收约 **76 GiB HBM**；
- Attention block pool 扩大 **71%**；
- 1k8k、并发 64 output throughput 提高 **18%**。

![Hybrid state 的容量分配](/assets/tpu-inference-hardware-software-codesign/figure-23.png)

> **图 7：KV 与 recurrent state 必须采用不同容量模型。** 来源：SemiAnalysis。

再把 recurrent state 以 BF16 存储、在 VMEM 内以 FP32 计算，可把 HBM footprint 减半，同时保留 FP32 arithmetic。1k8k、并发 512 提升 **15%**。这里需要验证的是数值稳定性，而不是只看吞吐。

### 6.2 Lane Layout 为什么能让 KV Page 翻倍

TPU vector tile 的 trailing dimension 是 128 lanes。原 FP8 KV layout 沿 head dimension 打包 K/V，packing factor 为 4。若每个 device 只有 1 个 KV head，实际只有 K 和 V 两项，占不满四个 packing slots，一半 lane 被浪费。

Sequence-on-lane layout 改为：

- Token/page dimension 放在 128-lane axis；
- Head dimension 放在 sublane axis；
- Head dimension 只需为 32 的倍数；
- 可用 KV pages 从 **5,141 增至 10,283**。

![Sequence-on-lane KV Layout](/assets/tpu-inference-hardware-software-codesign/figure-26.jpg)

> **图 8：从 head-on-lane 改为 sequence-on-lane，本质是让实际变化维度占满物理 lane。** 来源：SemiAnalysis。

代价是低并发每 token latency 增加约 3%。但在 8k1k、并发 128 下：

- 吞吐提高 16.5%；
- Median TTFT 降低 95%。

TTFT 的巨大下降不是 kernel 快了 95%，而是额外 KV 容量消除了等待可用 page 的排队。

可以写成：

$$
TTFT=T_{service}+T_{KV\ admission\ queue}
$$

容量跨过临界点后，第二项可能从主导项接近归零。

### 6.3 Page Size 是容量、Padding 和流水深度的共同参数

删除旧 hybrid page-size alignment constraint 后，batched attention 可使用 256-token 等合适的 2 的幂 page size，1k8k、并发 512 提升约 **7%**。但该路径不含 prefix caching，不能与后续 checkpoint mode 混为一谈。

Page 太大：

- Internal fragmentation 增加；
- 短 sequence 浪费容量；
- 单次 fetch 占据更多 VMEM。

Page 太小：

- Page table/metadata 增加；
- Gather 次数和 address generation 开销增大；
- DMA transaction 变碎。

因此 page size 是 workload-dependent 的硬件/软件共同参数。

### 6.4 Fetch Block 与 Compute Block 为什么不应相等

Pallas kernel 用 double buffering 隐藏 HBM latency。旧 RPA v3 heuristic 在 v7x decode 中把 KV fetch block 和 compute block 都设为约 16k tokens，导致当前 compute tile 已占满 VMEM，几乎没有空间给下一块 prefetch。

优化后：

- Fetch block：16k tokens，维持大事务效率；
- Compute block：4k tokens，降低 working set；
- VMEM 可同时容纳预取与计算 buffer；
- Qwen3-0.6B decode 从 64.9k 提升到 **96.3k tokens/s**，提升 49%，四次测试复现。

![Fetch/Compute Block 解耦](/assets/tpu-inference-hardware-software-codesign/figure-27.jpg)

> **图 9：数据搬运粒度和计算粒度服务于不同约束，不应由一个参数绑定。** 来源：SemiAnalysis。

Block-size sweep 呈倒 U 曲线：过小导致 command/loop overhead，过大挤压 prefetch buffer。原 tuning table 只能保存一个 block size，因此初版用 environment override，后续才扩展为双参数。这也说明性能可移植性依赖 tuning metadata schema，而不仅是 kernel code。

### 6.5 Prefix Cache 为什么需要保存 GDN Checkpoint

普通 Transformer 的 cached prefix 主要保存 KV blocks。Hybrid model 若只保存 KV，不保存 prefix 末端的 recurrent state，则从缓存点继续生成时无法恢复 GDN 的正确状态。

问题是 live recurrent state 会在后续 token 中原地更新。解决方法是分离：

- Checkpoint read slot：对应 cached prefix 末端；
- Live write slot：当前请求继续生成时更新；
- Block table：同时定位 KV block 与 recurrent checkpoint；
- Aligned cache granularity：确保 state checkpoint 与 KV block boundary 对齐。

![Hybrid Prefix Cache](/assets/tpu-inference-hardware-software-codesign/figure-25.jpg)

> **图 10：Checkpoint 和 live state 分离，避免继续生成覆盖可复用 prefix。** 来源：SemiAnalysis。

代价是必须保留完整 checkpoint pool，无法同时享受最紧凑的 per-request recurrent-state allocation。它是在 HBM 容量和 recompute avoidance 之间交换，只有 prefix hit rate 足够高才值得。

---

## VII. Ironwood：这些优化依赖哪些芯片机制

> **先说人话：** 前面的软件优化之所以可行，是因为 Ironwood 不只有 MXU。它还提供向量、稀疏处理、片上存储、双 die 和专用互连，让不同性质的工作可以真正并行，而不是在同一个执行单元上轮流运行。

### 7.1 256×256 MXU 的高峰值和高 Shape 税

Systolic array 把 weight 驻留在阵列中，activation 从边缘流入，partial sum 逐 cell 传播。TPUv2–v5 使用 128×128 MXU，即 16,384 MAC/cycle；TPUv6e 与 Ironwood 使用 256×256 MXU，即 **65,536 MAC/cycle**，理论每周期 FLOPs 提高 4 倍。

但小维度必须 padding 到阵列 tile：

$$
U_{shape}
=\frac{M\times N\times K}
{\lceil M/T_M\rceil T_M\times
 \lceil N/T_N\rceil T_N\times
 \lceil K/T_K\rceil T_K}
$$

$T_M,T_N,T_K$ 是实现所需 tile granularity。简化到某一关键维度：

- Head dimension 128 对 256-wide MXU：上限约 50%；
- Head dimension 64：上限约 25%；
- DeepSeek MLA 的 128+64=192 也无法自然填满 256。

![模型 Shape 与 MXU 利用率](/assets/tpu-inference-hardware-software-codesign/figure-30.jpg)

> **图 11：大阵列提高峰值，也放大 shape mismatch 的损失。** 来源：SemiAnalysis。

这解释了第三章中的 layout、packing、padding token routing 和 special bucket 为什么重要。软件不是在修补偶然 bug，而是在把模型 shape 映射到物理阵列 geometry。

### 7.2 SparseCore 是结构化的异构卸载点

SparseCore 原本面向 embedding 与 sparse operation，但 Qwen3.5 优化将其扩展到：

- Ragged gather/reduce；
- Top-k routing weight gather；
- Token permutation；
- Sort/metadata；
- Hierarchical ReduceScatter。

硬件价值取决于三件事：

1. 与 TensorCore 是否能真正并行；
2. 与 VMEM/HBM 的数据路径是否避免往返复制；
3. Offload threshold 是否覆盖不同 message size 的 crossover。

只增加异构单元而没有统一 scheduler、memory ownership 和 synchronization primitive，无法得到这些收益。

### 7.3 双 Logical Device 决定 Collective 的层次

Ironwood 每芯片两个独立 logical devices。ReduceScatter 因此先利用低成本 die-to-die link 合并芯片内 contribution，再跨 ICI 交换 partial sum。

![分层 ReduceScatter](/assets/tpu-inference-hardware-software-codesign/figure-17.jpg)

> **图 12：先局部归约，再跨芯片通信，减少进入慢层级的数据量。** 来源：SemiAnalysis。

配合 microbatch double buffering：

![ReduceScatter 流水](/assets/tpu-inference-hardware-software-codesign/figure-18.png)

> **图 13：MB0 进行跨芯片阶段时，MB1 可执行芯片内阶段。** 来源：[Google 与 SemiAnalysis](https://github.com/vllm-project/tpu-inference/blob/main/tpu_inference/kernels/collectives/hierrs_sc/kernel.py#L109)。

报告结果：

- 8k1k、并发 64–512：吞吐提高 4.1%–14.2%；
- 8k1k、并发 256：提高 8.5%；
- 1k8k、并发 512：提高 26.1%。

差异说明通信占比随 input/output ratio 与 concurrency 变化。不能从单一 workload 推导通用 fabric 收益。

---

## VIII. ICI Torus 与 Boardfly：节点内优化如何扩展到 Pod

> **先说人话：** 一颗芯片内部优化完成后，下一瓶颈是多颗芯片之间要走几跳、每跳多快、是否堵塞。Torus 用规则邻居连接换规模，Boardfly 用更多交换能力换更少跳数。

### 8.1 3D Torus 给了什么，又收了什么

Ironwood 使用 ICI 直接连接 TPU，绕过 host CPU、PCIe 和通用 NIC。基本单元是 4×4×4 的 64-chip cube，每颗芯片连接 ±X、±Y、±Z 六个邻居。

![Ironwood 3D Torus](/assets/tpu-inference-hardware-software-codesign/figure-31.jpg)

> **图 14：64-chip 3D torus 对应一个物理 rack。** 来源：SemiAnalysis。

先看一个维度。假设该维度排列了 $N$ 颗芯片，编号为 $0$ 到 $N-1$。如果它们只是首尾不相连的 Mesh，最远的两个端点是芯片 $0$ 和芯片 $N-1$，报文必须依次穿过中间节点：

```text
0 -> 1 -> 2 -> ... -> N-2 -> N-1
```

因此单维 Mesh 的最坏最短路径是 $N-1$ hops，数量级上常简写为约 $N$ hops。

Torus 增加一条 **wraparound link**，把最后一颗芯片 $N-1$ 与第一颗芯片 $0$ 直接连接，使这条线变成一个环：

```text
0 -- 1 -- 2 -- ... -- N-2 -- N-1
|                              |
+------------------------------+
```

此时，报文可以沿两个方向传输，并选择较短的一边。若两个节点在线性编号上的距离为 $d$，则 Torus 中的最短距离为：

$$
H_{torus}(d)=\min(d,N-d)
$$

两个方向距离相等时才达到最坏情况，所以单维 Torus 的最大最短路径为：

$$
H_{torus,max}=\left\lfloor\frac{N}{2}\right\rfloor
$$

例如，一个维度有 8 颗芯片。Mesh 中芯片 0 到芯片 7 需要 7 hops；加入 wraparound link 后，0 与 7 直接相邻，只需 1 hop。整个环上最远的节点对是 0 与 4，需要 4 hops。因此更严谨的说法是：**wraparound link 把单维最坏距离从 $N-1$ 降到 $\lfloor N/2\rfloor$，而不是让每一跳本身变快。**

对于 3D Torus，这个计算要分别应用到 X、Y、Z 三个维度，再把三个方向的最短距离相加。以 Ironwood 的 $4\times4\times4$ cube 为例：

- 不带 wraparound 的 3D Mesh，最坏距离为 $3+3+3=9$ hops；
- 带 wraparound 的 3D Torus，最坏距离为 $2+2+2=6$ hops。

这里的 $N$ 是**单个维度的节点数**，不是整个 cube 的 64 颗芯片。Torus 的优势是：

- 每个 endpoint 不需要连接大型 central switch；
- Link bandwidth 可随节点规模分布；
- 适合规则 collective 与 topology-aware sharding；
- TPUv5p 可在最多 8,960 chips 范围组合 TP、EP 和 FSDP-style DP。

代价是：

- 远端通信经过多个 hops；
- Route contention 与 collective mapping 更复杂；
- Small message latency 对 software schedule 敏感；
- 故障会破坏规则拓扑。

Google 用 twisted torus 降低平均 hop，并用 Optical Circuit Switch 跨 cube 重构连接。原文给出 Ironwood full superpod 最多 **9,216 chips、42.5 FP8 exaflops**，OCS 可在秒级绕过故障 link/chip。

![OCS 扩展与重构 Torus](/assets/tpu-inference-hardware-software-codesign/figure-32.jpg)

> **图 15：OCS 的价值同时包括扩展拓扑与故障重构。** 来源：SemiAnalysis。

### 8.2 Boardfly 为什么更像推理网络

先澄清标题：“更像推理网络”不是说 Boardfly 只能运行推理，也不是说 Torus 不适合推理。更准确的表述是：**Boardfly 优先改善的网络属性，与大规模 MoE Decode 和 Agentic 推理更敏感的指标高度一致。**

TPUv8 首次拆分为两种芯片：

- **TPU 8t：** 面向训练，继续使用 3D Torus；
- **TPU 8i：** 面向推理，改用 Boardfly。

这个选择反映了两类工作负载不同的通信重点。训练通常长期运行大批量计算，通信模式比较稳定，常见大规模 AllReduce、ReduceScatter 和 AllGather，主要追求持续带宽、规模和成本效率。推理特别是 MoE Decode，则会反复产生较小、目的地随 Token 改变的消息，并直接暴露给用户延迟，更关注每次通信的固定延迟、跳数和 p99。

#### 8.2.1 先理解两个词：Radix 与网络直径

**Radix** 是一个交换节点能够直接连接的端口或方向数量。Radix 越高，一个节点可直接到达的邻居越多，但交换芯片、SerDes、布线、功耗和成本也会增加。

**网络直径（Network Diameter）**是任意两个端点之间最短路径的最大 hop 数。它描述最远两个设备最多需要经过多少段链路或交换层级。

例如：

```text
低 Radix 邻居网络：
TPU A -> 邻居 -> 邻居 -> 邻居 -> TPU B
         多次逐跳转发

高 Radix 层次网络：
TPU A -> 本地交换域 -> 上级交换域 -> 远端本地交换域 -> TPU B
         用更多直接连接减少中间跳数
```

网络不是 hop 越少就一定越快，但 hop 数会进入基础延迟：

$$
T_{network}
\approx
T_{inject}
+H\times(T_{switch}+T_{link})
+T_{queue}
+T_{serialization}
$$

其中：

- $T_{inject}$：发送端启动、封装和注入消息的固定开销；
- $H$：路径 hop 数；
- $T_{switch}$：每跳交换与路由开销；
- $T_{link}$：每段链路传播和 PHY 开销；
- $T_{queue}$：竞争造成的排队，通常是尾延迟的主要不确定项；
- $T_{serialization}$：消息字节数除以链路带宽。

对大消息，$T_{serialization}$ 可能占主导；对小消息，字节很快发完，$T_{inject}$、$H$ 和同步等待更重要。MoE Decode 恰好经常属于后一类。

#### 8.2.2 如何读 Boardfly 这张图

![TPUv8i Boardfly](/assets/tpu-inference-hardware-software-codesign/figure-33.jpg)

> **图 16：Google 公布的 Boardfly 层次示意。** 图中表达的是三级高连接度组织；它不是完整的端口、交换芯片或物理布线图。来源：Google。

图中可读出三个层级：

1. **Board/Machine 层：每块板 4 颗 TPU 全连接。** 同板 TPU 之间有直接路径，适合最紧密的模型分片和局部 collective。
2. **Group/Rack 层：每组 8 块板全连接。** 一组共有 $4\times8=32$ 颗 TPU。跨板通信通过组内高连接度网络完成，不需要沿规则网格逐节点转发。
3. **Pod 层：36 个 Group 全连接。** 总规模为 $32\times36=1,152$ 颗 TPU。不同 Group 之间通过更高层连接互达。

这里的“全连接”应按 Google 图中的逻辑拓扑理解。公开材料没有在这张图中完整披露每条逻辑连接对应多少物理 SerDes、是否经过哪类交换芯片、链路复用方式和 oversubscription，因此不能仅凭图推导准确的 bisection bandwidth 或单芯片端口数。

Boardfly 借鉴 Dragonfly 类高 Radix 层次网络的思路：先在小范围内建立高连接度 Group，再用高层连接让不同 Group 以较少跳数互达。它不是把 1,152 颗 TPU 做成物理意义上的单层全互连，那需要不可接受的 $O(N^2)$ 链路数量。

#### 8.2.3 Torus 与 Boardfly 的根本交换

Torus 的每颗芯片只连接固定数量邻居。以 3D Torus 为例，每颗芯片沿 $\pm X$、$\pm Y$、$\pm Z$ 连接 6 个方向。优点是端口数固定、结构规则、扩展成本可控；缺点是远端消息必须经过多个中间节点。

Boardfly 则增加网络层次与交换连接度，让远端设备通过少数高 Radix 层级到达。其交换关系可以概括为：

| 维度 | 3D Torus | Boardfly |
|---|---|---|
| 直接邻居数 | 固定、较少 | 每层连接度更高 |
| 远端路径 | 沿 X/Y/Z 多跳前进 | 经本地组与高层连接快速跨组 |
| 网络直径 | 随规模扩大而增长较快 | 通过层次化连接压低 |
| 布线与交换 | 规则、分布式 | 更复杂，需要更多高 Radix 交换资源 |
| 成本重点 | Endpoint link 与 OCS/拓扑管理 | Switch、SerDes、光互连和每芯片网络附着 |
| 适合优化 | 大规模、规则、可拓扑感知的 collective | 频繁、小消息、目的地动态、尾延迟敏感的通信 |

原文称，在约 1,024–1,152 颗芯片的相近规模下，Boardfly 把网络直径从 3D Torus 的约 **16 hops** 降到约 **7 hops**，下降超过 50%。这里比较的是最远路径的量级，不表示每条消息都固定经过 7 或 16 hops，也不能直接推出延迟精确下降 56%；实际延迟还取决于路由、链路速率、拥塞和 collective 算法。

#### 8.2.4 为什么训练可以容忍 Torus，而推理更在意少 Hop

训练与推理并不是“大消息”和“小消息”的绝对二分，但典型工作点不同。

| 特征 | 大规模训练 | MoE/Agentic 推理 |
|---|---|---|
| 计算粒度 | 大 batch、大矩阵，单 step 较长 | Decode 每步计算较短，逐 Token 重复 |
| 通信模式 | 较稳定的 AllReduce/ReduceScatter/AllGather | Token-dependent EP routing、状态/KV 访问、小 collective |
| 消息特征 | 较大、规则、可提前规划 | 较小、频繁、目的地随请求和 Router 变化 |
| 主要目标 | 持续吞吐与集群利用率 | TTFT、TPOT、p99 和用户可见响应时间 |
| 延迟隐藏 | 可用大计算块和流水覆盖较多通信 | Decode critical path 短，固定延迟更难隐藏 |
| 不均衡 | 数据与模型划分相对稳定 | Expert 热点、请求长度和 Sub-agent burst 动态变化 |

训练中，一次较大的 collective 即使多走几跳，只要链路持续灌满且通信能与长时间矩阵计算重叠，系统仍可能获得较高吞吐。Torus 的规则结构还便于 topology-aware partition 和 collective scheduling。

Decode 则每生成一个 Token 都要重复多层计算。若每个 MoE 层都需要一次或多次跨设备交换，单次网络固定延迟会沿层数和输出 Token 数累积：

$$
T_{MoE\ network/request}
\approx
N_{output\ tokens}
\cdot N_{MoE\ layers}
\cdot T_{collective/layer}
$$

假设每层只多出几十微秒，乘以几十个 MoE 层和上千个输出 Token 后，也会成为明显的用户等待时间。这里不代入具体产品数字，因为实际执行可能有 overlap；公式表达的是累积方向。

#### 8.2.5 用一个 MoE Token 看 Boardfly 的价值

假设当前 Token 在 TPU A 上完成 Router，Top-2 结果选择 Expert 17 和 Expert 301，而两个 Expert 位于其他设备：

```text
TPU A
  -> 发送 Token activation 和 routing weight
  -> Expert 17 所在 TPU 执行 GEMM
  -> Expert 301 所在 TPU 执行 GEMM
  -> 返回两个 Expert 的加权结果
  -> TPU A 合并并继续下一层
```

这一路径的特点是：

- Payload 相对模型权重较小，固定延迟占比较高；
- 目的地由 Router 每次动态决定，不一定符合邻近拓扑；
- Top-k Expert 中最慢的返回路径决定当前 Token 能否继续；
- 热门 Expert 会形成 Incast 和队列；
- 该过程可能在每个 MoE 层、每个输出 Token 重复。

在 Torus 上，远端 Expert 可能经过更多中间 hops，并与其他流共享链路。Boardfly 通过减少最远路径和提供更高层的跨组连接，理论上可降低固定转发成本，并减少一条流长期占用多个中间链路的机会。

但少 hop 不能消除 Expert imbalance。若大量 Token 同时选择同一 Expert，其目标 TPU、入口链路或本地队列仍会成为热点。Boardfly 改善的是网络可达性和路径长度，不替代 Router load-balancing、Expert placement 与 admission control。

#### 8.2.6 为什么 Agentic 推理也受益

Agentic workload 除 MoE routing 外，还有三个特点：

- 多轮会话让同一请求反复进入 Decode；
- Sub-agent burst 会同时创建多个短请求，放大瞬时通信和 KV 分配；
- PD 分离和 KV pooling 可能需要跨设备或跨主机移动状态。

这些流量不像训练 collective 那样始终规则。网络路径更短、可用带宽更高，通常有利于降低状态访问与同步的尾延迟。但若 KV 仍主要经 host Ethernet/RDMA 而不是 ICI，Boardfly 不会自动加速那部分路径。评估时必须先画清 KV 实际经过的 fabric。

#### 8.2.7 19.2 Tb/s ICI 和 384 MB SRAM 为什么一起出现

TPUv8i 同时增加网络带宽和片上状态容量：

| 指标 | TPUv8i 披露/转述值 | 推理意义 |
|---|---:|---|
| Network diameter | 约 7 hops | 降低 collective 固定延迟与 p99 |
| ICI bandwidth | 19.2 Tb/s，前代约 2× | 支撑更宽 EP 与 KV movement |
| On-chip SRAM | 384 MB，前代约 3× | 增加 KV/working-state 片上驻留机会 |
| 对比 3D torus | 直径下降超过 50% | 更适合频繁、小粒度、延迟敏感通信 |

二者解决不同问题：

- **384 MB SRAM** 尽量让热点 KV、metadata 或工作集不离开芯片；
- **19.2 Tb/s ICI 与更少 hop** 处理无法留在本地、必须跨设备交换的数据。

理想策略是先通过 locality 减少网络流量，再让剩余网络流量走更短路径。若软件把不合适的数据长期占在 SRAM，或频繁迁移 KV，硬件容量与网络余量仍可能被浪费。

#### 8.2.8 为什么不是所有推理都需要 Boardfly

以下场景对 Boardfly 的收益可能有限：

- 模型完全放在少量 TPU 内，几乎没有跨组通信；
- Dense model 使用稳定的较大 collective，Torus 已能有效流水；
- 低并发请求受 HBM weight streaming 限制，网络不在关键路径；
- KV offload 走独立 host network，ICI 不是瓶颈；
- Expert placement 很差，热点计算而非网络 hop 才是主因。

因此正确结论不是“Boardfly 比 Torus 更先进”，而是：

> **Boardfly 用更高网络连接度和更高附着成本，换取更小网络直径；这个交换更符合大规模 MoE Decode 和 Agentic 推理对小消息延迟、动态目的地与 p99 的敏感性。**

#### 8.2.9 需要验证什么

Boardfly 的方向合理，但公开结构图和峰值规格还不足以证明端到端收益。至少需要测量：

- 64 B 到数 MB 的 message-size sweep，而不是只看峰值带宽；
- 同规模 Torus 与 Boardfly 的 EP All-to-All、AllGather、ReduceScatter p50/p99；
- 不同 Expert 热度分布下的 Incast、热点链路和队列深度；
- 1、2、4、7 hops 下的延迟与有效带宽；
- Oversubscription、bisection bandwidth 与跨 Group 流量比例；
- 故障和降级路由后的直径、带宽与尾延迟；
- Switch、SerDes、光互连、功耗和每芯片网络附着成本；
- 384 MB SRAM 的真实 KV/状态命中率，以及减少了多少 ICI/HBM 流量；
- 最终 AgentX 的 TTFT、TPOT、p99 与 cost per qualified task。

---

## IX. PD 分离与 KV 分层：状态离开 HBM 以后

> **先说人话：** 输入理解和逐 Token 生成对硬件的需求不同，所以可以放进两组 TPU。代价是中间产生的 KV Cache 必须可靠、快速地搬过去；HBM 放不下时，还要继续下沉到 DRAM 和 NVMe。

### 9.1 为什么 Prefill 与 Decode 应分开

Prefill 对整个输入序列做并行计算，通常 compute-heavy；decode 每一步只生成少量 token，需要重复读取 weight 和历史 KV，通常 memory/state-heavy。

$$
AI_{prefill}\gg AI_{decode}
$$

若两者共享同一 pool：

- Prefill burst 会干扰 decode latency；
- 一套硬件配置难以同时匹配两种 arithmetic intensity；
- Batch 与调度目标冲突；
- Capacity planning 被峰值耦合。

PD disaggregation 将其拆为独立 pool：

![Prefill-Decode 分离](/assets/tpu-inference-hardware-software-codesign/figure-35.png)

> **图 17：Prefill 生成初始 KV，再传给独立 Decode Pool。** 来源：DistServe。

拆分收益成立的条件是：

$$
T_{transfer}+T_{queue,new}
<
T_{interference,avoided}+T_{specialization,gain}
$$

如果 KV transfer 太慢、p99 抖动太大或跨 pool 排队增加，分离可能适得其反。

### 9.2 TPU-Sync 做了什么

TPU-Sync，原名 TPU-raiden，是 Google 外部化的 disaggregated KV transfer library：

- 支持 JAX 与 TorchTPU；
- 提取原生 `PJRTBuffer` hardware descriptor；
- 支持 zero-copy transfer；
- 支持 TPU KV cache 到 host DRAM 的 offload。

![TPU-Sync](/assets/tpu-inference-hardware-software-codesign/figure-36.jpg)

> **图 18：TPU-Sync 把 KV buffer ownership 和传输 primitive 暴露给外部 serving 栈。** 来源：[GitHub](https://github.com/google/tpu-sync)。

“Zero-copy”不等于“零成本”。仍需测量：

- Descriptor/control-plane 开销；
- DMA setup；
- Source/destination synchronization；
- Fabric bandwidth 与 contention；
- Buffer pinning 和内存回收；
- 故障时 ownership/fencing。

### 9.3 DRAM 与 NVMe Pool 如何形成状态平面

当 HBM 无法容纳所有 active KV 时，host DRAM 是下一容量层。Mooncake Store 的通用架构可以把多台 host 的 DRAM 聚合为逻辑 P2P pool，并可聚合 NVMe 或接入 WEKA/VAST。但截至 2026 年 9 月 9 日，Mooncake 已合入的是 TENT 中经 Host DRAM 中转的实验性 TPU/PJRT staging；生产 PJRT Adapter、JAX/PyTorch-XLA serving integration 与 TPU 专用 Store 语义尚未闭环。详细证据见[Mooncake Store 对 TPU 支持到哪一步](mooncake-tpu-support-status.md)。

![Mooncake Store TPU 支持](/assets/tpu-inference-hardware-software-codesign/figure-37.png)

> **图 19：Mooncake Store 与 TPU 支持计划。** 来源：[GitHub](https://github.com/kvcache-ai/Mooncake/issues/2662)。

![Mooncake DRAM P2P Pool](/assets/tpu-inference-hardware-software-codesign/figure-38.jpg)

> **图 20：跨 Host DRAM 形成统一 KV logical pool。** 来源：Mooncake。

状态层级可以写成：

```text
On-chip SRAM: 最低延迟、最小容量
  -> HBM: Active KV 与 weight
  -> Local host DRAM: Warm KV
  -> Remote DRAM P2P pool: Shared warm capacity
  -> Distributed NVMe: Cold KV / large prefix
  -> Persistent backend: Recovery / long-lived state
```

每向下一层移动，都要比较：

$$
V_{tier}
=P_{hit}\times T_{recompute\ avoided}
-
(T_{lookup}+T_{transfer}+T_{queue}+T_{evict})
$$

仅有容量而没有 locality policy，可能把 HBM capacity wall 变成 fabric tail-latency wall。

---

## X. Agentic AI：为什么优化目标必须改变

> **先说人话：** 普通 benchmark 常测一次问答；Agent 会反复思考、调用工具、追加历史并启动子 Agent。此时真正重要的是整项任务多久完成、历史命中多少、状态搬了多少，而不是某一步生成了多少 Token。

### 10.1 随机 8k1k 不能代表 Agentic Workload

原文概括的 Agentic 特征包括：

- 数十到数百轮的 multi-turn session；
- System prompt、tool schema 和历史结果导致 long context；
- 第 $n-1$ 轮被拼入第 $n$ 轮，prefix reuse 随轮次增加；
- 多个短生命周期 sub-agent 形成突发 state allocation；
- Tool call 令 accelerator 计算与外部等待交错。

随机输入 benchmark 刻意消除了 prefix sharing，因此无法度量 prefix cache、KV pooling 和 warm state 的价值。

![AgentX 工作负载](/assets/tpu-inference-hardware-software-codesign/figure-39.jpg)

> **图 21：AgentX 将评测从单轮 token serving 推向多轮状态复用。** 来源：DeepSeek、SemiAnalysis。

### 10.2 端到端时间不是单个 Token 时间

$$
T_{agent\ task}
=
\sum_{turn=1}^{N}
(T_{prefill}+T_{decode}+T_{KV}+T_{tool}+T_{queue})
+T_{retry/recovery}
$$

多轮场景中，某个每层 80 µs 的优化会重复很多次；同样，一个 KV miss 或 remote-tier p99 也会跨轮累积。系统应同时报告：

- TTFT、TPOT 和 E2E latency；
- Prefix/KV hit rate；
- 每轮新增与复用 token；
- KV bytes moved per generated token；
- Active sessions per chip/rack at fixed SLO；
- Cost per qualified agent task；
- Failure/retry amplification。

### 10.3 TPUv8i 的 SRAM 和 Boardfly 为什么对 Agentic 有意义

384 MB on-chip SRAM 可能容纳更多热点 KV 或 state，Boardfly 更少 hop 可能降低 remote expert/KV communication。但二者只有在以下条件同时成立时才转换为业务收益：

1. Runtime 能识别高复用 prefix 与 hot state；
2. Placement 避免频繁 migration；
3. Cache policy 在 sub-agent burst 下不抖动；
4. Collective 与 KV transfer 的 p99 可控；
5. 软件能把 SRAM 用于正确数据，而不是被静态 buffer 占满。

因此“更大 SRAM + 更低网络直径”只是能力，不是 AgentX 性能结论。

---

## XI. 架构决策表与验证路径

> **先说人话：** 本章把软件名词翻译成硬件问题。评审任何优化时，只问四件事：减少了什么工作，增加了什么容量或通信，收益出现在哪个负载，失败边界在哪里。

### 11.1 把第三章优化映射回硬件问题

| 优化 | 原始硬件问题 | 主要收益机制 | 新代价/边界 |
|---|---|---|---|
| DP Attention + EP | KV heads 少于 TP ranks | KV 按请求局部化，expert weight 保持分布 | 低并发 load imbalance、attention/MoE collective |
| 合并 metadata AllGather | 小 collective 固定延迟 | 少一次 launch/sync | Payload 融合与解析复杂度 |
| SparseCore routing | MXU 被不规则搬运占用 | 异构并行、释放 MXU | Offload threshold、VMEM、同步 |
| Hierarchical ReduceScatter | 芯片内外链路成本不同 | 先 die-to-die 归约，再 ICI | Pipeline buffer 与 topology mapping |
| GDN 代数重排 | VPU→MXU 串行依赖 | 改 dependency graph | 数值等价与调度复杂度 |
| GDN fusion | 中间结果往返 HBM | 片上复用、少 launch | Kernel 变大、编译与维护成本 |
| Compact recurrent state | 固定 state 按 KV block 过分配 | 回收 HBM | 与 prefix checkpoint pool 冲突 |
| BF16 state storage | HBM 容量/带宽 | 存储减半、VMEM FP32 计算 | 数值稳定性验证 |
| Sequence-on-lane | 128-lane padding | KV page 翻倍 | 低并发 latency +3% |
| Fetch/compute block 解耦 | VMEM 无 prefetch 空间 | 恢复双缓冲深度 | 双参数 tuning |
| Hybrid prefix cache | Live state 覆盖 checkpoint | 保留可复用 GDN state | 更多 HBM、alignment |
| PD 分离 | Prefill/decode 资源目标冲突 | 独立扩缩与调优 | KV transfer 与跨池 queue |
| DRAM/NVMe KV pool | HBM capacity wall | 增加 warm/cold state 容量 | Fabric tail、lookup、eviction、failure |

### 11.2 对硬件架构的直接启示

**计算阵列：** 峰值宽度必须与目标模型 shape 分布共同设计。更大的 systolic array 会提高规则矩阵峰值，也会提高小维度 padding 税。

**异构执行单元：** Sparse/data-movement engine 的价值由并发能力、memory access path 和软件 crossover policy 决定，不是 TOPS 指标。

**片上存储：** VMEM/SRAM 不只保存 operand，也决定 double buffering、kernel fusion、KV hotset 和 metadata。容量分配需要可观测、可动态调整。

**Die-to-die：** 多 die 设计应暴露 topology 与 locality，让 collective 先局部后全局。把多个 die 抽象成完全均匀设备会浪费物理层级。

**Scale-up fabric：** 训练偏向大消息带宽，MoE decode/Agentic 更关注小消息、hop、同步和 p99。Boardfly 表明训练与推理网络可能走向分化。

**Host memory/fabric：** KV offload 使 host DRAM、NIC/DPU、DMA、IOMMU、buffer ownership 与 failure fencing 进入推理关键路径。Host 不再只是 boot/control plane。

### 11.3 最小验证矩阵

| 层级 | 必测项 | 通过标准 |
|---|---|---|
| MXU | Shape sweep：64/128/192/256，prefill/decode | Delivered utilization 与 padding bytes 可解释 |
| SparseCore | Message/token count sweep，on/off 对照 | 找到稳定 crossover，不损害低负载 |
| VMEM | Buffer occupancy、spill、双缓冲深度 | 无隐性回退，容量与 pipeline 模型吻合 |
| HBM | Weight/KV/state 分项 bytes/token | Sustained bandwidth 与瓶颈归因闭合 |
| Die-to-die/ICI | Small/medium message p50/p99、hop sweep | 分层 collective 优于 flat baseline |
| Serving | 1k1k、8k1k、1k8k；并发 4–512 | Pareto 曲线而非单点提升 |
| Prefix cache | 真实多轮 trace、reuse ratio sweep | 节省 recompute 大于 checkpoint/lookup 成本 |
| PD | Aggregated 与 disaggregated 同 SLO 对照 | Specialization 收益覆盖 transfer/queue |
| KV tier | HBM/DRAM/remote DRAM/NVMe | 每层 hit rate、bytes moved、p99 和故障恢复可接受 |
| AgentX | Multi-turn + tool + sub-agent burst | Cost/qualified task 与 SLO 同时改善 |

### 11.4 三个 No-Go 条件

1. **模型 Shape 不可摊销。** 关键模型长期只有 25%–50% MXU utilization，专用 kernel 数量随模型线性增长。
2. **状态外部化制造尾延迟。** PD/KV pooling 节省 HBM，但 transfer、queue 和 recovery 使 p99 或任务完成率恶化。
3. **异构卸载缺少动态策略。** SparseCore/collective offload 在不同并发与 message size 下频繁跨越收益点，运行时无法稳定选择。

> 💡 **芯一视角：** 第三章真正展示的不是 TPU 有多少优化，而是峰值算力离产品性能有多远。每个百分点都在补一条供给链：shape 要喂饱 MXU，SparseCore 要清走不规则工作，VMEM 要容纳流水，ICI 要隐藏同步，KV 层级要避免排队。硬件规格表通常从这里开始沉默，系统性能恰好从这里开始说话。

## 配图说明

本文使用 21 张技术图片，分别覆盖：

- Ironwood logical device、SparseCore 与 MXU；
- DP Attention、MoE routing 和 hierarchical ReduceScatter；
- GDN dependency rewrite 与 kernel fusion；
- Hybrid state、sequence-on-lane、prefix cache 和 RPA pipeline；
- 3D torus、OCS 与 Boardfly；
- PD 分离、TPU-Sync、Mooncake KV pool 与 AgentX。

图片保存在 `assets/tpu-inference-hardware-software-codesign/`，来源和链接已在各图下注明。

## 结语

Qwen3.5-397B 在 TPU 上的优化可以压缩成一句话：**让规则矩阵计算持续留在 MXU，把不规则工作转移到 SparseCore/VPU，把可重叠的数据搬运放进流水，把不断增长的状态放到合适的容量层。** DP Attention、MoE routing、GDN fusion 和 Paged Attention 只是这条原则在不同数据结构上的具体实现。

Ironwood 的双 logical device、SparseCore、256×256 MXU、HBM 和 ICI 为这些优化提供了物理基础，也引入 shape、VMEM、collective 和 locality 约束。TPUv8i 用 Boardfly 与更大 SRAM 进一步针对推理，但能否兑现收益仍取决于 serving runtime 是否掌握数据 ownership、placement 和生命周期。

对 Agentic AI，最终边界已经离开单芯片。多轮 prefix reuse、sub-agent burst、PD transfer 和 KV 分层会把 host memory、fabric、storage 与恢复路径全部拉入关键路径。验证时必须固定质量与 SLO，报告 p99、cache hit、bytes moved 和 cost per qualified task。

**最终判断：第三章的所有优化都可以被归结为数据放置、物理形状、执行重叠和状态生命周期四类架构决策。只要沿这四条线检查，软件 PR 就不再是黑盒；反过来，任何无法映射到这四条线的性能宣称，也应要求更完整的测量边界。**

---

## 参考文献

1. [SemiAnalysis, TPU Inference Externalization Full Steam Ahead - InferenceX](https://newsletter.semianalysis.com/p/tpu-inferencex-full-steam)
2. [Google, Inside the Ironwood TPU co-designed AI stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack)
3. [vLLM TPU Inference Repository](https://github.com/vllm-project/tpu-inference)
4. [Google TPU-Sync](https://github.com/google/tpu-sync)
5. [Mooncake Store](https://github.com/kvcache-ai/Mooncake)
6. [Mooncake Store 对 TPU 支持到哪一步](mooncake-tpu-support-status.md)
7. [The Scaling Book: TPUs](https://jax-ml.github.io/scaling-book/tpus/)
8. [Ragged Paged Attention](https://arxiv.org/abs/2604.15464v1)
