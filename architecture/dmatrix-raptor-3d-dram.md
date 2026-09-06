---
title: "100 TB/s 不是更快的 DRAM：d-Matrix Raptor 如何删除 HBM 的片外内存边界"
description: "从垂直 I/O、bank-to-engine 映射和系统瓶颈，审计 Raptor 3D DRAM 的 100 TB/s 与 0.37 pJ/bit。"
date: 2026-09-05
updated: 2026-09-05
slug: dmatrix-raptor-3d-dram
status: published
github: true
public: true
wechat: draft
wechat_url:
cover: /assets/dmatrix-raptor-3d-dram/2026-08-24_2-56-50-728x355.jpg
series: "Y26W36"
content_type: analysis
---

# 100 TB/s 不是更快的 DRAM：d-Matrix Raptor 如何删除 HBM 的片外内存边界[Y26W36][解析]

***先给答案——Raptor 的价值不在于把 DRAM cell 做得更快，而在于把计算逻辑放到 DRAM 上方，缩短 memory-to-compute 路径。它可能让低 batch 的 MoE decode 更容易获得高吞吐，但 100 TB/s 仍只是局部器件带宽，不能直接等同于模型级 token/s。本文要回答三个问题：这组数字是否自洽，哪些 workload 真能受益，以及系统扩展后瓶颈会迁移到哪里。***

## 原始材料

**[d-Matrix's Raptor 3D DRAM Achieves SRAM-Class Bandwidth at 1/10th the HBM Power](https://wccftech.com/d-matrix-raptor-3d-dram-achieves-sram-class-bandwidth-at-1-10th-the-hbm-power/amp/)**

本文基于 Wccftech 对 d-Matrix 演示材料的转述进行架构分析。公开数字主要来自厂商披露，不等同于第三方 benchmark；缺少的产品组织、系统拓扑和 workload 配置均按“未披露”处理。

本文不复述产品材料，而是审计三条链能否同时闭合：**100 TB/s 如何从物理组织产生，0.37 pJ/bit 的测量边界是什么，以及单器件优势如何转换为模型级 token/s。**

> 💡 **芯一视角：** Raptor 的本质不是提高 DRAM cell 的速度，而是删除 package-scale PHY 与全局数据搬运，用超宽垂直 I/O、bank-level parallelism 和近存计算换取带宽。100 TB/s 是局部数据流的胜利；它能否变成模型级 token/s，取决于容量拼接、激活网络、KV 流量和流水线延迟。宣传页可以删除 PHY，系统图不能删除边界。

## 核心结论

- Raptor 是 **logic-on-DRAM 的计算存储融合器件**，不是传统意义上的 HBM 后继品。
- “删除 PHY”不是没有 I/O，而是删除面向封装级距离的高速 SerDes PHY、均衡、训练与 beachfront routing，改用超宽、低速、短距离的垂直接口。
- 100 TB/s 来自数百个 DRAM bank 与 256 个 tensor engine 的局部并行，不来自提高单 pin 速率。
- 0.37 pJ/bit 乘以 100 TB/s，对应约 296 W I/O 功耗。这个数字与报道中的约 300 W 自洽，但不包含完整 tensor compute、片间 fabric 和供电散热。
- 对比 2.4 pJ/bit 的 HBM I/O，Raptor 是约 6.5 倍，不是严格的 10 倍；13.5 倍需要把 HBM 后续片上搬运按约 5 pJ/bit 一并计入。
- 32 GB 容量处在 SRAM 与 HBM 之间。它能够容纳约 64B INT4 参数，却无法单独容纳 3T 模型。
- 该结构最适合低 batch、memory-bound 的 MoE decode，以及能够把计算固定在权重所在地的流水化系统。
- “1M context、3T-class model、约 1,000 TPS/user”不能由 100 TB/s 单点规格直接推出，必须披露模型稀疏度、KV 结构、卡数和 TPS 口径。

## 先建立一张阅读地图

这篇文章不是在复述产品发布材料，而是在做一条从器件到系统的审计链。建议先读每章开头的粗体判断，再回看公式和数字。

| 你想知道什么 | 对应章节 | 先记住的判断 |
|---|---|---|
| Raptor 改变了什么 | 1. 架构重构 | 它删除的是片外数据路径，不是 DRAM 本身 |
| 100 TB/s 是否算得通 | 2. 数字反推 | 峰值带宽可以闭合，但利用率依赖映射、聚合和调度 |
| 为何没有“免费带宽” | 3. 物理实现 | 热、刷新、ECC、良率和容量密度都会付费 |
| 哪类模型真正受益 | 4. Workload 审计 | 低 batch、memory-bound 的 MoE decode 最匹配 |
| 规模扩大后会怎样 | 5. 系统 Deep Dive | 瓶颈从权重读取转向 activation、queue、state 和故障恢复 |
| 如何验证结论 | 6. 验证路径 | 需要 sustained bandwidth、端到端能耗和模型级延迟数据 |

### 读者先掌握四个词

- **Peak bandwidth**：器件在理想并行访问下的峰值，不等于模型实际得到的带宽。
- **Sustained bandwidth**：在访问模式、调度和冲突约束下可持续获得的带宽。
- **Active parameters**：某个 token 实际需要读取的参数，而不是模型的总参数量。
- **KV traffic**：生成 token 时读取历史上下文状态产生的数据流量；长 context 下，它可能超过权重流量。

后文的公式都服务于一个判断：**不要从单点规格直接跳到端到端性能。**

### 公式怎么读

本文的公式主要有三种用途：

1. **单位换算**：例如把 TB/s 换成 bit/s，检查公开数字是否算术自洽。
2. **上限或下限估计**：例如用带宽除以每 token 的数据量，得到理想吞吐上限；这不是实测结果。
3. **结构关系表达**：例如把 token 延迟拆成权重、KV、计算、互连和同步五部分，用来定位瓶颈。

读公式时先看三件事：**每个符号代表什么、单位能否相消、这个式子描述的是峰值还是实际系统。** 其中 `≈` 表示量级近似，`≤` 表示上限，`≥` 表示下限；它们都不是产品实测承诺。

---

## 1. 架构重构：删除片外内存边界

**本章回答：Raptor 与 HBM 的根本差异是什么？**

### 1.1 Raptor 到底是什么

报道披露的 Raptor 由一层 N4 logic die 和一层定制 DRAM 通过 36 µm pitch 的 face-to-face 连接组成。公开规格如下。

| 参数 | 披露值 | 架构含义 |
|---|---:|---|
| 容量 | 32 GB | 高于纯 SRAM，低于多堆 HBM |
| 峰值带宽 | 100–100+ TB/s | 依赖大规模 bank 并行与垂直 I/O |
| 垂直 I/O 能效 | 0.37 pJ/bit，称为 measured | 不是整卡能效 |
| Logic 工艺 | TSMC N4 | 顶层 tensor compute 与控制逻辑 |
| 3D 连接 | Face-to-face，36 µm pitch | 缩短 memory-to-compute 距离 |
| Tensor engine | 256/chiplet | 与 DRAM bank 做局部映射 |
| DRAM bank | 840 | 无法直接按每 engine 四 bank 对称分配 |
| DRAM 输出 | 32 B/bank/column | 三个 bank 一次提供 96 B |
| 内部 flit | 128 B | 需要聚合和重排 |
| Logic 功率密度 | 约 0.5 W/mm² | 不代表总功耗只有数百瓦以内 |
| 工作结温 | 最高约 105°C | 推动 refresh 间隔缩短到 4 ms |
| Refresh 带宽损失 | 1.37% | 平均吞吐口径，未必代表尾延迟 |
| ECC | Reed–Solomon $T=2$ + CRC | 每 128 B 纠正两个 symbol error |
| Spare bank | 约 8–9% | 用 MUX chain 替换缺陷 bank |
| Microbank | 1,366 rows，5.33 MB | 支持细粒度刷新、修复与并行访问 |

从产品分类看，Raptor 更接近一个带有大容量本地 DRAM 的 near-memory accelerator：

```text
传统 GPU/HBM

HBM stack -> HBM PHY -> package/interposer -> GPU PHY
          -> memory controller -> global NoC -> cache -> tensor core

Raptor

local DRAM bank -> short vertical I/O -> local data path -> tensor engine
```

前者把 HBM 作为独立存储器；后者把 DRAM bank 变成计算阵列的物理组成部分。

#### 原始材料主张什么

文章主张 Raptor 在 SRAM、HBM 之间找到一个新工作点：

- 带宽接近 SRAM accelerator 的数量级；
- 容量明显大于片上 SRAM；
- I/O 能耗明显低于 HBM；
- 通过 3D 集成摆脱 HBM PHY beachfront 限制。

#### 原始材料没有证明什么

当前公开信息没有证明：

- 整个 accelerator 的能耗只有 HBM 系统的十分之一；
- 任意 workload 都能持续获得 83–85 TB/s；
- 32 GB 单元能够低延迟扩展到数十或数百个；
- 1M context 下约 1,000 TPS/user 是单请求串行 decode 速度；
- 3D bonding 的成本与量产良率已经优于 HBM。

### 1.2 HBM 的带宽墙在哪里

HBM 的问题不是 DRAM cell array 缺少内部并行度。真正的瓶颈是这些并行度必须通过有限的 die edge、PHY、interposer routing 和 memory controller 暴露给计算 die。

#### Beachfront 限制

HBM PHY 需要占据 accelerator die 边缘。每增加一个 HBM stack，都要同时增加：

- PHY macro；
- die edge 长度；
- interposer 信号走线；
- memory controller；
- 电源、时钟与信号完整性预算。

这和海边地产类似：HBM stack 可以继续增加，但 accelerator die 的“海岸线”不会跟着无限增长。技术结论是，带宽扩展逐渐被 PHY 面积和封装 perimeter 限制，而不是只被 DRAM 容量限制。

#### 高速接口的能耗

HBM 相对 DDR 已经采用宽接口和较低 pin speed，但相对 face-to-face 3D 连接，它仍要驱动更长、更重的电气路径。

每 bit 动态能量可粗略写成：

$$
E_{bit}
\approx
C_{path}V^2
+E_{clock}
+E_{PHY}
$$

其中：

- $E_{bit}$ 是传输 1 bit 数据的能量，单位通常是 pJ/bit；
- $C_{path}$ 是整条电气路径看到的等效电容，包含 bump、interposer、package trace 和 receiver input；
- $V$ 是摆幅电压；$C_{path}V^2$ 表示驱动这条路径的主要动态能耗项；
- $E_{clock}$ 是时钟分配的摊销能耗；$E_{PHY}$ 是 SerDes、均衡和接收端等接口电路的摊销能耗。

这个式子不是 Raptor 的精确功耗模型，而是在解释为什么缩短路径通常有利：路径越长、负载越大，$C_{path}$ 越大，驱动与时钟能耗越高。

#### 数据通过 HBM PHY 后还没到计算单元

HBM 报告的接口能效通常不包含数据进入 accelerator 后的全部搬运。一个 weight tile 可能继续经过：

```text
HBM PHY
  -> memory controller
  -> global NoC
  -> L2/cache
  -> local SRAM
  -> tensor engine operand buffer
```

因此，同样是“pJ/bit”，至少有两个边界：

1. **Memory I/O energy**：只计算 HBM link；
2. **Compute-visible data delivery energy**：计算从 DRAM 到 tensor engine 的完整路径。

Raptor 的优势主要来自第二个边界：不仅垂直 I/O 更短，而且 tensor engine 被放到 bank 附近，减少后续全局搬运。

### 1.3 删除 PHY 到底删除了什么

“No PHY”是一句传播效果很好的标题，但工程上并不精确。Raptor 仍然需要 I/O driver、receiver、clocking、timing control 和错误检测。

它删除的是面向 package-scale 距离的高速 PHY，包括其中相当一部分：

- serializer/deserializer；
- 高摆幅 transmitter；
- 高速 receiver；
- training 与 deskew；
- clock recovery 或复杂时钟分配；
- equalization；
- DBI 与高速链路编码；
- 受 die edge 限制的 PHY macro。

替代方案是大量短距离、低速率的垂直 wire：

$$
B_{total}
=
N_{wire}\times r_{wire}
$$

其中 $B_{total}$ 是总带宽，单位为 bit/s；$N_{wire}$ 是并行数据线数量；$r_{wire}$ 是每根线的有效传输速率，单位为 bit/s。两者相乘得到所有数据线每秒搬运的 bit 数。

HBM 依赖有限数量的 wire 运行在较高 $r_{wire}$；Raptor 通过 3D 集成大幅提高 $N_{wire}$，允许每根 wire 以更低速率工作。这个公式表达的是**带宽的来源**，没有包含协议开销、刷新、bank 冲突或调度空泡，所以它更接近峰值而不是 sustained bandwidth。

这会同时改善三个指标：

- 单 bit I/O 能耗下降；
- 单位 footprint 的 I/O 密度上升；
- memory channel 可以更细粒度地绑定本地计算单元。

需要强调：36 µm pitch 仍不是“无限互连密度”。它决定可放置的垂直连接数量、供电连接数量以及信号/电源比例。Raptor 的 100 TB/s 必须在这套 bump budget 中同时容纳数据、地址、控制、时钟、电源、地与冗余连接。

> 💡 **芯一视角：** Raptor 改变的是数据路径的物理尺度和职责边界。HBM 把高并行 DRAM 收敛到有限 PHY 后再送入全局计算阵列；Raptor 把计算摊到 bank 上方，让内部并行度就地消费。收益来自减少长距离电气路径和全局搬运，代价是计算、容量和调度从此被绑定在同一物理组织中。

---

## 2. 数字反推：100 TB/s 与 0.37 pJ/bit

**本章回答：公开数字能否在带宽、接口宽度和功耗之间互相闭合？**

### 2.1 100 TB/s 如何产生

100 TB/s 等于：

$$
100\times10^{12}\ \mathrm{B/s}
\times8
=
8\times10^{14}\ \mathrm{bit/s}
$$

这里的 `B/s` 是字节每秒，`bit/s` 是比特每秒；因为 1 byte = 8 bit，所以要乘以 8。这个换算只是在统一单位，尚未说明这些 bit 能否被计算单元持续消费。

如果内部一次消费 128 B flit，每秒需要处理：

$$
\frac{100\times10^{12}}{128}
\approx
7.81\times10^{11}\ \mathrm{flit/s}
$$

若 256 个 tensor engine 均匀分担，每个 engine 平均对应：

$$
\frac{100\ \mathrm{TB/s}}{256}
\approx
390.6\ \mathrm{GB/s}
$$

显然，这不可能由一条中央总线提供。合理组织只能是高度分布式的数据路径：

```text
Bank group 0  -> local channel 0  -> Tensor Engine 0
Bank group 1  -> local channel 1  -> Tensor Engine 1
...
Bank group N  -> local channel N  -> Tensor Engine N
```

因此，100 TB/s 是以下三个条件的乘积：

$$
B_{effective}
=
B_{bank-parallel}
\times U_{mapping}
\times U_{schedule}
$$

其中：

- $B_{bank-parallel}$：全部 bank 同时服务时的物理带宽；
- $U_{mapping}$：模型布局与 bank/channel 映射效率；
- $U_{schedule}$：运行时请求能否持续填满这些 channel。

两个 $U$ 都是 0 到 1 之间的利用率系数。它们把峰值带宽折算成更接近系统可用的带宽：映射不均匀会让一部分 bank 空闲，调度不连续会让 channel 出现空泡。若两个利用率分别为 0.9 和 0.8，100 TB/s 峰值只对应约 $100\times0.9\times0.8=72$ TB/s 的这一阶估计。

报道使用 83–85% effective bandwidth utilization。若峰值为 100 TB/s，则持续带宽约为：

$$
B_{sustained}
\approx83\text{–}85\ \mathrm{TB/s}
$$

这个利用率对规则权重流可能成立，但不能自动外推到 KV 随机访问、embedding lookup 或 expert hotspot。

### 2.2 840 个 Bank 如何喂给 256 个 Tensor Engine

Raptor 的 bank 数量和 tensor engine 数量并不天然对齐：

$$
840\neq256\times4=1024
$$

文章披露每个 DRAM bank 每个 column access 提供 32 B。三个 bank 组成一个 channel 时，一次只有：

$$
3\times32\ \mathrm{B}=96\ \mathrm{B}
$$

而 tensor engine 或其内部接口需要 128 B flit：

$$
96\ \mathrm{B}\neq128\ \mathrm{B}
$$

如果简单 overfetch 到 128 B，利用率只有：

$$
U=\frac{96}{128}=75\%
$$

更合理的办法是跨多个 DRAM transaction 聚合和重排。四次 96 B 正好等于三个 128 B flit：

$$
4\times96\ \mathrm{B}
=384\ \mathrm{B}
=3\times128\ \mathrm{B}
$$

这可以避免固定 25% 浪费，但需要：

- aggregation buffer；
- flit boundary 重排；
- channel steering；
- sequence tracking；
- ECC codeword 对齐；
- bank 不同时 ready 时的 backpressure。

这也是为什么“pitch-matched”不等于一对一硬绑定。物理上靠近只解决 wire distance，逻辑上仍需处理 bank 数、访问粒度和 engine 消费粒度不匹配。

### 2.3 0.37 pJ/bit 与“十分之一功耗”的口径

功耗由带宽和每 bit 能量相乘得到：

$$
P=B_{bit}\times E_{bit}
$$

其中 $P$ 是功率，单位为 W；$B_{bit}$ 是每秒传输的 bit 数，单位为 bit/s；$E_{bit}$ 是每 bit 的能量，单位为 J/bit。两者相乘后 bit 会相消，得到 J/s，也就是 W。

Raptor 在 100 TB/s 下的 I/O 功耗为：

$$
P_{Raptor,I/O}
=
8\times10^{14}
\times0.37\times10^{-12}
\approx296\ \mathrm{W}
$$

这与文章中的约 300 W 一致。

若用 2.4 pJ/bit 的 HBM I/O 提供相同 100 TB/s：

$$
P_{HBM,I/O}
=
8\times10^{14}
\times2.4\times10^{-12}
=1,920\ \mathrm{W}
$$

因此纯接口能效差距是：

$$
\frac{2.4}{0.37}
\approx6.5\times
$$

如果把 HBM 到 tensor engine 的后续搬运也计入，并采用文章给出的约 5 pJ/bit：

$$
P_{HBM,path}
=
8\times10^{14}
\times5\times10^{-12}
=4,000\ \mathrm{W}
$$

此时差距为：

$$
\frac{5}{0.37}
\approx13.5\times
$$

所以标题中的“十分之一 HBM 功耗”是一个区间化表达：

| 边界 | 相对能效 |
|---|---:|
| Raptor vertical I/O 对 HBM I/O | 约 6.5× |
| Raptor vertical I/O 对 HBM 完整数据路径 | 约 13.5× |
| 标题传播口径 | 约 10× |

0.37 pJ/bit 并不包含完整系统功耗。至少还要加入：

$$
P_{total}
=
P_{DRAM-array}
+P_{refresh}
+P_{vertical-I/O}
+P_{tensor}
+P_{local-NoC}
+P_{chiplet-fabric}
+P_{VR-loss}
$$

如果不统一功耗边界，“13.5×”只能说明数据搬运方向正确，不能直接说明每 token 能效提高 13.5 倍。

> 💡 **芯一视角：** 三组数字能够说明架构方向，却不能单独证明产品性能。100 TB/s 需要规则并行访问维持高利用率；0.37 pJ/bit 只覆盖垂直 I/O；约 300 W 是接口算术，不是整机功耗。判断成立的最小证据，是相同访问模式下同时披露 sustained bandwidth、端到端数据路径能耗与 tensor utilization。

---

## 3. 物理实现：带宽不是免费的

**本章回答：为了获得局部高带宽，Raptor 付出了哪些热、可靠性和制造代价？**

### 3.1 为什么 Logic 必须放在 DRAM 上方

3D 堆叠有两种基本方向。

```text
方案 A：DRAM-on-logic       方案 B：logic-on-DRAM

Cold plate                 Cold plate
DRAM                       Logic
DRAM                       DRAM
Logic                      Package
Package
```

在 DRAM-on-logic 中，计算逻辑产生的热量需要穿过温度敏感的 DRAM 才能到达 cold plate。随着温度上升，DRAM cell 漏电增加、retention time 缩短、refresh 频率上升。

Raptor 选择 logic-on-top：

```text
Cold plate
    ↓
N4 logic die
    ↓ face-to-face links
custom DRAM die
    ↓
package/substrate
```

这样做的优势是：

- 高功耗 logic 最接近 cold plate；
- 热量不必先穿过 DRAM；
- 顶层逻辑可以承受更高局部热通量；
- DRAM 更容易维持在设计温度以内。

但问题没有消失，只是改变了形态：

- 顶层 logic 的电源如何低阻抗送达；
- 垂直信号连接和 power/ground 连接如何分配；
- DRAM 仍会受到 logic 的热耦合；
- face-to-face 后的测试、返修和 known-good-die 管理更困难；
- die warpage 与热机械应力需要长期可靠性验证。

文章给出的约 0.5 W/mm² 是 logic power density，不是整个 stack 的总功耗。若逻辑面积达到数百平方毫米，计算部分本身仍可能是数百瓦级。

### 3.2 4 ms Refresh、ECC 与良率设计

#### 为什么 Refresh 加快八倍

Raptor 面向约 105°C junction。DRAM 温度升高后 retention time 下降，因此 refresh interval 从常见的约 32 ms 缩短到 4 ms：

$$
\frac{32\ \mathrm{ms}}{4\ \mathrm{ms}}
=8
$$

文章宣称 refresh 只损失 1.37% 带宽。要做到这一点，必须依靠：

- 小粒度 microbank；
- bank-level refresh；
- staggered refresh scheduling；
- 其他 bank 隐藏当前 bank 的刷新时间；
- 足够深的请求队列。

但平均带宽损失 1.37%，不表示 p99 service time 也只增加 1.37%。如果 tensor engine 与少量 bank 强绑定，refresh 会表现为周期性局部 bubble。

#### 为什么需要小 Microbank

报道给出的 microbank 规格是 1,366 rows、5.33 MB。小 bank 可以：

- 增加并行 bank 数量；
- 缩小一次 refresh 的阻塞范围；
- 缩小缺陷隔离单元；
- 允许 spare bank 替换；
- 减少部分 row activation 的无效数据。

代价是外围电路占比增加，包括 decoder、sense amplifier、row buffer、repair MUX 和控制状态。

公开数字中仍有一个组织层级缺口：

$$
840\times5.33\ \mathrm{MB}
\approx4.48\ \mathrm{GB}
$$

它与 32 GB 总容量不一致。这意味着 840 banks、5.33 MB microbank 和 32 GB 很可能分别位于 chiplet、slice、layer 或 card 的不同层级。没有完整 organization diagram，不能把它们直接相乘重建器件。

#### Reed–Solomon $T=2$ 与 CRC

Raptor 在 logic die 上使用 Reed–Solomon $T=2$，声称每 128 B 可纠正两个 symbol error，并使用 CRC 检测更广泛的错误。

其目标不只是普通随机 bit flip，还可能包括：

- 垂直连接局部失效；
- column 或 bank 相关 burst error；
- bonding defect；
- 高温下的 retention error；
- repair 后的数据路径异常。

这里的 symbol 不一定是一 bit。若使用 $GF(2^m)$，每个 symbol 包含 $m$ bits。公开资料还需要补充 symbol 宽度、parity overhead、decoder latency、scrub 策略以及 detected-but-uncorrectable error 的传播方式。

#### Spare Bank 与堆叠良率

3D stack 的简单良率近似为：

$$
Y_{stack}
\approx
Y_{logic}\times Y_{DRAM}\times Y_{bond}
$$

只要任一 die 或 bonding interface 存在致命缺陷，整个 stack 都可能报废。Raptor 预留约 8–9% spare banks，通过 MUX chain 重映射缺陷区域，目的是把部分 die-level defect 降级为可修复 bank defect。

其代价包括：

- 额外 DRAM 面积；
- repair map；
- MUX 延迟和功耗；
- 修复路径造成的时序偏差；
- spare 用尽后的降级或报废策略。

因此，“高良率、低成本”不能只由小于四层堆叠推出，最终仍需要 wafer yield、bonding yield、spare consumption distribution 和测试成本。

### 3.3 SRAM、HBM 与 3D DRAM 的真实交换

| 属性 | SRAM Accelerator | HBM4 系统 | Raptor 3D DRAM |
|---|---:|---:|---:|
| 报道示例容量 | 约 2–4 GB | 约 192 GB | 32 GB |
| 峰值带宽 | 约 300 TB/s | 约 18–20 TB/s | 约 100 TB/s |
| 接口能效 | 约 0.1 pJ/bit | 约 2.4–2.5 pJ/bit | 0.3–0.37 pJ/bit |
| 容量密度 | 最低 | 最高 | 报道称约为 HBM4 的一半 |
| 主要约束 | 面积、漏电、成本 | PHY、beachfront、功耗 | 热、良率、容量、系统拼接 |

“SRAM-class bandwidth”可以成立，因为 100 TB/s 已进入数百 TB/s 的数量级；“SRAM-class latency”则没有证据。

Raptor 仍保留 DRAM 的基本行为：

- activate/precharge；
- row-buffer hit/miss；
- bank conflict；
- refresh；
- retention；
- ECC decode；
- aggregation latency。

准确表述应是：

> 💡 **芯一视角：** Raptor 接近 SRAM 的 aggregate bandwidth 与接口能效，但保留 DRAM 的访问时序、刷新和可靠性约束。它不是同时获得 SRAM、HBM 和 DRAM 的全部优点，而是用 3D 封装、热设计、冗余和软件约束购买一个新的 bandwidth-capacity 工作点。

---

## 4. Workload 审计：器件带宽如何变成 Token

**本章回答：哪些 workload 能把器件带宽转化为模型级收益？**

### 4.1 为什么它适合 LLM Decode

#### Decode 首先是权重整读问题

在低 batch decode 中，每生成一个 token，通常需要读取一次活跃权重。理论 aggregate throughput 上限为：

$$
TPS_{aggregate}
\le
\frac{B_{effective}}
{W_{active/token}}
$$

这里 $TPS_{aggregate}$ 是所有并发请求合计的 token/s；$B_{effective}$ 是实际可用带宽，单位为 bytes/s；$W_{active/token}$ 是生成一个 token 需要读取的活跃权重，单位为 bytes/token。用“每秒能搬运的字节数”除以“每个 token 要搬运的字节数”，单位正好得到 token/s。

这个式子假设权重读取是唯一瓶颈，而且每个字节只读取一次；真实系统还会受到计算、KV、互连、同步和利用率影响，因此它只能作为理想上限。

取有效带宽 85 TB/s：

| 每 token 活跃权重 | 带宽上限 |
|---:|---:|
| 10 GB | 8,500 tok/s |
| 20 GB | 4,250 tok/s |
| 40 GB | 2,125 tok/s |
| 80 GB | 1,062 tok/s |
| 160 GB | 531 tok/s |

这解释了 Raptor 对 MoE decode 的吸引力。一个模型可以拥有数万亿总参数，但每个 token 只激活少量 experts。若活跃权重约 80 GB，85 TB/s 恰好对应约 1,000 tok/s 的带宽上限。

#### Dense 模型不会得到同样结果

3T 参数以 INT4 存储，完整权重约为：

$$
3\times10^{12}
\times0.5\ \mathrm{B}
=1.5\ \mathrm{TB}
$$

若每个 token 都读取全部权重：

$$
TPS_{dense,upper}
\approx
\frac{85\ \mathrm{TB/s}}
{1.5\ \mathrm{TB}}
=56.7\ \mathrm{tok/s}
$$

所以“3T-class、1,000 TPS”几乎必然依赖稀疏激活、多单元并行或特殊 TPS 口径。

#### 它改变了 Batch 的经济性

GPU decode 常通过 batch 让多个 token 分摊同一次权重读取：

```text
权重读取昂贵
-> 必须提高 batch
-> 提高 aggregate throughput
-> 单用户等待 batch 发车
```

Raptor 提高带宽/容量比后，单 token 独自支付权重读取的成本显著下降：

```text
权重读取便宜
-> 小 batch 也可接受
-> 降低排队与发车间隔
-> 改善单用户 TPOT
```

这可能比峰值 aggregate TPS 更重要。推理产品真正稀缺的不是“机房总 token”，而是在成本约束下可交付的低延迟 token。

### 4.2 3T 模型与 1M Context 性能审计

报道中的演示图声称，Raptor 在 1M context 下可为 3T-class model 提供约 1,000 TPS/user，并给出 GLM 5.2 最高约 3,153 TPS/user、Kimi K3 最高约 988 TPS/user 的图示。

这组数字缺少以下配置：

- 模型总参数与 active parameters；
- 权重精度与 KV 精度；
- Raptor 单元数量；
- pipeline、tensor、expert parallel 布局；
- attention 类型；
- speculative decoding 的 draft/acceptance 配置；
- TPS 是单流、aggregate 还是 accepted token rate；
- TTFT、TPOT 与并发用户数。

#### 1M Context 的 KV 约束

对标准 attention，单 token 的历史 KV 读取量可写为：

$$
B_{KV/token}
=
2LSH_{KV}db
$$

这个式子估算生成一个 token 时扫描历史 KV 所需读取的字节数。每个符号的含义是：

- $L$：层数；
- $S$：上下文长度；
- $H_{KV}$：KV head 数；
- $d$：head dimension；
- $b$：每元素字节数。

前面的 2 表示每个位置同时有 K 和 V 两份数据。乘法关系也很直观：层数越多、历史 token 越长、KV head 越多、每个 head 的维度越大、精度占用的字节越多，KV 流量就越大。这个式子解释了为什么 1M context 不能只看权重带宽。

当 $S=10^6$ 时，即使使用 GQA，KV 读取也可能压过权重读取。要达到约 1,000 TPS/user，通常还需要至少一种机制：

- MLA 或其他 KV compression；
- 极少 KV heads；
- sliding-window/local attention；
- recurrent/linear attention；
- KV shard 上的近存 attention；
- speculative decoding；
- 多卡并行扫描 KV。

因此端到端 token latency 必须写成：

$$
T_{token}
=
T_{weight}
+T_{KV}
+T_{compute}
+T_{fabric}
+T_{sync}
$$

这里 $T_{token}$ 是生成一个 token 的端到端时间；右侧分别是读取权重、读取历史 KV、执行计算、跨设备/片上互连传输，以及多单元同步所花的时间。这个拆分不是严格的时序模拟，因为其中一些阶段可以重叠；它的用途是防止把某一项优化误说成整体延迟同比例下降。

Raptor 直接优化的是 $T_{weight}$，也可能通过近存计算优化部分 $T_{KV}$，但文章没有提供足够数据证明整个公式都按同样比例下降。

> 💡 **芯一视角：** 对低 batch MoE decode，Raptor 最有价值的不是把 aggregate TPS 再推高一点，而是降低单 token 独占权重带宽的成本。但 3T、1M context 和 1,000 TPS/user 同时出现时，必须先问 active parameters、KV bytes/token 和 TPS 口径；否则三个大数字只是同框，不是因果链。

---

## 5. 系统 Deep Dive：从 32 GB 单元到模型级服务

**本章回答：从一个 32 GB 单元扩展到几十个单元后，新的系统瓶颈是什么？**

### 先把系统想成四层

本章不从公式开始，而是先固定四个对象：

```text
模型容量：权重和 KV 能否放下？
    ↓
数据流：activation、权重和 KV 在哪些单元之间移动？
    ↓
时延：每一层需要几次跨单元往返？
    ↓
可靠性：某个单元故障时，状态能否继续服务？
```

单个 Raptor 解决的主要是“本地权重读取”。扩展到几十个单元后，系统必须同时解决这四层问题。后文的顺序也是容量 → 数据流 → 时延 → 排队 → 故障，公式分别对应其中一层。

**本章先给结论：** 单元数量增加不会让 100 TB/s 简单相加。权重容量会推动横向扩展，activation 和 KV 会制造跨单元流量，流水线会把吞吐与单用户延迟分开，热点和故障则决定系统能否稳定运行。

### 5.1 从单器件扩展到系统后，瓶颈移到哪里

#### 5.1.1 容量下限只是起点

单个 32 GB Raptor 大致只能容纳：

| 权重格式 | 可容纳参数量 |
|---|---:|
| FP16/BF16 | 约 16B |
| INT8/FP8 | 约 32B |
| INT4 | 约 64B |

3T INT4 权重需要的理论最少单元数约为：

$$
N_{Raptor}
\ge
\frac{1.5\ \mathrm{TB}}
{32\ \mathrm{GB}}
\approx47
$$

这个式子只做容量除法：分子 1.5 TB 是 3T 参数按 INT4 存储后的理论权重体积，分母 32 GB 是一个 Raptor 单元的容量，结果 $N_{Raptor}$ 是理论最少单元数。因为这里没有扣除 KV、预留空间、冗余和容量单位换算损失，所以 47 不是部署建议，而是“至少需要多少存储容器”的下界。

这还没有加入 KV Cache、冗余、embedding、runtime workspace 和容量碎片。更接近部署现实的容量约束是：

$$
N_{unit}
\ge
\left\lceil
\frac{
C_{weight}
+C_{KV}
+C_{workspace}
}{
C_{unit}(1-r_{reserve})
}
\right\rceil
$$

其中 $N_{unit}$ 是实际所需单元数，$C_{weight}$、$C_{KV}$ 和 $C_{workspace}$ 分别是权重、KV 和运行时工作区容量，$C_{unit}$ 是单元原始容量；$r_{reserve}$ 是预留比例，包含 spare、坏块隔离、热备、碎片和滚动升级空间。分母中的 $(1-r_{reserve})$ 表示真正可用于业务数据的容量。因此，47 只是“所有 bit 都能放权重”的数学下限，不是可运维系统的卡数。

#### 5.1.2 Fabric 搬运对象从权重变成 Activation

这里的 **fabric** 可以先理解成连接多个 Raptor 单元的片上/片间数据网络；**activation** 是正在网络中流动的中间张量，不是模型权重，也不是 KV 历史状态。单元越多，权重越容易就地保存，但 token 的中间结果越需要跨单元移动，这就是瓶颈迁移。

当系统扩展到几十个 Raptor 单元后，瓶颈从本地 DRAM 转向：

1. **Activation fabric**：每层或每个 expert 的输入输出如何传输；
2. **Expert routing**：热门 expert 是否压垮局部单元；
3. **Pipeline latency**：几十个 stage 的固定 hop latency 是否累积；
4. **Reduction**：tensor parallel 或 attention shard 如何归约；
5. **Capacity placement**：权重和 KV 如何避免频繁迁移；
6. **Failure recovery**：任一 3D stack 故障时如何恢复模型状态。

以 MoE 为例，如果每个 token 在 $L_{MoE}$ 个稀疏层分别选择 $k$ 个 expert，hidden width 为 $D$，每元素 $b$ bytes，dispatch 和 return 各传一次，则 activation fabric 的一阶流量可以写成：

$$
V_{act/token}
\approx
2L_{MoE}kDb\alpha
$$

$V_{act/token}$ 是每个 token 在 expert 之间来回搬运的 activation 字节数；$L_{MoE}$ 是 MoE 层数，$k$ 是每层选择的 expert 数，$D$ 是 hidden width，$b$ 是每个元素的字节数。前面的 2 表示 dispatch 和 return 各传一次；$\alpha\ge1$ 表示路由元数据、对齐、重传和拓扑复制开销。

举一个只用于量级判断的假设：$L_{MoE}=64$、$k=2$、$D=16{,}384$、$b=2$、$\alpha=1$，得到约 8 MiB/token；在 1,000 token/s aggregate rate 下约为 8.4 GB/s。这个例子说明平均字节数可能不大，但消息数量多、依赖短、路径长，仍可能受延迟和排队影响。

这个数字远低于 100 TB/s 本地权重带宽，却不能据此忽略 fabric。原因是 MoE 流量由大量逐层、小粒度、有依赖的消息构成。系统可能不是被链路峰值带宽卡住，而是被 hop latency、serialization、credit backpressure 和 tail congestion 卡住。

> **证据边界：** 上述 8 MiB/token 是基于明确假设的量级模型，不是 Raptor 已披露的链路流量。实际值取决于模型层数、hidden width、top-k、并行布局、量化和 expert 是否跨单元复制。

#### 5.1.3 吞吐扩展不等于单用户延迟扩展

设流水线有 $M$ 个 stage，第 $i$ 个 stage 的服务时间为 $s_i$，跨 stage 固定开销为 $h_i$。稳态 aggregate throughput 的上限由最慢 stage 决定：

$$
Throughput_{pipeline}
\le
\frac{1}{\max_i(s_i)}
$$

其中 $s_i$ 是第 $i$ 个流水线 stage 处理一个工作单元所需的服务时间，$\max_i(s_i)$ 是最慢 stage 的服务时间。稳态时，最快的 stage 也不能让整个流水线超过最慢 stage 的处理速度，所以吞吐上限是其倒数；如果最慢 stage 用时 2 ms，上限就是约 500 个工作单元/s。

但单个 token 的端到端 latency 下限更接近：

$$
T_{token}
\ge
\sum_{i=1}^{M}s_i
+
\sum_{i=1}^{M-1}h_i
$$

这里 $M$ 是 stage 数，$s_i$ 是每个 stage 的计算/服务时间，$h_i$ 是相邻 stage 之间的固定传输或边界开销。单 token 必须依次经过这些 stage，因此端到端时间至少要累加所有 stage 和边界开销；这就是为什么 aggregate throughput 提升，不代表单用户 TPOT 也按相同比例提升。

增加 stage 可以提升容量和多用户稳态吞吐，却会累积单 token 的边界延迟。不同用户的 token 可以填满流水线；同一用户的下一 token 通常仍依赖上一 token 的结果。于是，aggregate TPS 可以很好看，而 per-user TPOT 并不会按卡数线性改善。

因此，系统带宽不是简单相加：

$$
B_{system}
\neq
N\times100\ \mathrm{TB/s}
$$

更准确的吞吐上限是本地内存、计算、fabric、同步和最慢 stage 的共同最小值。

#### 5.1.4 MoE 热点首先表现为排队，而不是平均带宽不足

对 expert $e$，设 token 到达率为 $\lambda_e$，服务率为 $\mu_e$，利用率为：

$$
\rho_e
=
\frac{\lambda_e}{\mu_e}
$$

其中 $\lambda_e$ 是 expert $e$ 每秒收到的 token 数，$\mu_e$ 是它每秒能够处理的 token 数，$\rho_e$ 是利用率。比如 $\lambda_e=90$、$\mu_e=100$ 时，$\rho_e=0.9$；此时平均还有 10% 余量，但队列对突发流量已经很敏感。这个式子描述的是单个 expert 的局部拥塞，不是全系统平均带宽。

当 $\rho_e$ 接近 1 时，队列等待会按近似 $\rho_e/(1-\rho_e)$ 的形态快速上升。即使全系统平均带宽仍有余量，少量热门 expert 也足以抬高 p99 TPOT。这解释了为什么 100 TB/s 本地带宽不能替代：

- expert replication；
- capacity-aware routing；
- admission control；
- token reordering；
- 热点迁移和负载反馈。

这里的关键不是让所有单元平均繁忙，而是避免任一 token 路径经过接近饱和的 expert。

#### 5.1.5 单元数增加会放大故障域

如果一次请求必须经过 $N$ 个不可替代单元，且单元可用率均为 $A_{unit}$，最简单的串联系统近似为：

$$
A_{request}
\approx
A_{unit}^{N}
$$

这里 $A_{unit}$ 是一个单元在观察窗口内可用的概率，$N$ 是一次请求必须经过且不可替代的单元数，$A_{request}$ 是请求成功完成的近似概率。把它写成幂，是因为这里暂时假设这些单元近似串联、故障相互独立；真实系统若有副本、绕行或重试，结果会更好。

例如仅作敏感性说明，若 $A_{unit}=99.9\%$、$N=47$，则 $A_{request}\approx95.4\%$。真实系统会使用冗余，不应把这个结果当成产品可用率；它说明的是，没有 expert 副本、spare stage、故障绕行和状态恢复时，单元级可靠性无法直接外推为模型级可靠性。

Raptor 系统至少需要区分三种恢复对象：

1. **权重状态**：静态、可复制，适合预置副本和快速重路由；
2. **KV 状态**：按 session 增长，恢复需要复制、重算或迁移；
3. **流水线在途状态**：寿命短，但失败会造成 token replay 和尾延迟尖峰。

> 💡 **芯一视角：** 单器件的 100 TB/s 把“读权重”从主瓶颈降级后，系统不会变得无瓶颈，而会进入 message、queue 和 state 主导的区域。真正需要证明的不是 47 个单元有 4.7 PB/s，而是热门 expert 不排队、每层边界不过度累积、任一单元故障不丢失长上下文 session。

---

### 5.2 与计算型 KV 状态平面的结合

Raptor 可以成为两类状态平面的物理候选：权重平面负责让大体积、低变化的参数留在计算附近；KV 平面负责让随 session 增长的历史状态留在 attention 附近。两者减少的是不同数据流，不能用同一个“近存计算”标签替代系统设计。

**这一节回答另一个问题：如果把 KV 也放到近存位置，究竟减少了哪一段数据搬运？** 关键不是“KV 也在 DRAM 里”，而是历史 KV 是否在本地完成扫描，以及跨单元只传递多大的归约结果。

#### 5.2.1 3D DRAM 权重平面

MoE expert 权重常驻本地 DRAM，由本地 tensor engine 完成 FFN：

```text
token activation
    -> scale-up fabric
    -> Raptor local expert
    -> local weight stream + tensor compute
    -> output activation
```

外部 fabric 只搬 activation，不搬 expert weights。若每个 token 的 activation 为 $O(D)$，而 expert weight 为 $O(D^2)$，数据流收益非常明显。

#### 5.2.2 计算型 KV 状态平面

如果 tensor engine 支持 attention primitive，也可以让历史 KV 常驻 DRAM，在本地执行：

$$
QK^T
\rightarrow
\operatorname{softmax}
\rightarrow
PV
$$

这里 $QK^T$ 计算当前 query 与历史 key 的相似度，`softmax` 把相似度变成归一化权重，$PV$ 用这些权重对 value 做加权求和。若历史 K/V 留在其所在单元，外部网络不必把整段历史搬到 query 所在单元；但本地单元仍然要完成完整扫描和归约，计算并没有凭空消失。

外部只传输当前 token 的 $Q/K_{new}/V_{new}$ 和 context vector。这样可以把历史 KV 的跨设备流量从：

$$
O(SD_{KV})
$$

降低为：

$$
O(D)
$$

这个复杂度变化不是“把 attention 算得更少”，而是把对历史序列的扫描留在 KV 所在位置，只跨设备传输当前 query 和压缩后的归约结果。

#### 5.2.3 跨 KV Shard 的精确 Online Softmax

KV 一旦按 sequence 或 head 分到多个 Raptor 单元，每个 shard 不能独立 softmax 后直接平均。对 shard $j$ 的 score 集合 $S_j$，需要维护局部三元组：

$$
m_j=\max(S_j)
$$

其中 $S_j$ 是第 $j$ 个 KV shard 上的 attention score 集合，$m_j$ 是该 shard 的最大 score。它是数值稳定计算 softmax 的局部最大值。

$$
l_j=\sum_{x\in S_j}\exp(x-m_j)
$$

$l_j$ 是减去局部最大值后的指数和，也可以理解为该 shard 的归一化分母部分。

$$
o_j=\sum_{x\in S_j}\exp(x-m_j)V_x
$$

$o_j$ 是该 shard 对 value 的加权和，也就是归一化前的输出分子部分。于是每个 shard 只需发送 $(m_j,l_j,o_j)$，而不必发送整段历史 KV。

全局归约先计算：

$$
m=\max_j(m_j)
$$

再合并：

$$
l=\sum_j l_j\exp(m_j-m)
$$

$$
o=\sum_j o_j\exp(m_j-m)
$$

最终 context 为：

$$
context=\frac{o}{l}
$$

最后用全局分子 $o$ 除以全局分母 $l$，得到 attention context。这里的难点是不同 shard 的最大值不同，所以必须先用全局 $m$ 重新缩放各 shard 的 $l_j$ 和 $o_j$；不能分别 softmax 后再做简单平均。

这样每个 shard 跨 fabric 发送的是 $(m_j,l_j,o_j)$，通信量主要随 head dimension 和 shard 数变化，而不再随历史长度 $S$ 线性增长。但代价转移到：

- 每层、每个 query head 的低延迟归约；
- 数值格式和累加精度；
- straggler shard；
- shard 故障后的 partial reduction 恢复；
- GQA、MLA 和不同 attention 结构下的映射。

这条归约协议是 KV 状态平面能否成立的核心，不是可选优化。

#### 5.2.4 每层往返延迟是隐藏的硬边界

如果权重平面和 KV 平面是两组独立设备，每个 Transformer layer 可能经历：

```text
weight plane: produce Q/Knew/Vnew
    -> fabric
KV plane: append KV + scan history + reduce softmax
    -> fabric
weight plane: output projection + FFN
```

单层延迟至少包含：

$$
T_{layer}
=
T_{QKV}
+T_{Q\rightarrow KV}
+T_{attention}
+T_{context\rightarrow W}
+T_{FFN}
$$

其中 $T_{QKV}$ 是生成 query、key、value 的时间，$T_{Q\rightarrow KV}$ 和 $T_{context\rightarrow W}$ 是两个状态平面之间的跨网络往返，$T_{attention}$ 是读取历史 KV 并完成 attention 的时间，$T_{FFN}$ 是前馈网络时间。这个分解的用途是定位每层的时间预算，而不是声称这些项完全串行。

若声称单用户达到 1,000 token/s，则 TPOT 约为 1 ms。以 80 层模型为例，平均每层全部计算、两次跨平面传输和同步预算只有：

$$
\frac{1\ \mathrm{ms}}{80}
=
12.5\ \mathrm{\mu s/layer}
$$

这是极强的系统约束。它意味着实现必须至少采用一种方法：

- 权重计算与 KV attention 在同一 Raptor 单元或同一低延迟域内共置；
- 按 layer group 建立 macro-pipeline，避免每层跨远端网络往返；
- 使用足够并发隐藏跨平面延迟，但此时指标更接近 aggregate TPS；
- 通过 speculative decoding 一次接受多个 token，但必须披露 acceptance rate；
- 改变 attention 或模型结构，减少必须访问完整 KV 的层数。

> **证据边界：** 12.5 µs/layer 是由“1,000 token/s 单用户、80 层、严格自回归”推导的平均总预算，不是对 Raptor 实测延迟的断言。它的用途是检验 TPS 口径是否与物理通信路径相容。

#### 5.2.5 状态所有权比计算能力更难

完整 KV state plane 需要回答的不只是“能否算 $QK^T$”，还包括谁拥有和移动状态：

| 状态对象 | 推荐所有权 | 主要约束 |
|---|---|---|
| Expert 权重 | 固定 Raptor 单元，可复制 | 热点、副本一致性、版本切换 |
| Session KV page | session-layer 固定归属 | 容量增长、碎片、故障恢复 |
| Prefix KV | 内容寻址共享池 | 引用计数、租户隔离、淘汰 |
| 新增 KV | 当前 token 执行路径 | 原子追加、顺序与可见性 |
| Online-softmax partial | 单层临时状态 | 低延迟归约、超时与重试 |

KV 容量约束可写为：

$$
N_{session}
\le
\frac{
C_{KV,pool}-C_{reserve}
}{
\mathbb{E}[C_{KV/session}]
}
$$

其中 $N_{session}$ 是 KV 池可以同时容纳的 session 数，$C_{KV,pool}$ 是 KV 池总容量，$C_{reserve}$ 是为故障、碎片和突发增长保留的容量，$\mathbb{E}[C_{KV/session}]$ 是一个 session 平均占用的 KV 容量。它表达的是容量上限，不是吞吐上限；长 context 或 session 长度分布变宽时，平均值还可能掩盖尾部容量压力。

当 context 持续增长时，系统可能先耗尽容量而不是带宽。KV migration 又会重新引入本来试图删除的数据搬运：迁移一个长 session 的收益，必须覆盖复制 KV、更新路由和暂停请求的成本。

#### 5.2.6 三种可行部署形态

| 形态 | 数据路径 | 优点 | 主要边界 |
|---|---|---|---|
| 同栈融合 | 权重与 KV 在同一 Raptor | 最少跨平面往返 | 容量竞争，调度耦合 |
| 独立权重池 + KV 池 | 两类单元专用化 | 资源可独立扩展 | 每层 fabric latency 与归约 |
| Layer-group 混合 | 若干层的权重和 KV 共置 | 减少远端往返，保留流水化 | 放置复杂，故障恢复粒度大 |

对追求低 TPOT 的系统，同栈融合或 layer-group 混合更符合因果链；对高并发吞吐，独立池化可能通过并发隐藏延迟，但必须把 aggregate TPS 与 per-user TPOT 分开报告。

#### 5.2.7 目前仍缺少的产品证明

公开材料尚未证明 Raptor 已经具备完整 KV state plane 所需的：

- online softmax；
- 跨 shard 的 $(m,l,o)$ 精确归约；
- page allocator；
- prefix sharing；
- session-layer affinity；
- KV migration；
- attention p99 latency 控制。

它给出了很有吸引力的物理底座，但系统软件、状态协议和跨层调度仍然需要单独设计。技术可行性不等于当前产品已经实现，更不等于在 1M context 下达成可预测的 per-user latency。

> 💡 **芯一视角：** 权重平面和 KV 平面共同遵循一个原则：不要搬动大而稳定的数据，而要把小而短命的请求送到数据所在地。真正的难点不是矩阵乘，而是每层往返、精确归约、状态所有权和故障恢复。若这些协议不能在微秒级预算内闭合，100 TB/s 只会把等待时间从内存控制器搬到 fabric 队列。

---

## 6. 验证路径：什么证据能够形成结论

**本章回答：如果要把本文的判断从推演升级为结论，还缺哪些数据？**

### 6.1 器件级

- 顺序、随机、stride、hotspot 下的 sustained bandwidth；
- read、write、read-modify-write 的独立能效；
- 0.37 pJ/bit 是否包含 DRAM array、ECC 和 local delivery；
- 25°C、85°C、105°C 下的 refresh、带宽与错误率；
- bank conflict 和 96 B/128 B 聚合效率；
- ECC correction latency 与不可纠正错误率。

### 6.2 封装级

- 36 µm face-to-face bonding yield；
- logic die、DRAM die 与完整 stack 面积；
- 300 W I/O 加 tensor compute 后的总功耗；
- cold plate 到 logic/DRAM 的热阻；
- power delivery droop 与 simultaneous switching noise；
- spare bank remap 后的 timing skew。

### 6.3 系统级

- 1、8、64 个单元的 scaling efficiency；
- 单用户 TPOT、aggregate TPS 和并发度三者的完整曲线；
- MoE expert load imbalance 与动态路由；
- 1M context 下的实际 KV bytes/token；
- 相同模型、精度、batch、SLO 下与 HBM4 的端到端比较；
- 性能、功耗、成本和良率的统一 TCO 模型。

最终需要观察的不是单点峰值，而是条件分布：

$$
P(
T_{token}>x
\mid
N,
batch,
context,
routing,
temperature,
failure
)
$$

只有当卡数、上下文、负载和温度上升时，token latency 仍然保持可预测，100 TB/s 才真正转化成 serving value。

> 💡 **芯一视角：** 最小验证集不是再给一个峰值柱状图，而是在相同模型、精度、batch 和 SLO 下，同时披露 sustained bandwidth、active weights/token、KV bytes/token、卡数、TPOT 分布和墙上功耗。任何一项缺失，都会给“1,000 TPS”留下改变口径的空间。

---

### 6.4 配图建议

> **图 1：HBM 横向 PHY 与 Raptor 垂直 I/O 对比**  
> 左侧画 HBM stack、PHY、interposer、GPU NoC；右侧画 logic-on-DRAM 和 bank-to-engine 局部路径。强调被删除的是 package-scale PHY 与全局搬运，不是所有 I/O 电路。

> **图 2：100 TB/s 的能耗算术**  
> 使用三根柱：0.37 pJ/bit 对应 296 W；2.4 pJ/bit 对应 1.92 kW；5 pJ/bit 对应 4 kW。明确标注三者的系统边界不同。

> **图 3：840 Banks 到 256 Tensor Engines 的粒度重组**  
> 展示 3 bank × 32 B = 96 B，以及 4 × 96 B = 3 × 128 B 的聚合过程。

> **图 4：Logic-on-top 热路径**  
> 展示 cold plate、N4 logic、face-to-face interconnect、custom DRAM 和 package，并标注 105°C、4 ms refresh。

> **图 5：从 32 GB 单元扩展到 3T MoE 系统**  
> 展示权重分片、activation fabric、KV plane 和 pipeline stage，突出瓶颈从本地内存转移到跨单元通信。

---

## 结语

Raptor 揭示了 AI 推理芯片的一条重要分叉：未来的竞争不只是“谁能封装更多 HBM”，而是谁能重新定义计算看到数据的物理距离。

HBM 的优势是容量、生态与通用性。它的代价是 PHY、beachfront 和从 memory controller 到 tensor engine 的多级搬运。Raptor 用 3D 集成把这条路径压缩为垂直连接，再用 bank 与 tensor engine 的共同设计，把 DRAM 内部并行度直接暴露给计算。这就是 100 TB/s 和 0.37 pJ/bit 背后的真实机制。

但器件带宽不是系统吞吐。32 GB 容量意味着大型模型必须跨几十个单元部署；100 TB/s 本地带宽会把瓶颈推向 activation fabric、KV 扫描、expert imbalance、pipeline latency 和故障恢复。文章目前最有说服力的是器件方向与 I/O 能耗算术，最缺的是端到端边界一致的第三方测量。

实践上的判断标准很简单：不要只问它有多少 TB/s，要问这些带宽在什么访问模式、什么温度、什么模型布局和什么并发下能够持续；不要只问 pJ/bit，要问这个数字从 DRAM cell 算到哪里；不要只看 1,000 TPS，要同时看 active parameters、KV bytes/token、卡数与 TPOT。

**100 TB/s 解决的是本地数据供给。真正决定 Raptor 能否改变推理系统的，是它能否让计算长期留在数据所在的位置。**

---

如果这篇分析对你有价值，欢迎点赞、在看或转发。技术勘误和补充证据请通过公开仓库 Issue 提交。

**工程芯一**

---

## 参考文献

1. [d-Matrix's Raptor 3D DRAM Achieves SRAM-Class Bandwidth at 1/10th the HBM Power](https://wccftech.com/d-matrix-raptor-3d-dram-achieves-sram-class-bandwidth-at-1-10th-the-hbm-power/amp/)
2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
3. [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)

除明确引用的公开数字外，文中的带宽、容量、流量、延迟和可用率计算均为基于已声明假设的架构推导，不代表 d-Matrix 产品实测或官方实现说明。