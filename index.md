---
layout: default
title: AI Infra Knowledge
---

# AI Infra Knowledge

Publicly approved AI infrastructure architecture analysis.

## Architecture

### [100 TB/s 不是更快的 DRAM：d-Matrix Raptor 如何删除 HBM 的片外内存边界](architecture/dmatrix-raptor-3d-dram/)

从垂直 I/O、bank-to-engine 映射和系统瓶颈，审计 Raptor 3D DRAM 的 100 TB/s 与 0.37 pJ/bit。

### [MemServe 解读：统一 KV 生命周期，而不是一颗独立的 Memory Pool 芯片](architecture/memserve-elastic-memory-pool-analysis/)

从 MemPool API、分布式 Prompt Tree、PD-Caching 演进和单机 H800 实验，分析 MemServe 的真实贡献、证据边界，以及对 Scale-up Device-Memory Pool 与池内 Attention 芯片的启示。

### [Mooncake Store 对 TPU 支持到哪一步：底层通了，产品链路尚未闭环](architecture/mooncake-tpu-support-status/)

审计 Mooncake TENT 的 TPU/PJRT staging、Host DRAM 中转路径、实机缺陷与 TPU-Sync 关系，区分已合入能力和生产可用边界。

### [从一个 Token 出发：硬件架构师如何读懂 TPU 推理优化](architecture/tpu-inference-hardware-software-codesign/)

先补齐 Attention、KV Cache、MoE 和 GDN 的最少模型知识，再把 TPU 推理优化还原为计算、内存、互连和状态管理问题。

