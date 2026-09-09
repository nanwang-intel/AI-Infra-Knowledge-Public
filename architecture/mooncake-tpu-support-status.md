---
title: "Mooncake Store 对 TPU 支持到哪一步：底层通了，产品链路尚未闭环"
description: "审计 Mooncake TENT 的 TPU/PJRT staging、Host DRAM 中转路径、实机缺陷与 TPU-Sync 关系，区分已合入能力和生产可用边界。"
date: 2026-09-09
updated: 2026-09-09
slug: mooncake-tpu-support-status
status: published
github: true
public: true
wechat: draft
wechat_url:
cover: /assets/mooncake-tpu-support-status/cover.jpg
series: "Y26W37"
content_type: analysis
---

# Mooncake Store 对 TPU 支持到哪一步：底层通了，产品链路尚未闭环[Y26W37][解析]

***截至 2026 年 9 月 9 日，Mooncake 已经合入 TPU 的底层传输支持，但准确地说，这是 Mooncake TENT Transfer Engine 的实验性 TPU staging 能力，还不能等同于“Mooncake Store 已经可以在 TPU 上开箱即用地构建生产级分布式 KV Cache”。***

过去判断一个 KV Cache 系统是否支持某种加速器，容易只看仓库中是否出现对应的 `Transport`。TPU 把这个问题拆得更清楚：识别 TPU buffer、把数据搬出 HBM、经过数据中心网络、在远端写回 HBM，以及让 JAX/PyTorch serving runtime 管理 KV 生命周期，是五个不同层级。Mooncake 当前只完成了其中一部分。

## 原始材料

- [Mooncake RFC #2662：Add TPU Transport Support via PJRT](https://github.com/kvcache-ai/Mooncake/issues/2662)
- [Mooncake PR #2733：Add TPU (PJRT) staging support to TENT](https://github.com/kvcache-ai/Mooncake/pull/2733)
- [Mooncake PR #2815：Fix silent TPU data corruption](https://github.com/kvcache-ai/Mooncake/pull/2815)
- [Mooncake TPU/PJRT Platform README](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/tent/src/platform/tpu/README.md)
- [Google TPU-Sync](https://github.com/google/tpu-sync)

本文不把 RFC、已合入代码和未来路线图混为一谈，而是围绕一个问题做证据审计：**Mooncake 今天已经能在 TPU 上完成什么，距离生产级 KV Store 还缺什么。**

## 本篇文章分为六个部分

**1. 当前状态：区分 TENT staging 与 Mooncake Store 支持**

2. 数据路径：TPU HBM 为什么必须经过 Host DRAM

3. 已合入机制：Platform、Transport、ProxyManager 与 PJRT Adapter

4. 实机教训：4 MiB 之后的静默损坏与子区间带宽

5. TPU-Sync：可能的长期数据面，但尚未完成 Mooncake 对接

6. 最小验证路径：从“代码存在”走到“生产可用”

---

## I. 当前状态：底层传输已合入，Store 语义未闭环

2026 年 7 月 4 日，Mooncake 合入 PR #2733，为 TENT 增加 TPU/PJRT staging。对应代码至少已经进入 v0.3.12 代码线，构建开关默认为关闭。

但这里必须区分三个对象：

| 对象 | 职责 | 当前 TPU 状态 |
|---|---|---|
| TENT Transfer Engine | 路由请求、分块、选择 TCP/RDMA、编排 Host staging | TPU staging 已合入 |
| Mooncake Store | KV 对象、索引、逻辑内存池、淘汰和后端层级 | 没有完成 TPU 专用 Store 集成 |
| Serving Integration | 从 JAX/PyTorch-XLA 取得 KV buffer，注册、释放并维护生命周期 | 主仓库尚未包含 |

PR #2815 的模块清单也明确标注：修改覆盖 Transfer Engine 与 Common，没有覆盖 Mooncake Store、Integration、Python Wheel 和 P2P Store。因此，“Mooncake 支持 TPU”若不注明层级，会把字节搬运能力误写成完整 KV 系统能力。

![Mooncake TPU 支持 RFC](/assets/mooncake-tpu-support-status/figure-01-rfc.png)

> **图 1：Mooncake TPU RFC 截至调查日仍保持 Open。** 已合入的 TENT staging 是 RFC 的一部分，不代表所有长期目标完成。来源：[Mooncake RFC #2662](https://github.com/kvcache-ai/Mooncake/issues/2662)。

> 💡 **芯一视角：** Mooncake 已经搭好 TPU 数据搬运的编排骨架，但生产 Adapter、Serving 集成和 Store 语义没有一起交付。判断支持状态时，关键不是仓库里有没有 `TpuTransport`，而是 KV buffer 从框架所有权到远端恢复能否完整闭环。

---

## II. 数据路径：TPU HBM 为什么必须经过 Host DRAM

当前 Mooncake 设计假设 TPU HBM 不能由 Host NIC 直接寻址。跨节点传输因此采用 staged two-copy model：

```text
源 TPU HBM
  -> PJRT D2H
源 Host DRAM staging buffer
  -> TCP 或 RDMA
目标 Host DRAM staging buffer
  -> PJRT H2D
目标 TPU HBM
```

总延迟可以拆成：

$$
T_{transfer}
=
T_{D2H}
+T_{host\ network}
+T_{H2D}
+T_{pipeline/queue}
$$

Mooncake 的 `ProxyManager` 把请求切成默认 4 MiB chunk，并使用 staging buffer 和异步状态机编排各阶段。若双缓冲有效，设备拷贝与网络传输可以部分重叠，所以端到端时间不一定等于四项简单相加；稳态吞吐仍受最慢阶段约束：

$$
BW_{end-to-end}
\lesssim
\min(BW_{D2H},BW_{network},BW_{H2D},BW_{DRAM})
$$

这条路径的架构收益是复用现有 TCP/RDMA，不占用 TPU ICI 承载异步 KV 流量。代价是两次 Host staging、额外 DRAM 带宽、NUMA/NIC 亲和性，以及 buffer 注册和故障回收。

![Mooncake Store 逻辑内存池](/assets/mooncake-tpu-support-status/cover.jpg)

> **图 2：Mooncake Store 的逻辑内存池愿景。** 该图说明通用 Store 架构，不证明 TPU serving integration 已完成。来源：Mooncake。

---

## III. 已合入机制：TENT 如何认识 TPU

PR #2733 增加了四个关键组件：

| 组件 | 作用 | 明确边界 |
|---|---|---|
| `MTYPE_TPU` 与 `tpu:N` | 标记 TPU memory type 和设备序号 | 只是地址与拓扑元数据 |
| `TpuPlatform` | 识别 device token，调用 D2H/H2D | 不分配 TPU HBM，不支持 D2D |
| `TpuTransport` | 宣告 `gpu_to_dram`/`dram_to_gpu` 能力 | 只执行本机 staging，不是网络 transport |
| `ProxyManager` | 分块、双缓冲并串联本地与远端阶段 | Host-to-host 仍由 TCP/RDMA 完成 |

TPU HBM 由 Serving Framework 分配并注册给 TENT。Mooncake 不把 PJRT 作为构建依赖，而是在运行时加载一个外部 Adapter：

```text
libmooncake_tpu_pjrt.so
```

构建与运行条件为：

```bash
cmake -S . -B build \
  -DUSE_TENT=ON \
  -DUSE_TPU=ON

export MC_TPU_PJRT_LIB=/path/to/libmooncake_tpu_pjrt.so
```

Mooncake 主仓库定义了 Adapter C ABI，并提供 Mock Adapter，但当前 README 的 “Not yet included” 仍列出：

- JAX/PyTorch-XLA serving integration；
- DMA-mapped pinned staging buffer；
- 真正异步的 device DMA；
- TPU 实机 benchmark。

所以代码能够编译，不表示运行时具备生产 PJRT Adapter；Adapter 缺失时，TPU 操作会返回错误状态。

---

## IV. 实机教训：正确性修复了，性能路径仍需重构

PR #2815 记录了第一次真实 TPU VM 验证暴露的问题。`ProxyManager` 以 4 MiB 为默认 chunk，第二个 chunk 开始把 `token + chunk_offset` 传给 Adapter。初版 ABI 只识别 buffer base address，导致内部地址被误判为 Host pointer。

后果不是崩溃，而是更危险的静默错误：

```text
Chunk 0：正确
Chunk 1..N：把 PJRT opaque token 当普通地址 memcpy
最终状态：返回 COMPLETED，但 KV 内容错误
```

修复要求 Adapter 识别 registered buffer 内的 interior pointer，并在 `TpuTransport` 中强制每次 staging 恰好只有一侧属于 TPU memory。测试也改成 poisoned token 与 shadow storage，避免普通 Host pointer 掩盖错误。

实机还暴露了性能问题：

| 路径 | PR #2815 披露值 | 证据边界 |
|---|---:|---|
| 整 buffer D2H | 约 18.5 GB/s | v5p-8、libtpu 0.0.32、PJRT C API 0.83 |
| PJRT sub-range D2H | 约 0.8 GB/s | 同一调查中的子区间路径 |
| 当前 staging 推测上限 | 约 1 GB/s/Chip | 设计讨论，不是跨节点 E2E 测量 |

当前 ABI 中 D2H/H2D 调用是同步的；跨节点 hop 也没有在该实机测试中覆盖。因此不能由单机局部修复推导 PD 分离的吞吐与 p99。

> 💡 **芯一视角：** 这次缺陷说明，Mock 的价值取决于是否保留真实硬件的语义，而不只是返回正确字节。性能问题也同样直接：如果框架按 chunk 调度，而底层只有 whole-buffer fast path，两层各自“合理”的设计组合起来仍会得到约 20 倍落差。

---

## V. TPU-Sync：可能的长期数据面，但尚未完成对接

Google TPU-Sync 的目标比 Mooncake 当前 Adapter ABI 更完整：它直接提取 JAX array 和 PyTorch tensor 的原生 `PjRtBuffer` 描述符，并提供 Host offload、异步 pipeline、共享内存持久化和跨节点 KV 迁移。

![Google TPU-Sync](/assets/mooncake-tpu-support-status/figure-02-tpu-sync.jpg)

> **图 3：TPU-Sync 定位为 TPU KV 编排、多层内存与跨节点数据传输库。** 官方同时注明仍在积极开发，不建议通用部署。来源：[Google TPU-Sync](https://github.com/google/tpu-sync)。

TPU-Sync README 宣称，针对生产级 KV block shape 的优化路径可达到约 200 GB/s 双向 DMA 带宽。这个数字不能与 Mooncake PR 中的 0.8 GB/s 直接比较：前者是 TPU-Sync 自身披露的优化双向 pipeline，后者是特定 PJRT sub-range D2H 路径；方向、硬件、buffer shape 和测量边界不同。

Mooncake RFC 在 2026 年 9 月更新称，开发者正与 TPU 团队合作一套“更高效的新 TPU library”。从功能方向看，TPU-Sync 是合理候选，但当前没有公开的 Mooncake 合入 PR 证明：

- `libmooncake_tpu_pjrt.so` 已由 TPU-Sync 正式实现；
- Mooncake Store 已调用 TPU-Sync 管理 KV block；
- TENT 与 TPU-Sync 的 buffer ownership 和 completion semantics 已对齐；
- 200 GB/s 能在 Mooncake staged E2E 路径复现。

因此应把两者关系写成“潜在对接方向”，而不是“Mooncake 已经建立在 TPU-Sync 上”。

---

## VI. 最小验证路径

从实验性 transport 走到生产级 TPU KV Store，至少需要以下验证：

| 层级 | 必测项 | 通过条件 |
|---|---|---|
| Device copy | Whole-buffer/sub-range，4 MiB 到 1 GiB | 带宽曲线可解释，无 chunk cliff |
| Buffer ownership | JAX 与 PyTorch-XLA 注册、释放、重启 | 无悬挂 token、重复释放和错误复用 |
| Pipeline | Chunk size、buffer depth、同步/异步 DMA | Device copy 与网络能稳定重叠 |
| Host memory | DRAM 带宽、NUMA、DMA-map 与 RDMA MR | 不把 Host DRAM 变成新瓶颈 |
| Network | TCP/RDMA 单流、多流、p50/p99 | 跨节点结果与局部模型闭合 |
| KV semantics | Block、Prefix、Eviction、Warm restart | 正确性覆盖异常与恢复路径 |
| PD serving | TTFT、TPOT、KV bytes moved | 分离收益覆盖 transfer 与 queue |
| Agentic trace | 多轮、长前缀、Sub-agent burst | Cost/qualified task 与 p99 同时改善 |

三个 No-Go 条件：

1. **子区间 DMA 仍停留在约 1 GB/s 量级。** Host network 再快也无法补偿设备侧 staging。
2. **Serving Framework 不掌握 buffer 生命周期。** Transport 正确也会被 stale token 和错误回收破坏。
3. **KV 外部化降低 HBM 占用，却提高任务 p99。** 容量收益没有覆盖两次 staging、网络和排队成本。

> 💡 **芯一视角：** 峰值带宽证明一条路径能跑多快，真实 KV trace 才说明系统是否总能走上那条路径。没有 sub-range、p99、故障恢复和 buffer ownership，`USE_TPU=ON` 只是编译选项，不是生产结论。

## 配图说明

本文使用三张证据型图片：

- Mooncake TPU RFC 状态与目标；
- Mooncake Store 逻辑内存池架构；
- TPU-Sync 官方定位与能力说明。

## 结语

Mooncake 对 TPU 的支持已经从路线图推进到可审计的底层代码：TENT 能识别 TPU memory，通过 PJRT Adapter 在 HBM 与 Host DRAM 之间搬运数据，再复用 TCP/RDMA 完成跨节点传输。这证明架构路径可行，也明确了 ICI 与数据中心网络的职责边界。

最大缺口不在是否存在 `TpuTransport`，而在生产 Adapter、JAX/PyTorch-XLA 集成、异步 pinned DMA、Mooncake Store KV 语义和跨节点实机数据。PR #2815 已经证明，真实 PJRT buffer 的地址语义和 sub-range 性能都不能由 Mock 或 whole-buffer benchmark代替。

**最终判断：现在可以说 Mooncake 具备实验性 TPU staging 基础，不能说 Mooncake Store 已经完整支持 TPU。只有当 TPU-Sync/PJRT 数据面、Serving buffer ownership 和 Store 生命周期在真实 PD/KV 工作负载下共同通过正确性、p99 与故障恢复验证，产品链路才算闭环。**

---

如果这篇分析对你有价值，欢迎点赞、在看或转发。技术勘误请通过仓库 Issue 或文章讨论入口提交。

**工程芯一**

---

## 参考文献

1. [Mooncake RFC #2662, Add TPU Transport Support via PJRT](https://github.com/kvcache-ai/Mooncake/issues/2662)
2. [Mooncake PR #2733, Add TPU PJRT Staging Support to TENT](https://github.com/kvcache-ai/Mooncake/pull/2733)
3. [Mooncake PR #2815, Fix Silent TPU Data Corruption](https://github.com/kvcache-ai/Mooncake/pull/2815)
4. [Mooncake, TPU PJRT Platform for TENT](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/tent/src/platform/tpu/README.md)
5. [Mooncake, Supported Communication Protocols](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/getting_started/supported-protocols.md)
6. [Google, TPU-Sync](https://github.com/google/tpu-sync)