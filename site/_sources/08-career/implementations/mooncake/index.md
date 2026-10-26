# Mooncake 与 KV Cache 存储：28 天学习手册

跨项目学习先从[全库总目录](../../../README.md)和 [Mooncake 白话介绍](../../../04-storage/implementations/mooncake/overview.md)开始，执行[统一学习计划](../../study-plan.md)。本页保留 Mooncake 单项目的深读导航，以下 28 天安排是专项备选。

这套中文资料面向推理基础设施、KV Cache 存储与高性能通信岗位。目标是在 **28 天、约 72 小时**内建立完整知识体系，能够沿源码解释一次 KV 写入和读取，设计分布式缓存系统，并用实验判断缓存是否真的提高了满足延迟目标的吞吐。

学习主线是：**理解 KV 的计算语义 → 算清容量与搬运成本 → 掌握 Mooncake 数据路径 → 接入推理引擎 → 验证正确性与性能 → 完成系统设计和面试表达**。

## 使用范围与版本基线

- 整理日期：2026-10-24。
- Mooncake 源码基线：`dd1217b6c461491e7c73ff29b4ffad215e0a9ac0`。
- Store 实现主线：`MOONCAKE_STORE_BACKEND=cpp`；通信主线：经典 Transfer Engine。Store-RS、TENT、Conductor、EP/PG 等在知识地图中标出边界和进阶入口。
- LMCache 部分同时说明进程内模式与 MP 模式。本文引用的仓库 MP 集成指南要求 LMCache v0.4.5 或更新版本，并需要构建可选 Mooncake 扩展；它不是任意版本组合的兼容性保证。
- 论文架构、当前源码行为、教学设计建议分别标注。性能算例均为计算示例；本手册没有把未运行的 GPU/RDMA 实验当作实测结果。

如果使用另一个 commit 或发行版，先核对 API、Connector、数据布局和配置，尤其是 `soft_pin_action`、Store 后端选择及 LMCache MP 接入方式。不要把多篇不同年代的启动命令拼成一套部署。

## 学完应该能做到什么

| 能力 | 合格标准 |
|---|---|
| 数学与容量 | 从模型配置计算每 token、每请求、每 rank KV 字节数，说明 GQA、MLA、TP 的适用条件 |
| 缓存正确性 | 解释前缀依赖、键命名空间、租约、完整写入可见性和异步缓冲区生命周期 |
| Mooncake 源码 | 找到 `Client::Put/Get`、Master 元数据、分配/回收、Transfer Engine 批传输的调用关系 |
| 系统集成 | 区分直接 PD 传输、共享 Store、LMCache 分层管理、SGLang HiCache |
| 性能分析 | 分解 TTFT/ITL；比较重算与拉取；设计消融实验和故障实验 |
| 系统设计 | 给出容量、带宽、元数据规模、SLO、降级和恢复方案 |
| 面试表达 | 用“结论—机制—代价—证据”回答，而不是背组件名或宣传数字 |

## 学习材料导航

| 章节 | 内容 | 使用方法 |
|---|---|---|
| [01 · 28 天学习计划](study-plan.md) | 每日任务、时间、交付物、每周验收 | 按天执行；以验收结果决定是否进入下一阶段 |
| [02 · 知识地图与设计图](../../../04-storage/implementations/mooncake/architecture.md) | 全景图、部署图、读写时序、PD 流程、状态和数据布局 | 第一周建立框架，第四周闭卷重画 |
| [03 · KV Cache 基础与性能模型](../../../03-kv-cache/capacity-and-reuse.md) | Attention、容量、分页、命中语义、收益公式、调度 | KV 存储岗位必读 |
| [04 · Mooncake 内核与源码路线](../../../04-storage/implementations/mooncake/internals.md) | Store、Master、TE、租约、副本、淘汰、故障恢复 | 带着问题逐段阅读源码 |
| [05 · LMCache 与推理引擎集成](../../../04-storage/implementations/mooncake/integrations.md) | MP/进程内模式、HiCache、Connector、CacheBlend | 重点理解各层职责和数据所有权 |
| [06 · 实验与性能诊断](../../../07-practice/implementations/mooncake/labs.md) | 本机推导、Linux TCP、RDMA、集成、故障与消融 | 不同硬件条件选择对应路线 |
| [07 · 面试题与参考回答](questions.md) | 50 道高价值问题、追问方向、常见错误 | 录音复述，再对照答案纠错 |
| [08 · 系统设计、毕业项目与参考资料](system-design.md) | 容量规划示例、项目验收、阅读清单 | 形成可展示的项目报告 |

## 先建立这六个判断

1. **KV Cache 是模型中间状态。** 同样的文本片段处在不同前缀、模型或位置编码配置下，通常不能直接共享 KV。
2. **缓存容量和请求吞吐不是同一个指标。** 大缓存可能增加远程读取、排队和元数据成本；应比较满足 TTFT/ITL SLO 的有效吞吐。
3. **共享缓存和 PD 分离是两个维度。** PD 可以直接通过 TE 搬运，未必经过 Store；共享 Store 也可以服务未做 PD 分离的实例。
4. **LMCache 和 Mooncake 可以组合。** 前者衔接推理引擎的 KV 生命周期、分层管理与加载，后者提供共享对象存储和数据传输。
5. **零拷贝有明确边界。** `get_into` 可以减少 Python 对象复制，RDMA 可以绕开远端 CPU 数据搬运；它们不自动消除布局转换、GPU 同步和所有内存拷贝。
6. **缓存失效可以重算，错误命中不能使用。** 超时、租约过期、校验失败或兼容性不满足时，应回退到可验证的计算路径。

## 当前 macOS 工作区如何使用

本机适合阅读源码、做容量计算、写策略模拟、阅读和构建文档。Mooncake 原生部署和性能实验按官方支持的 Linux 环境安排；macOS 上的 Docker 虚拟机不能替代真实 GPU/RDMA 网络。

没有 GPU 时，可以完成 TCP 对象读写、元数据/租约分析和负载模拟；实验报告必须注明没有验证真实 KV 的 GPU 布局、跨 rank 协调和端到端推理性能。学习目标可以完成到设计和源码层面，性能结论需要补齐相应硬件证据。
