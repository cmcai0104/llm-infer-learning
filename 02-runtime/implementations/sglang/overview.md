# SGLang：把请求调度、前缀复用与分层缓存连起来

[返回运行时主题](../../README.md) · [同主题对照：vLLM](../vllm/overview.md) · [相关主题：KV 管理](../../../03-kv-cache/README.md)

SGLang 是完整的推理服务系统。它同样需要把请求送进 GPU、调度 Prefill/Decode、管理内存和返回输出。学习重点是：**前缀复用怎样影响调度，缓存怎样从 GPU 延伸到主机和外部存储，以及这些状态怎样与执行保持一致。**

## 1. 为什么前缀树适合这些请求

考虑三个输入：

```text
A = 系统提示 + 合同甲 + 问题一
B = 系统提示 + 合同甲 + 问题二
C = 系统提示 + 合同乙 + 问题三
```

A/B 共享更长前缀，A/C 只共享系统提示。Radix 树把共同 token 路径存成共享分支，分叉位置对应输入差异。查找一个新请求时，沿树寻找可复用的最长前缀。

真实 radix 实现可以压缩路径、按一定粒度组织节点；**树节点是索引，不是 KV tensor 本身**。节点关联相应的设备位置、host 状态、引用和其他信息。找到节点后，仍要确认数据在哪、能不能用。

这就是 RadixAttention 思路值得学的地方：让共享前缀与请求运行时联系起来。当前 SGLang 还有 Unified Cache、混合模型组件和多种后端，不能仅靠早期论文还原所有现行实现。

## 2. 一次请求怎么走

```text
HTTP / TokenizerManager
→ tokenized request
→ Scheduler
→ TpModelWorker / ModelRunner
→ 模型与 attention backend
→ token IDs
→ DetokenizerManager
→ 流式文本
```

在常见部署中，Scheduler、TpModelWorker、ModelRunner 可以位于同一个 GPU worker 进程。职责框图的一个方框不等于一个独立进程。

| 对象 | 管什么 | 为什么要分开 |
|---|---|---|
| `Req` | 单个请求的输入、输出、采样、结束条件与缓存关联 | 记录请求长期状态 |
| `ScheduleBatch` | 当前调度选择与批内请求状态 | 决定这一轮哪些工作运行 |
| `ForwardBatch` | 模型执行需要的张量、索引和运行模式 | 把调度信息转成 GPU 可消费的输入 |

先沿普通文本、单卡、常规采样路径读懂这三者，再增加 TP、投机解码、PD 等分支。

## 3. 内存分配、前缀管理、HiCache 各管一层

**物理内存池和 allocator** 负责“哪些槽位空闲，数据 tensor 在哪里”。**前缀缓存**负责“哪段输入能匹配，哪些状态要保留，共享者还有几个”。**HiCache**进一步处理“设备、主机和外部后端之间怎么保存与加载”。

```text
请求 / Scheduler
  ↕
前缀索引与状态管理
  ↕
GPU KV pool ↔ Host KV pool ↔ 外部存储
```

当前基线中，可从 `mem_cache/registry.py` 找到缓存构造路径，再看 `UnifiedRadixCache` 及其组件。不同模型和配置存在特定分支；不要假设所有部署都使用同一种树、同一种 KV 形状。

## 4. HiCache 的三层怎么配合

常见叫法为：L1=GPU，L2=主机内存，L3=外部后端。这里 L2 不是处理器的硬件 L2 cache。

1. 新请求匹配 GPU/host 上已有的前缀。
2. 对尚未覆盖的连续前缀查询 L3。
3. 根据等待预算和可用空间，把远端数据预取到 host。
4. 把可消费部分加载到 GPU，完成同步。
5. 执行剩余输入，并按策略保存新状态。

外部后端可以是 Mooncake Store 等。主机缓存通常属于具体实例或其管理范围；把 host cache 配大，不等于自动获得跨机器共享能力。

### 预取该等多久

| 策略 | 白话理解 | 适合关注的指标 |
|---|---|---|
| `best_effort` | 能赶上就用，尽量别为等缓存拖延可运行请求 | TTFT、实际加载前缀 |
| `wait_complete` | 等预取走完再决定执行范围 | 慢存储下的尾延迟 |
| `timeout` | 给缓存一个时间预算，过时就按有效结果推进 | 超时率、节省计算与等待代价 |

“等得更久所以命中更多”并不能证明优化成功。要看多省的 Prefill 是否大于多等的时间。

### 什么时候写到下一层

`write_through` 倾向更早备份，`write_through_selective` 根据条件选择要备份的数据，`write_back` 将备份推迟到淘汰等阶段。具体异步行为和支持组合按版本核对。写回任务仍在使用内存时，不能因为 GPU 缓存想淘汰就直接归还槽位。

## 5. 跨 rank 为什么不能各算各的

假设 TP 的 rank 0 成功加载了 4 个 chunk，rank 1 只加载了 3 个。一个 rank 跳过 4 个 chunk、另一个跳过 3 个，会使执行进度和通信不一致。

普通 full-attention 路径需要在参与者之间确定共同安全前缀，常见办法是取有效长度的最小值。还要保证 collective 调用顺序一致。混合 group、滑窗与递归状态要按各自契约判断，不能只对所有数字机械取最小值。

## 6. 重叠调度为什么难

CPU 可以在 GPU 算 batch n 时准备后续工作，提高流水线利用率。但下一轮生成需要前一轮结果，取消、采样约束、缓存回收也有依赖。

因此 `event_loop_overlap` 的价值不是“让所有步骤并行”，而是把能提前做的工作安排好，同时保留必要同步。读代码时关注结果队列、事件、在途引用和禁用 overlap 的特殊路径。

## 7. 与 LMCache、Mooncake 怎么接

两条常见候选路线：

```text
SGLang → HiCache → Mooncake Store
SGLang → LMCacheUnifiedRadixCache / MP 集成 → LMCache → 存储后端
```

第二条在本地源码基线有独立 LMCache daemon 的接入说明。两条路线的资源所有者和策略执行位置不同，不应理解成默认层层叠加。选择后确认该版本的参数互斥、模型支持、持有者存活和缓存格式。

PD 场景还可能直接使用 Mooncake TE 把 KV 从 Prefill 端送到 Decode 端，和共享 Store 是不同的数据路径。

## 8. 源码与岗位重点

路径相对于 SGLang 上游源码根目录：

| 入口 | 阅读重点 |
|---|---|
| `python/sglang/srt/managers/scheduler.py` | 正常/重叠循环、批次推进和输出处理 |
| `python/sglang/srt/managers/schedule_batch.py` | 请求与调度对象 |
| `python/sglang/srt/model_executor/model_runner.py` | 模型执行与设备状态 |
| `python/sglang/srt/mem_cache/registry.py` | 实际选择哪条缓存路径 |
| `python/sglang/srt/mem_cache/unified_radix_cache.py` | 匹配、引用、淘汰、load-back |
| `python/sglang/srt/managers/cache_controller.py` | host/device/外部缓存任务协调 |
| `python/sglang/srt/mem_cache/storage/` | Mooncake、LMCache 等集成边界 |

推理引擎岗重点读调度和运行时；KV 存储岗重点读缓存构造、预取、加载完成和跨 rank 一致性。对应的[架构教材](runtime.md)、[KV 教材](../../../03-kv-cache/implementations/sglang/prefix-cache.md)和[HiCache 专题](../../../03-kv-cache/implementations/sglang/tiered-cache.md)提供详细展开。

## 9. 面试自检

能否解释“前缀树节点存在但 GPU 数据已经不在”的正常情况？能否画出从远端对象到 GPU 可用状态的过程？能否给出等待预取损害 P99 的例子？能否说明页分配与 Radix 管理可以共存？

回答这些问题，比只说“SGLang 用 RadixAttention，所以更快”更能证明理解。

## 专题深读资料

跨项目学习使用[统一 28 天计划](../../../08-career/study-plan.md)。以下保留 SGLang 单项目的完整教材与选修路线。

### 专项教材导航

| 文档 | 内容 | 建议使用时间 |
|---|---|---|
| [01：知识地图](knowledge-map.md) | 学习大纲、依赖关系、岗位能力与优先级 | D1，总复盘时重读 |
| [02：28 天计划](../../../08-career/implementations/sglang/study-plan.md) | 每日任务、交付物、每周验收与硬件替代方案 | 每天 |
| [03：架构与运行](runtime.md) | 进程架构图、请求时序、调度循环、源码阅读路线 | D4–D14 |
| [04：推理基础](../../../01-foundations/compute-and-memory.md) | Attention、性能模型、调度、GPU 优化、并行与生产工程 | D1–D10、D22–D26 |
| [05：KV Cache](../../../03-kv-cache/implementations/sglang/prefix-cache.md) | 容量计算、分页、Radix、生命周期、兼容性、混合模型 | D11–D16 |
| [06：HiCache、Mooncake 与 LMCache](../../../03-kv-cache/implementations/sglang/tiered-cache.md) | 分层缓存、共享存储、传输、框架对比、PD 分离与选型 | D15–D23 |
| [07：实验手册](../../../07-practice/implementations/sglang/labs.md) | 七组实验、启动命令、缓存探测、对照设计与 profiling | 贯穿全程 |
| [08：面试问答](../../../08-career/implementations/sglang/questions.md) | 36 道题的回答要点、追问与系统设计演练 | D7 起穿插，D27 集中 |
| [09：参考资料与验收](../../../08-career/implementations/sglang/assessment.md) | 论文、官方资料、实验报告模板、最终评分与项目讲述 | 按主题阅读，D28 验收 |

继续深学 SGLang，可按知识地图 → Runtime → 前缀缓存 → 分层缓存阅读。表内编号用于对应原教材，天数属于专项计划，不覆盖全库统一计划。

### 专项学习目标

完成学习后，应能够：

1. 从源码解释 SGLang 的请求生命周期，区分服务层、调度层、模型执行层和缓存层。
2. 计算权重、KV Cache、并行切分与传输的近似成本。
3. 解释 Continuous batching、Chunked prefill、CUDA Graph 和 CPU/GPU overlap。
4. 完成冷缓存、GPU 命中、CPU 命中、外部缓存命中实验，解释收益和退化。
5. 区分 Mooncake Transfer Engine、Mooncake Store、HiCache 和 LMCache 的职责。
6. 围绕 SLO、Goodput、容量和故障恢复设计一个推理服务。
7. 用源码与实际实验支持面试回答，明确区分理论推演和实测结果。

一个月的目标是形成可独立分析和验证的核心能力。所有模型、硬件后端和 CUDA kernel 的深度开发需要后续专题学习。

## 版本依据与正确使用方式

- 编写与核对日期：2026-10-24。
- 本地仓库 HEAD：`30dc2f94303e339431cbd51489b2d1719678b0c1`。
- 以核对时的工作区源码为主要依据，同时参考官方文档；未假定工作区所有文件均与 HEAD 完全一致。
- HEAD 是源码定位依据，不等于某个已发布 wheel 或容器版本。
- 本手册中的 GPU 命令经过静态核对，**没有在本次文档整理中运行 GPU benchmark**。
- 理论容量、传输下界和示例数据不代表实测性能。
- 配置开关、支持矩阵、默认值和集成方式会随版本变化。复现实验时必须记录实际软件版本与完整命令。

### 当前源码中的两个关键变化

1. 当前默认缓存构造路径已转向 `UnifiedRadixCache`。旧教程中的 `RadixCache`、独立 `HiRadixCache` 需要映射到当前的树核心、组件和控制器。
2. 当前 SGLang 的 LMCache 集成使用 `LMCacheUnifiedRadixCache` 和独立多进程服务。当前代码限制它与 HiCache、PD 分离、投机解码等功能组合使用。

事实入口：

- [缓存工厂与默认实现选择](https://github.com/sgl-project/sglang/blob/30dc2f94303e339431cbd51489b2d1719678b0c1/python/sglang/srt/mem_cache/registry.py)
- [LMCache 兼容性检查](https://github.com/sgl-project/sglang/blob/30dc2f94303e339431cbd51489b2d1719678b0c1/python/sglang/srt/arg_groups/kv_cache_hook.py)
- [LMCache 接入说明](https://github.com/sgl-project/sglang/blob/30dc2f94303e339431cbd51489b2d1719678b0c1/python/sglang/srt/mem_cache/storage/lmcache/README.md)

## 文档与实验环境

这些文档是仓库内独立的中文学习资料，不依赖官方文档站点构建。

- 教材间链接使用资料库相对路径，源码与上游文档链接固定到相应提交，适合在 IDE 或 GitHub 中浏览。
- Mermaid 图需要支持 Mermaid 的 Markdown 查看器；不支持时仍可阅读图定义与相邻说明。
- 数学公式需要支持 LaTeX 数学渲染的查看器。
- 源码导航主要链接到文件和函数名，避免随源码变更而快速失效的固定行号。

当前 macOS 环境适合阅读、计算练习、图示和报告编写。本文 GPU 实验统一按 Linux + NVIDIA 环境设计；涉及 RDMA 的性能实验需要相应网卡、网络和驱动。

实验中的服务只应在隔离的学习环境运行。SGLang 和 Router 示例绑定回环地址；Mooncake 等辅助服务的监听范围需要另行检查，不要把无认证的管理或缓存接口暴露到公网。

## 推荐学习习惯

每读一个模块，回答六个问题：

1. 输入是什么？
2. 输出是什么？
3. 它持有什么状态？
4. 谁可以修改这些状态？
5. 哪个同步边界保证数据可用？
6. 什么时候可以安全释放资源？

每做一次优化，保存四样东西：基线、单一变量改动、端到端指标、解释结果的源码或 trace 证据。

最终验收见 [09：参考资料与验收](../../../08-career/implementations/sglang/assessment.md)。
