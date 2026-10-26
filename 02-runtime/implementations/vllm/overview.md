# vLLM：请求怎样变成高效的 GPU 执行

[返回运行时主题](../../README.md) · [前置基础](../../../01-foundations/concepts.md) · [同主题对照：SGLang](../sglang/overview.md)

vLLM 的核心工作是：**把许多处于不同生成阶段的请求，组织成 GPU 能高效执行的任务，同时管好 KV 和请求生命周期。** 它既是推理服务系统，也是一条学习调度、内存、kernel 和缓存集成的完整路径。

## 1. 从一个实际问题开始

假设同时来了三个请求：A 已经开始生成，B 刚提交 100-token 输入，C 提交了 20,000-token 长文档。如果每次只处理一个请求，GPU 可能用不满；如果 C 一次算完才服务 A，A 的流式输出会卡住。

vLLM 需要每轮决定：A 生成多少新 token，B/C 各处理多少输入，显存能否容纳，哪些前缀已算过，是否需要等待外部 KV。

这里的 batch 不是用户提交的固定 Python list。**Continuous Batching** 允许请求完成后退出，新请求在后续轮次进入；活跃集合会持续变化。

## 2. 先理解五个主要角色

| 角色 | 白话解释 | 读代码时要追的问题 |
|---|---|---|
| 入口与 Engine Client | 把输入变成引擎请求，接收输出 | 请求何时进入队列、取消消息如何到达？ |
| EngineCore | 驱动每一轮调度和执行 | 是否同步等待，哪里维护在途 batch？ |
| Scheduler | 决定本轮推进哪些请求、多少 token | token、请求数、KV 容量分别怎样限制？ |
| KVCacheManager / BlockPool | 管 GPU KV 的分配、命中、引用、回收 | 逻辑位置怎么映射物理块，哪些块不能淘汰？ |
| Worker / Model Runner | 准备 GPU 输入并执行模型 | 索引与布局如何传给 attention，何时取回结果？ |

```text
请求
→ 输入处理 / Engine Client
→ EngineCore
→ Scheduler ↔ KVCacheManager
→ Worker / Model Runner
→ 模型层与 attention backend
→ 采样 / 状态更新 / 输出返回
```

这是一张职责图。真实部署可能跨进程，异步路径也不保证严格逐步等待。当前基线的 V1 引擎有原 runner 与 Model Runner V2 两条执行器路径；MRV2 不是另一套“vLLM V2 引擎”。

## 3. 四个常见优化分别解决什么

### PagedAttention：让 KV 不必占一整块连续显存

KV 按 block 存放，attention 通过映射找到所需位置。请求增长时追加块，结束后归还可回收块；共享前缀可以引用同一组只读块。

收益是降低碎片和过度预留、支持更灵活的共享。代价包括 block table、索引处理与适配这种布局的 kernel。它没有把 GPU 变成无限内存。

### Prefix caching：以前算过的前缀尽量不重算

同样模型和兼容配置下，链式块哈希用于识别可复用的完整前缀。匹配成功并满足可用性条件后，调度器减少需要重新计算的输入范围。

这与分页是两件事：分页解决“放在哪里”，前缀缓存解决“是不是已经算过”。只讨论分页无法解释缓存命中的正确性。

### Chunked Prefill：别让一个长输入长时间挡住其他请求

将长 Prefill 分成多轮推进。教学例子：token 预算为 8，A/B 各做一个 Decode token，C 本轮处理 6 个输入 token。下一轮再决定 C 的剩余工作。

块越小不一定越好。过小会降低 kernel 效率、增加调度开销；过大可能伤害 Decode ITL。真实策略还受优先级、预算、模型与 KV 资源影响。

### FlashAttention：减少 attention 中间结果的显存访问

FlashAttention 通过分块和在线 softmax 等方法，减少大中间张量的 HBM 读写。它处理一次 attention 计算内部的效率。它与请求调度、分页管理、跨请求复用可以同时存在。

## 4. 一次外部缓存命中怎样接回执行

假设输入长 8,192 token，外部缓存可能覆盖前 6,144 token：

1. Scheduler/Connector 确认本地已命中的范围，查询额外可复用范围。
2. 为目标 KV 准备合适的 GPU block，并传递 block、请求、rank 等元数据。
3. Worker 侧通过 Connector 触发加载，处理传输、重排和同步依赖。
4. 确认实际可用范围后，按协议只计算未覆盖输入。不能把“查询返回命中”直接写成“这些 token 全部完成”。
5. 新 KV 的保存可能异步进行。请求结束以后，必须等仍引用相关 block 的保存任务满足释放条件。

Connector 因而是执行协议的一部分，涉及调度器侧和 worker 侧。写一个 `put/get` 适配器并不代表已经正确接入 vLLM。

**错误例子**：外部查到 6,144 token，加载第 4,096 token 后网络失败，却仍按 6,144 token 跳过 Prefill。Store 可能只是返回了正确的失败状态，但引擎已经消费了缺失或旧数据。

## 5. 什么时候重点优化 vLLM 本身

| 现象 | 优先检查 | 需要的证据 |
|---|---|---|
| 低并发很快，高并发首字慢 | 排队、准入、token 预算、长 Prefill | 请求时间线、等待队列、Prefill 分布 |
| 显存够用但 GPU 有空洞 | CPU 准备、同步、执行器、批次组织 | CPU/GPU trace |
| 上下文越长 Decode 越慢 | KV 读取、attention kernel、batch、并行通信 | 固定 batch/长度的前向测量 |
| 开远端缓存后偶尔错误 | key、布局、block 生命周期、加载失败处理 | 对照输出、分片状态、完成事件 |
| 频繁抢占/重算 | KV 预算、并发、输出长度、共享情况 | 抢占率、活跃 token 数与 block 使用 |

不要同时修改十个参数。先确定一项瓶颈，建立基线，再做单因素对照。

## 6. 源码怎么读

以下路径相对于 vLLM 上游源码根目录，具体链接与逐段解释见[源码教材](request-walkthrough.md)。

| 顺序 | 路径 | 阅读产物 |
|---|---|---|
| 1 | `vllm/entrypoints/llm.py` | 一次离线生成怎样进入引擎 |
| 2 | `vllm/v1/engine/core.py` | 一轮调度、提交、等待、更新的图 |
| 3 | `vllm/v1/core/sched/scheduler.py` | 请求状态和资源预算表 |
| 4 | `vllm/v1/core/kv_cache_manager.py`、`block_pool.py` | 匹配、分配、引用与释放路径 |
| 5 | `vllm/v1/worker/gpu_worker.py` 及所选 runner | 输入映射到 GPU tensor 的过程 |
| 6 | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | 外部缓存与调度、执行的契约 |

推理引擎岗要能解释整轮执行；KV 存储岗优先读 3、4、6，同时知道 GPU 数据最终怎样被消费。Kernel 岗再深入 backend 和具体算子。

## 7. 面试里怎样算“掌握了”

能手推一个同时有 Prefill 和 Decode 的小调度例子；能画逻辑 token 到物理 block 的映射；能解释抢占、取消、异步保存时为什么不能立即复用内存；能把一次 TTFT 变慢拆成可观测阶段。

一句项目介绍可以这样说：“我主要从 vLLM 的请求执行链路理解调度与 KV 管理，再跟踪 Connector 的加载和完成契约。重点验证的是减少了多少实际 Prefill，以及是否引入额外等待或错误回收。”接下来要用自己的源码记录或实验支撑。

## 专题深读资料

下面保留原有教材导航，表内编号用于对应原教材。按主题学习使用[统一计划](../../../08-career/study-plan.md)；需要持续深入 vLLM 时再使用[专项计划](../../../08-career/implementations/vllm/study-plan.md)。

### 四本深读教材

| 教材 | 可以直接学到什么 | 对应计划 |
| --- | --- | --- |
| [07 推理基础](../../../01-foundations/attention-and-inference.md) | Transformer shape、Prefill/Decode、KV、显存、Roofline、指标及计算题 | D01～D04 |
| [08 vLLM 源码精读](request-walkthrough.md) | 入口、请求状态、EngineCore、Scheduler、KV、packed inputs、MRV2、采样、connector | D05～D12 |
| [09 性能与分布式](../../../06-serving/implementations/vllm/performance-and-parallelism.md) | Benchmark、profiling、FlashAttention、kernel、编译、并行、PD、量化和服务工程 | D13～D24 |
| [10 实践与参考答案](../../../07-practice/implementations/vllm/walkthrough.md) | CPU 实验、GPU 安装/使用/压测步骤、源码作业、计算题和月末报告 | 全程配套 |

CPU 教学脚本：[inference_lab.py](../../../07-practice/cpu/inference_lab.py)，含 Attention/KV 等价、链式前缀 hash、调度、容量四个实验。它不依赖 vLLM 或 PyTorch，不替代真实 GPU benchmark。

### 配套资料导航

| 文档 | 内容 | 推荐用法 |
| --- | --- | --- |
| [01 学习大纲与计划](../../../08-career/implementations/vllm/study-plan.md) | 28 天逐日任务、阅读章节、月内选修、岗位分支和验收标准 | 每天执行 |
| [02 知识地图与设计图](architecture.md) | 核心知识地图、V1 架构、启动流程、请求时序、状态机、调度与部署图 | 建立整体认识，读源码时对照 |
| [03 面试问题与参考回答](../../../08-career/implementations/vllm/questions.md) | 原理、定量分析、引擎、GPU、分布式、缓存、系统设计与排障 | 先独立回答，再看参考答案 |
| [04 KV Cache：vLLM、LMCache、Mooncake](../../../03-kv-cache/implementations/vllm/integration.md) | 内存模型、复用正确性、多级缓存、传输、PD 分离、框架对比、故障处理 | 重点专题，建议精读 |
| [05 实验与项目](../../../07-practice/implementations/vllm/experiments.md) | 基线、调度、缓存、PD、故障实验与结果模板 | 将知识转成可展示的项目 |
| [06 官方资料与版本说明](../../../08-career/implementations/vllm/references.md) | 同版本本地官方资料、论文、源码定位与版本记录 | 核对实现和兼容性 |

### 单项目的三条深读路径

- **vLLM 专项路径**：基础计算 → 请求源码 → 性能与分布式，穿插实践作业，再用缓存专题和问答复习；整体学习仍按全库统一计划。
- **面试路径**：基础教材的计算 → 源码链路 → 性能取舍 → 问答 → 项目报告。
- **KV Cache 专项路径**：基础教材 F4～F6 → 源码教材 R6/R10 → KV 专题 → 实践 L6 → 实验 E4～E9。

## 学完需要能够做什么

1. 画出一条请求从 API 到 GPU、再返回客户端的执行过程。
2. 估算权重与 KV 显存，解释 batch、序列长度和并行方式对容量的影响。
3. 解释 Continuous Batching、Chunked Prefill、PagedAttention、FlashAttention 的不同作用。
4. 用负载与 trace 证明瓶颈，并以延迟目标下的 Goodput 评价优化。
5. 区分 vLLM 本地前缀缓存、LMCache 多级复用、Mooncake 传输与共享存储。
6. 解释缓存何时可复用、何时可释放，以及读取失败后如何安全回退。
7. 完成至少一份包含基线、假设、实验、收益边界和正确性验证的报告。

## 使用约定

- 内容以 **vLLM V1 的架构概念**为主。图中的箭头表示职责与数据依赖，不保证等同于任一版本的逐行调用关系。
- Mermaid 图可在 GitHub 或支持 Mermaid 的 Markdown 阅读器中渲染；原始图代码也可直接编辑。
- `P0`：必须掌握；`P1`：进阶必学；`P2`：按岗位选学。
- 面试题属于常见考察主题整理，不代表任何公司的真实题库或出现频率统计。
- “参考回答”包含前提与边界。性能问题通常没有脱离硬件、模型、负载和 SLO 的固定答案。
- CPU 教学实验可以在本机运行；GPU/分布式实验中的数值示例是计算示例或待验证的假设，不能当作实测收益。

## 资料边界

首次整理日期：2026-10-23。本次补充时本地源码已经可读，源码教材核对基准为 `f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`。已核对入口、引擎、调度、KV、runner 和 connector 的关键符号；切换提交后需重新确认实现。

当前快照存在 V1 引擎中的原 runner 与 Model Runner V2 两条路径，08 专门解释了区别。LMCache/Mooncake 部分依据所列官方资料，未在本机验证所有版本组合。

在线 `stable`、`latest` 和 LMCache 开发文档会更新。尤其是 LMCache 的进程内模式与 MP 模式、vLLM connector 接口、Mooncake 后端能力，不能跨版本混用配置。参考文档中的性能数字也不能直接当作自己的收益预期。
