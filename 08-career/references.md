# 按问题查资料：专题索引与版本基线

[返回主题](README.md) · [岗位路线](roles.md) · [学习计划](study-plan.md)

## 1. 新结构与原资料的关系

全库按基础、运行时、KV 管理、存储、传输、Serving、实验和职业学习八个主题组织。每个主题的 `README.md` 讲共同问题和取舍；`implementations/<项目>/` 放版本相关的实现解释、源码和案例。原有教材已迁移到主要问题所属主题，文中的旧章节号和题号保留作定位标识，不表示全库阅读顺序。

共同概念以[公共基础](../01-foundations/concepts.md)作为首次阅读入口。出现框架差异时回到对应实现和源码，不把共同介绍当作统一 API 规范。

## 2. 按问题找一份主读材料

| 你要解决的问题 | 主读材料 | 什么时候补另一份 |
|---|---|---|
| Attention、形状与推理成本 | [Attention 与推理](../01-foundations/attention-and-inference.md) | 存储预算再看 [KV 容量与复用](../03-kv-cache/capacity-and-reuse.md) |
| 请求怎样被调度和执行 | [运行时主题](../02-runtime/README.md) | 深读 [vLLM](../02-runtime/implementations/vllm/request-walkthrough.md)或 [SGLang](../02-runtime/implementations/sglang/runtime.md) |
| 分页、前缀树与状态池 | [KV 管理](../03-kv-cache/README.md) | 对照 [Radix/状态池](../03-kv-cache/implementations/sglang/prefix-cache.md)与 [block/集成](../03-kv-cache/implementations/vllm/integration.md) |
| GPU/host/远端怎么衔接 | [分层控制案例](../03-kv-cache/implementations/sglang/tiered-cache.md) | 独立服务路线看 [缓存服务内核](../03-kv-cache/implementations/lmcache/internals.rst) |
| lookup、staging、锁与完成 | [缓存服务内核](../03-kv-cache/implementations/lmcache/internals.rst) | 远端保护看 [Store 对象协议](../04-storage/implementations/mooncake/internals.md) |
| 目录、对象与 payload 的关系 | [存储主题](../04-storage/README.md) | 上层衔接看 [后端契约](../04-storage/implementations/lmcache/backend-contracts.rst) |
| RDMA、拓扑与传输完成 | [传输主题](../05-transfer/README.md) | 深入 [TE 实现](../05-transfer/implementations/mooncake/transfer-engine.md) |
| 多卡、kernel、CUDA Graph、量化 | [Serving 与性能](../06-serving/README.md) | 深入 [引擎性能案例](../06-serving/implementations/vllm/performance-and-parallelism.md)和[计算与内存](../01-foundations/compute-and-memory.md) |
| 压测、观测与性能归因 | [实验方法](../07-practice/README.md) | 深入 [缓存观测](../06-serving/implementations/lmcache/observability.rst)或 [运行时实验](../07-practice/implementations/sglang/labs.md) |
| 无 GPU 练习 | [CPU 练习与答案](../07-practice/implementations/vllm/walkthrough.md) | 缓存策略看 [对象与传输实验](../07-practice/implementations/mooncake/labs.md) |
| Linux TCP、TE、Store 实验 | [对象与传输实验](../07-practice/implementations/mooncake/labs.md) | 独立缓存服务集成看 [MP 实验](../07-practice/implementations/lmcache/labs.rst) |
| 从预算到系统设计 | [设计主线](interview-and-design.md) | 深入 [缓存服务设计](implementations/lmcache/system-design.rst)与 [共享存储设计](implementations/mooncake/system-design.md) |

设计图按主题归位：[引擎架构](../02-runtime/implementations/vllm/architecture.md)、[运行时知识地图](../02-runtime/implementations/sglang/knowledge-map.md)、[缓存状态图](../03-kv-cache/implementations/lmcache/state-diagrams.rst)、[存储与传输图](../04-storage/implementations/mooncake/architecture.md)。建议按题目自己重画。

## 3. 各方向面试题怎么裁剪

| 主方向 | 主题库 | 补充重点 |
|---|---|---|
| 引擎 | [vLLM 问答](implementations/vllm/questions.md)中的引擎、GPU、并行 | [SGLang](implementations/sglang/questions.md)的 Radix、重叠、HiCache |
| KV 存储 | [LMCache 问答](implementations/lmcache/questions.rst)中的生命周期、后端、设计 | [Mooncake](implementations/mooncake/questions.md)的读写、租约、传输 |
| 通信/GPU | vLLM 的执行与通信、Mooncake 的 TE | 补对象大小、拓扑、测量边界与异步失败 |
| 平台 | 四套中的 SLO、故障、容量与设计题 | 用自己的流量和发布场景作答 |

统一复习先过[面试主线](interview-and-design.md)。原题库中重复的容量、指标、前缀题不用重复安排四次。

## 4. 本次整理使用的源码基线

整理日期：2026-10-26。以下提交对应本地已有源码，用来核对关键路径；不是推荐的可任意组合部署版本。

| 项目 | 源码提交 | 本资料的主线 |
|---|---|---|
| vLLM | [`f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`](https://github.com/vllm-project/vllm/tree/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41) | V1、KV manager、Scheduler、原 runner/MRV2、Connector |
| SGLang | [`30dc2f94303e339431cbd51489b2d1719678b0c1`](https://github.com/sgl-project/sglang/tree/30dc2f94303e339431cbd51489b2d1719678b0c1) | Python runtime、Unified Cache、HiCache、MP 集成 |
| LMCache | [`ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be`](https://github.com/LMCache/LMCache/tree/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be) | MP、L1/L2、controller/adapter；进程内模式作版本对照 |
| Mooncake | [`dd1217b6c461491e7c73ff29b4ffad215e0a9ac0`](https://github.com/kvcache-ai/Mooncake/tree/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0) | C++ Store、经典 TE；其他后端作进阶入口 |

源代码和原项目文档的链接尽量固定到这些提交；全库内部导航用相对路径。旧文章引用的在线 `latest/stable` 页面仍可能变化。

### 跑实验前记录什么

```text
四个项目中实际使用者的 tag/commit、wheel、扩展构建方式
模型 revision、tokenizer/模板、LoRA、RoPE、KV dtype
TP/PP/DP、group、block/chunk size、layout
Connector 类、MP/进程内、Store 后端和传输后端
GPU、CPU、主机内存、NUMA、NIC、驱动与运行时
各层容量、在途限制、复制、准入、预取/等待策略
负载、冷热状态、正确性口径与 SLO
```

“LMCache 支持 Mooncake”不能替代这张兼容性表。尤其是原生扩展、Connector callback、布局和参数名，跨版本可能变化。

## 5. 按问题读论文

| 资料 | 要带走的问题 |
|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Q/K/V 如何参与计算；与当前 decoder-only serving 有何边界 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 为什么访存而不只是 FLOPs 决定性能 |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 动态 KV 如何减少浪费并共享 |
| [SGLang](https://arxiv.org/abs/2312.07104) | 多次调用的前缀结构怎样进入运行时设计 |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked Prefill 怎样改变延迟与吞吐关系 |
| [DistServe](https://arxiv.org/abs/2401.09670) | 为什么围绕 goodput 和 SLO 做 PD 资源分配 |
| [Mooncake](https://arxiv.org/abs/2407.00079) | 缓存、通信、调度与过载准入怎样一起设计 |

读完每篇写四句话：瓶颈是什么、用什么换掉哪项成本、实验在哪些条件下成立、什么场景会失效。再到当前代码核对，避免把论文原型等同于今天的完整实现。

## 6. 资料维护约定

新增内容先判断它解决哪个技术问题：共同概念放对应主题，实现细节放该主题的 `implementations/<项目>/`。同一篇内容只保留一个主要位置，其他主题通过链接引用；不要恢复框架级顶层目录。新功能写明版本和支持边界，新实验注明推导、模拟或实测。

新增链接时，在 `Infra` 目录中实际检查能否打开。引用上游文件用提交固定链接，不沿用从原项目复制过来的 `../../源码目录`。新同学应能从根目录开始完成阅读，而不需要先拥有相同的本机源码布局。
