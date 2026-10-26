# 官方资料与版本说明

[返回导航](../../../02-runtime/implementations/vllm/overview.md)

## 1. 如何使用这些资料

资料检索日期为 2026-10-23。下列页面是学习依据，不代表已经验证某个组合能够部署运行。`stable` 和不带版本的站点会变化；实验必须固定 release/commit，并保存与之匹配的配置。

本文档中的“现有能力”指所链接官方文档中介绍的能力。兼容性仍取决于模型、GPU 后端、attention backend、KV dtype/layout、并行拓扑、connector 与框架版本。

## 2. vLLM

| 编号 | 资料 | 阅读重点 |
| --- | --- | --- |
| V1 | [Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview/) | API、EngineCore、Worker 进程职责 |
| V2 | [Scheduler Interface](https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/interface/) | 每轮请求到 token 数的调度结果 |
| V3 | [Optimization and Tuning](https://docs.vllm.ai/en/stable/configuration/optimization/) | Chunked Prefill、抢占、参数影响 |
| V4 | [Automatic Prefix Caching Design](https://docs.vllm.ai/en/stable/design/prefix_caching/) | 链式 hash、block pool、引用、驱逐 |
| V5 | [Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/) | Full/SWA/Mamba 等不同缓存需求 |
| V6 | [KVConnectorBase V1](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/base/) | scheduler/worker 接口、异步完成 |
| V7 | [Disaggregated Prefilling](https://docs.vllm.ai/en/stable/features/disagg_prefill/) | PD 目标、connector、实验性与兼容性 |
| V8 | [MooncakeStoreConnector](https://docs.vllm.ai/en/stable/features/mooncake_store_connector_usage/) | 共享 Store 与 P2P connector 的区别 |
| V9 | [Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) | TP/PP 与部署 |
| V10 | [Data Parallel Deployment](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/) | 稠密副本与 MoE DP/EP 的不同耦合 |
| V11 | [Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/) | 实际指标和命中统计口径 |
| V12 | [Benchmark Serving CLI](https://docs.vllm.ai/en/stable/cli/bench/serve/) | 到达率、并发、分位数、Goodput |
| V13 | [LMCache Connector API](https://docs.vllm.ai/en/stable/api/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector/) | 加载/保存 hook 与 graph 约束 |

源码定位：[vllm-project/vllm](https://github.com/vllm-project/vllm)。本次已核对本地提交 [`f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`](https://github.com/vllm-project/vllm/tree/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41) 的关键路径，详见 [08 源码教材](../../../02-runtime/implementations/vllm/request-walkthrough.md)。本地文件链接与在线 `stable` 不一定是同一版本。

## 3. LMCache

| 编号 | 资料 | 阅读重点 |
| --- | --- | --- |
| L1 | [Architecture Overview](https://docs.lmcache.ai/developer_guide/architecture.html) | 多级缓存与引擎集成的整体概念 |
| L2 | [Integration](https://docs.lmcache.ai/developer_guide/integration.html) | lookup、retrieve、store |
| L3 | [Multiprocess Overview](https://docs.lmcache.ai/mp/) | 独立服务、共享 L1、传输接口 |
| L4 | [MP Architecture](https://docs.lmcache.ai/mp/architecture.html) | server、storage manager、状态/锁、控制器 |
| L5 | [Secondary KV Storage](https://docs.lmcache.ai/mp/l2_storage/index.html) | L1/L2、异步写入与预取 |
| L6 | [MP Mooncake Store Backend](https://docs.lmcache.ai/mp/l2_storage/mooncake_store.html) | Mooncake 作为 L2 后端、构建和内存要求 |
| L7 | [Chunk-Size Negotiation](https://docs.lmcache.ai/mp/chunk_size_negotiation.html) | chunk 与引擎 KV 几何的对齐 |
| L8 | [CacheBlend](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html) | 非前缀复用与选择性重计算 |
| L9 | [Layerwise Transfer — Legacy](https://docs.lmcache.ai/legacy/kv_cache_optimizations/layerwise.html) | 层级流水原理；不是 MP 配置手册 |
| L10 | [MP Configuration](https://docs.lmcache.ai/mp/configuration.html) | 实验所选版本的参数入口 |
| L11 | [MP Tracing](https://docs.lmcache.ai/mp/observability/traces.html) | L1/L2/APC 命中分母与阶段延迟 |

源码定位：[LMCache/LMCache](https://github.com/LMCache/LMCache)。

部分进程内模式页面已经提示 deprecated，并推荐 MP。MP 文档中也存在开发分支要求，不应只看到“推荐”就假设已安装 release 支持所有特性。本手册主要解释稳定概念，不混抄两套配置。

MP 的 L1/L2 是 LMCache 的分层命名，**不等同于 GPU 芯片的 L1/L2 硬件缓存**。远端或 L2 后端是否持久化，必须看具体介质和恢复机制，不能仅由层级名称推断。

## 4. Mooncake

| 编号 | 资料 | 阅读重点 |
| --- | --- | --- |
| M1 | [Architecture](https://kvcache-ai.github.io/Mooncake/design/architecture.html) | 引擎之下的通信与存储基础设施 |
| M2 | [Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html) | Segment、注册 Buffer、BatchTransfer、Transport |
| M3 | [Mooncake Store](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html) | 不可变对象、Put/Get、replica、lease、驱逐 |
| M4 | [Metadata Management](https://kvcache-ai.github.io/Mooncake/design/store/metadata-management.html) | Master 元数据面与客户端数据面 |
| M5 | [vLLM KV Storage and Sharing](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/kv-cache-storage.html) | 直接使用 Store 的集成 |
| M6 | [vLLM V1 PD Integration](https://kvcache-ai.github.io/Mooncake/deployment/integrations/vllm/vllm-integration-v1.0.html) | MooncakeConnector 与 PD 传输 |

源码定位：[kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)。

Mooncake 论文描述的是 KV 中心化的完整 serving 架构，包括全局调度和过载决策。开源 Transfer Engine 或 Store 是其中可组合的基础设施，单独接入它们不等于复现论文的全套调度策略。

## 5. 核心论文

| 编号 | 论文 | 要回答的问题 |
| --- | --- | --- |
| P1 | [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | 为什么分页与共享可以提高有效 batch？ |
| P2 | [DistServe](https://arxiv.org/abs/2401.09670) | 为什么以延迟目标下的 Goodput 优化 PD 资源配置？ |
| P3 | [Mooncake](https://arxiv.org/abs/2407.00079) | 如何围绕 KV 位置、复用与过载组织服务？ |
| P4 | [CacheBlend](https://arxiv.org/abs/2405.16444) | 非前缀文档的 KV 为什么不能直接拼接？ |

论文结果来自指定模型、硬件和工作负载。记录其约束、基线、指标和消融实验，比记住加速倍数更有用。

## 6. 实验版本记录模板

```text
日期：
vLLM release / commit：
LMCache release / commit / in-process 或 MP：
Mooncake release / commit：
模型仓库 / revision / tokenizer / chat template：
GPU 型号与数量 / 显存 / GPU-NIC 拓扑：
OS / driver / CUDA / PyTorch / NCCL：
attention backend / dtype / KV dtype / KV layout：
TP / PP / DP / EP：
connector 名称与完整配置：
block size / 实际协商后的 chunk size：
CPU / NUMA / 主存与 pinned-memory 预算：
网络 / 介质 / 有效吞吐：
压测输入输出分布 / 并发或到达率 / SLO：
warmup / 缓存冷热 / 重复次数：
```

## 7. 与本地源码配套的官方阅读路线

下面资料已在对应版本的 vLLM 源码仓库确认存在，链接固定到该提交。先学 07～10 的中文正文，再核对实现、参数和扩展能力；不要求在 28 天内通读全部内容。

| 阅读顺序 | 上游固定版本资料 | 应带着的问题 |
| --- | --- | --- |
| 1 | [Quickstart](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/getting_started/quickstart.md) | 离线与在线最小使用方式是什么？ |
| 2 | [安装总览](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/getting_started/installation/README.md)、[环境准备](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/getting_started/installation/python_env_setup.inc.md) | 平台、Python 和 wheel 是否匹配？ |
| 3 | [架构概览](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/arch_overview.md) | API、Core、Worker 如何分工？ |
| 4 | [离线推理](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/offline_inference.md) | 批量输入和输出如何组织？ |
| 5 | [配置](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/configuration/README.md)、[调优](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/configuration/optimization.md) | token、请求数和显存预算怎样关联？ |
| 6 | [前缀缓存设计](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/prefix_caching.md)、[使用](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/automatic_prefix_caching.md) | key、引用和驱逐如何配合？ |
| 7 | [混合 KV](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/hybrid_kv_cache_manager.md) | 不同层型为什么不能统一套普通 KV 公式？ |
| 8 | [Attention backend](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/attention_backends.md)、[分页 Attention](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/paged_attention.md) | 数据布局和 kernel 怎样配合？ |
| 9 | [MRV2](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/model_runner_v2.md) | persistent state、异步和输入准备为什么重构？ |
| 10 | [编译](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/torch_compile.md)、[CUDA Graph](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/cuda_graphs.md) | 动态形状与捕获范围怎样影响收益？ |
| 11 | [Profiling](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/contributing/profiling.md)、[指标](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/metrics.md) | 怎样区分排队、执行和同步？ |
| 12 | [TP/PP](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/parallelism_scaling.md)、[DP](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/data_parallel_deployment.md) | 容量、吞吐和单请求延迟怎样权衡？ |
| 13 | [EP](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/expert_parallel_deployment.md)、[上下文并行](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/context_parallel_deployment.md) | 专家和上下文分别怎样切分？ |
| 14 | [PD](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/disagg_prefill.md) | 传输与资源分离什么时候值得？ |
| 15 | [Mooncake P2P](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/mooncake_connector_usage.md)、[Store](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/mooncake_store_connector_usage.md) | 一次交接与共享存储有什么区别？ |
| 16 | [量化](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/quantization/README.md)、[KV量化](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/quantization/quantized_kvcache.md) | 显存、kernel 和质量怎样一起验证？ |
| 17 | [投机解码](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/speculative_decoding/README.md) | 接受率和验证成本怎样决定收益？ |
| 18 | [模型接入](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/contributing/model/basic.md)、[注册](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/contributing/model/registration.md) | forward 和权重怎样接到引擎？ |
| 19 | [多模态](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/contributing/model/multimodal.md)、[LoRA](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/lora.md)、[结构化输出](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/features/structured_outputs.md) | 特殊输入与采样怎样改变执行状态？ |
| 20 | [分布式排障](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/serving/distributed_troubleshooting.md) | 多卡失败如何分离通信与模型问题？ |

基础原理论文可补充：

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Q/K/V 与多头注意力。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：分块、在线 softmax 与 IO 分析。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：精确投机采样的接受/拒绝修正。

本地教学代码：[CPU 四实验](../../../07-practice/cpu/inference_lab.py)。它用于推演概念，不是官方性能基准，也不是生产调度器。
