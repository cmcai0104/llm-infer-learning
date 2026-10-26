# 参考资料、实验报告与最终验收

[返回导航](README.md) · [学习计划](02-28-day-plan.md) · [面试问答](08-interview-qa.md)

## 1. 阅读方法

读论文先回答“问题是什么、优化了哪项成本、假设是什么、什么场景会失效”；读源码再确认“当前版本怎样实现、有哪些组合限制”。

论文、官方文档与本地源码可能属于不同时间点。涉及参数、默认值与支持矩阵时，优先核对实际执行版本。

以下链接来自前期官方资料检索与仓库引用；文档结构检查不等于对外部网站内容进行持续验证。

## 2. 推荐顺序

| 顺序 | 资料 | 阅读问题 |
|---|---|---|
| 1 | [SGLang 论文](https://arxiv.org/abs/2312.07104) | 多次生成调用为什么产生复用机会？RadixAttention 解决什么？ |
| 2 | [PagedAttention / vLLM 论文](https://arxiv.org/abs/2309.06180) | 动态 KV 如何减少碎片并支持共享？ |
| 3 | [SGLang Benchmark 与 Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling) | Serving、offline、固定 batch 的测量边界？ |
| 4 | [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefill 如何改变吞吐与延迟？ |
| 5 | [HiCache 系统设计](https://docs.sglang.io/docs/advanced_features/hicache_design) | 匹配、预取、回写和多 rank 同步如何配合？ |
| 6 | [HiCache 最佳实践](https://docs.sglang.io/docs/advanced_features/hicache_best_practices) | 如何选择容量、预取和后端？ |
| 7 | [Mooncake 架构](https://kvcache-ai.github.io/Mooncake/design/architecture.html) | TE、Store 与其他组件的职责边界？ |
| 8 | [Mooncake Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine/index.html) | Segment、注册内存和异步传输如何工作？ |
| 9 | [Mooncake Store](https://kvcache-ai.github.io/Mooncake/design/mooncake-store.html) | 对象、元数据、副本和数据传输如何分离？ |
| 10 | [LMCache 集成](https://docs.lmcache.ai/developer_guide/integration.html) / [Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) | 当前 connector、服务模式与匹配版本？ |
| 11 | [LMCache Mooncake 后端](https://docs.lmcache.ai/mp/l2_storage/mooncake_store.html) | 两个框架如何组合，RDMA 有何内存条件？ |
| 12 | [CacheBlend](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html) | 非前缀复用为什么要处理计算依赖？ |
| 13 | [DistServe](https://arxiv.org/abs/2401.09670) | 如何围绕 Goodput 和 SLO 选择 PD 与资源？ |
| 14 | [Mooncake 论文](https://arxiv.org/abs/2407.00079) | KV-centric 调度、资源池和过载控制如何协同？ |
| 15 | [SGLang PD 文档](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) | 论文思想在当前工程中的约束是什么？ |

补充官方入口：

- [SGLang 安装](https://docs.sglang.io/docs/get-started/install)
- [LMCache 多进程存储层](https://docs.lmcache.ai/mp/l2_storage.html)
- [SGLang 项目](https://github.com/sgl-project/sglang)
- [Mooncake 项目](https://github.com/kvcache-ai/Mooncake)
- [LMCache 项目](https://github.com/LMCache/LMCache)

## 3. 仓库内事实依据

| 主题 | 本地入口 |
|---|---|
| 缓存模块分层 | [mem_cache README](../../python/sglang/srt/mem_cache/README.md) |
| 当前默认缓存构造 | [registry.py](../../python/sglang/srt/mem_cache/registry.py) |
| Unified Cache 组件 | [组件说明](../../python/sglang/srt/mem_cache/unified_cache/components/README.md) |
| HiCache 层级与策略 | [设计文档](../../docs/docs/advanced_features/hicache_design.mdx) |
| Mooncake 配置与持有者 | [接入说明](../../python/sglang/srt/mem_cache/storage/mooncake_store/README.md) |
| LMCache MP 使用方式 | [接入说明](../../python/sglang/srt/mem_cache/storage/lmcache/README.md) |
| LMCache 当前限制 | [kv_cache_hook.py](../../python/sglang/srt/arg_groups/kv_cache_hook.py) |
| 统一缓存 key | [hicache_key_scheme.py](../../python/sglang/srt/mem_cache/hicache_key_scheme.py) |
| PD 分配策略 | [PD 文档](../../docs/docs/advanced_features/pd_disaggregation.mdx) |
| Benchmark 测量边界 | [Benchmark 指南](../../docs/docs/developer_guide/benchmark_and_profiling.mdx) |
| 测试组织 | [test README](../../test/README.md) |

## 4. 论文笔记模板

```text
论文 / 版本：
问题：
基线：
主要瓶颈：
核心方法：
假设与适用范围：
正确性条件：
实验 workload：
指标与硬件：
主要结果：
潜在反例：
与当前 SGLang 源码的对应：
仍需自己验证的结论：
```

不要直接把论文中的峰值收益写成自己的实验结论。

## 5. 实验报告模板

### 5.1 问题与假设

描述一个可证伪的假设，例如：

> 在具有较长共享前缀的负载下，外部缓存加载的关键路径成本低于重算前缀的成本，从而改善 TTFT。

同时描述反例：低复用率、短输入、网络慢、并发争用或热点变化。

### 5.2 环境

| 项目 | 实际值 |
|---|---|
| 日期、操作者 | 待填写 |
| SGLang commit / wheel | 待填写 |
| PyTorch、CUDA、driver | 待填写 |
| LMCache、Mooncake、Router | 待填写 |
| 模型 ID、revision、dtype | 待填写 |
| GPU、CPU、RAM、NUMA | 待填写 |
| NIC、网络、传输后端 | 待填写 |
| 并行策略 | 待填写 |

### 5.3 Workload

- 输入/输出长度分布。
- 实际输出长度与 EOS 行为。
- 到达率、并发上限、总请求数。
- 共享前缀比例、长度与热点分布。
- 冷/热缓存构造方式。
- 模型、JIT、Graph 的预热方式。
- 测试重复次数与随机种子。

### 5.4 配置

保留完整启动和压测命令，以及：

- GPU KV 容量。
- Host/外部缓存实际容量。
- Page/chunk 大小。
- 写策略、预取策略与超时。
- Parallel sizes、Graph、overlap。

### 5.5 结果

| 配置 | TTFT p50/p95/p99 | ITL p95/p99 | Output tok/s | Goodput | GPU/Host 内存 | 错误/拒绝 |
|---|---|---|---:|---:|---|---|
| 基线 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 实验组 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

补充缓存证据：

- 各层命中 token 数。
- 真正少计算的 token 数。
- 查到但未使用的缓存。
- 查询、传输、H2D 和同步耗时。
- 每请求传输字节。
- 网络与 Host 内存带宽。

### 5.6 正确性

- 与无缓存/无优化路径比较。
- 按场景检查输出、logprob 或任务质量。
- 检查模板、token、位置、LoRA、dtype/layout 和模型版本。
- 记录取消、超时、重启和失败行为。

### 5.7 解释与局限

说明为什么指标变化，提供源码或 trace 证据。列出未覆盖的硬件、模型、负载和失败场景。

Profiler 有开销，正式性能结论应用非 profiling 运行确认。

## 6. 最终交付物

1. 一张请求架构图和一张运行时序图。
2. 一张 KV 索引、物理池与生命周期图。
3. 一张 HiCache / Mooncake / LMCache 选型图。
4. 一份容量与传输成本计算。
5. 至少一份完整对照实验报告。
6. 一份系统设计方案，包含 SLO、容量、路由、隔离与失败恢复。
7. 一份能够口头讲述的源码路线。

## 7. 评分标准

| 维度 | 分值 | 达标表现 |
|---|---:|---|
| 原理 | 20 | 解释 Prefill/Decode、性能模型和优化成本 |
| 源码 | 20 | 跟踪请求、调度、执行和缓存生命周期 |
| 实验 | 25 | 有控制变量、正确性、分位数指标与证据 |
| KV 专题 | 20 | 容量、共享、分层、兼容性与传输判断准确 |
| 系统设计 | 15 | 方案包含资源模型、SLO、失败路径和取舍 |

建议总分达到 80，并且不能在正确性上存在关键误解，例如错误跨模型复用 KV、提前释放在途 buffer、把外部查询命中当作设备就绪。

## 8. 五项口头验收

- 10 分钟讲清一次请求，以及 Prefill、Decode、分页、前缀复用。
- 从入口跟到 Scheduler、ModelRunner 和缓存。
- 独立计算给定模型的 KV 体积和传输下界。
- 用实验解释一个优化为什么有效，以及何时失效。
- 比较 HiCache + Mooncake、LMCache 和直接重算的适用条件。

## 9. 简历与面试表达

推荐：

> 针对某模型和指定工作负载，定位了某瓶颈，设计了某对照实验；在固定硬件与 SLO 下观察到实际变化，并分析了适用范围与退化条件。

填写实际数据，不使用没有运行过的百分比或倍数。不把“阅读过源码”“设计了方案”写成“上线过系统”。

## 10. 后续学习

完成首月后，按岗位选择一条深入：

- Runtime：复杂调度、公平性、取消/回退、线上回放。
- KV：跨拓扑共享、混合状态、压缩、缓存感知路由。
- 通信：RDMA、NCCL、MoE All-to-All、拓扑与容错。
- Kernel：Attention、融合、量化、CUDA/Triton 与硬件计数器。
- 平台：多租户、容量预测、弹性、灰度和故障演练。
