# SGLang 与推理 Infra：28 天学习手册

这套资料围绕一条主线展开：

**一次请求如何执行 → 请求如何调度 → KV Cache 如何管理 → 数据如何跨设备流动 → 如何解释性能 → 如何设计生产系统。**

适用读者：熟悉 Python 和基本深度学习，希望学习 SGLang，并准备推理 Infra 岗位面试。

建议投入：28 天，每天 2–3 小时，合计约 56–84 小时。重点覆盖自回归 LLM serving；多模态、LoRA、结构化输出、Diffusion 和底层 kernel 开发作为扩展方向。

## 1. 阅读导航

| 文档 | 内容 | 建议使用时间 |
|---|---|---|
| [01：知识地图](01-knowledge-map.md) | 学习大纲、依赖关系、岗位能力与优先级 | D1，总复盘时重读 |
| [02：28 天计划](02-28-day-plan.md) | 每日任务、交付物、每周验收与硬件替代方案 | 每天 |
| [03：架构与运行](03-architecture-and-runtime.md) | 进程架构图、请求时序、调度循环、源码阅读路线 | D4–D14 |
| [04：推理基础](04-inference-fundamentals.md) | Attention、性能模型、调度、GPU 优化、并行与生产工程 | D1–D10、D22–D26 |
| [05：KV Cache](05-kv-cache.md) | 容量计算、分页、Radix、生命周期、兼容性、混合模型 | D11–D16 |
| [06：HiCache、Mooncake 与 LMCache](06-hicache-mooncake-lmcache.md) | 分层缓存、共享存储、传输、框架对比、PD 分离与选型 | D15–D23 |
| [07：实验手册](07-hands-on-labs.md) | 七组实验、启动命令、缓存探测、对照设计与 profiling | 贯穿全程 |
| [08：面试问答](08-interview-qa.md) | 36 道题的回答要点、追问与系统设计演练 | D7 起穿插，D27 集中 |
| [09：参考资料与验收](09-references-and-assessment.md) | 论文、官方资料、实验报告模板、最终评分与项目讲述 | 按主题阅读，D28 验收 |

只想先建立整体认识，可以按 `01 → 03 → 05 → 06` 阅读；要实际执行学习计划，按 `02` 安排。

## 2. 一个月后的目标

完成学习后，应能够：

1. 从源码解释 SGLang 的请求生命周期，区分服务层、调度层、模型执行层和缓存层。
2. 计算权重、KV Cache、并行切分与传输的近似成本。
3. 解释 Continuous batching、Chunked prefill、CUDA Graph 和 CPU/GPU overlap。
4. 完成冷缓存、GPU 命中、CPU 命中、外部缓存命中实验，解释收益和退化。
5. 区分 Mooncake Transfer Engine、Mooncake Store、HiCache 和 LMCache 的职责。
6. 围绕 SLO、Goodput、容量和故障恢复设计一个推理服务。
7. 用源码与实际实验支持面试回答，明确区分理论推演和实测结果。

一个月的目标是形成可独立分析和验证的核心能力。所有模型、硬件后端和 CUDA kernel 的深度开发需要后续专题学习。

## 3. 版本依据与正确使用方式

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

- [缓存工厂与默认实现选择](../../python/sglang/srt/mem_cache/registry.py)
- [LMCache 兼容性检查](../../python/sglang/srt/arg_groups/kv_cache_hook.py)
- [LMCache 接入说明](../../python/sglang/srt/mem_cache/storage/lmcache/README.md)

## 4. 文档与实验环境

这些文档是仓库内独立的中文学习资料，不依赖官方文档站点构建。

- Markdown 链接使用仓库相对路径，适合在 IDE 或 GitHub 中浏览。
- Mermaid 图需要支持 Mermaid 的 Markdown 查看器；不支持时仍可阅读图定义与相邻说明。
- 数学公式需要支持 LaTeX 数学渲染的查看器。
- 源码导航主要链接到文件和函数名，避免随源码变更而快速失效的固定行号。

当前 macOS 环境适合阅读、计算练习、图示和报告编写。本文 GPU 实验统一按 Linux + NVIDIA 环境设计；涉及 RDMA 的性能实验需要相应网卡、网络和驱动。

实验中的服务只应在隔离的学习环境运行。SGLang 和 Router 示例绑定回环地址；Mooncake 等辅助服务的监听范围需要另行检查，不要把无认证的管理或缓存接口暴露到公网。

## 5. 推荐学习习惯

每读一个模块，回答六个问题：

1. 输入是什么？
2. 输出是什么？
3. 它持有什么状态？
4. 谁可以修改这些状态？
5. 哪个同步边界保证数据可用？
6. 什么时候可以安全释放资源？

每做一次优化，保存四样东西：基线、单一变量改动、端到端指标、解释结果的源码或 trace 证据。

最终验收见 [09：参考资料与验收](09-references-and-assessment.md)。
