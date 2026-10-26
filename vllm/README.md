# vLLM 与推理 Infra 学习手册

这套文档面向希望理解 vLLM、开展性能优化并准备推理 Infra 面试的工程师。主线是：**模型计算 → GPU 执行 → KV 管理 → 请求调度 → 分布式缓存与服务 → 性能验证**。

学习周期已压缩为 **28 天，每天约 3 小时**。先读下面四本教材正文，再用知识地图、KV 专题和面试题巩固。源码教材已核对本地提交，实践教材附带无 GPU 也能运行的实验。

## 从这里开始学习

| 教材 | 可以直接学到什么 | 对应计划 |
| --- | --- | --- |
| [07 推理基础](07-inference-foundations.md) | Transformer shape、Prefill/Decode、KV、显存、Roofline、指标及计算题 | D01～D04 |
| [08 vLLM 源码精读](08-vllm-source-walkthrough.md) | 入口、请求状态、EngineCore、Scheduler、KV、packed inputs、MRV2、采样、connector | D05～D12 |
| [09 性能与分布式](09-performance-distributed-serving.md) | Benchmark、profiling、FlashAttention、kernel、编译、并行、PD、量化和服务工程 | D13～D24 |
| [10 实践与参考答案](10-practice-and-solutions.md) | CPU 实验、GPU 安装/使用/压测步骤、源码作业、计算题和月末报告 | 全程配套 |

CPU 教学脚本：[inference_lab.py](labs/inference_lab.py)，含 Attention/KV 等价、链式前缀 hash、调度、容量四个实验。它不依赖 vLLM 或 PyTorch，不替代真实 GPU benchmark。

## 文档导航

| 文档 | 内容 | 推荐用法 |
| --- | --- | --- |
| [01 学习大纲与计划](01-study-plan.md) | 28 天逐日任务、阅读章节、月内选修、岗位分支和验收标准 | 每天执行 |
| [02 知识地图与设计图](02-knowledge-map-and-architecture.md) | 核心知识地图、V1 架构、启动流程、请求时序、状态机、调度与部署图 | 建立整体认识，读源码时对照 |
| [03 面试问题与参考回答](03-interview-qa.md) | 原理、定量分析、引擎、GPU、分布式、缓存、系统设计与排障 | 先独立回答，再看参考答案 |
| [04 KV Cache：vLLM、LMCache、Mooncake](04-kv-cache-vllm-lmcache-mooncake.md) | 内存模型、复用正确性、多级缓存、传输、PD 分离、框架对比、故障处理 | 重点专题，建议精读 |
| [05 实验与项目](05-experiments.md) | 基线、调度、缓存、PD、故障实验与结果模板 | 将知识转成可展示的项目 |
| [06 官方资料与版本说明](06-references.md) | 同版本本地官方资料、论文、源码定位与版本记录 | 核对实现和兼容性 |

## 三条阅读路径

- **28 天完整路径**：按 01 的每日表阅读 07 → 08 → 09，贯穿 10 的作业，再用 04、03 专项复习。
- **面试路径**：07 的计算 → 08 的链路 → 09 的取舍 → 03 问答 → 10 的项目报告。
- **KV Cache 专项路径**：07 的 F4～F6 → 08 的 R6/R10 → 04 全文 → 10 的 L6 → 05 的 E4～E9。

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
