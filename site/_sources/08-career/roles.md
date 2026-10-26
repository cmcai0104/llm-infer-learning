# 按岗位方向学习：什么必须熟练，什么可以后学

[返回主题](README.md) · [组件选型](../06-serving/component-selection.md) · [下一篇：统一计划](study-plan.md)

岗位名称并不统一，先看 JD 中实际职责：是改 Scheduler、做缓存后端、调 RDMA/kernel，还是维护集群服务？以下按工作内容分四条路线，可以主修一条、辅修一条。

本页用于方向分流。P7–P8 型专家的系统设计、交付与技术影响要求统一见[专家面试总纲](expert-interview-guide.md)，尤其注意 KV 存储的跨层协议、容量成本和故障闭环。

## 1. “熟练掌握”分四层

1. **能解释**：用自己的话说明为什么需要、解决哪项成本。
2. **能推演**：给一个请求、容量或失败场景，能手算和画状态变化。
3. **能定位**：能找到相关源码、指标和日志，验证自己的理解。
4. **能证明**：用正确性测试、基线、消融或故障实验支持结论。

P0 主题应做到前 3 层，并至少有一个端到端项目做到第 4 层。P1 能分析方案与代价；P2 知道问题边界及阅读入口即可。

## 2. 四个方向的能力矩阵

| 能力 | 推理引擎 | KV Cache 存储 | 通信/GPU 性能 | Serving 平台 |
|---|---|---|---|---|
| Prefill/Decode、指标、KV 容量 | P0 | P0 | P0 | P0 |
| 调度、batch、抢占、取消 | P0 | P1，执行契约 P0 | P1 | P1 |
| GPU page、引用、前缀语义 | P0 | P0 | P1 | P1 |
| chunk/key/layout/跨 rank 正确性 | P1 | P0 | P1 | P1 |
| 分层、预取、准入、淘汰 | P1 | P0 | P1 | P1 |
| lease、可见性、副本、失败回收 | P1 | P0 | P1 | P1 |
| 注册内存、异步生命周期、拓扑 | P1 | P0 基础 / P1 深调优 | P0 | P1 |
| kernel、Roofline、GPU profiling | P1 | P1 | P0 | P2 |
| 路由、SLO、限流、扩缩容、可观测性 | P1 | P1 | P1 | P0 |
| 权重加载、MoE、EP、特殊模型 | 按 JD 升级 | P2，模型状态兼容性 P1 | 按 JD 升级 | 按服务模型升级 |

## 3. 方向 A：推理引擎与运行时

**日常问题**：一轮处理哪些请求，如何混合长短输入，GPU 为什么空闲，抢占后怎样恢复，取消和异步执行怎样不破坏状态。

推荐以 vLLM 或 SGLang 为主，另一个做对照：

```text
公共基础
→ 一个引擎的完整请求路径
→ Scheduler / KV manager / Runner
→ batch、chunked prefill、并行与执行重叠
→ 外部缓存 Connector
→ 同条件比较另一个引擎
```

必须能交付：

- 一张请求状态图，覆盖等待、运行、抢占、结束和取消。
- 一个预算受限的手推调度例子，解释 token 预算与 KV 容量不是同一限制。
- 一份 trace 分析，指出 CPU 准备、GPU 执行、同步或通信中的实际瓶颈。
- 一个对照实验，说明改动改善了吞吐还是 TTFT/ITL，损害了什么。

先读[运行时主题](../02-runtime/README.md)和[Serving 与性能](../06-serving/README.md)，再选实现深读：[vLLM 源码](../02-runtime/implementations/vllm/request-walkthrough.md)、[性能与分布式](../06-serving/implementations/vllm/performance-and-parallelism.md)、[SGLang Runtime](../02-runtime/implementations/sglang/runtime.md)。

## 4. 方向 B：KV Cache 存储与缓存系统，重点路线

**日常问题**：哪些状态值得保存，怎样准确匹配，数据存哪里，加载是否比重算划算，生命周期是否安全，节点故障后怎样不把回源压垮。

### P0：六项必须能独立讲完整的内容

| 主题 | 面试应能做到 | 可以准备的证据 |
|---|---|---|
| 语义与容量 | 算每 token/rank 容量，构造错误共享反例 | 公式、模型配置、兼容性用例 |
| 引擎契约 | 分清 lookup、可读、已加载、GPU ready | Connector 时序图 |
| 数据组织 | 解释 page/chunk/object/group、前缀 hash 和布局 | 对象格式与映射图 |
| 读写状态 | 解释 staging→发布、lease/引用、部分失败 | Put/Get 与释放路径记录 |
| 成本与策略 | 比较加载/重算，分析准入、淘汰、预取 | 算例、冷/热/低复用实验 |
| 故障与退化 | miss 重算、坏数据拒用、超时排空、限制回源 | 故障矩阵与压力曲线 |

### P1：决定设计深度的内容

副本的故障域、Master HA 与数据副本的区别、元数据规模、小对象压力、内存注册复用、NUMA/NIC 拓扑、热点复制、缓存感知路由、跨 rank/group 协调、配额与租户隔离、SSD 分层。

这里不要求第一天就写 RDMA 驱动或 attention kernel。但必须知道网络带宽、H2D、布局和同步可能成为存储收益的上限。

### 工程基础：不能只会讲框架

| 基础 | 至少熟练到什么程度 |
|---|---|
| C++ 资源管理 | 用 RAII 表达所有权，理解 move、智能指针和析构；能解释异步回调为什么可能引用已释放对象 |
| 并发与同步 | 区分锁保护的数据和设备完成事件；能分析并发读写、锁顺序、引用与发布，知道原子变量不自动让整个协议线程安全 |
| Python 与扩展边界 | 理解对象寿命、buffer/tensor 指针、跨进程地址不通用；知道不能依赖 GIL 保护独立进程、native 线程或设备访问 |
| 内存与 Linux | 区分虚拟地址、物理页、锁页、NUMA、进程 RSS；能排查 host OOM、memlock 与重复 buffer 预算 |
| 网络与 RPC | 区分超时、断连、重试、幂等、部分成功；理解批量调用不等于事务 |
| 数据结构 | 会分析哈希表、前缀树、LRU/LFU 的时间与空间开销，能说明小对象多时元数据为什么可能比带宽先成为瓶颈 |

可用三个小练习检查：实现带容量上限的对象缓存并证明活跃读者不会被淘汰；为同 key 并发加载做在途请求合并并处理失败唤醒；设计异步写入取消后的安全回收状态机。先写清接口与不变量，再选语言，不需要复制生产系统的所有功能。

### 阅读与实践主线

```text
引擎怎样消费 KV
→ LMCache 怎样组织/保护 KV
→ Mooncake 怎样发布/保护/搬运对象
→ HiCache 怎样把层级与调度衔接
→ 完整请求成本和失败恢复
```

推荐产物是一个“共享 KV 层评估报告”，至少覆盖版本与格式、容量/网络预算、冷热分层命中、错误与超时处理、端到端 SLO。没有 GPU 时，可完成协议、CPU/TCP、预算与模拟部分，明确剩余验证。

先读[KV 管理](../03-kv-cache/README.md)→[分布式存储](../04-storage/README.md)→[数据传输](../05-transfer/README.md)。实现深读：[LMCache 内核](../03-kv-cache/implementations/lmcache/internals.rst)、[Mooncake 内核](../04-storage/implementations/mooncake/internals.md)、[HiCache](../03-kv-cache/implementations/sglang/tiered-cache.md)、[系统设计题](implementations/lmcache/system-design.rst)。

## 5. 方向 C：高性能通信与 GPU 性能

这一方向内部再选主攻点：

- **通信侧**：buffer 注册/复用、批量提交、完成队列、GDR、NUMA、NIC/PCIe、多路传输、取消后的生命周期。
- **GPU 侧**：矩阵形状、算术强度、HBM 带宽、kernel launch、融合、attention backend、CUDA Graph、计算与传输重叠。

共同要求是能区分“设备理论峰值”“微基准”“模型前向”“端到端请求”四种测量边界。一个 kernel 快 20%，不代表 TTFT 快 20%；要看它在关键路径上占多少。

必须交付一份含正确性、预热、形状、同步方式和误差范围的测量。通信岗还需小对象延迟、大对象吞吐、并发和故障测试；GPU 岗还需代表性 shape、精度/误差和生成代码或 trace 依据。

先读[数据传输](../05-transfer/README.md)与[性能分析](../06-serving/README.md)。实现深读：[vLLM 性能教材](../06-serving/implementations/vllm/performance-and-parallelism.md)、[Mooncake TE 与实验](../07-practice/implementations/mooncake/labs.md)。CacheBlend、EP、TENT 仅在相关职位上继续深入。

## 6. 方向 D：Serving 平台与生产工程

**日常问题**：请求发到哪里，如何设 SLO，热点怎样分散，什么时候扩容，服务重启是否造成命中骤降，故障时如何限流而不是无限重试。

P0 包括工作负载建模、开环/闭环压测、goodput、排队与准入、路由、配额、监控、发布与回滚。要把 Prefill 能力、Decode 活跃请求容量、缓存容量和网络能力分别规划。

必须交付一份流量和故障方案：正常负载、突发、低命中、节点退出四种状态下，看什么指标，触发什么降级，怎样恢复。缓存失效后大量请求回到 GPU 重算，是这类系统特有的容量风险。

先读[Serving 与性能](../06-serving/README.md)和[实验方法](../07-practice/README.md)。实现深读：[SGLang 实验](../07-practice/implementations/sglang/labs.md)、[LMCache 性能运维](../06-serving/implementations/lmcache/observability.rst)、[Mooncake 系统设计](implementations/mooncake/system-design.md)。

## 7. 取舍建议

若目标明确为 KV 存储，建议把学习时间主要放在容量/正确性、LMCache/Mooncake 路径、性能与失败验证，给引擎接口留足时间。不要把一个月耗在所有模型 kernel、所有并行策略和全部新后端上。

若暂时没有目标岗位，先按统一计划完成基础、运行时、缓存、存储与传输的主题介绍，再以“最想调试哪一类问题”选择主线。最后用[面试与项目验收](interview-and-design.md)检验，而不是用看完的文档数量衡量。
