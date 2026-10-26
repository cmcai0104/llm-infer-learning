# 03 · KV Cache 管理：让正确的状态在正确的时间可用

[总目录](../README.md) · [前置：运行时](../02-runtime/README.md) · [下一主题：分布式存储](../04-storage/README.md)

本主题回答“什么能复用、怎样管理、何时有收益”。先建立缓存语义和生命周期，再看框架怎样用块、树、分层控制器或独立服务实现。

重点深读：[KV 存储详细教材](../04-storage/kv-cache-storage.md)、[LMCache 与 Mooncake 联合链路](../04-storage/implementations/lmcache/mooncake-walkthrough.md)。数据映射和完成协议见[设计图册 D5–D6、D8–D9](../01-foundations/design-atlas.md)。

## 1. 四个问题决定一次复用是否成立

1. **是不是同一份计算状态？** 核对 token 前缀、模型、位置及适用的命名空间信息。
2. **能不能按当前布局消费？** 核对 dtype、层/head、并行切分、页/chunk 和设备格式。
3. **数据是否完整且仍受保护？** 查到条目不代表写入已发布，也不代表读取期间不会被回收。
4. **加载是否值得？** 比较省掉的计算与查询、排队、传输、转换、同步的关键路径。

容量和复用语义详见[容量与复用](capacity-and-reuse.md)、[状态与成本模型](semantics-and-costs.rst)。第一次选一篇深读，另一篇用于补充，不重复安排基础课。

## 2. 分开理解三个层面

| 层面 | 解决的问题 | 常见方案 |
|---|---|---|
| 物理分配 | 状态放在哪些内存页，如何减少浪费 | 分页、block pool、引用计数、必要时 copy-on-write |
| 逻辑索引 | 哪段历史状态与新请求匹配 | 链式 block hash、前缀树、Radix 结构 |
| 分层控制 | 保留、写回、预取、加载哪些状态 | GPU/host/远端分层、准入、淘汰、等待预算 |

这三层可以组合。分页解决分配问题，前缀树解决匹配和共享结构问题，二者并不互斥。远端 Store 只接受 key 和字节时，语义匹配仍需要上层完成。

## 3. 用生命周期理解异步加载

```text
候选命中
→ 验证可复用范围
→ 取得读保护并保留目标空间
→ 加载/转换/设备同步
→ 向运行时发布可用状态
→ 消费结束
→ 释放引用或读保护
```

这是教学状态模型，不是某个项目的统一 API。失败时要区分“确认没有成功，可以重算”和“仍有在途操作，需要先隔离或等待”。不能让重算结果与迟到的远端写入争用同一个已复用 buffer。

## 4. 缓存策略的取舍

| 策略 | 可能带来的收益 | 要验证的反例 |
|---|---|---|
| 更积极地写回 | 增加后续请求可复用范围 | 一次性请求耗尽网络和 host 内存 |
| 更积极地预取 | 把加载移出关键路径 | 错误预取抢占有效流量和目标空间 |
| 更大 chunk | 减少元数据与小 I/O 开销 | 部分命中浪费搬运，等待粒度变粗 |
| 保留热门前缀 | 降低重复 Prefill | 热点变化、租户挤占和过期数据 |
| 等待远端缓存 | 避免重复计算 | 短输入或拥塞时比立即重算更慢 |

评价时同时记录请求/token 命中率、实际减少的计算、读取字节数、加载等待和最终延迟，不能只报告一个 hit rate。

## 5. 在同一问题下比较具体实现

| 关注点 | 实现资料 | 阅读时要核对 |
|---|---|---|
| GPU block 管理与外部加载边界 | [vLLM 缓存与集成](implementations/vllm/integration.md) | 命中范围、页所有权、Connector 完成契约 |
| 前缀结构与状态池 | [SGLang 前缀缓存](implementations/sglang/prefix-cache.md) | 树节点、引用、evictable 状态与混合模型边界 |
| 运行时内部的分层控制 | [HiCache 与后端](implementations/sglang/tiered-cache.md) | 预取、写回、TP 一致性与布局 |
| 独立缓存管理服务 | [LMCache 概览](implementations/lmcache/overview.md)、[内核](implementations/lmcache/internals.rst)、[状态图](implementations/lmcache/state-diagrams.rst) | lookup、staging、锁、发布、释放和进程边界 |

要理解数据放到共享池之后的对象保护，继续读[分布式存储](../04-storage/README.md)。要验证省下的 Prefill 是否超过搬运成本，按[实验主题](../07-practice/README.md)设计冷/热、短/长、成功/失败对照。
