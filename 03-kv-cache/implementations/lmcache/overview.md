# LMCache：把 KV 的保存、查找、加载和回收管起来

[返回 KV 管理主题](../../README.md) · [分层控制对照](../sglang/tiered-cache.md) · [相关主题：存储](../../../04-storage/README.md)

LMCache 主要解决：**引擎算出的 KV 怎样离开 GPU 保存下来，又怎样在下一次请求需要时正确、及时地回到引擎。** 它理解 token/chunk、引擎 block、缓存层和加载协议；下层对象存储通常只理解 key 与 bytes。

本篇以当前资料对应的 MP（multiprocess，独立进程）架构为主。原有 RST 教材保留在[专题目录](../../../08-career/implementations/lmcache/index.rst)，进程内模式用于理解旧部署和版本差异。

## 1. 为什么有了对象存储还需要 LMCache

假设 vLLM 里有一段 4K-token KV，分布在许多 GPU block，某些 block 还被多个请求共享。直接写一个 `store.put(key, bytes)` 有四个缺口：

1. 从哪些 block 取数据，如何组织各层和 rank 的内容？
2. key 怎样对应完整前缀，查到多少 chunk 才能减少计算？
3. 读取结果写回哪些新的 GPU block，何时允许 attention 使用？
4. 异步写入未结束时，请求取消、block 被淘汰怎么办？

LMCache 与引擎 Connector 要一起填补这些契约。后端越快，上层协议错误也不会自动消失。

## 2. MP 模式的主要角色

```text
推理引擎（GPU KV、调度器）
        ↕ Connector：请求、block、完成状态
LMCache server
  ├─ lookup / prefetch / retrieve / store
  ├─ L1 Manager：管理本地对象和保留状态
  ├─ PrefetchController：安排从 L2 加载
  └─ StoreController：安排向 L2 保存
        ↕ L2 adapter
外部存储，例如 Mooncake Store
```

这里常见的 CPU 配置中，LMCache 的 L1 是主机内存，L2 是外部存储；引擎 GPU 在这个命名体系外。它与 HiCache 的 L1=GPU、L2=host 不同。讨论层级时最好直接说“GPU、host、远端”。

MP 的好处是缓存服务和引擎进程可独立管理；代价是 IPC、进程健康、锁和资源所有权更复杂。server 独立存活也不代表所有内容永远保留，仍受配额、淘汰和后端故障影响。

## 3. 四个动作按时间顺序看

### Lookup：查到多少前缀有希望复用

输入是一组前缀身份和相关执行信息。查询可能需要检查 L1，或者触发 L2 查询和预取。任务被接受的 ack 只说明请求已登记。

在当前 MP lookup 协议中，pending 与最终命中结果不同。多 server/多分片结果还要聚合为共同可用前缀，不能把各自命中数量相加。

### Prefetch：提前搬到受管理的本地内存

从后端加载，先放到未发布的 staging 空间，完成后再进入可读状态。需要为在途数据和读者保留资源；查询之后对象可能被淘汰或后端可能失败，所以不能只信早先的存在性结果。

### Retrieve：把真正可用的状态交回引擎

把对象搬到目标 GPU block，完成布局映射及设备同步。调度器才能依据已准备好的有效范围减少 Prefill。控制请求通过 IPC，不意味着 GPU payload 也一定逐字节走同一个 RPC。

### Store：保存新产生、值得保留的 KV

引擎暴露可保存范围，LMCache 将其组织为缓存对象，先在相应本地层完成，再按策略提交后端。GPU 计算完成、L1 完成、L2 完成是不同的事件。

## 4. 用一个具体请求串起来

假设输入 1,024 token、chunk=256，共 4 个 chunk。L1 有第 1 个，L2 有前 3 个，第 4 个没有：

```text
查找：L1 [有 无 无 无]，L2 [有 有 有 无]
保护：为计划使用的对象建立必要保留
预取：把第 2、3 个加载进 L1
恢复：把可用的前 3 个放进 GPU
计算：计算未覆盖输入，按引擎契约处理边界和首 token
保存：把新产生的、符合策略的对象送入缓存
```

如果第 2 个加载失败，第 3 个的成功不能直接变成“共命中 512 token”。普通前缀复用需要连续性；引擎最终跳过多少 token 还受块边界和首 token 处理影响。

如果 rank 0 有 3 个 chunk、rank 1 只有 2 个，还要先确认共同有效范围。

## 5. Key 需要同时表达内容和分片

当前基线的 `ObjectKey` 包含：

```text
chunk_hash
model_name
kv_rank
object_group_id
cache_salt
```

这些字段帮助定位内容、模型、分片、group 和共享范围，但字段存在不代表所有兼容性都已自动解决。模型 revision、dtype、LoRA、位置配置等如何隔离，需要沿调用链核对。

也不能只比较 key 字符串。取到对象以后，接收者仍要按正确布局解释 bytes。同一段逻辑 KV，按层排列和按页排列可能是不同物理格式。

## 6. 为什么需要 staging、读保留和完成回调

想象把一块空内存登记到目录，随后才慢慢写入数据。另一个请求查询命中后立刻读，就会读到半成品。因此典型生命周期是：

```text
分配 staging → 填充数据 → 完成并发布 → 读保留 → 使用 → 释放保留 → 可淘汰
```

在当前 L1 设计中，写者可以有自己的 staging；并发同 key 的发布需要解决谁成为 resident、谁释放多余副本的问题。读保留用于阻止正在使用的对象被普通淘汰逻辑回收。

超时清理也不能跳过设备生命周期。一个任务在业务上不再等待，不表示数据拷贝一定停止。安全做法要把取消、完成确认、回调和内存所有者连接起来。

## 7. 接 Mooncake 后改变了什么

Mooncake L2 adapter 把 LMCache 的存储操作映射到共享 Store。Store 管对象位置、副本和读写协议，TE 处理对应数据搬运。

获得的是共享资源与数据传输能力，同时增加：

- 元数据查询与远端排队。
- L1、staging、Store segment 的叠加内存预算。
- 本地锁、远端租约和异步任务之间的生命周期协调。
- 热点、网络故障、后端容量不足的降级处理。

因此既要测对象存取带宽，也要测完整请求 TTFT、ITL、goodput。对象 benchmark 更快，未必代表推理加载的关键路径更短。

## 8. CacheBlend 为什么要单独学

精确前缀复用要求完整前缀条件兼容。RAG 文档顺序变了，或者前文变了，普通 KV 不能随意拼接。

CacheBlend 探索复用片段并选择性重算，使它们适应新的上下文。这里需要关注算法支持、位置处理、重算比例及质量。不能把它归结成“存储把对象拼起来”。

KV 存储岗先掌握精确前缀的正确性，再研究高级复用；性能报告需要同时给出质量与延迟结果。

## 9. 源码阅读顺序

路径相对于 LMCache 上游源码根目录：

| 入口 | 要看懂什么 |
|---|---|
| `lmcache/integration/vllm/lmcache_mp_connector.py` | 调度/worker callback 如何连接请求状态与完成事件 |
| `lmcache/v1/multiprocess/` | server、请求会话、lookup、store/retrieve 的调度 |
| `lmcache/v1/distributed/api.py` | ObjectKey、group、布局、预取类型 |
| `lmcache/v1/distributed/l1_manager.py` | staging、resident、读写保留 |
| `lmcache/v1/distributed/storage_manager.py` | L1、controller、adapter 如何组合 |
| `lmcache/v1/distributed/l2_adapters/` | 异步后端协议和 Mooncake 适配 |

对照上游 `docs/design/` 的对应目录读设计约定，再读实现。旧材料的 `lmcache/v1/cache_engine.py` 在本基线对应进程内目录 `lmcache/inprocess/v1/cache_engine.py`，不要沿旧路径推断 MP 架构。

## 10. 深读与面试验收

| 需求 | 资料 |
|---|---|
| 画数据流和生命周期 | [知识地图](state-diagrams.rst)、[LMCache 内核](internals.rst) |
| 理解 Mooncake 后端与锁契约 | [存储与 Mooncake](../../../04-storage/implementations/lmcache/backend-contracts.rst) |
| 判断收益和排查问题 | [性能与运维](../../../06-serving/implementations/lmcache/observability.rst) |
| 做实验 | [实验手册](../../../07-practice/implementations/lmcache/labs.rst) |
| 练习面试与系统设计 | [面试题](../../../08-career/implementations/lmcache/questions.rst)、[系统设计](../../../08-career/implementations/lmcache/system-design.rst) |

应能回答：“lookup 成功后为什么仍可能 miss？”“对象什么时候发布？”“谁保护 in-flight buffer？”“为什么多一层缓存也可能更慢？”最好能用一次请求的时间线，把四个回答连起来。
