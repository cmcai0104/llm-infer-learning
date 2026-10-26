# 05 · LMCache 与推理引擎集成

## 1. 先按职责理解框架

| 系统 | 主要职责 | 它不替上层完成什么 |
|---|---|---|
| vLLM / SGLang | 执行模型、管理 GPU KV、请求调度、attention kernel、并行 | 不因接入远端缓存就自动得到所有跨节点容错与最优路由 |
| LMCache | 衔接引擎 KV 生命周期，组织/查找/保存/加载 KV chunk，管理层级与后端 | 不直接替代模型执行器，也不保证所有后端均持久 |
| SGLang HiCache | 将 radix/prefix 管理扩展到 GPU、host、外部存储 | 不代表所有层都用同样布局或延迟预算 |
| Mooncake Store | 对象 key、资源池、分配、副本、租约、回收、后端访问 | 不理解任意 token 前缀、模型 revision 或 logits 语义 |
| Mooncake TE | 高性能字节传输、内存注册、拓扑与协议处理 | 不提供完整的分布式 key-value 生命周期 |
| Conductor | 汇总事件形成前缀位置索引，供路由查询 | 不搬运 KV payload，不单独决定完整 SLO 调度 |
| NIXL 等传输抽象 | 为上层提供数据移动能力；可用于部分缓存后端/集成 | 不能仅凭有传输 API 推断有完整缓存系统 |

“LMCache 和 Mooncake 谁更好”需要先确认比较的层次。常见组合是 LMCache 管理推理相关的 KV 生命周期，Mooncake 提供共享存储与网络搬运；某些场景也可以直接使用原生 Store Connector。

## 2. 三条需要熟练区分的路径

### 路径 A：直接 PD 传输

```text
Prefill GPU KV → Mooncake Transfer Engine → Decode GPU/接收缓冲区
```

重点是已有请求的状态交接、接收地址、完成通知、布局和跨 rank 协调。直接 PD 传输本身不等于一个可长期查询的全局前缀缓存池。

入口：[vLLM PD](../deployment/integrations/vllm/disagg-prefill-decode.md)、[SGLang PD](../deployment/integrations/sglang/pd-disaggregation.md)。

### 路径 B：推理引擎原生 Store Connector

```text
引擎 prefix/block 管理 → MooncakeStoreConnector → Mooncake Store
```

适合用原生集成做共享缓存、offload，或框架支持的 PD 方式。要查清当前 Connector 如何形成 key、对齐 page、打包层/rank，以及何时允许读取和释放 GPU block。

入口：[vLLM KV Store](../deployment/integrations/vllm/kv-cache-storage.md)。

### 路径 C：LMCache + Mooncake Store

```text
推理引擎 → LMCache Connector → LMCache KV 管理/分层
→ Mooncake 后端 → Store / TE
```

这条路径有更多缓存管理能力和可替换后端，同时也增加进程边界、缓存副本、锁与数据布局转换。需要从端到端测量它的价值。

入口：[LMCache 集成](../deployment/integrations/lmcache/index.md)。

## 3. LMCache 进程内模式与 MP 模式

检索时很容易同时看到 2025 年博客与当前文档。旧架构解释了重要概念，但启动方式和具体组件不能直接混用。

| 维度 | 进程内模式：历史路径 | MP 模式：本手册重点 |
|---|---|---|
| 引擎侧入口 | `LMCacheConnectorV1` 等相应版本 Connector | `LMCacheMPConnector` |
| 运行位置 | LMCache 逻辑加载到推理进程相关执行路径 | 独立 LMCache server，通过 Connector 连接 |
| 存储组织 | `LMCacheEngine`、`StorageManager`、本地/远端 backend | L1 Manager、StoreController、PrefetchController、L2 adapter |
| Mooncake 入口 | Python remote connector，`mooncakestore://...` | `mooncake_store` L2 adapter，依赖可选 native 扩展 |
| 配置 | `LMCACHE_CONFIG_FILE`、YAML、`remote_url` 等 | `lmcache server` 参数与 `--l2-adapter` 配置 |
| 资源边界 | 与引擎进程的内存和生命周期耦合较强 | 进程可独立管理，但增加 IPC、锁、服务生命周期 |

当前 LMCache 在线文档把进程内 Mooncake 页面标为 deprecated，并推荐 MP；仓库仍保留两条集成指南。维护旧系统时应按已锁定版本阅读旧指南，不能只改 Connector 名称就完成迁移。

**重要的 L1/L2 命名差别**：

- SGLang HiCache 常用：L1=GPU，L2=host，L3=外部 Mooncake。
- LMCache MP 常用 CPU 配置：L1=CPU，L2=外部存储；GPU 属于引擎侧。
- “L2”只表达框架内的相对层级，不代表所有框架的同一硬件，更不代表 CPU L2 cache。

## 4. LMCache MP 写路径

典型 CPU L1 配置中的概念步骤：

1. 引擎计算出 GPU KV，并按 Connector 协议告诉 LMCache 哪些 token/block 可保存。
2. 完成必要 GPU 同步，把 KV 放入 LMCache 可管理的 L1 对象。
3. `StoreController` 感知可保存对象，向配置的 L2 adapter 异步提交。
4. Mooncake adapter 将数据保存到 Store；其下是对象元数据协议与 TE/相应后端数据移动。
5. 根据完成状态更新对象生命周期，解除相应锁或在途引用。

这里至少有三个不同事件：GPU kernel 完成、L1 store 完成、远端 L2 store 完成。PD 场景的消费者何时能开始，必须以当前协议的 ready/telemetry 语义为准。

把“本地 store API 返回”直接解释成“任意远端 decode 节点立刻可读”会导致竞态。异步写回还需要错误计数、重试预算和 buffer 在途保护。

## 5. LMCache MP 读路径

1. `LOOKUP` 检查可复用前缀，先查 L1。
2. 对 L1 未命中的 key，`PrefetchController` 查询 L2 adapters。
3. 如果后端有对象，预取到 L1，并在相应协议下为待执行的 `RETRIEVE` 保护对象。
4. Connector 把可用 KV 写回引擎目标 GPU blocks，处理布局与流同步。
5. 引擎只跳过已真正准备好且跨 rank 一致的连续前缀。

**Lookup 命中不等于可执行命中**。后端可能在查询后淘汰或故障；预取可能超时；目标 GPU block 可能不够；某个 rank 可能没加载完。必须以最终可消费的前缀长度更新调度状态。

不要假设 `LOOKUP/RETRIEVE` 一定通过 RPC 逐字节传 GPU payload。控制请求和数据通路可能分离，具体由 MP 支持的 transfer mode 与 Connector 决定。

## 6. Mooncake MP adapter 的工程约束

仓库 [MP 集成指南](../deployment/integrations/lmcache/vllmv1-lmcache-mp-integration.md)说明：

- 使用该指南的 MP adapter 需要 LMCache v0.4.5 或更新版本。
- 需要构建可选 `lmcache.lmcache_mooncake` 扩展，不能假设普通 wheel 已包含。
- 构建需明确 Mooncake headers/library 来源，避免 Python wheel 与 native library 版本错配。
- RDMA 模式需要有效、连续的 L1 内存描述符供 adapter 注册/使用；不能因为传了 `"rdma"` 就假定数据自动走 RDMA。
- 示例采用 `metadata_server: "P2PHANDSHAKE"`，因此不依赖 HTTP metadata endpoint；若改为 HTTP，需要同时处理 endpoint 和端口冲突。

保留一份版本矩阵：

| 项目 | 必须记录 |
|---|---|
| vLLM/SGLang | tag/commit、CUDA/ROCm、Connector 类与来源 |
| LMCache | tag/commit、MP 或进程内、native 扩展构建选项 |
| Mooncake | tag/commit、cpp/rs、TE/TENT 与实际编译后端 |
| 模型 | 模型 revision、tokenizer、adapter、KV dtype、RoPE |
| 布局 | TP/PP、block/chunk size、layout、page 对齐 |
| 资源 | GPU、CPU L1、Store segment、staging、SSD 配额 |
| 网络 | 地址、协议、metadata 方案、设备、NUMA、MTU 等 |

这张表比一组脱离版本的环境变量更值得在面试中展示。

## 7. SGLang HiCache：另一条重要知识线

HiCache 将前缀树与多级缓存结合。节点管理相应前缀，热数据在 GPU/host，更冷的数据可以存入 Mooncake。

### 预取策略

- `best_effort`：当 GPU 已可执行 prefill 时尽量不再等待，优先延迟。
- `wait_complete`：等待预取完成，有利于复用但会增加慢存储带来的等待。
- `timeout`：在时间预算内等待，平衡命中收益与 TTFT。

当前设计文档给出按 token 数调整超时的形式：

```text
timeout = base + per_ki_token × tokens_to_fetch / 1024
```

实际参数要根据对象大小、网络排队与 SLO 调整；不能只增加等待时间追求更高命中率。

### 写回策略

`write_through`、`write_through_selective`、`write_back` 控制是否每次写、达到热度再写、或淘汰时写回。异步写回减少前台等待，但不能让已提交任务引用的内存先被释放。

### 多 rank 同步

如果 TP 的各 rank 得到不同可用前缀长度，某个 rank 可能跳过计算，而其他 rank 仍需要计算相同部分，导致状态不一致。HiCache 使用同步操作对命中及成功预取长度达成共识，典型选择是取各 rank 的最小有效值。

### Page-first 与 layer-first

GPU 推理自然按层执行，而对象存储常希望把一页完整 KV 合并为少数大 I/O。`layer first`、`page first`、`page first direct` 等布局在 I/O 聚合、CPU↔GPU 传输粒度与重排成本间取舍。不要只看远端带宽而遗漏 layout 转换。

详见 [HiCache 设计](../design/hicache-design.md)。

## 8. CacheBlend：非前缀复用为什么需要额外机制

RAG 中常有可重复文档片段，但文档顺序、前面的 system prompt 或其他文档会改变。普通 prefix cache 只能保证相同完整前缀条件下的精确复用。

CacheBlend 研究/实现的是复用预计算片段并选择性重算部分 token，使片段在新的上下文中结合。需要同时考虑跨片段 attention 影响、位置处理和质量评估；具体支持模型和算法选项应查当前 LMCache 文档。

**不能回答成**“Mooncake 把若干 KV 对象拼接起来，就能无损复用任意 RAG 文档”。Store 负责存 bytes，不能修复模型语义。

评估 CacheBlend 至少有两条轴：

- 性能：TTFT、节省的计算、重算比例、搬运成本。
- 质量：目标任务准确率/检索问答指标、长上下文表现、不同文档顺序与边界的鲁棒性。

入口：[当前 CacheBlend 文档](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html)；旧版参数保留在 [Blending](https://docs.lmcache.ai/kv_cache_optimizations/blending.html)。

## 9. KV Cache 存储岗位最需要练的六个集成难点

| 难点 | 要能描述的方案 | 失败时的可观测信号 |
|---|---|---|
| 跨实例兼容性 | namespace、模型版本、dtype/layout、TP/PP、token hash | “命中但输出错误”、形状/大小不匹配 |
| 对象生命周期 | GPU block 引用、L1 lock、Store lease、异步引用同时管理 | 偶现损坏、非法访问、租约失败 |
| 预取与 deadline | 预取预算、连续前缀、部分完成、取消后 drain | 高命中低收益、P99 TTFT 上升 |
| 热点与路由 | 本地副本、冷热选择、队列感知、限流 | 单节点 NIC 打满、请求偏斜 |
| 缓存击穿 | 同 key 在途合并、有限等待、失败恢复 | GPU 同时重算、重复写、Master RPC 突增 |
| 多层内存预算 | L1、segment、staging、GPU page 单独核算 | OOM、memlock 失败、反复淘汰/加载 |

## 10. 怎样选型

- 需要低延迟 PD 交接，已有明确消费者：先研究直接 TE Connector。
- 需要跨实例复用、共享容量：研究原生 Store Connector 或 HiCache。
- 需要 LMCache 的多后端、缓存管理或 CacheBlend 能力：研究 LMCache + Mooncake，优先匹配当前 MP 文档与版本。
- 需要全局缓存位置辅助路由：研究 Conductor 与已有 router 的结合。
- 只有单实例、短 prompt、低命中：先测本地 prefix caching，再判断外部存储的复杂性是否有收益。

这些是设计起点。最终选择由模型兼容性、SLO、硬件、故障处理与消融实验决定。
