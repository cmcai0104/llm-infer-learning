知识地图与设计图
================

图中实线用于表示主要调用或数据路径，虚线用于表示管理、元数据或可选联系；
具体含义由每幅图的说明限定。架构图省略了平台差异与错误分支，
不能据此推断所有配置都支持任意两节点直传。
图表保留可读字号；超过正文宽度时可在图表区域横向滚动。

全景知识地图
------------

.. mermaid::
   :name: learning-knowledge-map
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
     A["KV Cache / 推理 infra"] --> B["推理语义"]
     A --> C["LMCache 运行时"]
     A --> D["存储与传输"]
     A --> E["性能与生产"]
     B --> B1["Prefill / Decode / Attention"]
     B --> B2["GQA / MLA / Hybrid / TP"]
     B --> B3["前缀依赖 / 位置 / key / layout"]
     C --> C1["Connector / Scheduler / Worker"]
     C --> C2["Lookup / Prefetch / Retrieve / Store"]
     C --> C3["L1 staging / 锁 / 事件 / 会话"]
     D --> D1["DRAM / SSD / S3 / 远端内存"]
     D --> D2["Mooncake Store / Transfer Engine"]
     D --> D3["RDMA / PCIe / NUMA / GDS"]
     E --> E1["TTFT / ITL / Goodput / 尾延迟"]
     E --> E2["准入 / 淘汰 / 背压 / 路由"]
     E --> E3["隔离 / 故障 / 可观测 / 正确性"]
     C2 --> F["高级复用"]
     F --> F1["CacheBlend / 压缩 / Layerwise"]
     F --> F2["P/D 分离 / 跨实例共享"]

对应阅读：推理语义见 :doc:`03-foundations`；运行时见
:doc:`04-lmcache-internals`；存储见 :doc:`05-storage-and-mooncake`；
生产问题见 :doc:`06-performance-and-operations`。

MP 架构：谁决定、谁执行、谁存储
--------------------------------

.. mermaid::
   :name: learning-mp-architecture
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart TB
     U["请求 / tokenizer"] --> S["vLLM Scheduler"]
     subgraph ENGINE["推理引擎进程"]
       S --> SA["LMCacheMPSchedulerAdapter"]
       W["vLLM Worker"] --> WA["LMCacheMPWorkerAdapter"]
       W <--> G["GPU paged KV"]
     end
     subgraph MP["独立 LMCache 服务"]
       R["RPC / Session / Modules"]
       L["LookupModule"]
       T["Transfer modules"]
       SM["StorageManager"]
       L1["L1Manager / allocator"]
       P["PrefetchController"]
       ST["StoreController"]
       AD["L2 adapters"]
       R --> L --> SM
       R --> T --> SM
       SM <--> L1
       SM --> P
       L1 --> ST
       P <--> AD
       ST --> AD
     end
     SA -->|"LOOKUP / 状态"| R
     WA -->|"STORE / RETRIEVE / 上下文"| R
     G <-->|"设备传输；不等于普通 RPC payload"| T
     AD <--> FS["文件 / 对象存储 / 远端 Store"]
     CO["可选 Coordinator"] -.-> R
     OBS["HTTP / Metrics / Traces"] -.-> SM

常见配置中 L1 是主机 DRAM，L2 是外部后端。当前实现还支持其他 L1 backing，
因此 L1/L2 是 LMCache 的管理层级，不是 CPU 硬件 cache 层级。
MP 也有 engine-driven transfer，图中的跨进程传输并不限定由 server 发起。

进程内模式对照
--------------

.. mermaid::
   :name: learning-inprocess
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
     V["vLLM / LMCacheConnectorV1"] --> E["inprocess LMCacheEngine"]
     E --> H["TokenDatabase"]
     E --> GC["GPU connector"]
     E --> BS["inprocess StorageManager"]
     BS --> B["CPU / Disk / Remote backend"]

这条路径适合理解旧教程。它与 MP 共享一些底层能力，但
``ObjectKey``、L2 adapter 和 MP server 参数不能直接套用到进程内配置。

冷请求：计算后存储
------------------

.. mermaid::
   :name: learning-cold-request
   :config: {"sequence": {"useMaxWidth": false}}

   sequenceDiagram
     participant S as Scheduler
     participant M as LMCache server
     participant W as Worker/GPU
     participant L as L1 manager
     participant B as L2 backend
     S->>M: LOOKUP
     M-->>S: ack
     S->>M: QUERY_PREFETCH_STATUS
     M-->>S: 0 个可复用 chunk
     S->>W: 调度 prefill
     W->>W: 计算 KV / 记录设备完成事件
     W->>M: STORE 元数据与传输依赖
     M->>L: reserve_write，创建 staging
     M->>W: 等待生产者事件并发起 GPU 到 L1 传输
     Note over M,L: 实际 copy 完成后才发布
     M->>L: finish_write，admit resident
     L->>B: StoreController 异步写 L2（若配置）
     B-->>L: 完成后释放所持读锁
     W-->>S: 继续生成；请求完成与存储完成分开处理

这是依赖关系图，并不表示每次 STORE 都同步阻塞 scheduler。
只允许存储引擎声明已完成的有效 token 范围；尾部不完整 chunk 和 decode
写回时机由具体路径及策略决定。

热请求：L1/L2 查找、预取、回填 GPU
----------------------------------

.. mermaid::
   :name: learning-warm-request
   :config: {"sequence": {"useMaxWidth": false}}

   sequenceDiagram
     participant S as Scheduler
     participant M as LookupModule
     participant P as PrefetchController
     participant L as L1
     participant B as L2 adapters
     participant W as Worker/GPU
     S->>M: LOOKUP
     M->>P: 提交 grouped keys
     M-->>S: ack（并非预取完成）
     P->>L: reserve_read
     P->>B: lookup_and_lock
     B-->>P: 命中 bitmap
     P->>P: 汇总层级结果，规划不重复的数据来源
     P->>L: 为需要的 L2 数据 reserve_write
     P->>B: load 到 L1 staging
     B-->>P: 完成 bitmap
     P->>L: admission 并保留读锁
     S->>M: QUERY_PREFETCH_STATUS
     M-->>S: 完成后的可复用前缀（未完成时为 None）
     S->>W: 分配物理块并传递加载计划
     W->>M: RETRIEVE
     M->>W: L1 到 GPU；发出完成事件
     W->>W: 等待数据依赖后执行剩余计算
     M->>L: 读完成后释放；清理未使用的尾锁

关键区别是 **lookup ack、预取完成、GPU 可读** 三个时刻。多 server 的命中聚合
在本基线采用共同可用前缀，不能取“最快 server 的最长结果”。

L1 对象状态与生命周期
---------------------

.. mermaid::
   :name: learning-l1-lifecycle
   :config: {"state": {"useMaxWidth": false}}

   stateDiagram-v2
     [*] --> Absent
     Absent --> Staging: reserve_write(key, tag)
     Staging --> Resident: finish_write / admission
     Staging --> Discarded: 另一 writer 已发布同 key
     Staging --> Reclaimable: 写锁过期
     Reclaimable --> Absent: 回收
     Resident --> Reading: reserve_read
     Reading --> Resident: finish_read（仍有副本）
     Reading --> Absent: 最后读者结束且对象为 temporary
     Resident --> Absent: 无锁时淘汰或删除
     Discarded --> Absent: 释放此 staging，其他 resident 不受影响

图中 staging 是 writer 的私有对象，resident 是同 key 的公共对象；
“Discarded→Absent”只描述被丢弃的 staging 生命周期。多个 writer tag
可以同时拥有同 key 的 staging；不应把这幅简图理解为整个 key 的单一互斥状态机。
锁过期也不等于 GPU/DMA 自动停止，设备操作的完成与内存回收必须另行协调。

LMCache 与 Mooncake 组合
------------------------

.. mermaid::
   :name: learning-mooncake-integration
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
     G["推理 GPU"] <-->|"D2H / H2D"| L["LMCache L1 buffer"]
     L <--> A["Mooncake native L2 adapter"]
     A --> C["嵌入式 Mooncake Client"]
     C -.->|"分配 / 副本位置 / 租约"| M["Mooncake Master"]
     C --> T["本地 Transfer Engine"]
     T <-->|"数据面：TCP / RDMA"| RT["远端 Transfer Engine"]
     RT <--> R["已注册的远端 segment"]
     T -.-> MS["segment / endpoint metadata"]
     RT -.-> MS
     M -.-> R

Master 管对象位置与生命周期；Transfer Engine metadata 管 segment、
buffer 和 endpoint 信息，两者不是同一层元数据。
本图展示常见的主机 L1 路径，不承诺该 adapter 自动执行 GPU 到远端 GPU 的
GPUDirect RDMA，也不把控制 RPC 画成大对象传输路径。

P/D 分离与跨请求复用
--------------------

.. mermaid::
   :name: learning-pd-disaggregation
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart TB
     R["Router"] --> P["Prefill engine"]
     R --> D["Decode engine"]
     P -->|"本次请求 KV 交接：例如 NIXL"| D
     P <--> LP["Prefill 的 LMCache server"]
     D <--> LD["Decode 的 LMCache server"]
     LP <--> SH["可选共享存储 / P2P"]
     LD <--> SH

P/D 分离解决阶段独立扩缩与干扰；跨请求缓存解决重复 prefill。
当前仓库的 MP P/D recipe 使用 ``MultiConnector`` 组合 NIXL 与 LMCache，
并要求两个引擎各自配套 server；它不是“把 Mooncake Store 开关打开就自动完成 P/D”。

性能分析因果链
--------------

.. mermaid::
   :name: learning-performance
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart LR
     H["更多缓存命中"] --> S["少做重复 prefill"]
     H --> IO["更多查询 / 网络 / H2D"]
     S --> T["TTFT 与 goodput"]
     IO --> T
     IO --> Q["队列 / 内存 / 锁竞争"]
     Q --> T
     Q --> D["Decode ITL"]
     POL["准入 / 路由 / chunk / 背压"] --> H
     POL --> Q

所以命中率只描述复用机会，不能单独证明性能收益。
