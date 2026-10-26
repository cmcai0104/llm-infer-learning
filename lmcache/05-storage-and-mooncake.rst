KV 存储、传输与 Mooncake 专题
=============================

从普通 KV 存储到推理状态存储
----------------------------

普通 key-value 服务关心 key/value 的一致性与读写延迟；
LLM KV Cache 还要求理解上层的 **前缀依赖、张量布局、分片完整性和设备生命周期**。
对象很大、访问有强烈前缀与会话局部性，可重算但错误命中不可接受。
存储层可以不理解 attention，但集成层必须将这些语义编码为可靠契约。

例如一个 256-token chunk 可能有数十 MiB，且同一个逻辑 chunk 分成多个 rank/group
对象。Redis 风格的小对象每秒操作数不等价于这种 workload 的性能。
还需要量化大对象带宽、batch 开销、内存拷贝次数、查找放大和尾延迟。

存储层级如何选
--------------

.. list-table::
   :header-rows: 1
   :widths: 18 25 27 30

   * - 层级
     - 优点
     - 代价/风险
     - 适用场景
   * - GPU HBM / APC
     - 可直接供 attention 使用
     - 容量贵，与权重和运行时争显存
     - 最近使用、即将 decode 的 KV
   * - 本机 DRAM
     - 容量较大、低网络开销
     - H2D/D2H、NUMA、pinning 与 CPU 内存压力
     - 多轮会话、同机跨引擎复用
   * - 远端 DRAM
     - 汇聚内存池、跨节点共享
     - 网络和控制服务、故障与热点
     - 共享文档、长上下文、弹性计算池
   * - 本地 NVMe / 文件
     - 容量成本较低
     - IOPS、吞吐、写放大、文件元数据及 page cache 干扰
     - 次热对象、重启后的复用
   * - 共享文件/对象存储
     - 大容量、跨节点访问、可选耐久性
     - RTT、请求费用、吞吐与恢复流量
     - 长尾缓存、预热、可容忍较长加载的场景

“远端”未必比“磁盘”慢，“本地”也未必最快；有效速度由对象大小、
排队、介质、拓扑与软件路径共同决定。直接给所有层级固定性能排名会误导选型。
L2 可配置多个 adapter，但默认策略不是一个自动学习最快层的智能层级系统。

必须分开的三类策略
------------------

**准入（admission）**：值不值得存？一次性长 prompt 会占满缓存，可能挤掉高复用前缀。
**淘汰（eviction）**：空间不够时删谁？LRU 易实现，但忽略对象大小、重算成本和未来价值。
**放置/路由（placement/routing）**：存在哪、请求去哪？能减少搬运，但会与计算负载均衡冲突。

用于讨论的价值密度可以写为：

.. math::

   V_i =
   \frac{P_i(\mathrm{reuse}) \cdot
   \left(T_i(\mathrm{recompute}) - T_i(\mathrm{load})\right)}
   {\mathrm{bytes}_i}

这是启发式设计模型，不是 LMCache 默认算法。还应扣除写入流量、复制成本和
机会成本。前缀依赖使孤立的后缀 chunk 即使还在也可能无法使用；
淘汰与存储策略应考虑有效前缀而不只是对象 hit 数。

Write-through、异步 write-behind 与 eviction-time write-back 是不同策略。
缓存层允许丢弃可重算对象，不意味着所有写路径都需要数据库级事务；
但用户必须知道收到“存储完成”时到底完成了哪一层。

Mooncake 的三个相关层次
------------------------

1. **Transfer Engine**：搬运已注册地址范围，提供 segment、batch transfer、
   transport 和完成状态，不负责判断一段 KV 是否能用于某个 prompt。
2. **Mooncake Store**：管理 key→对象/副本位置、分配、生命周期、租约和存储池；
   Client 通过 Transfer Engine 直接读写 payload。
3. **上层集成/路由**：LMCache、SGLang HiCache、vLLM connector 或 Conductor
   将前缀身份、对象命中、实例位置连接到推理调度。

Mooncake 论文中的完整 KV-centric serving 方案不能和“安装了 Transfer Engine
库的应用”画等号。当前仓库还有 PG/EP、Reshard 等组件，KV 存储岗位首先掌握
Store/Transfer Engine 即可，MoE 通信可作为 P2 专项。

Store 的读写与一致性边界
------------------------

常规 Put 的概念流程：

.. code-block:: text

   Client -> Master: PutStart，申请空间与副本位置
   Client -> 目标 segment: Transfer Engine 写 payload
   Client -> Master: PutEnd，报告完成
   Reader -> Master: GetReplicaList，获得可读副本与保护条件
   Reader <- 目标 segment: 直接读取数据

Master 不转发大对象。元数据服务不可用可能阻止新查找或分配，但不是因为所有字节
必须经过 Master 的网卡。Transfer Engine 的 metadata 与 Store Master 的对象目录
也不是同一个职责。

普通 Put 以不可变缓存对象为主要模型；本基线还存在 ``Upsert`` 路径，因此不要说
“Mooncake 永远不能改值”。对于内容寻址的 KV 复用，应使用清晰的模型版本和对象身份，
不要覆盖旧 key 让在途读者读取另一版本。

需要区分：

* **读租约**：在限定时间内保护读取所需副本，解决并发淘汰/删除风险。
  超出租约的读取可能失败；不能把它当对象固定生存 TTL。
* **soft pin**：影响淘汰优先级/允许条件，不是永远不丢；本基线读操作不延长其固定生命周期。
* **副本**：提高容错与热点带宽，但容量与写流量增加，实际副本状态仍需确认。
* **持久化**：只有满足介质和完成协议时才具备；全内存多副本不防集群断电。
* **HA Master**：需要实际选主与恢复部署；“一个 Master 进程”不会自动变成高可用控制面。

对 grouped objects，Mooncake 可提供共享租约与 best-effort 生命周期处理；
它不等价于推理引擎对所有 TP/group 的完整性判定，也不自动构成任意多对象事务。

以上依据 Mooncake 的
``docs/source/design/store/mooncake-store.md``、
``docs/source/design/architecture.md`` 与
``docs/source/api-reference/python/mooncake-store.md``。

LMCache + Mooncake 的实际接缝
-----------------------------

本基线 MP 使用 ``mooncake_store`` native L2 adapter。
入口为 ``lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py``，
C++ 后端在 ``csrc/storage_backends/mooncake/``。
不要将它和旧进程内 remote connector 的 YAML 混用。

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 责任
     - 边界
   * - LMCache
     - 引擎侧语义、chunk/group/key、L1 buffer、预取/存储任务和设备完成
   * - native adapter
     - 把 L2 请求转换为 Mooncake 操作，管理异步任务与操作线程池
   * - Mooncake Store
     - 对象位置、副本、segment 空间、读写保护与对象生命周期
   * - Transfer Engine
     - 在可访问 buffer 之间传输 bytes，报告实际完成/失败
   * - 部署者
     - 模型/分片/layout 一致性，租户授权，容量、拓扑、ABI 和服务可达性

``type``、``num_workers``、``per_op_workers``、``eviction`` 由 LMCache 消费，
其余配置作为字符串值传给 Mooncake 的 ``setup_internal(ConfigDict)``。
LMCache 不替 Mooncake 自动填齐所有参数。操作线程池可分 lookup/retrieve/store/delete，
防止大流量写占满查询队列，但线程更多不一定提高带宽。

RDMA 模式要求有效的连续 L1 内存描述符；缺失时初始化报错，而非悄悄退到 TCP。
文档建议 ``--no-l1-use-lazy``，使注册前 L1 已完成分配。
常见路径为 GPU↔LMCache 主机 L1↔远端 segment；
即使 Store 支持设备内存，也不能据此宣称这个集成路径自动消除了 H2D/D2H。

完整配置与构建前提见 :doc:`/mp/l2_storage/mooncake_store`；
可控实验见 :doc:`07-labs`。安装 Mooncake Python wheel 不代表 LMCache 的
C++ Mooncake 扩展已经构建，后者有独立的 ``BUILD_MOONCAKE=1`` 开关与头文件/库依赖。

RDMA 面试必须说清的内容
-----------------------

**MR 与注册**：NIC 需要可 DMA 的内存区域及访问权限。注册通常涉及页固定、
地址映射和设备资源，不宜每个小对象反复注册。复用注册 slab 能摊销开销，
但要控制 pin 住的内存总量，避免宿主机内存压力。

**QP/CQ 与异步完成**：发送工作请求不等于工作完成；完成队列报告传输状态。
本地 buffer、远端地址及相关 key 的有效期必须覆盖整个传输。
应用还要把传输完成与“对象正式可读”关联起来。

**单边与双边**：RDMA READ/WRITE 可以让远端 CPU 不逐次参与 payload 拷贝；
连接管理、元数据、分配、完成处理和故障恢复仍需要软件。
零拷贝应说明省掉了哪一次 bounce-buffer copy，而不是“数据完全不移动”。

**GPUDirect RDMA**：NIC 可直接访问满足条件的 GPU memory，
受驱动、硬件、IOMMU/PCIe 拓扑、注册方式和 transport 支持限制。
GDS 是存储到 GPU 的数据路径，NVLink 是互连，不是三个可互换的名字。

**NUMA 与多网卡**：GPU、NIC、DRAM 跨 socket 会增加路径长度与争用。
多 NIC 聚合需要足够并发、合理 striping 和拓扑选择，不保证单对象无限线性扩展。
小对象更受 RTT/提交开销影响，大对象更受有效带宽影响。

**可靠性**：连接恢复、超时和路径重试不能暴露部分写；
failover 也不保证 exactly-once。正确做法是可识别任务状态、幂等或安全去重，
并确保未知/失败的 buffer 不作为有效 KV 返回。

与其他框架如何对比
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 38 42

   * - 组件
     - 主要职责
     - 与 LMCache/Mooncake 的关系
   * - vLLM APC
     - 引擎内 GPU 前缀复用
     - 和 LMCache 互补；实验要防止把 APC 收益算给外部 cache
   * - SGLang RadixCache / HiCache
     - radix 前缀索引及分层 KV 管理
     - 可接 Mooncake Store；不同路径不应同时无规划地重复 offload
   * - LMCache
     - 跨引擎集成、KV 生命周期与分层复用
     - 可以把 Mooncake Store 作为 L2，不替代整个推理引擎
   * - Mooncake Store
     - 分布式对象池与生命周期
     - 可以为 LMCache 或其他上层缓存系统服务
   * - Mooncake Transfer Engine
     - 地址/segment 层的数据传输
     - 可直接用于 P/D，也用于 Store 数据面；不等于 Store
   * - NIXL
     - 可插拔数据传输抽象及后端
     - 可用于 P/D 或 LMCache 某些 L2 路径；不自动提供前缀缓存语义
   * - 缓存感知 router / Conductor / Coordinator
     - 利用命中位置做路由或协调
     - 索引可陈旧，实际加载必须验证；不应成为 payload 中转站

SGLang HiCache 的常见文档命名为 GPU L1、host L2、外部存储 L3，
LMCache MP 常见命名为 host L1、后端 L2。
面试时先说明物理介质，再说层号，避免把同一个“L2”误当同一个位置。

延伸阅读
--------

* LMCache：:doc:`/mp/l2_storage/index`、:doc:`/mp/p2p`、
  :doc:`/mp/coordinator`、:doc:`/mp/disaggregated_prefill`。
* Mooncake 基线：
  `架构 <https://github.com/kvcache-ai/Mooncake/blob/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0/docs/source/design/architecture.md>`_、
  `Store <https://github.com/kvcache-ai/Mooncake/blob/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0/docs/source/design/store/mooncake-store.md>`_、
  `HiCache <https://github.com/kvcache-ai/Mooncake/blob/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0/docs/source/design/hicache-design.md>`_。
