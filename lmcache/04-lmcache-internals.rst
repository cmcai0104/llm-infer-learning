LMCache 架构与源码主线
======================

先建立阅读边界
--------------

本章描述学习基线的 MP 模式。先读 ``docs/design/`` 中与模块路径对应的文档，
再读实现与测试。每条路径都追问三个问题：谁持有数据、谁宣布完成、谁负责释放。
运行图见 :doc:`02-map-and-diagrams`。

.. list-table::
   :header-rows: 1
   :widths: 23 43 34

   * - 层
     - 源码入口（相对仓库根目录）
     - 阅读目标
   * - 引擎集成
     - ``lmcache/integration/vllm/lmcache_mp_connector.py``
     - vLLM callback 如何委派给 adapter
   * - 调度与 worker
     - ``lmcache/integration/vllm/vllm_multi_process_adapter.py``
     - 命中计划、block 分配、加载和存储完成
   * - 查找状态
     - ``lmcache/integration/vllm/mp_lookup_state.py``
     - callback、deadline、多 server 聚合与清理
   * - server 组合
     - ``lmcache/v1/multiprocess/server.py``
     - module、服务与存储管理器的装配
   * - 查找/预取
     - ``lmcache/v1/multiprocess/modules/lookup.py``
     - lookup 与消费一次的预取结果
   * - 设备传输
     - ``lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py``
     - STORE/RETRIEVE、事件依赖与对象生命周期
   * - 存储协议
     - ``lmcache/v1/distributed/api.py``
     - ObjectKey、分组 key、内存布局与预取类型
   * - L1
     - ``lmcache/v1/distributed/l1_manager.py``
     - staging、resident、读写保留与淘汰约束
   * - 跨层控制
     - ``lmcache/v1/distributed/storage_manager.py``
     - 组合 L1、controller、adapter 的公共入口
   * - L2 扩展
     - ``lmcache/v1/distributed/l2_adapters/base.py``
     - 异步提交、通知、结果和解锁契约

Connector 不只是 put/get 包装
-----------------------------

调度器需要知道多少输入可以跳过计算，worker 需要知道加载到哪些物理 GPU block，
两者之间需要传递元数据。Connector 把这些引擎语义转成缓存请求，
同时处理引擎结束请求、抢占、APC 已覆盖的 token 和异步存储。

因此一个“支持 KV 缓存的后端”不等于“已经接好推理引擎”。
后端即使能正确保存 bytes，若 connector 错误标记 computed tokens、
把数据搬到旧 block，或提前释放 page，仍会产生模型错误。

建议先在 ``lmcache_mp_connector.py`` 找
``get_num_new_matched_tokens``、``start_load_kv``、``wait_for_save``、
``get_finished`` 等 callback，再沿委派进入 adapter。
名字和 vLLM 版本有关；学习目标是掌握调度与数据完成的关系。

一次 lookup 的精确含义
-----------------------

``LookupState`` 管理每个 server 的 LOOKUP 与 QUERY_PREFETCH_STATUS：

1. 发送 LOOKUP。server 接受任务的 ack 只是确认查找/预取任务已登记。
2. 等该 server 自己的 ack 后再查状态，避免查询先于登记、被当成不存在的任务。
3. 返回 ``None`` 表示还没完成；返回整数表示最终命中 chunk 数。
4. 在本基线的多 server 聚合中，取共同的最短命中前缀，
   并释放其他 server 多出来、最终不会 retrieve 的尾部锁。
5. 已确定的结果不再被迟到响应改写；超时和结束请求必须有清理路径。

当前实现用 future callback 减少 scheduler step 间的等待，
但仍限制状态查询频率，避免“返回 pending→马上再查”的 RPC 忙循环。
查询结果有消费语义，重复查询不一定幂等；复用 request id 也必须遵守会话生命周期。

设计依据：``docs/design/integration/vllm/mp_lookup_state.md``。
该文档还记录了 mid-lookup abort 等已知边界；不要把目标契约说成所有路径已经完美覆盖。

key、chunk 与 group
-------------------

本基线 ``ObjectKey`` 明确包含：

.. code-block:: text

   chunk_hash: bytes
   model_name: str
   kv_rank: int
   object_group_id: int = 0
   cache_salt: str = ""

``chunk_hash`` 来自内容/前缀身份；``kv_rank`` 区分物理 KV 分片；
``object_group_id`` 区分同一 chunk 的不同对象组；
``cache_salt`` 支持命名隔离。模型修订号、dtype、LoRA 身份等是否由调用方
正确编码或限制，必须沿实际输入链检查，不能只看字段名就假定兼容。

``TokenHasher`` 在 ``lmcache/v1/multiprocess/token_hasher.py``；
本基线构造函数默认 ``blake3``，也有与 vLLM hash 接口兼容的路径。
不同语言、进程和部署必须统一算法、seed、输入编码和 namespace。

一个 hybrid chunk 可对应多个 object group 和 KV rank。
``GroupedObjectKeys`` 把命中信息组织成行列矩阵：

.. code-block:: text

                   chunk 0  chunk 1  chunk 2
   full group/r0      1        1        1
   full group/r1      1        0        1
   another group/r0   1        1        1
   another group/r1   1        1        1

若都按普通 full-attention 前缀要求解释，只能安全复用 chunk 0。
不能把所有 1 相加当作可跳过的 token，也不能跨过 chunk 1 的缺口。
滑窗 group 使用窗口语义，递归状态需要兼容的状态快照，不能机械沿用这个简化 AND。
相关类型在 ``api.py``，策略在 bitmap 与 prefetch policy 实现。

L1 写入：先 staging，再发布
---------------------------

``reserve_write(key, tag)`` 为新数据预留 staging。staging 对读者不可见，
直到真正完成数据填充并调用 finish 接口才进行 admission。
这防止 lookup 看到“地址已经分配，但内容还没搬完”的对象。

不同 writer tag 可以各自 staging 同 key；第一个发布者成为 resident，
后来的完成者发现已有 resident 时丢弃自己的 staging。
``finish_write_and_reserve_read`` 可在同一临界区发布并取读保留，
避免在两步之间被淘汰。相同 tag 的写入仍受排他约束。

``reserve_read`` 保持对象在读期间有效；读完必须 ``finish_read``。
正在使用、传输中或保留中的对象不能随意被普通淘汰策略回收。
过期写锁可使 abandoned staging 进入回收范围，但内存生命周期仍需配合实际设备完成。

三个容易混淆的标识：

* writer tag：区分对同 key 的不同 staging 写者。
* object key：标识缓存内容及分片。
* L1 owner id：多 L1 配置中标识分配对象的 manager，完成回调必须回到原 owner。

owner id 不是内容 key，也不是跨机器持久化的地址。
设计依据：``docs/design/v1/distributed/l1_manager.md`` 与
``docs/design/v1/distributed/storage_manager.md``。

L2 预取：不是固定串行扫描所有层
--------------------------------

``PrefetchController`` 在 L1 通过读保留观察对象，在 L2 通过
``lookup_and_lock`` 观察对象；把各层 bitmap 合并后规划来源。
每个计划单元必须在所选来源中受保护，且不能重复加载。
加载先进入 L1 staging，成功 admission 后成为供 retrieve 使用的对象。

默认 L2 prefetch policy 在多个后端同时拥有对象时选择第一个 adapter；
这不是自动测 RTT 后挑最快后端。默认加载对象为 temporary，在读者结束后删除；
``retain`` 策略才保留。默认 store policy 向所有配置的 L2 adapter 写入并保留 L1；
``skip_l1`` 则在写入后删除 L1 副本。见 :doc:`/mp/configuration`。

本基线的预取准入按 L1 内存预算控制：
同时计算 staging bytes 与已经 lookup、尚未分配的最坏加载 bytes，
避免一批任务在“当前内存看起来很空”时全部涌入。不能只用任务个数限制，
因为 1 个 128K 请求和 1 个 1K 请求的占用完全不同。

各 adapter 完成后可分别 admission，不必等最慢后端才发布已经成功的数据。
最终对成功且受保护的数据折叠出可用前缀；失败、取消和 shutdown
根据已记录的锁/预留集合清理。具体故障能否重试、可否部分服务要读调用方契约。

L2 adapter 的重要契约
---------------------

核心方式是 ``submit → 等待通知 → 查询结果``。需要掌握：

* store、lookup、load 有不同完成通知 fd，避免事件被错误路由。
* pending 与“已完成但全 miss/失败”的 bitmap 不同。
* lookup 结果是已找到且受锁保护的对象；load 仍须验证完成结果。
* 部分 query 结果只消费一次；store 完成队列用 pop 排空。
* ``submit_unlock`` 的接口要求 adapter 保证最终成功，controller 不替它重试。
* controller 持有调用方 buffer 的相应保留，直到 adapter 完成后才能释放。

这是接口契约。具体后端的锁可能依赖租约或弱化实现，故障时仍需核对适配层是否满足它；
例如不能因为后端提供 exists，就假定原子地阻止了下一刻淘汰。
设计依据：``docs/design/v1/distributed/l2_adapters/overall.md``。

设备拷贝、布局与异步完成
------------------------

GPU paged KV 与 L1 连续 chunk 的布局往往不同，需要 gather/scatter 或转换。
本基线把布局识别集中在 ``normalize_kv_and_discover_format``，
其他模块通过 ``EngineKVFormat`` 与公开 helper 查询，不应到处猜 tensor 维度。

LMCache-driven 路径可以使用 staging buffer 加 kernel，符合条件时也可选
直接批量 copy 路径。选择取决于 layout、设备运行时/驱动和内存 backing；
“所有复制都要消耗 SM”与“所有配置都能 direct copy”都不正确。

一条正确依赖链是：

.. code-block:: text

   引擎写 KV 完成事件
       -> 传输流等待
       -> gather/copy 或 direct copy
       -> 传输完成事件/回调
       -> 发布 L1 / 允许 GPU 消费 / 释放对应引用

RPC future 完成可能只代表 handler 提交了设备工作，不能无条件当作 DMA 完成。
同理，删除 Python 变量不代表底层 buffer 已经安全回收。
对 engine-driven 路径，传输发起位置不同，但上述依赖关系仍要满足。

抢占、APC 与混合模型
--------------------

抢占可能释放并复用 GPU page，异步 store 若继续读取旧 block id，会把其他请求的
KV 写入错误 key。恢复时 computed-token 进度与可加载范围也必须一致。
阅读 ``docs/design/integration/vllm/mp_preemption_correctness.md``，
理解为什么需要低并发 reference、高并发无缓存、冷缓存、热缓存、抢占热缓存的阶梯验证。

多种注意力 group 的 block size 和状态表示不同，chunk 协商与 group bitmap 是
正确性机制，而不仅是性能优化。SGLang 还有自身 radix-tree 生命周期；
阅读 ``docs/design/integration/sglang/lmcache-unified-radix-cache.md``，
不要把 vLLM scheduler 的细节直接投射过去。

源码阅读练习
------------

1. 从 ``get_num_new_matched_tokens`` 追到最终命中整数，标出每个异步边界。
2. 从 ``store`` 追到 L1 admission，找到在哪确认设备数据已经完成。
3. 找到 prefetch 失败时释放 L1 staging、L1 读锁与 L2 锁的位置。
4. 阅读 ``tests/v1/distributed/test_l1_manager.py`` 的 staging 行为，
   用公开接口解释“两个 writer、一个 resident”的预期。
5. 阅读 ``tests/v1/test_mp_lookup_state.py``，解释 ack 顺序与重复查询风险。

如果只能记住函数名却讲不出拥有者与释放者，继续追踪同一个请求即可，不必扩大阅读范围。
