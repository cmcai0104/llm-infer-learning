系统设计题、源码导航与阅读清单
==============================

设计题一：长文档问答的共享 KV 层
--------------------------------

需求与假设
~~~~~~~~~~

假设一个 RAG 服务使用 32 层、8 KV heads、128 head dimension、BF16 的普通 GQA 模型。
每次输入 8K tokens，常见文档前缀 6K；请求峰值 20 req/s；热文档前缀
约 1,000 个。目标是降低 TTFT，同时不给 decode 尾延迟造成明显回退。
这些是练习输入，不是 LMCache/Mooncake 的公开 benchmark。

继续向面试官确认：6K 是在同一 system prompt、模型版本与位置下的精确前缀吗？
热文档是否真的共享相同 tokenization？输出长度与 burst 多大？
SLO 是 p95 还是 p99？跨租户是否允许共享？网络多少、可用率要求多少？
若 system prompt 不同，普通前缀缓存的有效共享可能大幅下降。

容量预算
~~~~~~~~

沿用 128 KiB/token：

* 每条 6K 前缀为 0.75 GiB；1,000 条共 750 GiB 逻辑 KV。
* 若保留 2 份副本，为 1,500 GiB payload。
* 若 resident 预算只占物理池的 70%，最低约需 2,143 GiB；
  剩余 30% 要明确分配给 staging、元数据、碎片、系统与突发，而不是视为免费容量。
* chunk=256 时，每前缀 24 个 chunk，合计 24,000 个逻辑 chunk。
  TP=4 且每 rank 各存一个对象、单 group 时为 96,000 个对象身份，
  双副本对应 192,000 个物理副本记录。

一个对象身份可以有多个副本，不要把对象数与副本数混为一谈。
若有多个 group 或使用不同 chunk 策略，对象数继续变化。
对于这个均匀分片例子，每个 rank/chunk 的 payload 为 8 MiB。

带宽与回写预算
~~~~~~~~~~~~~~

若 20 req/s 全部从远端读满 6K 前缀：

.. math::

   B_{\mathrm{read}} = 20 \times 0.75 = 15\ \mathrm{GiB/s}
   \approx 128.85\ \mathrm{Gbit/s}

这是聚合 payload 读取，不含协议、同机 H2D、控制面和流量波动。
单 200G 的 25 GB/s 线速约为 23.28 GiB/s；若按 70% 有效利用预算，
约 16.30 GiB/s，已经十分接近 15 GiB/s 的持续读负载。
不能在这个余量下再承诺突发、故障与复制都毫无影响。

若 10% 请求产生新的完整 8K 前缀，每条 1 GiB，
双副本写出的逻辑写流量为 ``20 × 10% × 1 × 2 = 4 GiB/s``。
读取与写入是否争同一方向、同一 NIC/PCIe/存储池，要按实际拓扑拆开，
不能简单把全双工链路两方向相加就宣布超过线速；
但 DRAM 带宽、CPU 和交换网络仍可能共享瓶颈。

每秒需要查找约 ``20 × 24 × 4 = 1,920`` 个 rank/chunk 身份；
批处理可减少 RPC 数，但没有消除目录工作。还需计入 store、续约、淘汰与重试流量。
若平均 staging 停留 100 ms，15 GiB/s 意味着平均约 1.5 GiB 在途；
给 p99 与 burst 预留额外空间。

组件与请求流程
~~~~~~~~~~~~~~

.. mermaid::
   :name: learning-rag-system
   :config: {"flowchart": {"useMaxWidth": false}}

   flowchart TB
     API["Gateway / 鉴权 / 限流"] --> ROUTE["缓存感知 Router"]
     ROUTE --> ENGINE["兼容的 vLLM 实例池"]
     ENGINE <--> LOCAL["每节点 LMCache MP / host L1"]
     LOCAL <--> POOL["Mooncake Store / 远端内存池"]
     POOL -.-> MASTER["Master / HA 控制面"]
     INDEX["可选缓存目录 / 事件索引"] -.-> ROUTE
     LOCAL -.-> INDEX
     OBS["Metrics / Traces / SLO"] -.-> ENGINE
     OBS -.-> LOCAL
     OBS -.-> POOL

1. Gateway 认证，建立允许使用的租户/模型 namespace。
2. Router 同时看队列、前缀位置与加载成本，选择候选实例。
3. 引擎先识别本地 APC；LMCache 查询额外可复用范围。
4. L1/L2 在保留与预算约束下加载，检查所有所需 rank/group。
5. 把命中范围放入 GPU，执行剩余 prefill/decode。
6. 对可重用数据异步存储；事件索引用于路由，真实读取仍验证对象状态。

降级与一致性
~~~~~~~~~~~~

对象 miss 或可安全恢复的读取失败走重算；不兼容布局、坏数据或未知完成状态
不能继续当命中。Store 节点故障时尝试有效副本，否则降低 cache 依赖并限制回源。
Master HA、存储副本和 engine 重启分别测试，不能互相替代。

模型 revision 或 LoRA 变更使用新命名空间；对旧对象执行受控过期/清理策略。
同内容跨租户共享必须先明确授权和侧信道边界，默认按租户隔离更容易论证。
元数据陈旧允许引发 miss，但不允许错误加载。

验收标准
~~~~~~~~

比较 APC-only、L1、远端缓存三种配置在相同 trace 下的：
TTFT/ITL p50/p95/p99、goodput、GPU 利用与抢占、实际传输量、命中 token、
host/staging 使用、失败率及输出正确性。
测一次节点退出时的尾延迟与恢复时间；若缓存平均收益高但失败时压垮 GPU，
方案还没有达到生产要求。

设计题二：P/D 分离 + 跨请求复用
--------------------------------

不要先选择传输库，先识别两种数据需求：

* 本次请求：Prefill 计算出的 KV 要交给 Decode，交接失败可能需要明确失败而非重算。
* 后续请求：已计算的前缀可存入共享层，让任意兼容的 Prefill worker 复用。

当前 LMCache MP recipe 是 ``MultiConnector`` 中组合 ``NixlConnector`` 与
``LMCacheMPConnector``，两边引擎各配一个 LMCache server。参考
:doc:`/mp/disaggregated_prefill` 的版本前提和 real-blocks 修复要求。
另一个系统可用 Mooncake Transfer Engine 做交接，但不能直接套用 NIXL 的配置。

容量推导继续区分：请求级 KV 交接速率、跨请求读写速率、Decode 在途序列 KV、
不同 TP/PP 布局是否兼容、交接期间双份 buffer 和副本保存周期。
如果 Prefill 与 Decode 并行度不同，必须有已验证的 layout/resharding 支持；
“传输 bytes 成功”不意味着布局自动转换。

面试追问：哪个阶段产生首 token？Router 如何转发 stream？
何时允许 Prefill 释放原始 GPU KV？Decode 超时后清理谁的对象？
跨请求 cache 写入失败是否影响当前请求？这些答案必须来自具体协议，而非抽象方框。

设计题三：RAG 片段顺序变化
---------------------------

同一个文档被放在不同 system prompt、不同文档前后时，普通精确前缀命中减少。
有三类方案：

1. 规范模板与文档顺序，提高真实前缀重复率；代价是可能改变业务提示设计。
2. 只缓存稳定的共同前缀，其余重算；简单且容易验证。
3. 采用 CacheBlend 等非前缀复用与选择性重算；需模型/任务质量验证和额外运行时支持。

比较时同时报告质量与时延，而不是只统计“文档块在存储里存在”的比例。
不能直接对旧 K 做位置旋转就宣称所有层与新上下文一致。

源码导航：按问题找文件
----------------------

以下均为 LMCache 仓库相对路径。先读 design，再看实现，最后看测试；
这些是入口，不代表要求一个月内精读整个目录。

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - 问题
     - 设计/文档
     - 实现/测试入口
   * - vLLM 查命中怎么推进？
     - ``docs/design/integration/vllm/mp_lookup_state.md``
     - ``lmcache/integration/vllm/mp_lookup_state.py``；
       ``tests/v1/test_mp_lookup_state.py``
   * - key 与多 group 如何表示？
     - ``docs/design/v1/distributed/storage_manager.md``
     - ``lmcache/v1/distributed/api.py``；
       ``lmcache/v1/multiprocess/token_hasher.py``
   * - 谁保护正在写的数据？
     - ``docs/design/v1/distributed/l1_manager.md``
     - ``lmcache/v1/distributed/l1_manager.py``；
       ``tests/v1/distributed/test_l1_manager.py``
   * - 跨层预取如何规划与限流？
     - ``docs/design/v1/distributed/storage_controllers/prefetch_controller.md``
     - ``lmcache/v1/distributed/storage_controllers/prefetch_controller.py``
   * - L2 adapter 要遵守什么？
     - ``docs/design/v1/distributed/l2_adapters/overall.md``
     - ``lmcache/v1/distributed/l2_adapters/base.py``
   * - 如何接 Mooncake？
     - :doc:`/mp/l2_storage/mooncake_store`
     - ``lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py``；
       ``csrc/storage_backends/mooncake/``
   * - GPU KV layout 从哪里识别？
     - ``docs/design/v1/gpu_connector/layout-invariant.md``
     - ``lmcache/v1/gpu_connector/utils.py``；
       ``lmcache/v1/gpu_connector/kv_format/``
   * - copy 为什么有不同路径？
     - ``docs/design/v1/multiprocess/lmcache_driven_transfer_copy_paths.md``
     - ``lmcache/v1/multiprocess/object_group_transfer.py``
   * - 抢占恢复如何验证？
     - ``docs/design/integration/vllm/mp_preemption_correctness.md``
     - ``lmcache/integration/vllm/vllm_multi_process_adapter.py``
   * - SGLang 集成哪里不同？
     - ``docs/design/integration/sglang/lmcache-unified-radix-cache.md``
     - ``lmcache/integration/sglang/``
   * - 旧版 LMCacheEngine 在哪里？
     - :doc:`/legacy/index`
     - ``lmcache/inprocess/v1/cache_engine.py``；
       ``lmcache/inprocess/v1/token_database.py``

本仓库测试可能依赖 GPU/native 扩展与 fixtures，不应把离线练习当作跑过这些测试。
具体测试环境与命令以仓库 ``AGENTS.md`` 和 :doc:`/developer_guide/contributing` 为准。

Mooncake 对照导航（Mooncake 仓库相对路径）：

* ``docs/source/design/architecture.md``：组件边界与控制/数据分离。
* ``docs/source/design/store/mooncake-store.md``：Put/Get、对象与租约。
* ``docs/source/design/transfer-engine/index.md``：segment 与传输 API。
* ``docs/source/design/hicache-design.md``：SGLang 分层缓存、预取与布局。
* ``docs/source/api-reference/python/mooncake-store.md``：真实 Python 返回值与前提。
* ``mooncake-store/``、``mooncake-transfer-engine/``：深入对应 C++ 实现。

进一步阅读顺序
--------------

第一层：可运行的系统
~~~~~~~~~~~~~~~~~~~~

* :doc:`/getting_started/quickstart`、:doc:`/getting_started/compatibility`：
  先确定版本与运行路径。
* :doc:`/mp/index`、:doc:`/mp/configuration`：部署边界与真实默认值。
* :doc:`/mp/observability/index`：把架构中的各个阶段变成可测事件。
* :doc:`/mp/l2_storage/mooncake_store`：组合两套系统，而不是孤立学 API。

第二层：带着问题读论文与设计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

建议检索并阅读以下准确标题；论文中的实验只适用于其报告的配置，
不能替代本机复现，也不要把论文设计全都说成当前默认实现：

* **Efficient Memory Management for Large Language Model Serving with PagedAttention**：
  物理页如何共享？何时 copy-on-write？没有外部存储时容量瓶颈在哪里？
* **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving**：
  为什么把 KV 放在资源调度中心？存储、传输与调度各解决什么？
* **SGLang: Efficient Execution of Structured Language Model Programs**：
  radix 共享如何与程序结构/请求复用结合？
* `CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
  <https://dl.acm.org/doi/10.1145/3651890.3672274>`_：
  压缩和 streaming 分别省掉什么，何时编码代价更高？
* `CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion
  <https://arxiv.org/abs/2405.16444>`_：
  非前缀复用为什么需要选择性重算，质量如何评价？
* **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**：
  为什么优化 attention kernel 的 I/O 不等于解决跨请求 KV 存储？

第三层：选一个方向深挖
~~~~~~~~~~~~~~~~~~~~~~

* **KV 存储工程师**：adapter、异步 I/O、slab/allocator、配额、租约、HA 与故障恢复。
* **推理运行时工程师**：connector、scheduler、APC、TP/group、设备事件和抢占。
* **传输工程师**：RDMA 注册、QP/CQ、NUMA、batch、拓扑与 buffer 生命周期。
* **性能工程师**：可复现 benchmark、trace 关键路径、goodput、成本模型与质量约束。

一个月的合理目标是掌握一条完整链路和一个深挖点，而不是背完全部框架的所有参数。
准备面试时，用自己实际完成的实验与证据回答，明确区分实测、阅读结论和设计假设。
