优化、可观测与生产设计
======================

先定义指标再谈收益
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - 指标
     - 定义/测量口径
     - 误用
   * - TTFT
     - 从指定起点到首个有效输出 token；注明是否包含网络与排队
     - 非流式 HTTP 总耗时当作 TTFT
   * - ITL / TPOT
     - token 间延迟；TPOT 是一段生成的平均每 token 时间
     - 平均值代替 p99 或掩盖暂停
   * - Goodput
     - 满足 TTFT/ITL/成功率等 SLO 的有效请求或 token 每秒
     - 只统计总吞吐而忽略超时
   * - 请求命中率
     - 至少有可用命中的请求数/请求数，或全命中比例；必须注明定义
     - 把一次命中 1 chunk 算成整个请求全命中
   * - Token 命中率
     - 可复用 token/参与查询的 token；明确 APC 与外部 cache 的分母
     - 合并重叠来源的计数导致重复计算
   * - 实际读取带宽
     - 完成传输的 payload bytes/指定时间窗口
     - 使用提交字节数、压缩前逻辑大小冒充物理流量
   * - 缓存成本
     - DRAM/SSD/网络/存储请求费用及省下的 GPU 计算
     - 只看命中不计写入与副本流量

平均命中率不能说明尾延迟。建议分开报告 cold、L1 hit、L2 hit、partial hit、
lookup timeout、load failure/recompute；按 prompt 长度、租户与并发分桶。
监控生产请求时应避免把原文、token 内容或无限 request id 放进指标标签。

正确的性能实验阶梯
------------------

1. 无 LMCache 的引擎基线：固定模型、dtype、batch、并发、输入/输出分布。
2. 引擎 APC：确认本地前缀复用能带来什么。
3. APC + LMCache 冷缓存：测 store 与集成开销，不能称为“零收益所以失败”。
4. APC + LMCache 热缓存：区分 GPU 命中与真正从 LMCache 读取。
5. 新引擎进程连接仍存活的 LMCache：排除旧 GPU APC 的主要干扰。
6. 更大工作集/更高并发：测试超过 GPU 或 L1 容量后的实际价值。
7. 独立测试失败、取消、抢占和重启；不能只验证 happy path。

每组先预热，再重复多轮，保留原始输出、版本、启动配置和指标快照。
请求数要能支撑所报分位数；几十个请求不足以可靠说明生产 p99。
混合模型要增加 group/layout 支持和输出一致性验证，不能只迁移普通模型结论。

吞吐瓶颈如何拆
--------------

记录查询、后端读取、CPU buffer 等待、网络、H2D、剩余 prefill、首 token
和后续 decode 的时间。存在 overlap 时用 trace 的关键路径解释端到端延迟，
不要将多个重叠 span 直接求和。

.. code-block:: text

   端到端慢
     ├─ lookup 慢：RPC 排队 / 元数据压力 / 过多小对象 / 查询轮询
     ├─ load 慢：后端 I/O / 网络拥塞 / 热点 / CPU copy / 注册
     ├─ H2D 慢：NUMA / PCIe / layout 转换 / chunk 太碎 / 设备同步
     ├─ hit 不可用：前缀断裂 / group 缺失 / rank 不齐 / APC 已覆盖
     └─ decode 慢：copy 和计算争带宽 / 过大 batch / 抢占

Little 定律 ``在途量 ≈ 吞吐 × 平均停留时间`` 可以估算 buffer 预算。
例如 20 GiB/s 的持续输入、0.1 s 的平均 staging 停留，平均需要约 2 GiB staging；
还必须给尾延迟、突发、lookup 未分配预算和双缓冲留余量。
该近似不允许把无限排队当作带宽扩容。

主要优化项及代价
----------------

**Chunk 大小**：大 chunk 摊薄 hash、RPC 与元数据开销，提高顺序带宽，
但尾部不可复用、部分命中和过取增加；小 chunk 更灵活，却可能被提交开销限制。
本基线常用默认 256 tokens，但应遵守 block/group 协商并按 workload 验证。

**Batch 与流水**：合并 lookup、批量传输和流水可以提高利用率；
过大的 batch 延迟首个可消费结果，并放大失败/取消时占用。
流水稳态接近各阶段最大耗时，而不是各阶段之和，但会有启动、收尾和资源争用。

**Layerwise**：加载下一层与当前层计算重叠，减少关键路径上暴露的传输时间。
要有正确的 per-layer 数据依赖、layout 与支持的模型路径；
不是把 ``async`` 加在函数名上就自然 overlap。

**压缩/量化**：减少字节数但增加编码/解码时间、scale 元数据和可能的质量损失。
在带宽受限路径上有潜力，在 PCIe/GPU 已空闲而编码昂贵时可能变慢。
比较完整的 ``encode + transfer + decode`` 与原始路径，并报告任务质量。
CacheGen、普通 KV dtype 量化与任意 serde 插件不是同一个算法。
见 `CacheGen <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/cachegen.rst>`__、
`序列化 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/serde.rst>`__。

**CacheBlend**：复用非前缀片段并选择性重算以恢复跨片段依赖。
必须评价真实 RAG/多文档任务的质量和时延，不能宣称任意拼接与完整 prefill
逐位等价。MP 与进程内启用方式不同，见 `CacheBlend <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/kv_cache_optimizations/cacheblend.rst>`__。

**Lazy offload**：减少对未来未必重用数据的立即写出，但需与引擎淘汰及抢占生命周期协同；
不能等 page 已复用再去读取旧 KV。见 `Lazy offload <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/lazy_offload.rst>`__。

**缓存感知路由**：可用候选代价估计
``排队 + 缺失前缀重算 + 加载/传输``，而不是永远选 hit 最长的实例。
这是设计建议；目录陈旧时需要实际 lookup 验证，并控制热点和租户公平性。

故障矩阵
--------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - 现象
     - 检测
     - 正确处理原则
   * - lookup 超时
     - per-RPC deadline、server health、队列长度
     - 终结本次计划或安全降级；迟到回复不能覆写结果；释放未用锁
   * - L2 部分读失败
     - 完成 bitmap、bytes、adapter 错误
     - 只计成功且语义完整的前缀；不要使用未完成 buffer
   * - 请求取消
     - request/session 状态与在途操作
     - 停止新工作，等待或隔离已提交 DMA，再释放引用
   * - L1 空间不足
     - resident/staging/lookup 预算、eviction eligible bytes
     - 背压、减少预取或准入；不要强制回收正在读的数据
   * - 存储节点退出
     - membership、读错误、副本状态
     - 选择有效副本或重算；限制重算风暴与重新复制流量
   * - 引擎抢占
     - preemption 事件、page 生命周期
     - 避免旧 store 读到新请求页面；恢复 token 进度与 KV 必须一致
   * - 模型/布局变更
     - namespace、版本和 layout 检查
     - 隔离或拒绝不兼容对象，不能把 shape 相同视为语义相同
   * - 读完但质量异常
     - 对照输出、校验和、token/key/layout/rank 记录
     - 从身份和生命周期排查；校验和无法发现“正确 bytes 对错请求”

“回退到重算”需要 connector 与 scheduler 共同支持，并且未使用坏数据；
不能任意捕获异常后继续 forward。某些部署要求立即报错而非回退，
例如严格 P/D 的传输失败策略。缓存 miss、load failure、deadline exceeded
在指标中仍应可区分，哪怕某一接口层最终把它们都折叠成零命中。

安全、隔离与生产就绪
--------------------

KV 可能携带敏感上下文信息。至少设计模型/租户命名空间、服务身份认证、
网络访问控制、加密需求、数据保留与删除、审计，以及避免跨租户命中侧信道。
``cache_salt`` 能隔离 key，但如果任意用户可冒充另一人的 salt，它不是访问控制。

内存预算要包括：
``resident + staging + 在途传输 buffer + allocator/注册开销 + 运行时 + 余量``。
磁盘预算要包括副本、临时文件与写放大；用文件后端测试时注明 OS page cache 状态。

生产前最小检查：

* 用固定版本验证 runtime/ABI、connector 加载来源、model/layout/group 支持。
* 区分 request RPC、HTTP 管理、Mooncake Master 与 metadata 的端口和网络边界。
* 限制预取、写回、连接、线程和租户资源，保证 read/lookup 不被大写入饿死。
* 验证 graceful drain：停止接新任务，完成/取消在途操作，解锁后释放内存。
* 明确缓存丢失后的恢复负载和 SLO，不只测试进程能重新启动。
* 建立 TTFT/ITL、实际 token 命中、staging、带宽、失败和锁/会话泄漏的观测面。

观测入口见 `Observability <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/observability/index.rst>`__、
`HTTP API <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/http_api.rst>`__。
HTTP 200 本身不一定代表所有后端健康，应阅读具体 endpoint 的响应语义。
