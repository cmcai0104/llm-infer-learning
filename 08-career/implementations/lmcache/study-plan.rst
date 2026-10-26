28 天学习计划
=============

这是 LMCache 单项目深读计划。跨项目学习优先执行
`全库统一计划 <../../study-plan.md>`_，
本表用于补充缓存管理与后端集成专项任务。

每天约 2 小时，周末每天约 3.5 小时，四周共约 68 小时。每天分配为概念学习
30%、源码或实验 50%、复述与记录 20%。先保证 P0 的闭环，不以读完全部文件为目标。
本计划共 28 天，不额外要求预备周。

第一周：推理、容量与缓存语义
----------------------------

阅读 `公共基础 <../../../03-kv-cache/semantics-and-costs.rst>`__ 与 `知识地图 <../../../03-kv-cache/implementations/lmcache/state-diagrams.rst>`__。

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - 日期
     - 学习与动作
     - 当日产物/验收
   * - D1
     - 建立版本记录；读总览；画 Prefill→Decode
     - 用 3 分钟解释 LMCache 解决的重复计算问题
   * - D2
     - 从 attention 方程推导 K/V 的作用；比较 MHA、GQA、MQA
     - 手算 32 层、8 KV heads、8K 上下文的容量
   * - D3
     - 学习 PagedAttention、APC、逻辑块与物理块
     - 画两个请求共享前缀、独占尾块的示意图
   * - D4
     - 分析 token、模型、位置和上下文依赖；运行离线哈希练习
     - 证明相同后缀不等于相同 KV；列出 key 隔离维度
   * - D5
     - 学习 chunk 与 block；比较不同 chunk 大小
     - 算 255/256/257 tokens 在 chunk=256 下的完整块数
   * - D6
     - 阅读 MP quickstart、兼容性文档；有 GPU 者完成最小请求
     - 环境清单与日志；无 GPU 者完成容量及带宽计算脚本
   * - D7
     - 做 Q1–Q12；闭卷复画知识地图
     - 一页基础笔记，列出仍不能解释的三个问题

第二周：LMCache 源码主线
------------------------

阅读 `LMCache 内核 <../../../03-kv-cache/implementations/lmcache/internals.rst>`__。按设计文档→公共接口→主调用→测试的顺序，
不要从一个几千行文件的第一行机械读到底。

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - 日期
     - 学习与动作
     - 当日产物/验收
   * - D8
     - 读 connector 与 scheduler/worker adapter
     - 标出调度器查命中与 worker 搬数据的边界
   * - D9
     - 读 LookupState、LookupModule
     - 画 LOOKUP ack、查询完成、超时、尾锁释放时序
   * - D10
     - 读 ObjectKey、GroupedObjectKeys、PrefetchTaskSpec
     - 手工画两 rank、两 group、三 chunk 的命中矩阵
   * - D11
     - 读 L1Manager staging/resident 契约与测试
     - 解释并发写同 key、读中淘汰、过期写的处理
   * - D12
     - 读 retrieve/store 与设备事件、内存所有权
     - 画 buffer 从申请到 DMA 完成再到释放的生命周期
   * - D13
     - 读 PrefetchController、StoreController、L2AdapterInterface
     - 解释 None/pending 与空 bitmap/miss 的区别
   * - D14
     - 复盘冷缓存、L1 命中、L2 命中三条路径；做 Q13–Q25
     - 写 800 字源码导读，附符号与设计文档位置

第三周：存储、Mooncake 与性能
-----------------------------

阅读 `存储与 Mooncake <../../../04-storage/implementations/lmcache/backend-contracts.rst>`__、`性能与运维 <../../../06-serving/implementations/lmcache/observability.rst>`__；
按 `实验手册 <../../../07-practice/implementations/lmcache/labs.rst>`__ 选择硬件允许的实验。

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - 日期
     - 学习与动作
     - 当日产物/验收
   * - D15
     - 比较 DRAM、SSD、对象存储、远端内存；学习准入与淘汰
     - 给三种 workload 选择 tier，并解释成本
   * - D16
     - 读 Mooncake Store 架构、Put/Get、租约与副本
     - 画 Master 不经手 payload 的读写时序
   * - D17
     - 学习 RDMA、MR、QP/CQ、NUMA、GPUDirect
     - 解释“提交成功、传输完成、对象可读”三个时刻
   * - D18
     - 阅读 LMCache Mooncake native adapter 配置/实现
     - 标出 L1 注册、配置转发、操作线程池边界
   * - D19
     - 设计并运行 APC 与 LMCache 归因对照
     - 冷/热状态清单；token 命中率与 TTFT 表
   * - D20
     - 改变 chunk、并发或共享前缀比例中的一个变量
     - 一张带原始记录的曲线；无硬件者提供敏感性分析
   * - D21
     - 做 Q26–Q40；分析命中高但变慢的案例
     - 5 分钟讲清读取与重算的盈亏条件

第四周：高级主题与面试交付
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 42 50

   * - 日期
     - 学习与动作
     - 当日产物/验收
   * - D22
     - 学习 P/D 分离、NIXL、MultiConnector
     - 分开画当前请求 KV 交接与跨请求复用
   * - D23
     - 学习 CacheBlend、CacheGen/serde、量化与质量评估
     - 给出一个适用场景和一个不适用场景
   * - D24
     - 学习混合模型、MLA、滑窗、抢占恢复
     - 说明为什么总容量不能机械除以 TP
   * - D25
     - 故障桌面演练：超时、节点退出、半写、取消、租户热点
     - 故障→检测→回退→资源释放的检查表
   * - D26
     - 完成系统设计题，预算容量、流量、元数据与冗余
     - 2–4 页方案和架构图
   * - D27
     - 45 分钟模拟面试：基础 10 分钟、源码 10 分钟、设计 25 分钟
     - 对每个回答标注“有证据/需验证/不知道”
   * - D28
     - 修正薄弱点；整理实验、源码导航和面试速记
     - 完成下述毕业验收，并保留可复现命令

毕业验收与时间调整
------------------

每项按 0/1/2 评分：0=讲不清；1=能解释；2=有计算、源码或实验支撑。

1. 正确计算 KV 容量，并解释 GQA、TP、MLA 的适用条件。
2. 画出 MP 冷写、热读、L2 预取与完成事件。
3. 解释 key、layout、跨 rank 完整性和取消后的资源释放。
4. 画出 LMCache + Mooncake 的控制与数据路径。
5. 用测量或带假设的模型判断何时重算更划算。
6. 设计容量、带宽、SLO、隔离和故障回退，并回答追问。

达到 10/12 且没有 0 分，说明已具备初步岗位讨论能力；这不是生产经验的替代。
每天只有 1 小时时，保留 D1–D21 的 P0 内容和 D26–D28，压缩 P1/P2 深挖，
仍在 28 天内完成。已有分布式存储经验者把基础时间转移到 KV 语义、设备同步和引擎集成。
