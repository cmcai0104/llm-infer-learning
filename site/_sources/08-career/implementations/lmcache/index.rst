LMCache 中文学习指南
====================

跨项目学习先读 `全库总目录 <../../../README.md>`_ 和
`LMCache 白话介绍 <../../../03-kv-cache/implementations/lmcache/overview.md>`_，按
`统一学习计划 <../../study-plan.md>`_ 推进。
本页保留 LMCache 专题深读导航，以下 28 天安排是单项目备选。

这套资料面向希望进入 LLM 推理基础设施、KV Cache 存储与数据传输岗位的工程师。
目标是在 **28 天、约 70 小时** 内，从容量计算和缓存正确性出发，读懂 LMCache
主路径，解释它如何与 Mooncake 组合，并完成一份有实验依据的系统设计。
默认掌握 Python、基本 C++、操作系统与网络基础；不要求先读完 Transformer 论文。

阅读基线与边界
--------------

资料核对日期为 2026-10-26，源码基线为：

* `LMCache ff9185d419e1 <https://github.com/LMCache/LMCache/tree/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be>`_
  （2026-10-23）：以推荐的 multiprocess（MP）独立服务模式为主。
* `Mooncake dd1217b6c461 <https://github.com/kvcache-ai/Mooncake/tree/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0>`_
  （2026-10-23）：重点是 Store 与 Transfer Engine。

这是针对上述提交的学习资料，不代表任意 PyPI 版本的功能承诺。旧文章中的
``lmcache/v1/cache_engine.py`` 在当前树中对应
``lmcache/inprocess/v1/cache_engine.py``；MP 主线使用另一组 server、module、
StorageManager 接口。实验前记录自己的提交、包版本和运行时，
并阅读 `兼容性说明 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/getting_started/compatibility.rst>`__。

全文区分三种内容：标明源码路径的内容描述本基线实现；公式与取舍讨论是通用原理；
明确写为“教学模型”“设计建议”的内容不是 LMCache 已实现的算法。
性能数字若无实测说明，均为带假设的算例。

学习大纲与岗位优先级
--------------------

.. list-table::
   :header-rows: 1
   :widths: 12 28 35 25

   * - 优先级
     - 模块
     - 必须理解
     - 可展示成果
   * - P0
     - 推理与 KV 基础
     - Prefill/Decode、MHA/GQA/MLA、容量、TTFT/ITL、前缀依赖
     - 独立推导容量与传输下界
   * - P0
     - LMCache 主路径
     - Connector、lookup、prefetch、retrieve、store、会话与完成事件
     - 不看资料画请求时序图
   * - P0
     - KV 存储正确性
     - key、layout、分片、staging、锁、淘汰、失败与回收
     - 解释三种竞争条件和解决办法
   * - P0
     - Mooncake 与网络
     - Master/Client/Transfer Engine、RDMA、注册内存、控制与数据路径
     - 画出 LMCache + Mooncake 组合图
   * - P0
     - 性能与系统设计
     - 读取/重算盈亏、带宽预算、命中率口径、SLO、回源风暴
     - 一份可复现实验或离线预算报告
   * - P1
     - 分布式推理
     - P/D 分离、TP 完整性、缓存感知路由、抢占恢复
     - 设计跨节点共享缓存
   * - P1
     - 高级复用
     - CacheBlend、压缩、layerwise、混合注意力、滑窗与递归状态
     - 说明适用条件及质量验证
   * - P2
     - 专项深挖
     - GDS/Device-DAX、多 L1、native adapter、PG/EP、调度策略扩展
     - 选择一个方向做源码专题

“熟练掌握”意味着：能讲清概念，能算数量级，能找到代码证据，能设计反例，
能用实验区分原因。只记住配置项或背“RDMA 零拷贝”不满足这一标准。

如何使用
--------

单项目深读可按 `专项计划 <study-plan.rst>`__ 推进；先读 `公共基础 <../../../03-kv-cache/semantics-and-costs.rst>`__，再对照
`知识地图 <../../../03-kv-cache/implementations/lmcache/state-diagrams.rst>`__ 阅读源码。无 GPU 的 Mac/普通笔记本可完成容量、
前缀哈希、时延模型和源码练习；GPU 推理与 RDMA 实验需要相应 Linux 环境。
不具备硬件时，把结果明确写为“理论估算/未测”，不要填造吞吐数据。

面试准备优先读 `面试问答 <questions.rst>`__ 的 Q1–Q40，再完成
`系统设计 <system-design.rst>`__ 的容量与故障设计题。
已有 Mooncake 基础者可先读 `LMCache 内核 <../../../03-kv-cache/implementations/lmcache/internals.rst>`__，补足推理引擎与缓存之间的契约。

专题目录
--------

* `01 专项学习计划 <study-plan.rst>`__
* `02 知识地图与设计图 <../../../03-kv-cache/implementations/lmcache/state-diagrams.rst>`__
* `03 推理与缓存基础 <../../../03-kv-cache/semantics-and-costs.rst>`__
* `04 LMCache 内核 <../../../03-kv-cache/implementations/lmcache/internals.rst>`__
* `05 存储与 Mooncake <../../../04-storage/implementations/lmcache/backend-contracts.rst>`__
* `06 性能与运维 <../../../06-serving/implementations/lmcache/observability.rst>`__
* `07 实验手册 <../../../07-practice/implementations/lmcache/labs.rst>`__
* `08 面试问答 <questions.rst>`__
* `09 系统设计与阅读 <system-design.rst>`__
