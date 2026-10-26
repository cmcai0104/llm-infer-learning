# Mooncake：KV 对象怎样共享、怎样安全地高速搬运

[返回存储主题](../../README.md) · [上层后端契约](../lmcache/backend-contracts.rst) · [相关主题：传输](../../../05-transfer/README.md)

Mooncake 学习主线分成两件事：**Store 管对象在哪里、是否可读、何时能回收；Transfer Engine 管字节怎样从源内存到目标内存。** 对 KV Cache 存储岗，这两层都要熟练，同时理解上层 Connector 对内容、布局和完成状态的要求。

本文以 C++ Store 和经典 TE 为主。论文中的完整 KV-centric 服务架构、Store-RS、TENT、Conductor、EP/PG 等有各自边界，进阶再展开。全部专题与单项目备选计划见[深读目录](../../../08-career/implementations/mooncake/index.md)。

## 1. 从两台推理机器的重复计算说起

机器 A 刚算完一份长文档，机器 B 马上收到相同前缀。如果 KV 只在 A 的 GPU 上，B 可能重复 Prefill。一个共享池可以让 A 保存状态，B 按 key 查询并取回。

但共享池需要解决的不只是“有块大内存”：对象放在哪个节点？写到一半能不能读？满了删谁？读的时候能不能删？节点失联怎么办？Master 故障后谁继续管目录？

Store 处理这些对象与资源问题。上层负责把模型、前缀与布局语义编码进兼容的对象体系。

## 2. Master、Client、TE 分别做什么

| 角色 | 管什么 | 白话理解 |
|---|---|---|
| Store Master | 对象元数据、资源位置、可读副本、租约、分配和回收等 | 管目录与分配，不中转每块大数据 |
| Client / 资源提供者 | 调用读写协议，持有或贡献内存等资源 | 真正拥有数据的进程，以及要访问数据的进程 |
| Transfer Engine | 注册内存、发现可访问区域、提交和完成传输 | 在明确的源和目标之间搬 bytes |
| 传输元数据机制 | 交换 segment、endpoint、访问描述等 | 让 TE 知道如何访问对方 |

```text
              查询 / 分配 / 完成发布
读写 Client  ------------------------> Store Master
    │
    └──── 数据传输 ────> 资源提供者的内存 / 适用存储后端
```

大块 payload 不经 Master 中转，能减轻 Master 带宽压力，但 Master 仍可能被小对象数量、RPC、锁竞争、元数据内存和回收工作压满。

Store Master 与 TE metadata server 也不是同一个概念。即使某个部署将服务合在一起运行，职责仍然不同。

## 3. 一次 Put 为什么要分阶段

普通写入的概念流程：

1. Client 告诉 Master：要写哪个 key、多大、期望什么副本配置。
2. `PutStart` 分配位置并创建写入中的状态。
3. Client 向目标资源写数据；内存路径使用相应传输能力。
4. 汇总完成状态，执行 `PutEnd` 发布可读副本，或按失败路径 `PutRevoke` 撤销相应状态。
5. 读者通过目录获得已完成副本。

如果第二步就让读者使用，会读到未写完的数据。如果写者崩溃后永不清理，会泄漏空间。因此“可读性”和“失败回收”都要有明确规则。

这不是多 key 事务。batch 操作也需要逐项处理成功与失败。

当前普通 `put` 路径在对象已存在时可能成功返回而不覆盖旧内容。不能拿两次 `put(same_key, different_value)` 当成更新数据库。内容寻址缓存应让同 key 表示同内容；更新需求需研究版本键或适用的 `upsert` 语义。

## 4. 一次 Get 为什么网络成功还不够

1. 查询可读副本与相应读保护。
2. 选择副本，读取到目标 buffer。
3. 检查传输状态，以及适用的完整性校验、租约等条件。
4. 成功以后交给上层处理布局与 GPU 同步。

读取期间，地址可能面临回收或节点失效。当前 Client 的租约检查会影响最终成功与否。诊断校验发现数据损坏也应拒绝使用。

`get_into` 直接写调用者的内存，失败时 buffer 可能已经部分改变。**里面有 bytes 不代表这次 Get 成功**。上层必须使用明确完成状态决定是否允许 GPU 消费。

## 5. 四个容易混淆的生命周期概念

| 概念 | 解决的问题 | 不保证什么 |
|---|---|---|
| 读租约 lease | 正常读取期间提供有限保护窗口 | 节点不崩溃、数据永久有效 |
| 业务 TTL | 应用希望多久以后不再复用 | 不等同于读租约时长 |
| soft pin / hard pin | 影响对象被普通淘汰的资格或优先级 | 不等于持久化，也不等于跨机容灾 |
| 内存注册 | 让通信路径能访问一块内存 | 不自动延长 Python/C++ 对象寿命 |

以当前基线为例，读取延长普通读 lease 不等于延长 soft pin。存在性检查也可能影响读 lease；调试工具的高频查询可能改变回收行为。

“用户已经不要这个请求”与“NIC 已停止访问这块 buffer”是两个状态。超时后马上 free/reuse 是典型的偶现数据损坏来源。

## 6. 副本、放置与容量怎么理解

两份副本可以提高可用性或分散读取，但前提是实际成功创建，并放在符合目标的故障域中。同一台机器上的两个 segment 不是跨机器冗余。

规划内存要把以下项目加起来：

```text
Store 贡献的共享 segment
+ Client 本地 staging / transfer buffer
+ LMCache 或 HiCache 主机缓存
+ 元数据、分配器、对齐碎片、在途任务、系统预留
```

不能每层都配“32 GiB”，然后认为整机只消耗 32 GiB。副本会额外占用容量和写带宽；缓存池容量也不等于 GPU 活跃请求容量。

## 7. TE 与 RDMA 要理解到哪里

TE 通常需要知道源/目标内存区域、长度、方向和访问描述，提交传输后通过状态确认完成。Store 负责对象身份，TE 不替上层判断 token 前缀是否相同。

RDMA 让适用路径能在较少远端 CPU 数据搬运参与下访问远端注册内存。理解它至少要知道：

- **MR / 注册内存**：NIC 可以访问的内存区域与权限描述，注册有成本。
- **QP / CQ**：提交通信工作与获取完成信息的基本队列机制；具体上层如何封装取决于实现。
- **GDR**：在支持的硬件、驱动和拓扑条件下，让 NIC 直接访问 GPU 内存。不是打开一个字符串配置就必定生效。
- **NUMA / PCIe**：GPU、NIC、CPU 所在位置影响路径长度与共享带宽。
- **批量与流水线**：摊薄小任务开销、重叠阶段，但要保留正确依赖。

零拷贝必须说清省掉哪次拷贝。减少 Python bytes 拷贝、减少 host staging、绕过远端 CPU、GDR 是不同层次；都不意味着网络、内存总线或同步成本消失。

## 8. 存储方向怎样诊断性能

| 现象 | 可能原因 | 先看什么 |
|---|---|---|
| 大对象快，小对象慢 | 固定开销、RPC、锁、任务数量 | objects/s、元数据 CPU、批量粒度 |
| NIC 没满却很慢 | H2D、重排、同步、注册、单线程瓶颈 | 分段延迟与 CPU/GPU trace |
| 多 NIC 没线性提升 | NUMA、PCIe、内存带宽、热点、任务分配 | 拓扑、各 NIC 流量和队列 |
| 命中多但 P99 差 | 预取等待、排队、热点、共享总线争用 | 尾延迟与在途量 |
| 重启后缓存消失 | 数据持有者退出、无可用副本、仅存 DRAM | 哪个进程贡献 segment、实际副本状态 |

先分别测 TE、Store、引擎端到端，再寻找差异。不要用裸 RDMA 带宽直接推断 TTFT。

## 9. 源码路线与进阶边界

路径相对于 Mooncake 上游源码根目录：

| 阅读入口 | 要回答的问题 |
|---|---|
| `mooncake-store/src/real_client.cpp` | Python/高层客户端怎样初始化和管理资源 |
| `mooncake-store/src/client_service.cpp` | `Client::Put/Get` 怎样处理完成与失败 |
| `mooncake-store/src/master_service.cpp` | 对象状态如何创建、发布和回收 |
| `mooncake-store/include/lease.h` | 保护窗口如何授予与延长 |
| `mooncake-store/src/allocation_strategy.cpp` | 副本放哪里，有什么限制 |
| `mooncake-store/src/transfer_task.cpp` | 对象访问如何落到传输 |
| `mooncake-transfer-engine/include/transfer_engine.h` | 注册、提交、状态、释放的协议 |

先读[完整内核路线](internals.md)，再看 HA 和分层。Master HA 解决目录服务恢复，不自动恢复已经丢失的 DRAM payload；Conductor 提供前缀位置索引，也不是存储或完整调度器的替代品。

Store-RS/TENT 属于另一条实现主线；EP/PG 服务于专家并行与通信。KV 存储面试先把当前主线讲完整，再说明扩展组件解决什么问题。

## 10. 深读与验收

[知识地图](architecture.md)用于画图，[内核](internals.md)用于追状态，[集成](integrations.md)用于接上推理，[实验](../../../07-practice/implementations/mooncake/labs.md)用于验证，[面试题](../../../08-career/implementations/mooncake/questions.md)与[系统设计](../../../08-career/implementations/mooncake/system-design.md)用于复述。

合格标准是：能画一条 Put 和 Get 正常路径，再分别添加写者崩溃、读超时、节点退出、buffer 提前回收四种情况，指出谁检测、谁清理、什么时候重算、什么时候拒绝消费。
