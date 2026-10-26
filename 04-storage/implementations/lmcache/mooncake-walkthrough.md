# LMCache + Mooncake：从一次 KV 保存到下一次请求恢复

[存储详解](../../kv-cache-storage.md) · [设计图册](../../../01-foundations/design-atlas.md) · [LMCache 内核](../../../03-kv-cache/implementations/lmcache/internals.rst) · [Mooncake 内核](../mooncake/internals.md)

这篇把两套系统放在同一条数据链上讲。范围是 LMCache 的 MP 模式、native `mooncake_store` L2 adapter，以及 Mooncake C++ Store 的常见 host-memory 路径。LMCache 基线为 `ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be`，Mooncake 为 `dd1217b6c461491e7c73ff29b4ffad215e0a9ac0`；这组源码用于学习，不是经过本次实测的部署兼容性承诺。

## 1. 两者分别补哪块能力

假设引擎中有一份 8K-token KV，分散在许多 GPU 页。单独调用对象存储并不能回答：从哪些页读、按什么格式组织、什么 key 对应哪段前缀、恢复到哪些新页、何时允许少算 token。

| 层次 | 核心职责 | 不能单独保证 |
|---|---|---|
| 引擎与 Connector | 请求进度、页位置、可跳过范围、设备执行依赖 | 远端对象持久存在 |
| LMCache | chunk/group/key、L1 staging/resident、预取、保存与读写保留 | 所有部署天然兼容同一模型表示 |
| native adapter | 转换 key/buffer，提交任务，汇总完成结果 | 跨所有对象的事务、任意后端的强锁 |
| Mooncake Store | 对象目录、分配、发布、副本、租约与回收 | token 前缀的计算语义 |
| Transfer Engine | 注册地址、segment、READ/WRITE 与完成状态 | 上层对象发布与 GPU attention 可消费 |

两套系统的组合价值在于把引擎状态管理与共享存储连接起来；代价是更多边界、资源预算和失败组合。必须证明收益超过这些成本。

## 2. 部署与两条通路

![LMCache Mooncake 架构](../../../01-foundations/diagrams/d08-lmcache-mooncake.svg)

常见 host L1 路径是：

```text
GPU KV ↔ LMCache host L1 ↔ native Mooncake Client ↔ 远端 segment
```

LMCache 的 L1 在该配置中是 host 内存，L2 是外部后端；引擎 GPU 不在这个 L1/L2 命名体系里。HiCache 等系统采用不同层级命名，面试和设计文档中最好直接写 GPU、host、远端。

控制消息可以通过 IPC/RPC 传递，但设备 payload 不一定逐字节走同一控制消息。Master 管对象元数据，不承载所有 payload 转发。这里也不能直接宣称 GDR：即使底层支持 GPU 内存，这个 host L1 集成仍要按实际路径计算 D2H/H2D。

## 3. 一次保存：三次“完成”不能混为一谈

![保存与恢复运行链](../../../01-foundations/diagrams/d09-lmcache-io.svg)

### 3.1 引擎到 L1

1. 引擎确定可保存的 token 范围、GPU block 与设备依赖，Connector 传递相应信息。
2. LMCache 用对象 key 和 writer tag 预留 L1 staging。地址分配成功时数据还不可读。
3. 完成相应的设备数据传输和事件依赖。
4. L1 finish/admission 将完整数据发布为 resident；有些路径同时取得读保留，防止发布与后续使用之间被淘汰。

同 key 的不同 writer tag 可各有 staging；第一个有效发布者成为 resident，后续完成者按当前协议处理重复内容。不能把 writer tag、object key 和 L1 owner ID 当作同一种标识。

### 3.2 L1 到 Mooncake

1. 保存策略决定哪些对象送往哪些 L2 adapter；任务期间 L1 源对象必须保持有效。
2. `NativeConnectorL2Adapter.submit_store_task` 转换对象 key 与 memory view，向 native client 提交批量 set，并登记 future/task 对应关系。
3. C++ Mooncake connector 的批量 set 路径使用 `batch_put_from`。Store 层再执行适用的分配、传输与发布协议。
4. native 完成经 adapter 汇总成 L2 store 结果；controller 才能按策略解除本次源数据保留或改变 L1 留存状态。

因此至少有 **GPU 数据已产生、L1 可读、L2 保存完成** 三种事件。是否必须等 L2 才结束用户请求、是否保存全部对象、L1 是否继续保留，取决于集成和策略，不能从图推导一个统一答案。

### 3.3 Store 内部的发布协议

普通单对象 `Client::Put` 可以沿 `PutStart → 数据写入 → PutEnd/PutRevoke` 理解；批量 API 有独立的逐对象状态和 finalize 流程，不能把单对象函数图当成批量函数的逐行调用栈。

`PutStart` 得到位置还不代表对象已可读。写者退出后，还需要清理未完成状态和残留空间。申请 N 个副本也不能直接声称已得到 N 个跨主机完整副本，必须看实际结果和放置约束。

内容寻址下同 key 应对应同内容。本基线普通 `Put` 遇到已有对象可返回成功而不覆盖；要修改内容应研究更新语义或使用新版本 key。批量写部分成功时，整批失败不等于已经成功的对象都被回滚。

## 4. 一次恢复：lookup ack 不等于命中

### 4.1 查找与预取

1. 引擎侧提交 LOOKUP，server 的 ack 只表示任务已登记。
2. LMCache 在 L1 观察并保留对象，在 L2 查询候选；`StorageManager`/`PrefetchController` 根据合并信息规划来源。
3. 要从远端加载的对象先获得 L1 staging；native load 路径使用 `batch_get_into` 写入调用方 buffer。
4. 成功且被保留的对象进入 L1，形成结果 bitmap。最终命中还要按各 group/rank 的语义折叠。
5. 引擎侧查询最终状态：pending 与最终 0 命中不同；消费一次的结果也不能任意重复读取。

多 server 聚合时，普通前缀路径需要共同可用范围。对滑窗或递归状态，应按具体 group 协议判断，而不是简单把所有对象存在性相加。

### 4.2 L1 到 GPU

RETRIEVE 将实际可用数据恢复到目标 GPU 页，满足布局和设备依赖之后，才能让运行时使用相应范围。未覆盖输入、边界和首 token 计算继续由引擎处理。

完成后释放的是**本次读取的保留**，不是强制删除所有缓存。临时加载对象、长期保留对象和不同预取策略可能有不同回收行为。

### 4.3 一个具体反例

输入 1,024 token，chunk=256；L1 有 chunk0，远端有 chunk0–2，chunk3 缺失。若远端 chunk1 加载失败、chunk2 成功，普通连续前缀仍只能使用 chunk0，不能报告“成功拿到两个 chunk，所以跳过 512 token”。

第二个反例是 rank0 有 3 个 chunk，rank1 只有 2 个。不能按 rank0 的结果让所有 GPU 一起跳过 3 个 chunk。专家回答应同时说明对象级成功、模型级有效范围和引擎最终跳过范围。

## 5. 三层保护，三个不同保证

| 保护 | 在当前链路中的含义 | 不能推导出的保证 |
|---|---|---|
| LMCache L1 读写保留 | 管本地 staging/resident 的发布与使用 | 远端副本永不消失 |
| NativeConnectorL2Adapter 锁计数 | 客户端侧按 key 计数；unlock 减少本地计数 | 自动持有远端跨客户端强锁 |
| Mooncake 读租约 | Store 在限定窗口内保护适用读取 | 覆盖任意长排队、节点故障和所有设备任务 |

这是需要逐层核对源码的地方。`lookup_and_lock` 是上层接口名和契约；当前 native adapter 的查询实际提交 batch exists，并在客户端记录命中的锁计数，`submit_unlock` 也在本地减计数。Mooncake 的 exists/副本查询可影响 Store 读租约，但不能据此认为本地计数等于无期限的远端锁。

加载仍可能遇到副本丢失、租约过期、传输失败或不兼容数据。目标 buffer 可能已部分写入，不能只看“指针里有值”就消费。应用侧的等待时间、Store 租约窗口和设备任务寿命必须一起核算。

## 6. 原生扩展与 RDMA 前提

本基线 `mooncake_store_l2_adapter.py` 是配置与工厂入口，最终创建通用的 `NativeConnectorL2Adapter`，其 native client 由 Mooncake C++ connector 实现。

- LMCache 的 Mooncake native 扩展有独立构建前提，安装 Mooncake Python 包不等于该扩展已经存在。
- 工厂消费 `type`、`num_workers`、`per_op_workers`、`eviction` 等 LMCache 字段；其他配置传入 Mooncake setup。
- RDMA 模式要求有效的 L1 连续内存描述符；缺失时会报错，不能假设悄悄回退到 TCP。
- 当前 native 路径会预注册相应 L1 区域，并检查对象 buffer 是否处于允许的注册范围；注册不会替代对象所有权管理。
- 操作线程池可以分 lookup/retrieve/store/delete，但加线程不自动增加链路、内存或 H2D 带宽。

启动与构建命令使用[同版本 adapter 指南](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/l2_storage/mooncake_store.rst)，不要把旧进程内 connector 的 YAML 混入 MP 部署。

## 7. 专家应审查的资源预算

```text
引擎活跃 GPU KV
+ LMCache host resident
+ LMCache staging 与尚未完成 lookup 的最坏加载预算
+ Mooncake 贡献的 global segment
+ Mooncake 本地 buffer / 后端暂存
+ 元数据、注册资源、碎片与故障恢复余量
```

这些资源不一定全部独立分配，也不一定可以复用；必须按实际地址范围和所有者去重计算。把每个组件都配置为“32 GiB”，可能远超一台主机的可用内存。

预取准入不能只限制任务个数。当前预取设计同时考虑 staging 与已接受 lookup 尚未分配的最坏加载量，防止“现在看起来还很空”导致批量超卖。

## 8. 从现象定位到协议层

| 现象 | 优先区分 | 证据 |
|---|---|---|
| 查到很多，实际少算很少 | 候选命中、加载成功、group/rank 折叠、引擎边界 | 各阶段 token/bitmap 与完成记录 |
| L2 带宽高，TTFT 没下降 | 排队、staging、H2D、布局、同步 | 分段 trace、字节数、队列深度 |
| 偶发输出错误 | key/布局不兼容、提前发布、旧页复用 | namespace、对象身份、页代际、事件依赖 |
| 取消后内存不降 | 有效保留、在途任务、遗漏清理 | owner、引用、task/future 与结束事件 |
| 缓存故障拖垮 GPU | 重算放大、无界重试、准入不足 | miss 原因、回源速率、GPU 队列和 goodput |
| Store 有对象但无法加载 | 注册/位置过期、租约、部分失败 | Store 返回值、buffer 范围、加载结果 |

先还原状态和时间线，再判断是配置问题、实现缺陷或设计能力不足。不要只因为名字中有 lock、zero-copy、async 就把目标契约当作已验证保证。

## 9. 源码阅读与验证顺序

| 顺序 | 固定版本入口 | 要带走的答案 |
|---|---|---|
| 1 | [lookup 聚合设计](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/design/integration/vllm/mp_lookup_state.md) | ack、pending、最终结果、取消与多 server 聚合 |
| 2 | [StorageManager 协议](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/design/v1/distributed/storage_manager.md) | group/rank、bitmap、prefix/full 和消费一次的结果 |
| 3 | [L1 设计](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/design/v1/distributed/l1_manager.md) | staging、发布、读写保留与 owner |
| 4 | [预取控制器设计](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/design/v1/distributed/storage_controllers/prefetch_controller.md) | 来源规划、准入、部分完成和清理 |
| 5 | [Mooncake adapter 工厂](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py) | 配置边界、native client 与 L1 注册 |
| 6 | [native adapter](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py) | key/buffer、future、锁计数、通知与结果 |
| 7 | [C++ connector](https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/csrc/storage_backends/mooncake/connector.cpp) | batch put/get、返回值、注册和关闭顺序 |
| 8 | [Store Client](https://github.com/kvcache-ai/Mooncake/blob/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0/mooncake-store/src/client_service.cpp) | 单对象/批量发布、租约、部分失败与副本 |

动手入口：[LMCache 实验](../../../07-practice/implementations/lmcache/labs.rst)、[Store/TE 实验](../../../07-practice/implementations/mooncake/labs.md)。至少提交一次成功恢复、一次失配、一次部分加载失败和一次取消的状态记录；硬件不支持时先做推演，并明确未验证项目。
