# Serving 组件选型：在同一技术问题下比较实现

[返回主题](README.md) · [缓存管理](../03-kv-cache/README.md) · [存储](../04-storage/README.md) · [岗位路线](../08-career/roles.md)

比较前先确定层次。vLLM 与 SGLang 都是推理服务系统，适合比较同一模型和负载下的执行效果；LMCache 与 HiCache 在缓存管理层有重叠；Mooncake Store 与 TE 分别提供对象与传输能力。不能用一张“谁最快”的榜单回答所有问题。

## 1. 总体职责对照

| 维度 | vLLM | SGLang | LMCache | Mooncake Store / TE |
|---|---|---|---|---|
| 主要工作 | 请求调度、模型执行、GPU KV 管理 | 请求执行、前缀管理、缓存与调度协同 | KV 保存、查询、分层加载与引擎集成 | 对象资源池与字节传输 |
| 核心抽象 | Request、token budget、block、runner | Req、batch、radix/cache 状态、pool | chunk、key/group、L1/L2、完成协议 | object、replica、segment、lease、transfer |
| 是否直接执行模型 | 是 | 是 | 不承担完整推理引擎职责 | 不承担完整推理引擎职责 |
| 本地前缀复用 | block hash 等管理路径 | radix/unified cache 等管理路径 | 通过集成管理额外可复用范围 | Store 本身通常不解释 token 前缀 |
| GPU 内存的最终消费 | 模型/attention backend | 模型/attention backend | 与引擎约定恢复位置和同步 | 提供底层数据，需上层解释 |
| 跨实例共享 | 依赖具体 Connector 与后端 | 依赖 HiCache/集成与后端 | 依赖共享服务、后端和兼容格式 | 提供共享对象池；TE 也可仅用于直接交接 |
| 学习重点 | 调度到执行的全链路 | 前缀、分层状态与调度 | 查到、搬到、可消费之间的契约 | 读写发布、保护、回收、传输 |

某个项目新增一种能力，不代表所有版本、所有模型、所有组合都支持它。选型表应同时带上版本和实验条件。

## 2. vLLM 与 SGLang 怎么比较

两者都有动态批处理、KV 管理、前缀复用、优化 kernel、并行执行和外部缓存集成。差异体现在架构、策略、模型适配、具体 backend 与版本，而不只是论文中的标志性技术。

| 比较问题 | 应怎样做 | 不能直接得出的结论 |
|---|---|---|
| 哪个吞吐更高 | 固定模型、精度、硬件、长度分布、到达率、SLO | 一个最大 batch 的峰值不等于线上更优 |
| 谁的前缀复用更好 | 扫描共享长度、分叉模式、热点变化和内存压力 | 用树就一定比 hash 快 |
| 哪个 Decode 更快 | 固定活跃 batch、上下文长度、kernel 与并行方案 | 一个模型结果代表所有模型 |
| 哪个更适合外部缓存 | 验证当前 Connector、布局、故障和监控契约 | “已接 Mooncake”代表任意互通 |
| 哪个更适合团队 | 看模型支持、调试成本、运维能力、升级风险 | 仅按项目热度决定 |

**学习建议**：先选一个引擎完整追通请求路径，再用另一个框架比较同一个问题。一次同时读两个 Scheduler 大文件，容易陷入变量名差异，却没弄清共同约束。

## 3. PagedAttention、RadixAttention、FlashAttention

| 技术 | 核心问题 | 具体解释 | 主要代价/边界 |
|---|---|---|---|
| PagedAttention / 分页 KV | KV 怎么放、attention 怎么找到 | 用逻辑到物理块映射访问非连续 KV | 索引、分页 kernel、页尾浪费 |
| RadixAttention / 前缀树管理 | 哪些请求共享已经算出的前缀 | 用树表达公共 token 路径与分叉 | 树与引用维护、缓存状态协调 |
| FlashAttention | 一次 attention 内部怎么减少访存 | 分块计算与在线 softmax，避免巨大中间结果反复写回 HBM | 需要适配形状、dtype、硬件与布局 |

它们不处于同一个层次。系统可以用分页分配物理 KV，用前缀树决定复用，再由优化 attention kernel 执行计算。

## 4. LMCache 与 SGLang HiCache

二者都涉及分层缓存，但接入方式、资源边界和策略所在位置不同。

| 维度 | SGLang HiCache | LMCache MP |
|---|---|---|
| 与引擎关系 | 在 SGLang 缓存运行时中协调前缀和层级 | 独立服务，通过对应 Connector/集成连接引擎 |
| 常见层级叫法 | L1 GPU、L2 host、L3 外部 | CPU 配置常用 L1 host、L2 外部，GPU 在引擎侧 |
| 重点机制 | radix 状态、host/device pool、预取与写回、跨 rank 协同 | lookup/prefetch/retrieve/store、L1 保留、controller、adapter |
| 资源生命周期 | 需要追 SGLang 进程与缓存后端的所有权 | 需要追引擎、MP server、L2 各自所有权 |
| 后端组合 | 按 HiCache 支持的存储接口 | 按 MP adapter 和策略接口 |
| 适合先研究的情况 | 已用 SGLang，希望沿原生分层路线落地 | 希望使用 LMCache 管理能力或适配其后端生态 |

有相似概念不等于可以把两套 L1/L2 按数字串在一起。先确定谁负责同一块 KV 的保留、淘汰和加载，避免重复缓存预算和冲突策略。

## 5. LMCache 与 Mooncake

用一次恢复过程说明边界：

```text
LMCache：这个请求前 24 个 chunk 可以尝试复用；
         它们属于哪些 rank/group，应该恢复到哪些 GPU blocks？
Mooncake Store：这些对象有哪些可读副本，在哪里，访问保护是否有效？
TE：按给定源/目标和长度搬运数据，报告完成或失败。
LMCache / Connector：确认内容与布局可消费，通知引擎推进。
```

LMCache 并不要求所有后端都是 Mooncake；Mooncake 也能通过框架原生 Connector 使用。选择 LMCache + Mooncake，应有清楚理由，例如统一缓存管理、需要特定后端/策略、独立缓存进程。随后测量新增层级是否带来收益。

## 6. 本地复用、共享存储、PD 分离

| 方案 | 省什么/改变什么 | 新增什么成本 | 优先尝试的场景 |
|---|---|---|---|
| GPU 本地前缀缓存 | 同实例重复 Prefill | 占用 GPU KV 容量、淘汰维护 | 有稳定前缀复用，先做基线 |
| host 缓存 | 保留更多 KV，减少部分重算 | D2H/H2D、host 容量与布局转换 | GPU 容量紧、主机资源可用 |
| 远端共享缓存 | 跨实例复用与容量池化 | 查询、网络、尾延迟、资源与故障管理 | 长前缀、高复用、多实例 |
| 直接 PD 传输 | 分开配置 Prefill/Decode 资源 | 状态交接、网络、路由、两边负载平衡 | 两阶段干扰大、网络足够、资源可分配 |
| PD + 共享 Store | 同时交接和跨请求复用 | 组合的状态与调度复杂性 | 已有证据证明两项收益均成立 |

共享 Store 可以服务没有做 PD 分离的实例；PD 也可以不经过 Store。应分别做开关实验，不能把收益全部归给一个组件。

## 7. 同一 Store 不等于跨框架互读

让 vLLM 和 SGLang 指向同一个 Mooncake 地址，只解决“能访问同一资源池”的问题。直接复用还要求：

- key 的前缀哈希、编码、namespace 相容。
- 模型、tokenizer、位置、adapter 等计算语义相容。
- layer/group、TP/PP 分片、head、dtype、block/chunk 布局相容。
- 必要的重排/重分片由明确实现处理。
- 两端理解相同的就绪、失败和生命周期契约。

例如写者按“每层一组数据”保存，读者却按“每页包含所有层”解释，即使字节数恰好相同，输出仍会错误。共享存储从不自动修正这种差异。

## 8. 用场景做选择

### 场景 A：单 GPU、短问答、输入几乎不重复

先看引擎 batch、kernel 和显存使用。远端缓存的命中收益有限，却会增加写入和查询。可以先保留本地缓存对照，再决定是否增加层级。

### 场景 B：长文档、多轮提问、多个副本接入

先验证完整前缀是否稳定，然后测本地/host/远端分别减少多少计算。SGLang 可研究 HiCache + Store；vLLM 可研究原生 Store Connector 或 LMCache。还要用负载感知路由避免所有请求挤向一个热节点。

### 场景 C：长 Prefill 使生成中的用户卡顿

先检查 chunked prefill、token 预算和并发。若单实例内仍难以兼顾 TTFT/ITL，再评估 PD。把交接成本、Decode 容量和故障重试一起算进去。

### 场景 D：远端命中率很高，但 P99 变差

分解查询、远端读、H2D、重排和同步，检查热点、过多小对象、无效预取、NIC/PCIe 争用。考虑缩短等待、限制预取、热点副本或直接重算，而不是继续扩大缓存。

## 9. 一张公平实验表

| 对照 | 用途 |
|---|---|
| A：无跨请求复用 | 测完整重算基线；注明引擎是否可关闭及准确关闭范围 |
| B：只开本地前缀缓存 | 分离本地复用收益 |
| C：本地 + host | 测容量扩展是否抵消 H2D 成本 |
| D：本地 + host + 远端 | 测远端增量收益 |
| E：在相同缓存组合下开/关 PD | 单独判断 PD 是否改善目标 |

分别测冷启动、稳定热点、热点切换、低复用与故障。每组给出正确性、TTFT/ITL 分位数、goodput、实际省掉的 Prefill、分层加载量、失败/超时率。无实测时，结论只能写成待验证假设。
