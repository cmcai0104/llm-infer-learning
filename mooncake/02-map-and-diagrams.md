# 02 · 知识地图与设计图

本章图使用 Mermaid，可在本仓库 Sphinx 文档站渲染。图中“教学模型”表示为了理解而抽象的行为；涉及具体 API 的部分以 [源码路线](04-mooncake-internals.md) 为准。

## 1. 完整知识地图

```mermaid
flowchart TB
    R["推理 Infra / KV Cache 存储"]
    R --> A["计算与语义 P0"]
    R --> B["缓存与存储 P0"]
    R --> C["通信与硬件 P0/P1"]
    R --> D["分布式正确性 P0/P1"]
    R --> E["集成与调度 P0/P1"]
    R --> F["评估与工程 P0"]
    R --> G["Mooncake 进阶 P2"]
    A --> A1["Attention / Prefill / Decode"]
    A --> A2["MHA / GQA / MQA / MLA"]
    A --> A3["分页 / 前缀依赖 / dtype / TP"]
    B --> B1["HBM / DRAM / SSD / 远端池"]
    B --> B2["内容标识 / namespace / chunk"]
    B --> B3["准入 / 淘汰 / 预取 / 写回"]
    B --> B4["容量 / 碎片 / 副本 / 热点"]
    C --> C1["NUMA / PCIe / NVLink / HBM"]
    C --> C2["TCP / RDMA / QP / CQ / MR"]
    C --> C3["批处理 / 多网卡 / GDR / 完成语义"]
    D --> D1["对象状态 / 租约 / pin / 幂等"]
    D --> D2["部分失败 / fencing / HA / 恢复"]
    D --> D3["隔离 / 配额 / 校验 / 生命周期"]
    E --> E1["vLLM Connector / LMCache / HiCache"]
    E --> E2["PD 分离 / cache-aware routing"]
    E --> E3["SLO / 队列 / 准入 / 背压"]
    F --> F1["TTFT / ITL / goodput / 分位数"]
    F --> F2["负载模型 / 消融 / 故障注入"]
    F --> F3["C++ 并发 / Python 绑定 / 监控"]
    G --> G1["TENT / Store-RS / P2P Store"]
    G --> G2["Conductor / EP / PG / Reshard / RL"]
```

把图变成可执行知识表：

| 分支 | 核心问题 | 必须掌握的工程技能 |
|---|---|---|
| 计算 | 哪些计算可省，哪些必须保留？ | 根据模型实际 KV 张量计算容量，核对 prefix reuse 边界 |
| 存储 | 哪些对象值得留，放在哪里？ | 容量规划、分配器、变长对象、冷热分层、写放大分析 |
| 通信 | 数据何时真正能被 GPU 使用？ | 注册内存、跨节点传输、CUDA event/stream、NUMA 定位 |
| 正确性 | 如何保证读不到部分、过期或不兼容的 KV？ | 状态机、租约、引用计数、版本化 namespace、失败回退 |
| 调度 | 命中最多的节点一定最快吗？ | 比较队列等待、计算、传输、后续 decode 预算 |
| 评估 | 如何证明改动提升用户体验？ | 可重复负载、冷/热对照、SLO goodput、尾延迟和质量校验 |
| 实现 | 在哪里加功能、看测试？ | 追踪调用链、RAII、锁粒度、异步任务所有权、错误传播 |

## 2. 生态职责图

```mermaid
flowchart TB
    U["请求 / API 网关"] --> R["上层 Router / Scheduler"]
    R -. "可查询缓存位置" .-> C["Conductor：前缀索引"]
    R --> V["vLLM / SGLang：执行模型、GPU block、请求调度"]
    V --> N["原生 Mooncake Store Connector"]
    V --> L["LMCache 或 SGLang HiCache：KV 管理与分层"]
    V --> P["PD Connector：直接搬运 KV"]
    N --> S["Mooncake Store：对象、副本、租约、空间"]
    L --> S
    S --> T["Transfer Engine：字节传输、拓扑、批任务"]
    P --> T
    T --> H["DRAM / GPU / 网络 / 支持的存储后端"]
    V -. "缓存事件" .-> C
    S -. "对象事件" .-> C
```

**阅读要点**：LMCache 是一条可选集成路径，不是所有 Mooncake 用户都必须经过的组件。Conductor 提供缓存位置查询，上层路由器仍需结合负载和 SLO 做决策。

论文中的全局调度器是系统设计的一部分；不能由“安装 Store 和 TE”推断已获得论文里的完整准入、预测和调度策略。

## 3. Store 部署与控制面/数据面

```mermaid
flowchart LR
    subgraph CTRL["控制面"]
        M["Store Master<br/>key 到 replica、租约、分配和回收"]
        MD["Transfer metadata<br/>HTTP / etcd / P2P handshake 等"]
        HA["可选 HA 协调与恢复状态<br/>依部署配置"]
        M -.-> HA
    end
    subgraph A["节点 A"]
        CA["Store Client A / TE"]
        BA["本地缓冲区"]
        SA["贡献的 segment A"]
        CA --- BA
        CA --- SA
    end
    subgraph B["节点 B"]
        CB["Store Client B / TE"]
        SB["贡献的 segment B"]
        DB["可选 SSD tier"]
        CB --- SB
        CB --- DB
    end
    CA -. "PutStart / GetReplicaList 等 RPC" .-> M
    CB -. "挂载、心跳、元数据 RPC" .-> M
    CA -. "发现连接和内存描述" .-> MD
    CB -. "发布连接和内存描述" .-> MD
    CA == "数据直接传输，不经过 Master" ==> SB
```

Master 的对象元数据与 TE 的连接/内存元数据是两个逻辑概念。内置 HTTP metadata server 可以与 Master 同进程，但职责没有合并。HA 使用 etcd 也不代表每个 KV 对象的 payload 或每次数据读取都经过 etcd。

Client 可以既是访问者又是资源贡献者；图中的 segment 属于相应进程/节点的资源。退出贡献者可能让该处缓存副本消失。

## 4. Put 正常与失败路径

以下是 C++ Store 普通内存副本路径的简化图；磁盘/DFS 等副本具有各自完成规则。

```mermaid
sequenceDiagram
    participant App as 应用 / Connector
    participant C as Store Client
    participant M as Master
    participant S as 远端 segment
    App->>C: put(key, bytes)
    C->>M: PutStart(key, size, config)
    M->>M: 检查键和配额、分配副本、记录未完成状态
    M-->>C: replica descriptors
    C->>S: TE 写入已分配位置
    alt 数据传输成功并满足完成规则
        S-->>C: 传输完成
        C->>M: PutEnd
        M->>M: 副本变为可读
        M-->>C: 提交结果
        C-->>App: 成功
    else 传输失败
        C->>M: PutRevoke / 对应失败处理
        C-->>App: 错误
    end
    Note over M,S: Client 崩溃可能留下未完成对象，由超时清理路径处理
```

这个 start/end 协议让数据准备与发布可见性分离，但不应叫成跨对象数据库事务。读者不能使用仅分配而尚未完成的副本。

## 5. Get 与租约时序

```mermaid
sequenceDiagram
    participant App as 应用
    participant C as Store Client
    participant M as Master
    participant S as 远端副本
    App->>C: get / get_into
    C->>M: GetReplicaList(key)
    M->>M: 查可读副本并授予读租约
    M-->>C: descriptors + lease
    C->>S: TE 读入目标 buffer
    S-->>C: 完成 / 失败
    C->>C: 检查传输、适用的 checksum 和 lease
    alt 全部有效
        C-->>App: 返回可使用的数据
    else 未命中 / 失败 / 租约过期
        C-->>App: miss 或错误
        App->>App: 有界重试 / 重新查询 / 重算
    end
    Note over App,C: get_into 的目标内存可能已被写入；返回失败时不能消费它
```

租约保护正常删除/淘汰窗口；最终是否成功还依赖实际 I/O、客户端时间判断和故障状态。不能把 descriptor 缓存无限期复用。

## 6. 请求运行图：缓存复用与 PD

```mermaid
flowchart TB
    R["新请求：tokenize + namespace"] --> H["查找最长兼容前缀"]
    H --> LOCAL{"本地 KV 可用？"}
    LOCAL -->|"是"| READY["得到可用连续前缀"]
    LOCAL -->|"否/部分"| COST{"远端拉取比重算划算且满足 deadline？"}
    COST -->|"是"| FETCH["预取远端 KV，检查完成与各 rank 一致"]
    COST -->|"否"| MISS["从未命中位置计算"]
    FETCH -->|"成功的连续前缀"| READY
    FETCH -->|"失败或截止"| MISS
    READY --> PREFILL["只计算剩余 prompt<br/>可能保留末 token 用于 logits"]
    MISS --> PREFILL
    PREFILL --> PD{"Prefill 与 Decode 分离？"}
    PD -->|"否"| DEC["本实例 Decode"]
    PD -->|"是"| MOVE["直接 TE 或经 Store 交接 KV<br/>传递布局、请求和完成信息"]
    MOVE --> DEC
    PREFILL -. "依准入策略异步缓存" .-> POOL["KV 缓存池"]
    DEC --> OUT["流式 token / 结束 / 释放引用"]
```

这张图是通用运行模型，不规定某一 Connector 的实际回调顺序。直接 TE 与 Store 交接是不同路径，部署时只能使用框架和版本实际支持的协议。

## 7. LMCache MP 数据流

```mermaid
flowchart LR
    V["vLLM Worker<br/>GPU KV block"] <-->|"GPU 数据交换；依传输模式"| L1
    V -. "经 MP Connector 请求<br/>STORE / LOOKUP / RETRIEVE" .-> SERVER
    subgraph MP["LMCache 独立服务进程：典型 CPU L1 配置"]
        SERVER["请求处理 / KV 语义"]
        L1["L1 Manager<br/>CPU 内存、对象锁与生命周期"]
        SC["StoreController"]
        PC["PrefetchController"]
        AD["mooncake_store L2 adapter"]
        SERVER --> L1
        L1 --> SC
        SC --> AD
        AD --> PC
        PC --> L1
    end
    AD <-->|"对象 put/get 与数据传输"| MC["Mooncake Store<br/>Master + 资源池 + TE"]
```

MP 请求通道不等于 KV 大块数据必须序列化后穿过 RPC。实际 GPU 数据交换依赖支持的模式、内存映射和同步实现。此图画的是 CPU L1 常见配置；MP 还有特定版本支持的 GDS L1 等选项。

## 8. 对象与缓冲区状态：两套不同的生命周期

```mermaid
stateDiagram-v2
    [*] --> Absent
    Absent --> Writing: PutStart / 分配
    Writing --> Readable: 数据完成并提交
    Writing --> Absent: 撤销 / 超时清理
    Readable --> Readable: 查询并授予租约
    Readable --> Reclaimable: 满足租约和 pin 等回收条件
    Reclaimable --> Readable: 新访问可能重新保护
    Reclaimable --> Absent: 删除 / 淘汰 / 资源失效
```

这是教学对象状态机，不是 `ReplicaStatus` 枚举的逐项映射。不同副本可以处于不同状态；“可回收”也不是读不到数据，而是已满足回收资格。

```mermaid
flowchart LR
    A["分配 buffer"] --> B["注册内存 / 建立 MR"]
    B --> C["数据生产完成<br/>必要的 CUDA 同步"]
    C --> D["提交传输"]
    D --> E["在途：禁止释放或改写"]
    E --> F["确认完成或安全 drain"]
    F --> G["消费者可使用 / 复用"]
    G --> H["注销注册"]
    H --> I["释放所有者"]
```

超时只是等待方停止等待的信号之一；释放前必须确认底层不会再访问该内存。缓存对象生命周期与本次传输 buffer 生命周期不能互相替代。

## 9. Page、chunk、object、slice 的关系

```mermaid
flowchart TB
    TOK["逻辑 token 序列"] --> PAGE["推理引擎 page/block<br/>例如 16 tokens，教学示例"]
    PAGE --> CHUNK["上层缓存 chunk<br/>例如 256 tokens，教学示例"]
    CHUNK --> OBJ["Store object<br/>由 Connector 定义 key 与 bytes"]
    OBJ --> SLICE["Transfer Engine 传输 slice<br/>用于并行和路径调度"]
    CHUNK -. "各层、K/V、rank 的布局可能不同" .-> LAYOUT["layout / shape / dtype / rank metadata"]
```

几个粒度不能混用：模型 attention window、引擎 page、LMCache chunk、Store object、TE slice 的职责不同。一个 chunk 是否包含所有层、某个 rank 或某种 packed 布局，必须看当前 Connector。

## 10. 缓存感知路由与过载

```mermaid
flowchart LR
    REQ["请求"] --> ROUTER["Router：候选实例"]
    IDX["Conductor / cache index"] -->|"位置与命中估计"| ROUTER
    LOAD["队列 / GPU / 网络 / decode 负载"] --> ROUTER
    ROUTER --> SCORE["预测 queue + fetch + prefill + handoff"]
    SCORE --> OK{"可满足 TTFT 与 decode SLO？"}
    OK -->|"是"| SELECT["选取满足约束的低成本实例"]
    OK -->|"否"| POLICY["限流 / 排队预算 / 拒绝 / 服务降级"]
```

索引可能过期。路由命中预测是性能提示，不能代替 Store 读取与兼容性检查。最大化缓存命中率还必须考虑排队、热点与网络拥塞。
