# 知识地图与 vLLM 设计图

[返回导航](README.md) · [KV 专题](04-kv-cache-vllm-lmcache-mooncake.md) · [官方资料](06-references.md)

本章的图以 V1 为主线，表达职责、数据依赖和资源生命周期。进程数、调用位置、同步方式会随版本、executor、并行方案和异步调度配置变化。启动图和状态机是教学抽象，不是逐行调用图。

配套正文见 [07 推理基础](07-inference-foundations.md)、[08 当前源码精读](08-vllm-source-walkthrough.md)、[09 性能与分布式](09-performance-distributed-serving.md)。08 将本章的架构概念落实到当前本地函数，并解释 V1 引擎中的 Model Runner V2。

## 1. 总知识地图

### D01：学习依赖图

```mermaid
flowchart TD
    A["Python / C++ / Linux / 并发与内存"] --> B["PyTorch 与 GPU 执行"]
    T["Transformer / Attention / 自回归生成"] --> C["Prefill / Decode / KV Cache"]
    B --> R["计算量 / 访存量 / Roofline"]
    C --> R
    C --> K["分页 KV / Prefix Cache / 内存生命周期"]
    C --> S["Continuous Batching / Chunked Prefill"]
    K --> E["vLLM V1 Engine / Scheduler / Worker"]
    S --> E
    B --> E
    R --> O["FlashAttention / CUDA Graph / 编译 / 量化"]
    E --> O
    E --> P["TP / PP / DP / EP 与通信"]
    K --> X["KV Connector / LMCache / Mooncake"]
    P --> D["PD 分离 / 多副本路由 / 分布式缓存"]
    X --> D
    E --> M["指标 / Benchmark / Profiling"]
    O --> M
    D --> M
    M --> F["SLO / Goodput / 成本 / 容错"]
```

### 知识点与掌握深度

| 领域 | 核心知识点 | 必须能解释的联系 |
| --- | --- | --- |
| 模型 | Q/K/V、causal mask、FFN、RoPE、MHA/MQA/GQA | 模型结构如何决定 KV 容量与读取量 |
| 生成 | Prefill、Decode、采样、EOS、长度上限、停止串 | 模型 forward、采样和输出处理为什么是不同步骤 |
| 内存 | 权重、KV、激活、workspace、graph、碎片 | “模型能加载”与“并发可达标”为什么不同 |
| 调度 | 请求队列、token budget、连续批处理、抢占、公平性 | token 数、请求数和显存是不同约束 |
| KV | block table、引用、hash、复用、驱逐、预取 | 逻辑序列如何映射到物理内存 |
| 算子 | GEMM、Attention、归一化、融合、布局 | 带宽、计算和 launch 开销如何变化 |
| Runtime | stream/event、异步执行、CUDA Graph、编译 | 为什么异步提交不等于操作完成 |
| 并行 | TP/PP/DP/EP、长上下文并行、collective、NUMA、拓扑 | 通信成本为何会抵消加卡收益 |
| 缓存系统 | L1/L2、对象状态、租约、chunk、缓存感知路由 | 命中为何仍可能慢、为何不能随时驱逐 |
| 高级模型 | MoE、MLA、SWA、混合 Mamba、多模态、LoRA | 普通全注意力的容量公式与复用规则何时失效 |
| 服务 | streaming、取消、背压、限流、超时、重试、弹性 | 单请求优化为何不一定改善服务 Goodput |
| 验证 | 正确性、质量、负载分布、冷暖缓存、分位数、trace | 性能结论需要哪些证据与适用前提 |

## 2. V1 组件与进程架构

### D02：在线服务的组件图

```mermaid
flowchart TB
    U["客户端 / 压测器"] --> API
    subgraph AP["API 进程或进程组"]
        API["HTTP API / 校验 / 输入处理"]
        TOK["Tokenizer / 多模态预处理"]
        EC["Engine Client / 输出处理"]
        API --> TOK --> EC
    end
    subgraph CP["EngineCore 进程：通常每 DP rank 一个"]
        CORE["EngineCore 循环"]
        SCH["Scheduler"]
        KV["KV Cache Manager / Block Pool"]
        CS["Connector 调度侧：可选"]
        EX["Executor"]
        CORE --> SCH
        SCH <--> KV
        SCH <--> CS
        CORE --> EX
    end
    subgraph WP["GPU Worker：按 TP / PP 等拓扑组织"]
        W["Worker / 分布式执行"]
        MR["ModelRunner / 输入准备"]
        MODEL["模型层 / Attention / 采样"]
        MEM["GPU 权重 / KV / Workspace"]
        CW["Connector 执行侧：可选"]
        W --> MR --> MODEL
        MODEL <--> MEM
        MR <--> CW
        CW <--> MEM
    end
    EC <-->|"IPC；在线架构常用 ZMQ"| CORE
    EX <--> W
    CS -.->|"本轮传输元数据"| CW
    CW <-->|"KV 数据"| STORE["外部缓存 / 远端 Worker"]
    EC -->|"流式结果"| U
```

关键职责：

- **API/Engine Client**：接入、输入处理和输出流。API 并发能力不等于 GPU 的计算容量。
- **EngineCore/Scheduler**：决定谁进入本轮、计算多少 token、需要哪些 blocks。
- **KV Manager**：管理执行引擎的物理 KV blocks；外部缓存不自动替代它。
- **Executor/Worker/ModelRunner**：把调度决策变成设备上的输入、forward 和输出。
- **Connector**：把“外部存在什么”和“如何把数据搬入/搬出”与调度连接起来。

普通稠密模型可把 DP 理解为处理不同请求的副本；MoE 的 DP/EP 组合可能跨 rank 同步，不能一概视作完全独立。离线 `LLM` 接口不必经过 HTTP 路径。依据：V1、V6、V10。

## 3. 启动过程

### D03：服务就绪前的概念流程

```mermaid
flowchart TD
    A["解析配置与模型元信息"] --> B["建立引擎与 Worker 通信"]
    B --> C["初始化设备与并行组"]
    C --> D["加载模型权重与执行后端"]
    D --> E["估计非 KV 开销与可用显存"]
    E --> F["规划并分配 KV blocks"]
    F --> G["初始化缓存管理与可选 Connector"]
    G --> H["必要的编译 / warmup / graph capture"]
    H --> I["健康检查通过，服务可接收请求"]
```

不同版本可能交错执行 profiling、编译、KV 初始化和 warmup。学习时关注它们的依赖，而不是背诵一个固定顺序。

容量关系：

```text
设备可用显存预算
  >= 模型权重 + KV + 激活峰值 + workspace + graph/通信缓冲 + 其他开销
```

`gpu_memory_utilization` 是引擎规划显存的参数，不是 GPU 算力利用率，也不能替代对峰值开销的测量。

## 4. 请求运行过程

### D04：普通生成请求时序

```mermaid
sequenceDiagram
    autonumber
    participant U as 客户端
    participant A as API与Engine Client
    participant E as EngineCore
    participant S as Scheduler
    participant K as KV Manager
    participant W as Worker与ModelRunner
    participant G as GPU
    U->>A: prompt与生成参数
    A->>A: 校验、模板与tokenization
    A->>E: 提交请求
    E->>S: 加入等待队列
    loop 直到结束或取消
        E->>S: 生成本轮调度决策
        S->>K: 查找前缀、分配所需blocks
        K-->>S: blocks与可复用token信息
        S-->>E: 请求、token数量、block映射
        E->>W: 执行本轮任务
        W->>G: 输入准备、forward、适用时采样
        G-->>W: 结果与完成状态
        W-->>E: 执行结果
        E->>S: 更新计算进度、停止条件
        E-->>A: 新输出及状态
        A-->>U: 有输出时流式返回
    end
    S->>K: 释放请求引用
    Note over S,K: 外部异步传输可能延迟物理block回收
```

完整 Prefill 通常产生可采样首 token 的 logits。后续 Decode 用已经生成的 token 继续 forward。被切分的中间 Prefill 片段不一定产生用户可见输出。

全前缀命中也不应理解为“完全不执行模型”：某些路径需要重新处理尾部 token 以得到下一 token 的 logits；复用粒度和实现会影响剩余计算。

### D05：请求状态机

```mermaid
stateDiagram-v2
    [*] --> Waiting
    Waiting --> WaitingKV: 需要外部KV
    WaitingKV --> Waiting: 传输完成或回退重算
    Waiting --> Running: 满足调度与内存约束
    Running --> Running: 执行下一轮
    Running --> Preempted: 资源压力
    Preempted --> Waiting: 重新排队或恢复
    Running --> Finished: EOS或长度等停止条件
    Waiting --> Cancelled: 客户端取消或超时
    WaitingKV --> Cancelled: 取消或失败
    Running --> Cancelled: 取消或失败
    Finished --> Reclaiming
    Cancelled --> Reclaiming
    Reclaiming --> [*]: 引用与在途操作清理完成
```

这是概念状态机，名称不等同于源码 enum。请求逻辑结束、GPU 操作结束和缓存可回收可能发生在不同时间。

## 5. 一轮调度怎么理解

V1 的核心抽象可理解为“为请求分配本轮要计算的 token 数”，不必把 Prefill 和 Decode 维护成两个完全不同的执行系统。概念上，调度器让 `num_computed_tokens` 追赶当前需要计算的 token；投机 token 等特性会扩展这个目标。依据：V2。

假设每轮 token budget 为 8；A、B 各有一个 Decode token 待处理，C 有 10 个 Prefill token 待处理。忽略其他约束：

| 轮次 | A | B | C | 合计 |
| --- | --- | --- | --- | --- |
| 1 | Decode 1 | Decode 1 | Prefill 6 | 8 |
| 2 | Decode 1 | Decode 1 | 剩余 Prefill 4 | 6 |
| 3 | 若未结束则 Decode 1 | 若未结束则 Decode 1 | 若已采样则 Decode 1 | 至多 3 |

实际调度还受请求上限、KV 容量、优先级、外部 KV 就绪、encoder budget、抢占、异步执行和模型限制影响。因此“剩余 token budget”不代表一定能塞进更多请求。

| 约束 | 回答的问题 |
| --- | --- |
| token budget，常见配置如 `max_num_batched_tokens` | 本轮最多处理多少 token？ |
| `max_num_seqs` 等请求数约束 | 本轮最多容纳多少条序列？ |
| `max_model_len` | 单条序列的允许长度是多少？ |
| KV block 容量 | 当前及增长后的缓存是否放得下？ |
| 服务 admission/queue 限制 | 能否继续接受新请求？ |

新版本可能进一步拆分 scheduled/active/queued 等预算，以所选版本为准。不能把 token budget 当成静态 batch size。

## 6. 分页与共享

### D06：逻辑 block 到物理 block 的映射

```mermaid
flowchart LR
    subgraph A["请求 A 的逻辑块"]
        A0["A0：共同前缀"]
        A1["A1：共同前缀"]
        A2["A2：后缀 A"]
    end
    subgraph B["请求 B 的逻辑块"]
        B0["B0：共同前缀"]
        B1["B1：共同前缀"]
        B2["B2：后缀 B"]
    end
    subgraph P["GPU 物理 blocks"]
        P7["物理 7：ref=2"]
        P2["物理 2：ref=2"]
        P9["物理 9：ref=1"]
        P4["物理 4：ref=1"]
    end
    A0 --> P7
    B0 --> P7
    A1 --> P2
    B1 --> P2
    A2 --> P9
    B2 --> P4
```

物理编号不需要连续；共享的完整前缀块只存一份。分页解决分配与共享效率，不能把存储 KV 的逻辑字节数凭空消除。尾块仍可能有内部碎片；共享可变尾块需要额外的隔离或 copy-on-write 机制，不能无条件覆盖。

## 7. 外部缓存如何加入运行链路

### D07：外部命中与加载的时序

```mermaid
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant C as Connector调度侧
    participant K as KV Manager
    participant W as Connector执行侧
    participant X as LMCache或其他后端
    participant G as GPU执行
    S->>C: 查找本地未覆盖的可复用token
    C->>X: 查询兼容的缓存对象
    X-->>C: 命中范围或未命中
    C-->>S: 候选复用长度与加载状态
    S->>K: 预留目标blocks
    S->>W: 下发加载元数据
    W->>X: 取回KV
    X-->>W: 数据传输与完成通知
    W-->>S: 加载完成或失败
    alt 加载成功
        S->>G: 允许执行依赖该KV的计算
    else miss或加载失败
        S->>S: 撤销无效复用状态
        S->>G: 按支持的路径重算或终止请求
    end
    Note over W,G: layerwise实现可按层等待；并非全部传输都阻塞整个引擎
```

调度侧回答“哪些 token 可以复用、目标 blocks 在哪里”，执行侧完成实际搬运和同步。**lookup 命中不是 GPU 数据 ready**。异步加载时，其他请求可能继续运行。准确接口见 V6。

## 8. 多卡部署

### D08：两个稠密模型副本，每副本 TP=2 的示意

```mermaid
flowchart TB
    U["请求"] --> R["路由 / 排队 / admission"]
    R --> E0["EngineCore：副本 0"]
    R --> E1["EngineCore：副本 1"]
    subgraph DP0["副本 0，TP=2"]
        G0["GPU 0：每层部分权重与KV"]
        G1["GPU 1：每层部分权重与KV"]
        G0 <-->|"collective"| G1
    end
    subgraph DP1["副本 1，TP=2"]
        G2["GPU 2：每层部分权重与KV"]
        G3["GPU 3：每层部分权重与KV"]
        G2 <-->|"collective"| G3
    end
    E0 --> G0
    E0 --> G1
    E1 --> G2
    E1 --> G3
    E0 -.-> CACHE["可选：共享KV存储 / 缓存位置索引"]
    E1 -.-> CACHE
```

- **TP**：同一层切分；减少单卡负担，引入每层通信。
- **PP**：按层/阶段切分；有跨阶段传输和流水气泡。
- **DP**：稠密模型通常复制权重，处理不同请求；提高总服务能力。
- **EP**：MoE 专家分散；需要 token dispatch/combine，可能出现负载不均。
- **上下文并行**：按 token/context 维度分摊 Attention 工作或 KV 状态，面向长上下文；Prefill/Decode 的切分与通信方式可能不同，具体 CP/DCP 等能力需核对版本和后端。

图中的路由、共享缓存和位置索引是可组合的服务组件，不意味着 vLLM 默认部署自动具备完整的缓存感知集群路由。

## 9. 指标与瓶颈地图

| 指标 | 定义与边界 |
| --- | --- |
| TTFT | 从选定起点到首输出的时间；客户端口径包含网络、排队等 |
| ITL | 相邻输出 token 的间隔；流式 chunk 可能包含多个 token，需注明观测方式 |
| TPOT | 对输出数 N>1 的请求，通常取首输出之后的平均每 token 时间 |
| E2E | 请求开始到结束；结束消息和客户端解析可能形成额外尾部开销 |
| tokens/s | 指明输入、输出还是总 token；不能与 requests/s 混用 |
| Goodput | 单位时间内成功且满足指定延迟约束的请求数 |
| 命中率 | 明确按请求、token、chunk 还是字节；明确本地/外部查询的分母 |
| GPU utilization | 活跃程度的一个信号，不等于有效 FLOPs、HBM 利用或业务效率 |

若 TPOT 由首 token 到末 token 的间隔计算，则：

```text
首到末输出耗时 = (N - 1) × TPOT
请求 E2E ≈ TTFT + (N - 1) × TPOT
```

第二式近似忽略首末输出之外的尾部开销。单 token 请求不适用上述 TPOT 公式。Goodput 必须同时报告失败率、拒绝率和提交负载，避免只展示“幸存请求”。

### D09：性能排查决策图

```mermaid
flowchart TD
    A["TTFT / ITL / Goodput 未达标"] --> B["确认计时口径、负载与冷暖状态"]
    B --> C{"主要延迟在哪里"}
    C -->|"排队"| Q["检查到达率、admission、KV压力、路由"]
    C -->|"CPU或IPC"| H["tokenizer、调度、序列化、同步、日志"]
    C -->|"GPU执行"| G["检查kernel trace与计算/带宽"]
    C -->|"跨卡通信"| N["collective、拓扑、小消息、负载不均"]
    C -->|"缓存取回"| K["lookup、网络、H2D、布局转换、锁与队列"]
    Q --> V["提出一个可证伪的假设"]
    H --> V
    G --> V
    N --> V
    K --> V
    V --> E["保持其他条件固定，做对照实验"]
    E --> R["检查正确性、分位延迟、Goodput与成本"]
```

先区分排队、计算和搬运，再调参。指标给出症状，trace 和受控实验才能支持原因判断。
