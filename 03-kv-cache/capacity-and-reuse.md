# 03 · KV Cache 基础与性能模型

## 1. 从 Attention 解释 KV Cache

对标准 decoder-only Transformer 的某一层，输入隐状态经过投影得到 Q、K、V：

```text
Q = X Wq
K = X Wk
V = X Wv
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d) + causal_mask) V
```

在自回归生成第 t 个 token 时，新 token 的 Query 需要读取之前所有可见 token 的 K/V。过去位置的 K/V 在相同前缀和模型配置下不会因追加 token 而改变，因此可以缓存，避免每次重新执行整个历史前缀。

**为什么通常不缓存历史 Q？** 下一步主要需要新 token 的 Query 与历史 K/V 做 attention，历史 Query 通常不会再被用于新 token 的计算。其他模型架构可能有额外状态，不能用标准 KV 形式套所有网络。

**为什么不是“缓存后 attention 就是常数复杂度”？** KV Cache 消除了过去 token 的重复投影和前向计算，但普通全注意力 decode 仍需读取历史 K/V、计算当前 Query 对它们的 attention。上下文越长，相关流量和计算通常越大。

**为什么深层 KV 依赖整个前缀？** 第 1 层输入可能主要是 token 与位置，但后续层的隐状态已包含前面 attention 的结果。一个相同文本块接在不同前缀后，其深层 K/V 一般不同；位置编码也会影响兼容性。

## 2. Prefill、Decode 与指标

Prefill 一次处理多个 prompt token，具有较多矩阵乘法并行度；Decode 每条序列每步通常生成一个 token，需要读权重与不断增长的 KV。常见硬件下，长 prompt prefill 更容易受算力限制，小批量 decode 更容易受内存带宽限制，但这不是不可变的分类：batch、模型、attention kernel、量化、MoE 和上下文长度都会改变瓶颈。

| 指标 | 定义 | 解释时要说明什么 |
|---|---|---|
| TTFT | 请求发出至收到第一个输出 token 的时间 | 是否含网络、排队、tokenize、前缀查询和 KV 搬运 |
| ITL | 相邻输出 token 的时间间隔 | 分布和 P99，是否包含首 token |
| TPOT | 常见定义为 `(最后 token 时间 − 首 token 时间)/(输出 token 数 − 1)` | 单 token 输出时无定义；不同工具可能采用其他口径 |
| E2E | 请求发出至完整响应完成 | 固定输入/输出长度与终止条件 |
| 吞吐 | req/s、output tokens/s 等 | 不能把“复用的输入 token”当作新执行 FLOPs |
| Goodput | 每秒满足指定 TTFT、ITL/E2E 约束的成功请求数 | 指定判定规则、失败/拒绝是否计入分母、offered load |

只报告平均 TTFT 容易隐藏热点和长尾。只报告完成请求的延迟也容易隐藏拒绝、超时和排队丢失。

## 3. KV 容量公式

对于每层 KV 维度相同、标准 MHA/GQA/MQA、K/V 使用相同精度且不含额外压缩的情况：

```text
bytes_per_token = 2 × L × Hkv × d × element_bytes
KV_bytes = bytes_per_token × Σ(sequence_length_i)
```

其中 `L` 为 attention 层数，`Hkv` 为 KV heads 数，`d` 为每个 head 的维度，系数 2 对应 K 与 V。不能把 GQA 的 query heads 数 `Hq` 直接代入 `Hkv`。

如果层间配置不同，更通用的形式是：

```text
bytes_per_token =
    Σ_layer (K_elements_per_token × K_element_bytes
             + V_elements_per_token × V_element_bytes)
```

### 数量级算例

取 `L=32、Hkv=8、d=128、BF16=2 bytes`：

```text
2 × 32 × 8 × 128 × 2 = 131072 bytes = 128 KiB/token
8192 tokens × 128 KiB = 1 GiB/request
32768 tokens × 128 KiB = 4 GiB/request
64 条长度均为 8192 的活跃序列 = 64 GiB 原始 KV
```

这些数字没有包含权重、临时激活、CUDA graph、引擎预留、分页浪费、对齐、量化 scales 和副本。

若 GPU 可用 KV 预算是 40 GiB，最多可容纳 `40 × 8192 = 327680` 个此模型的 token 槽位；这不等价于固定支持 40 个任意长度请求。每个 decode 请求还会继续增长，需要预留或可控抢占。

### MHA、GQA、MQA、MLA

| 结构 | KV 特征 | 容量分析方法 |
|---|---|---|
| MHA | 通常 `Hkv = Hq` | 使用所有 KV heads |
| GQA | 多个 Q heads 共享一组 K/V heads | 使用较小的 `Hkv` |
| MQA | 通常一组 K/V heads | `Hkv=1`，注意 TP 下的复制 |
| MLA | 缓存压缩 latent 与可能独立的 RoPE key 等状态 | 按实现实际缓存张量求和，不能机械套 `2×Hkv×d` |
| Sliding Window / 混合注意力 | 部分层只保留窗口内状态 | 按每层窗口和全局层分别计算 |
| 混合状态模型 | 可能还有卷积、递归或状态空间状态 | 检查引擎缓存布局和额外状态 |

MLA 的教学估算可写成 `Σ_layer(latent_dim + rope_key_dim) × bytes`，但只适用于确实采用该压缩缓存布局的实现。有的执行路径会物化不同张量，必须核对实际内存。

### TP/PP/DP 与副本

- **TP**：如果 KV heads 在 TP ranks 均匀分片、没有复制，可把单请求全模型 KV 除以 TP。上例 TP=8 时每 rank 为 128 MiB。
- **例外**：`Hkv < TP`、MQA、MLA、某些 kernel/layout 或额外中间状态可能导致复制或不同分片，实际每 rank 开销不再严格等于总量除 TP。
- **PP**：每个 stage 保存本 stage 的相关层，按层数求和，不按 TP 规则推断。
- **DP**：独立模型实例会有独立活跃 KV，除非上层建立明确的共享存储/复用机制。
- **Store 副本**：是存储冗余，独立于 TP 分片。两个 TP shards 互补，不是彼此的数据备份。

## 4. PagedAttention 与前缀共享

PagedAttention 的关键是把逻辑 token block 映射到不必物理连续的 GPU block，降低外部碎片、支持动态增长和共享；attention kernel 通过 block table 读取。

设 block 大小为 `b`，请求长度为 `S`：

```text
allocated_tokens = ceil(S / b) × b
tail_waste_tokens = allocated_tokens − S
```

若尾部长度分布近似均匀，单条序列平均尾部浪费约 `(b−1)/2` 个 token。共享、混合 block 规格及运行时行为会改变真实数值。

两个请求共享已完成前缀时，多个 block table 可以指向同一物理块，引用计数用于避免过早释放。要改写共享块时，需要 copy-on-write 或等价保护。未完成尾块的复用与发布规则依赖引擎实现。

**分页和前缀缓存的关系**：分页提供分配/寻址机制，prefix cache 决定已计算内容能否跨请求复用，远端 Store 扩展保存位置。三者分别解决不同问题。

**FlashAttention 的关系**：优化 attention 运算中的内存访问和中间张量，尤其避免物化大的 attention 矩阵；并没有自动消除跨生成步骤需要保存的 K/V。它与分页、KV 复用可以协同。

## 5. 缓存键必须表达计算语义

概念上的链式前缀标识：

```text
prefix_hash_i = H(prefix_hash_(i−1), token_ids_in_chunk_i, semantic_extras)
object_key = encode(namespace, prefix_hash_i, rank_or_shard, layout_version)
```

这是设计模型，不是所有框架的逐字 key 格式。实际序列化应确定、无歧义、跨进程稳定，必要时采用 canonical 编码与合适的哈希算法。

兼容性检查至少考虑：

| 字段 | 为什么重要 |
|---|---|
| 模型架构与权重 revision | 同一模型名称下的权重更新也能改变 KV |
| Token IDs、chat template、tokenizer 版本 | 同一可见文本可能得到不同实际输入 |
| 之前的 prefix hash | 当前块相同不代表深层 KV 相同 |
| 位置、RoPE/scaling、attention 配置 | 改变模型对位置与上下文的计算 |
| LoRA/adapter 标识和版本 | adapter 会影响隐状态与 KV |
| 多模态内容/预处理身份 | 占位 token 相同不代表图像或 embedding 相同 |
| KV dtype、量化 scales、布局版本 | 相同字节长度不代表可直接解释 |
| TP/PP 切分或可执行转换协议 | rank 变化可能要求重分片或布局变换 |
| tenant / sharing group / cache salt | 控制允许的共享边界和配额 |

生成 temperature 通常不改变已有 prompt 的 KV，不能把所有推理参数都无差别加入 key。需要识别真正影响缓存内容与兼容性的参数。

Store 通常把 key 当不透明字符串，不能替应用保证键具有上述语义。缓存 salt/namespace 也不等于鉴权：它避免不当复用，访问控制仍需部署层实现。

## 6. 前缀命中到底省了什么

对长度 `S` 的 prompt，如果命中前 `P` 个 token：

- 可以省去已命中 token 的相关前向计算。
- 新的 suffix token 仍要执行各层计算，并 attend 到已命中的 prefix。
- full hit 也不保证 TTFT 为零：有调度、加载、末 token/logits 处理和首 token 采样。一些引擎会保留或重算最后一个 prompt token 来生成 logits。
- KV cache 不会自动缓存下一次请求的完整输出文本。

因此不能精确地把 prefill 节省时间写成 `P/S × 原 prefill 时间`。可以作为粗估，但最终要用实际 kernel、batch、长度和调度测量。

三个命中率口径：

```text
request_hit_rate = 至少命中一部分的请求数 / 请求总数
token_hit_rate = 可复用输入 token 数 / 输入 token 总数
byte_hit_rate = 实际复用 KV 字节数 / 所需 KV 字节数
```

最好再报告：按 tier 区分的命中、成功加载的连续前缀、跳过的实际 prefill token 数，以及节省的 GPU 时间。命中 100 个小请求可能不及命中一个超长请求有价值。

## 7. 拉取与重算的 break-even

比较同一个请求在相同负载下的两条路径：

```text
T_cold  = Q_cold + T_prefill_full + T_first_token_overhead
T_reuse = Q_reuse + T_lookup + T_load_critical
          + T_prefill_suffix + T_first_token_overhead
```

缓存值得使用的近似条件是：

```text
T_lookup + T_load_critical + (Q_reuse − Q_cold)
    < T_prefill_full − T_prefill_suffix
```

`T_load_critical` 是关键路径上不能被隐藏的部分。串行路径可以近似累加远端读取、CPU→GPU 和布局转换；稳定流水线可能接近若干阶段的最大值，但仍有启动、尾部和同步开销，不能一律把所有成本相加或全部重叠。

以 1 GiB KV、实测有效带宽假设为 20 GiB/s 的教学场景：

```text
仅网络传输下界 = 1 / 20 s = 50 ms
假设查询 2 ms + 主机到 GPU 40 ms + 布局/同步 8 ms
串行加载成本约 100 ms
若可省 prefill 700 ms：有潜在收益
若可省 prefill 60 ms：应考虑重算
```

这里的 20 GiB/s 是假设的有效 payload 带宽。200 Gb/s 的物理线速是 25 GB/s，约 23.28 GiB/s，应用有效带宽还会更低。

对于整个系统还应考虑新增缓存写流量、复制、后台预取占用和排队。单个请求的加载获益并不自动意味着集群吞吐提升。

## 8. Chunk 大小的取舍

小 chunk 带来更细的复用边界、更少多取数据，但产生更多 key、RPC、hash、元数据和完成事件。大 chunk 能摊薄固定开销、提高带宽，却会减少部分前缀复用、增加尾部浪费和单次等待。

应与引擎 page 对齐，并扫描不同 chunk 大小，记录：

- 保存/读取对象数和元数据 CPU；
- 真正可用的 prefix 长度；
- bytes/s、objects/s、P99 查询/读取延迟；
- 对齐/重排成本和多 rank 同步成本；
- 最终 TTFT、ITL 与 goodput。

不要把某个框架的默认 chunk 大小当成所有模型的最优值。

## 9. 分层、准入、淘汰与预取

| 决策 | 核心问题 | 常见方案与代价 |
|---|---|---|
| 准入 | 是否值得写入缓存？ | 二次访问准入、长度阈值、估计复用收益；减少冷数据污染但可能错过首次后续复用 |
| 放置 | 存在哪里、几个副本？ | 本地优先、空闲比优先、热点复制、故障域隔离 |
| 淘汰 | 满了先删谁？ | LRU 简单但会被扫描污染；LFU/TinyLFU 估计热度；考虑重算收益与体积 |
| 预取 | 提前拿多少、等多久？ | 根据预测、截止时间和可用 GPU buffer；误预取浪费带宽 |
| 写回 | 何时往下层写？ | write-through、selective、write-back；异步失败需可观测 |

一种教学价值分数：

```text
value_per_byte =
    (预计复用次数 × 每次节省的计算时间 − 写入/读取/复制成本) / 占用字节
```

它是设计启发式，不是 Mooncake 默认淘汰公式。保留热门系统提示、避免随机一次性长 prompt 冲走热点，往往比盲目增大缓存更有效。

## 10. 调度与排队

PD 分离让 Prefill 与 Decode 可分别分配硬件和 batch 策略，缓解相互干扰，但增加交接 KV、路由和状态管理。短 prompt、低并发、网络受限或负载失衡时，分离可能没有收益。

缓存感知调度的候选代价可估计为：

```text
T_candidate = queue_wait
              + cache_lookup_and_load
              + uncached_prefill
              + PD_handoff
```

同时约束 Decode 的 ITL/E2E。Conductor 的命中信息可以帮助估算，但调度器还要知道负载。排队 800 ms 的热缓存节点，可能输给排队 10 ms、重算 300 ms 的冷节点。

稳定系统的 Little 定律 `并发量 ≈ 到达率 × 平均停留时间` 有助于估算在途请求和 buffer 数。它不直接预测 P99，也不能在持续过载、队列无限增长时套用稳定态结论。

## 本章自检

能否回答：为什么不是所有 KV 都值得外存？为什么缓存命中但 TTFT 变差？为什么 `Hkv` 不能写成 `Hq`？为什么“字节相等”只是数据传输正确性，而不是不同模型间 KV 共享的充分条件？

下一章把这些原则映射到 [Mooncake 源码](../04-storage/implementations/mooncake/internals.md)。
