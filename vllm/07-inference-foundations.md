# 推理基础教材：从 Transformer 到服务成本

[返回导航](README.md) · [28 天计划](01-study-plan.md) · [下一本：源码教材](08-vllm-source-walkthrough.md)

本章不要求先读论文。目标是建立后续源码中每个 token、状态和字节的含义。F1～F8 对应计划前四天；计算题答案在本章和[实践教材](10-practice-and-solutions.md)中。

## F1. 一次文本生成到底在做什么

模型接受 token IDs，而不是直接接受字符串。一次常规文本生成可拆成：

```text
文本/消息
→ chat template（适用时）
→ tokenizer
→ token IDs
→ embedding
→ 多层 Transformer
→ 最后位置的 hidden state
→ vocabulary logits
→ 采样得到下一个 token
→ 将新 token 纳入历史，继续执行
→ detokenization 与输出
```

token 不等于汉字、单词或字节。相同字符串在不同 tokenizer、模板、特殊 token 设置下可能对应不同序列；这会影响长度预算、缓存身份、答案和速度。

例如用户问“解释 KV Cache”，模型看到的可能还包含 system 内容、角色分隔符和 assistant 前缀。计算输入长度时，要算实际送入模型的完整 token 序列。

三个不同的配置概念：

- **模型最大上下文能力**：模型及位置编码设计允许的范围。
- **服务允许的序列长度**：引擎配置对请求长度的限制。
- **本次最大生成长度**：该请求最多生成多少新 token。

这三者不能互换。限制生成 128 token 不代表模型只需要保存 128 token 的 KV。

**练习**：输入 4096 token、计划输出 512 token，普通全注意力生成末段会访问多长历史？  
**答案**：数量级接近 4608 token；精确到某一时刻，要区分已输出但还没作为输入 forward 的最后一个 token。

## F2. 看懂一个 Transformer 层的 shape

设：

| 符号 | 含义 |
| --- | --- |
| B | batch 中的序列数；变长打包时不一定显式保留这一维 |
| T | 本次输入长度 |
| Hq / Hkv | query heads / KV heads |
| d | head dimension |
| D | 模型 hidden size，常见情况下 `D = Hq × d` |
| F | FFN 中间维度 |

对于常见的 GQA decoder layer，概念形状为：

```text
X:          [B, T, D]
Q:          [B, T, Hq, d]
K, V:       [B, T, Hkv, d]
Attention:  [B, T, Hq, d]
合并 heads: [B, T, D]
```

实际 kernel 的维度顺序、打包方式和 KV 布局可能不同。

### Attention 的计算

对一个 head：

```text
scores = Q K^T / sqrt(d) + causal_mask
weights = softmax(scores)
output = weights V
```

因果 mask 确保位置 t 不能看 t 后面的输入。Prefill 中多个位置可以并行计算，不意味着模型可以偷看未来 token。

softmax 为数值稳定通常先减最大值：

```text
p_i = exp(s_i - max(s)) / Σ_j exp(s_j - max(s))
```

### 位置与 FFN

RoPE 通常作用于 Q/K，把位置相关旋转融入注意力匹配；具体模型可能使用不同位置编码。不同位置配置下的 K 即使对应相同 token，也不一定兼容。

常见 gated FFN 可抽象为：

```text
FFN(X) = (activation(X W_gate) * (X W_up)) W_down
```

Attention 负责上下文信息交互，FFN 是重要的参数与计算来源。推理优化不能只关注 Attention。

RMSNorm、残差连接和输出投影也参与 forward。它们往往单个算子计算量不大，但高频 launch 和中间读写会累积成可观开销。

**练习**：D=4096、Hq=32、Hkv=8，则 d 和每位置 K 的元素数是多少？  
**答案**：d=128；K 是 `8×128=1024` 个元素；K+V 是 2048 个元素，每层 BF16 为 4096 字节。

## F3. Prefill：并行处理 prompt

输入 prompt 长度 T，Prefill 在各层计算这 T 个位置的状态，保存未来生成需要的 K/V，并从末位置 logits 生成首个输出 token。

学习计算成本时可拆成两类：

1. 投影、FFN 等主要参数矩阵运算，随处理 token 数近似线性增长。
2. 全注意力 QK/AV，随着 query 与 key 长度乘积增长。

对普通全注意力，完整 Prefill 的注意力项有 O(T²) 的规模。FlashAttention 可显著减少中间结果存储和 IO，但不会把稠密全注意力的数学工作量直接变成线性。

大 Prefill 常能把矩阵乘法做得足够大，让 GPU 有较高计算效率；小 prompt、小模型、特殊后端仍可能受 launch 或其他开销限制。

### Chunked Prefill 的数学含义

把 T 个输入 token 拆成若干 chunk，每次计算本 chunk 的 query，同时访问此前已经保存的 K/V。

```text
chunk 1：query 来自 token 0..c-1，能看其因果历史
chunk 2：query 来自 token c..2c-1，能看之前 KV 和本 chunk 合法位置
……
```

分块改变调度与计算形状，不应改变普通因果模型允许访问的历史。正确的实现需要传入位置、历史长度、block table 和 mask 元数据。

**为什么分块影响 TTFT？** 首 token 通常要等最后的 prompt chunk 处理完；被拆成更多调度轮次后，可能增加等待，但也可能减少队列中其他请求的阻塞。实际 TTFT 取决于负载和策略。

## F4. Decode 与“生成 token / 已计算 token”的区别

假设 prompt 是 `[a,b,c]`：

| 阶段 | 本轮 forward 输入 | 本轮后新增的 KV | 本轮采样输出 |
| --- | --- | --- | --- |
| Prefill | a,b,c | a,b,c 的 KV | d |
| Decode 1 | d | d 的 KV | e |
| Decode 2 | e | e 的 KV | f |

所以刚输出 d 时，d 已在逻辑生成序列里，但通常尚未完成以 d 为输入的 forward，也尚无 d 的全部层 KV。读 `num_computed_tokens` 时必须保留这一区分。

对普通 Decode，query 长度通常是 1，KV 长度是当前可访问历史长度。缓存历史 KV 之后：

- 不再每次从头计算历史各层的隐藏状态。
- 新 token 仍需经过所有相关层。
- Attention 仍要读取历史 K/V 并计算结果。

因此 KV 是“计算换存储”的优化，长上下文生成不会因为缓存而变成 O(1) 的全部工作。

### 为什么不缓存历史 Q

新位置用自己的 Q 查询历史 K/V，历史 Q 通常不再参与新位置的 Attention。存储历史 Q 不能消除这条标准计算路径中的必要工作。

### 请求内 KV 与跨请求复用

请求内缓存依赖当前请求已经计算的历史。跨请求复用还要证明两个前缀在模型、token、位置、adapter 等条件下兼容。相同文档出现在不同前文后，不足以证明其深层 KV 相同。

**实践**：运行 `labs/inference_lab.py attention`，对同一给定输入比较“完整因果 Attention”与“逐 token 缓存 K/V”的结果。这个实验只模拟一个 Attention 层，帮助理解等价关系，不模拟 vLLM 的全部执行方式。

## F5. KV 显存：从公式到 block 取整

同构全注意力模型：

```text
每 token KV = 2 × L × Hkv × d × b
独立请求逻辑 KV = 每 token KV × ΣTi
```

2 表示 K 和 V，b 是每元素字节数。MHA/MQA/GQA 的差异体现在 Hkv；不能用 Hq 代替。

例子：L=32、Hkv=8、d=128、BF16 b=2：

```text
每 token = 131072 bytes = 128 KiB
8192 tokens = 1 GiB
```

### 分页取整

每块容纳 C 个 token，独立请求长度为 Ti：

```text
所需块数 = Σ ceil(Ti / C)
预留 token slots = C × 所需块数
```

例如 C=16，两请求长度 17 和 31：

```text
块数 = 2 + 2 = 4
实际 tokens = 48，预留 slots = 64
未用 slots = 16
```

尾块内部碎片仍然存在。分页主要缓解大段连续内存分配和过度预留问题，并支持按需增长和共享。

### 前缀共享

两请求前 32 token 相同，块大小 16，且模型语义兼容，则前两个完整块可能共享；不同后缀用不同块。物理总容量应按唯一活跃物理块计算，而非盲目相加所有逻辑序列长度。

### 公式何时不能直接用

- Sliding Window Attention 可能只保留窗口内必需状态。
- MLA 使用压缩 latent 等状态，布局不同。
- Mamba/混合模型有不同层型与状态。
- TP 分片可能复制少量 KV heads，PP 改变每卡层数。
- KV 量化还涉及 scale、元数据与布局约束。

这些情况下按实际每层、每 rank、每种缓存组求和。不要把“8 卡”直接当成 KV 物理占用除以 8。

## F6. 权重、激活与 GPU 总显存

先做容量分解：

```text
总显存需求 ≈ 权重 + KV + 激活峰值 + workspace
           + CUDA Graph / 通信 / 传输缓冲 + 其他开销
```

一个 8B 参数模型，按每参数 2 bytes 粗算权重为 16 GB，约 14.90 GiB。实际格式、额外参数、分片与精度会改变加载量。

训练中需要梯度、优化器状态与反向图；推理不需要这些，并不代表推理没有激活和临时内存。大 Prefill、logits/logprobs、图捕获或新功能仍可能抬高峰值。

### 为什么进程显存不下降不等于泄漏

引擎可能预先分配 KV pool；请求结束只把 block 退回可复用队列，而不是每次向 CUDA 释放内存。CPU/GPU allocator、graph 和合法缓存也可能持有内存。

判断泄漏应看：完成请求是否仍有不可解释的引用；长时间相同稳态负载下，活跃占用是否持续增长；是否存在在途操作或不可驱逐对象。

### 为什么 max length 不等于每请求预分配 max length

分页 KV 往往按当前进度和策略分配；最大长度是合法性与容量规划的约束之一，并不自动等于为每个新请求一次性分配全部长度。某些 admission 策略会检查整个序列能否放下，仍要与实际分配区分。

## F7. GPU 成本模型与 Roofline

计算性能至少受以下几类限制：

- SM/Tensor Core 的计算吞吐。
- HBM、L2、shared memory 等层次的访问效率。
- CPU launch 与调度供给。
- GPU 间通信和同步。
- CPU↔GPU、NIC↔GPU 等传输。

算术强度：

```text
I = FLOPs / 从相应存储层搬运的字节数
Roofline 上界 ≈ min(可用计算吞吐, 可用带宽 × I)
```

这是一种上界模型，不能代替测量。实际还有访存模式、occupancy、kernel 形状、依赖和通信。

### 一个仅用于推理的数量级例子

假设稠密模型主要参数 P=8B、BF16 权重，每步单卡确实读取约 16 GB 权重，设备可用带宽为 2 TB/s：

```text
仅权重读取下界 ≈ 16 GB / 2 TB/s = 8 ms
```

如果忽略 Attention、通信、launch 等，单 token 主体矩阵计算约 `2P=16 GFLOPs`。假设可用计算吞吐 200 TFLOP/s，对应理想计算下界约 0.08 ms。两者差距提示小 batch 时权重带宽可能更关键。

这些是给定假设的教学数字，不是任一 GPU 的实测值。

batch 增大到 B，同一份权重可服务更多 token，主体计算约 `2PB`，理想权重相关算术强度约为 `2B/b`。b=2 时约为 B FLOPs/byte。因此 batch 增大可能提高总 tokens/s，但每请求等待和执行时间也可能增大。

### 长上下文为什么改变结论

Decode Attention 每步还要访问历史 KV。上下文越长，KV 读取在总时间中的占比可能越高；不能一直用“所有 Decode 都只受权重带宽限制”解释系统。

**练习**：一个 KV 块从远端取回需要 30 ms，省掉 Prefill 只有 8 ms，是否应当无条件复用？  
**答案**：不应。还需看真实重叠和排队影响，但只凭命中不能证明有收益。

## F8. 采样、输出与服务指标

### logits 到 token

logits 是每个词表项的未归一化得分。常见生成策略包括 greedy、temperature、top-k、top-p，以及重复惩罚、允许词集合或语法约束。

这些策略的组合与执行顺序应以实现为准。temperature=0 的 greedy 也不意味着跨任意硬件、batch 或数值路径都保证完全相同结果。

结构化输出可能在采样前限制合法 token；stop string 检测往往还涉及 detokenization。模型完成一个 forward 与服务决定“请求结束”不是同一层职责。

### 指标的起止点

```text
请求发出
  → 网络与输入处理
  → 等待调度
  → Prefill / KV取回
  → 首输出
  → 后续输出
  → 最终完成
```

- **TTFT**：起点到首输出；注明客户端或服务端口径。
- **ITL**：连续 token 的间隔；网络 chunk 可能含多个 token。
- **TPOT**：一个请求首 token 后的平均每 token 时间。
- **E2E**：整个请求的完成时间。
- **Goodput**：单位时间内成功且满足指定延迟条件的请求数。

输出 N>1 时：

```text
首到末输出耗时 = (N - 1) × TPOT
E2E ≈ TTFT + (N - 1) × TPOT
```

第二式忽略结束消息等尾部开销。N=1 不能按上述式子计算 TPOT。

### 为什么平均值不够

若 99 个请求 TTFT=100 ms，一个请求=10 s，平均值约 199 ms，却掩盖了严重的尾延迟。还要按输入/输出长度、缓存命中与租户等维度分桶，避免负载组成变化被误判为优化。

Goodput 若使用平均 TPOT 约束，也不能自动保证每个 ITL 都平滑。需要根据产品体验选择指标。

### 本章自测

1. 首次输出 token 时，它自己的 KV 是否一定已计算？
2. 为什么相同请求长度下 GQA 比 MHA 可能需要更少 KV？
3. Prefill 并行计算为何不违反因果约束？
4. 分页为何仍有内部碎片？
5. 更高 tokens/s 为什么可能伴随更低 Goodput？

答案依次是：通常尚未；KV heads 更少；mask 限制可见历史；末块取整；延迟或失败导致更多请求不达 SLO。能结合上面的例子解释，再进入源码教材。
