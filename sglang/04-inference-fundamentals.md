# 推理基础、调度、GPU 优化与分布式

[返回导航](README.md) · [KV Cache 专题](05-kv-cache.md) · [实验手册](07-hands-on-labs.md)

## 1. Attention 与自回归生成

标准 Attention 的基本形式：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right)V
$$

其中因果 mask \(M\) 阻止当前位置读取未来位置。各层还包含位置编码、归一化、残差、MLP 等计算。

生成过程：

1. 对 Prompt 做 Prefill，计算各层状态和末位置的 logits。
2. 按采样规则选择首个输出 token。
3. 将新 token 输入下一次前向，读取已有 KV，产生下一个 token。
4. 达到 EOS、停止条件、长度预算或取消条件后结束。

RoPE 让位置参与 Q/K 的变换；缓存兼容性不仅取决于文字，还取决于位置与模型配置。

## 2. Prefill 与 Decode 的性能差异

Prefill 一次处理多个输入 token，线性层容易形成较大的矩阵乘法。Decode 每个活跃请求通常只新增一个 token，需要读取权重与历史 KV。

- Prefill 经常有较高算术强度，但长上下文 Attention、通信、小 batch 也可能成为瓶颈。
- 小 batch Decode 常受权重读取带宽限制。
- 大 batch 和长上下文会提高计算量与 KV 流量，瓶颈可能转移。
- 多卡部署还要考虑通信和 rank 间等待。

“Prefill 一定算力受限、Decode 一定带宽受限”只是粗略直觉，不能作为实测结论。

对 Dense 模型，线性层计算量可用：

$$
F_{\text{linear}}\approx 2\times P\times N_{\text{tokens}}
$$

这里 \(P\) 是参与这些线性层计算的参数量。该近似没有完整包含 Attention 的序列长度相关开销，也不能直接套用到所有 MoE 模型。

## 3. Roofline 与资源模型

算术强度：

$$
I=\frac{\text{FLOPs}}{\text{访问字节数}}
$$

粗略上界：

$$
\text{可达性能}\leq
\min(\text{峰值算力},I\times\text{有效带宽})
$$

使用时要注意：

1. 理论 HBM 带宽不是所有访问模式的可达带宽。
2. 小矩阵可能无法充分使用 Tensor Core。
3. Kernel launch、同步、CPU 调度不直接包含在简单 Roofline 中。
4. 通信与数据拷贝可能处于关键路径。
5. GPU 利用率不是“算力利用率”的同义词。

## 4. 指标与测量边界

| 指标 | 定义 | 易错点 |
|---|---|---|
| TTFT | 客户端发出请求到收到首个输出 token | 只统计 GPU Prefill，遗漏排队、分词、加载 |
| ITL | 相邻输出 token 到达间隔 | 网络缓冲和流式合并可能影响观测 |
| TPOT | 单请求首 token 后的平均每 token 时间 | 不等同于跨请求 ITL 分位数 |
| E2E | 请求发出到输出结束 | 只比较均值，忽视尾延迟 |
| Output throughput | 单位时间完成的输出 token 数 | 与输入加输出总吞吐混用 |
| Goodput | 满足规定 SLO 的请求或 token 吞吐 | 只看成功请求而忽略拒绝与超时 |

对 \(N>1\) 的单请求：

$$
TPOT=\frac{E2E-TTFT}{N-1}
$$

TTFT 的关键路径可以拆为：

$$
T_{\text{TTFT}}\approx T_{\text{排队}}
+T_{\text{输入处理}}
+T_{\text{未隐藏的缓存加载}}
+T_{\text{剩余 Prefill}}
+T_{\text{采样与返回}}
$$

实际存在重叠，不能把所有 trace 区间简单相加。

推荐报告 p50/p95/p99 TTFT、ITL，以及错误率、拒绝率和 Goodput。SLO 的定义必须固定，例如每请求 TTFT 与 TPOT 均达标，或对流式 ITL 使用额外约束。

## 5. 到达率、并发与排队

- 固定并发测试适合观察稳态吞吐与资源效率。
- 固定到达率测试适合观察服务容量和排队增长。
- 同时设置到达率和并发上限时，客户端可能被并发上限反压，实际 offered rate 低于目标。
- 突发流量、长度长尾和热点变化往往比平均输入更能暴露问题。

在稳定系统与一致统计边界下，Little 定律：

$$
N=\lambda W
$$

其中 \(N\) 是平均在系统中的请求数，\(\lambda\) 是有效到达率，\(W\) 是平均停留时间。这是容量思考工具，不是无条件适用的显存计算公式。

## 6. Continuous batching

静态 batch 常需等待整体结束；不同输出长度会造成空闲。Continuous batching 在迭代边界加入新请求、移除完成请求，提高资源利用。

调度要考虑：

1. 当前 Prefill token 预算。
2. 活跃请求数。
3. 当前与未来增长所需 KV。
4. 已有缓存的复用收益。
5. 队列等待与公平性。
6. Prefill 对 Decode 的干扰。

缓存友好策略可以提高复用，但若无限优先处理热点前缀，也可能造成长尾请求饥饿。应明确公平性与 SLO 约束。

## 7. Chunked prefill

把长 Prompt 分成多个块，控制单次 Prefill 的执行时间：

```text
不分块：
GPU： [很长的 Prefill....................][Decode][Decode]

分块的概念表示：
GPU： [Prefill chunk][Decode][Prefill chunk][Decode]...
```

真实实现也可能采用混合批次。分块不会自动减少需要处理的 Prompt 内容，主要改变调度粒度和干扰。

| Chunk 更小 | Chunk 更大 |
|---|---|
| 降低一次占用 GPU 的时间 | 更大的矩阵，计算效率可能更好 |
| 更多调度与 kernel 开销 | 可能增加 Decode 的 ITL |
| 有利于细粒度交错 | 可能有更高临时内存需求 |

实验应使用长输入和活跃 Decode 混合负载，不能只跑单个长 Prompt。

## 8. 准入、回退与取消

请求能否运行，不只看“当前空闲显存”：

- 是否存在可淘汰的空闲前缀。
- 是否有锁定或传输中的 KV。
- Decode 继续增长需要多少空间。
- 回退请求要重算，还是可以从其他层恢复。
- 取消后 GPU、网络与存储任务是否仍在运行。

取消不是简单删除一个队列对象。必须保证仍在执行的异步操作不会访问已经回收的内存。

## 9. GPU 执行优化

| 技术 | 主要优化对象 | 代价与限制 |
|---|---|---|
| CUDA Graph | CPU 发射大量 kernel 的开销 | shape、地址与控制流约束；capture 与额外内存 |
| CPU/GPU overlap | 调度与结果处理产生的 GPU 空隙 | 依赖和资源生命周期更复杂 |
| 算子融合 | 中间张量读写与 launch | 寄存器压力、可维护性与 shape 适用范围 |
| FlashAttention | Attention 的 HBM 流量与中间矩阵 | 硬件、dtype、head dimension 与 backend 限制 |
| 高效 Attention backend | 针对 Prefill/Decode 的算法与 kernel | 不同模型、形状和模式的兼容性 |

FlashAttention 使用分块与在线 softmax 等方法，避免完整 Attention 矩阵落地，数学上保持原 Attention。浮点计算顺序变化可能导致数值差异，不应默认 bitwise 一致。

FlashInfer 等库提供一系列推理 kernel；在 SGLang 中应区分上层接口、backend 选择和底层 kernel。

Kernel 学习路线：shape 与布局 → 数据流量 → launch → 向量化/分块 → 融合 → 数值验证 → 端到端验证。

## 10. 量化与投机解码

### 10.1 量化

- 权重量化减少权重容量与读取流量。
- 激活量化改变算子路径和数值范围。
- KV 量化减少缓存容量与读取字节。
- 实际收益取决于硬件、kernel、反量化成本和 batch。

验证至少包括任务质量、logprob 或合理的数值检查、显存、吞吐和尾延迟。压缩倍率不等于速度倍率。

### 10.2 投机解码

Draft 产生候选，目标模型批量验证，争取一次目标执行产生多个有效 token。

收益取决于：

- 候选接受率。
- Draft 的计算与权重成本。
- 验证的 batch 与 kernel 效率。
- 额外 KV、通信、调度开销。

只有正确实现验证与采样校正规则的算法才能保证目标分布。高接受率也不一定意味着端到端加速。

## 11. 分布式并行

| 技术 | 切分对象 | 主要代价 | 常见动机 |
|---|---|---|---|
| TP | 单层张量与计算 | 高频 collective | 单模型容量和单请求计算 |
| PP | 模型层 | 流水线气泡与阶段传输 | 大模型、跨节点放置 |
| Replica DP | 完整推理实例 | 权重复制、缓存分散 | 提高整体服务吞吐 |
| EP | MoE experts | Dispatch/combine、All-to-All | 分布专家权重与计算 |
| CP/DCP | 上下文或 KV 工作 | Attention/KV 相关通信 | 长上下文容量与带宽 |

SGLang 中的 Attention DP 与“复制整个模型的服务副本”不是同一个概念，需要按实际 group 和模型层布局理解。

扩展性分析顺序：

1. 单卡是否能容纳模型与必要状态。
2. 单请求延迟是否满足要求。
3. 每层通信频率和数据量。
4. GPU/NIC/NUMA 拓扑与跨节点比例。
5. 扩大并行后是否减少单卡计算却增加等待。
6. 同等 GPU 预算下，增加副本是否更有效。

## 12. MoE

MoE 的激活参数少，降低的是每 token 的部分计算量；以下成本仍存在：

- 专家权重的总体驻留与分布。
- Token 路由和专家负载偏斜。
- 跨 GPU dispatch/combine。
- All-to-All 与拓扑影响。
- 较小专家 batch 导致的 GEMM 效率问题。

面试时要同时解释激活参数、总参数、专家并行、路由负载和通信。

## 13. 生产系统能力

| 能力 | 需要解决的问题 |
|---|---|
| 限流与背压 | 防止排队失控与资源耗尽 |
| 超时与取消 | 保证用户离开后资源仍能正确清理 |
| 路由 | 平衡缓存局部性、排队时间和负载 |
| 可观测性 | 请求、调度、缓存、模型、网络的统一关联 |
| 灰度与版本 | 模型升级后旧缓存不能误复用 |
| 多租户 | 访问控制、配额、缓存命名空间与隐私 |
| 故障恢复 | 区分可重算前缀与丢失的活跃状态 |
| 容量规划 | 围绕 SLO 和 Goodput，而不是只看 GPU 利用率 |

## 14. 常用排查顺序

1. 固定版本、输入和负载，复现问题。
2. 检查请求是否充足、是否被客户端限流。
3. 分解 TTFT、ITL 与队列时间。
4. 看 CPU、GPU、拷贝、通信是否在等待。
5. 检查缓存命中来自哪一层，以及实际节省多少计算。
6. 只改变一个主要变量。
7. 比较端到端指标与正确性。

具体实践见 [实验七](07-hands-on-labs.md#9-实验七分析-trace-并写报告)。
