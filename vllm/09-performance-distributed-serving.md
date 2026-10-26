# 性能、分布式与服务工程教材

[返回导航](README.md) · [源码教材](08-vllm-source-walkthrough.md) · [实践教材](10-practice-and-solutions.md)

本章把“为什么这样设计”与“怎样证明有效”连接起来。P1～P10 覆盖性能测试、GPU 执行、并行、PD、模型扩展和服务工程；无需先写 CUDA 才能开始。

## P1. 先建立可比较的性能基线

一个 benchmark 结论至少需要六个条件：

```text
模型与精度
硬件与拓扑
输入/输出长度分布
请求到达过程
服务配置与缓存冷热
指标与正确性要求
```

只说“某配置 1000 tokens/s”，缺少以上条件，就难以比较或复现。

### 开放式与闭环负载

闭环固定并发：始终维持约 C 个在途请求，完成后才补下一个。服务变慢时发送速度也降低，因此可能掩盖生产流量不断到来的排队问题。

开放式到达率：按 λ requests/s 产生请求；若长期超过处理能力，队列增长。实际客户端仍可能受并发限制、连接数和 CPU 限制，所以要记录真实发送时间。

不要把客户端等待发送的时间从所有指标里悄悄删除，再声称服务端仍有稳定低延迟。

### 单变量对照

先测短短、长短、短长三类负载，再加入混合。比较 token budget 时，不同时改变精度、模型、prefix 命中和输出长度。对缓存实验，先区分引擎 warmup 和 KV warmup。

比较总吞吐时报告实际输出 token 数，避免某配置更早 EOS，工作量减少却被误认为同等负载加速。

### 质量是约束

量化、CacheBlend 或近似算法可能改变质量；不同数值执行路径也可能影响 greedy token。对语义应保持一致的优化，要用合适的数值/任务对照；不能为了得到性能数字而忽略错误。

本地资料：[优化与调参](../configuration/optimization.md)、[指标设计](../design/metrics.md)。

## P2. 如何用 profiling 找原因

先看服务时间轴：

```text
输入处理 → 排队 → KV准备 → GPU执行 → 输出处理 → 网络返回
```

再看 GPU 执行内部：矩阵运算、Attention、采样、数据搬运、collective、空洞。

| 工具/证据 | 适合回答 |
| --- | --- |
| 服务指标与日志 | 请求在哪里排队？是否有抢占、miss、load失败？ |
| PyTorch Profiler | PyTorch/CPU/CUDA 操作之间怎样关联？ |
| Nsight Systems | CPU线程、CUDA stream、kernel、memcpy、通信怎样重叠？ |
| Nsight Compute | 某个 kernel 的访存、吞吐、occupancy、stall 如何？ |
| 网络/存储与缓存 trace | lookup、IO、H2D、重排和等待分别有多大？ |

### 一个排查例子

症状：GPU utilization 不低，但 P99 TTFT 持续增长。

先检查输入长度和到达率是否升高；然后分解等待与计算。如果 Prefill 时间稳定但等待变长，优先看 admission 和容量；如果同样长度的 Prefill 变慢，才进一步看 GPU/通信/缓存争用。

### 计时陷阱

CUDA 提交通常异步，Python 函数返回时间不等于 GPU 完成时间。微基准需正确同步或用设备事件，但在生产主循环里每步强行 synchronize 又可能破坏原有重叠。

Profiler 本身有开销。选择代表性的窗口，并将带 profiler 的结果与正常运行结果区分。

**作业**：写一个可证伪的假设，例如“尾延迟来自外部缓存 lookup 排队”。同时写出若假设错误，哪条观察会反驳它；不要只写“准备调小 batch”。

## P3. FlashAttention 为什么减少 IO

普通 Attention 的直接实现可能把 scores 和 softmax 结果作为大矩阵写到 HBM，再读回来。T 很长时，中间矩阵规模 O(T²)，既占容量又产生大量 IO。

FlashAttention 用分块与在线归一化，让 query block 与 key/value block 在更快的片上存储中计算，维护必要统计量和输出累加，而不保存整个注意力矩阵。

### 在线 softmax 的基本思想

对一个 query，已经处理的 key 集合维护：

```text
m：已见 score 的最大值
l：Σ exp(score - m)
u：Σ exp(score - m) × value
```

加入新块 scores s、values v：

```text
m_new = max(m, max(s))
l_new = exp(m-m_new) × l + Σ exp(s-m_new)
u_new = exp(m-m_new) × u + Σ exp(s-m_new) × v
output = u_new / l_new
```

第一次处理块时适当初始化；因果 mask 仍要正确应用。调整旧累积的尺度，是分块后仍保持同一归一化含义的关键。

这解释了两个边界：

- 算法保持稠密 Attention 的数学目标，浮点顺序可能不同。
- 它减少中间存储与 IO，不表示全注意力的计算复杂度直接从 O(T²) 变 O(T)。

### 与分页 KV 的关系

PagedAttention 解决历史 K/V 的物理管理和寻址。一个 backend 可以从分页 blocks 取 KV，再执行高效分块 Attention；两者处于不同层面。

本地资料：[Paged Attention](../design/paged_attention.md)、[Attention backends](../design/attention_backends.md)。前者的具体 kernel 说明有其历史实现背景，不应据此断言当前所有模型走同一 CUDA kernel。

## P4. kernel 优化应掌握的基本因果关系

### 内存层次与访存

HBM 容量大、带宽高但访问成本仍显著；片上 L2/shared memory/register 的容量更有限。连续、合并的访问更容易高效利用带宽，布局不合适可能增加无用事务。

寄存器更多可以减少局部变量溢出，但每线程占用过大也可能限制同时驻留的线程/blocks。occupancy 高不是唯一目标，低 occupancy 也可能在充分数据复用时高效。

### 融合为什么有用

考虑“残差相加 → 写临时 Tensor → RMSNorm 读回”：

```text
独立实现：多个launch + 临时结果写入HBM再读出
融合实现：适用时在同一kernel中保留中间值
```

融合减少 launch 和中间 IO，但可能增加寄存器压力、编译复杂度或限制形状。应该看端到端路径，不能只把 kernel 数量变少当作成功。

### shape 决定效率

相同 FLOPs 的矩阵形状可能有不同 Tensor Core 利用、padding 和并行粒度。大 Prefill GEMM 的优化经验不能原样套到小 batch Decode。

### 正确的优化顺序

1. 定义输入 shape/dtype/布局和数值要求。
2. 建立参考与真实瓶颈证据。
3. 修改数据复用、融合或并行方式。
4. 检查生成代码与设备指标。
5. 验证代表性 shape 和服务端到端收益。

本月无需从零优化所有算子。先能解释 RMSNorm、RoPE、GEMM、Attention 的访问特点，再按岗位选择一个深入。

## P5. 编译、CUDA Graph 与异步执行

### 三种不同层次

| 机制 | 主要作用 | 常见成本 |
| --- | --- | --- |
| eager | 灵活地逐次执行算子 | Python/launch 开销 |
| `torch.compile` 等编译 | 图优化、融合、代码生成、形状专门化 | 编译时间、重编译、动态分支限制 |
| CUDA Graph | 捕获设备执行依赖并重放，减少重复launch | 捕获范围、地址/shape要求、额外内存 |

编译与 Graph 可以配合，不能把它们当作同一个开关。

### 为什么需要 padding 和多个捕获规模

真实请求数和 token 数一直变化，捕获图可能需要映射到支持的执行规模。padding 可以换取重放效率，却多做计算；捕获更多规模可以减少 padding，却增加启动时间和内存。

某些功能只能在局部图之间运行，因此存在 piecewise 等模式。外部 KV 的层级同步可能影响捕获边界，必须检查 connector 和 backend 的兼容性。

### 异步的真实成本

CPU 准备下一步、GPU 计算当前步、NIC 搬运 KV 可以重叠。但如果共享同一个 pinned buffer，CPU 提前改写会使 GPU 读到错误数据。

正确方案可以是稳定临时副本、双缓冲、事件同步或明确所有权协议。增加 stream 不能自动解决依赖；多个操作争用同一带宽也未必有收益。

本地资料：[torch.compile](../design/torch_compile.md)、[CUDA Graph](../design/cuda_graphs.md)、[MRV2](../design/model_runner_v2.md)。

## P6. 从线性层推导 TP，再理解 PP/DP/EP

### Column Parallel

线性层 `Y=XW`，W 按输出列切分：

```text
W = [W0, W1]
Y0 = XW0
Y1 = XW1
Y = [Y0, Y1]
```

每卡得到输出分片。如果后续运算能直接消费分片，不必立即 gather。

### Row Parallel

将输入特征和 W 的对应行切分：

```text
X = [X0, X1]
Y = X0W0 + X1W1
```

各卡计算部分和，再通过合适的归约通信得到结果。真实实现可能使用 All-Reduce 或与其他切分配合的 collective。

这解释了为什么 TP 常在每层引入通信，而不是只在请求开始时交换一次数据。

### PP 与 DP

PP 按层或阶段放在不同设备上；需要传 hidden states 并平衡各阶段。微批或不同请求可以流水，但单请求依赖不会消失，阶段不均也会产生等待。

稠密 DP 复制完整模型副本、处理不同请求；可提高总吞吐和隔离性，单请求不会因此自动利用所有副本的算力。

### EP 与上下文并行

MoE 用 router 选择 experts，EP 把 experts 分散到设备，token 需要 dispatch/combine。稀疏激活降低计算，不代表没有通信和负载倾斜。

上下文并行沿 token/context 维度分摊 Attention 或 KV，长上下文下有价值，但需要对应的历史信息交换和归约。Prefill 与 Decode 支持模式不同，查看所选 backend。

### 两个选择题

**8B 模型单卡放得下，四张卡要提高服务吞吐怎么办？**  
先比较四个副本与 TP=4 等方案，在同资源和 SLO 下测量。副本避免层间 TP 通信，TP 可能降低单请求计算或释放每卡权重空间；不能只凭模型大小作最终结论。

**跨机网络慢时能否一味增加 TP？**  
不能。需比较通信增长、PP 阶段传输、模型容量和目标延迟。拓扑会改变最优组合。

本地资料：[TP/PP 扩展](../serving/parallelism_scaling.md)、[DP](../serving/data_parallel_deployment.md)、[EP](../serving/expert_parallel_deployment.md)、[上下文并行](../serving/context_parallel_deployment.md)。

## P7. PD 分离如何影响容量和延迟

共置引擎中，长 Prefill 与 Decode 分享计算和调度。Chunked Prefill 可以控制干扰；PD 则把两阶段放到不同资源组，允许分别选择并行和容量。

收益可能来自：

- 更稳定的 Decode 尾延迟。
- 对 P/D 分别使用适合的硬件或并行配置。
- 独立扩容与更明确的资源隔离。

新增成本包括：

- KV 交接与布局转换。
- P/D 两边的队列和路由协调。
- 源端 blocks 保留、目的端空间预留。
- 失败、取消与重算回退。

### 定量判断

假设 Prefill 完成产生 1 GiB KV，网络理想吞吐为 50 GB/s，仅数据传输下界就约 21.5 ms。若应用对 TTFT 预算很紧，这笔开销不能忽略。

逐层流水可能隐藏部分传输，但首层数据依赖和共享瓶颈仍在。总时间应分析关键路径，不是简单减掉一个“overlap 开关”的常数。

比较共置与 PD 时固定总资源，报告 P:D 比例、网络和缓存节点成本。某方案减少 ITL 尖峰却降低低负载吞吐，也可以是合理取舍，但必须说明目标。

详细的完成/回收流程见 [KV 专题第 12～15 节](04-kv-cache-vllm-lmcache-mooncake.md)。本地资料：[PD](../features/disagg_prefill.md)。

## P8. 量化与投机解码：两种不同的加速思路

### 量化

一个简化的对称量化关系：

```text
q = clip(round(x / scale), qmin, qmax)
x_hat = q × scale
```

实际方案还有 per-channel/group scale、非对称零点、混合精度、校准和不同计算 kernel。压缩数据会引入误差，反量化及元数据又会增加执行工作。

需要分别分析：

- 权重量化：主要改变权重容量、带宽和矩阵计算路径。
- 激活量化：影响数值范围和 kernel，通常更需关注输入分布。
- KV 量化：改变长上下文容量和读取量，误差可能影响后续注意力与生成。

存储从 16-bit 到 4-bit，不意味着服务吞吐必然提高 4 倍。

### 投机解码

draft 提出 K 个候选，目标模型一次验证多个位置，希望减少每个有效输出 token 的目标模型执行成本。

收益取决于：

```text
每个有效token成本
≈ (draft成本 + verify成本 + 协调成本) / 本轮实际接受与补充的token数
```

假设每层条件接受率近似为固定 α，且采用适用的完整验证/补充规则，理想期望输出数可以用 `1+α+...+α^K` 理解。这是教学近似，真实接受率有条件依赖，也受长度和策略限制。

精确 speculative sampling 需要接受/拒绝修正来保持目标分布；普通贪心验证、近似方法和不同 draft 结构不能混为一谈。高接受率也不保证加速：draft 可能过贵，或者大 batch 下目标验证原本已经高效。

本地资料：[量化](../features/quantization/README.md)、[KV 量化](../features/quantization/quantized_kvcache.md)、[投机解码](../features/speculative_decoding/README.md)。

## P9. 模型接入与高级功能怎样影响引擎

### 接入一个稠密模型需要检查什么

1. 配置如何描述层数、heads、位置编码和特殊结构。
2. checkpoint 参数名、形状和分片如何对应本地模块。
3. 是否使用正确的并行线性层和 Attention 接口。
4. forward 是否能接受引擎的 packed input/positions。
5. logits、采样或 pooling 的接口是否匹配任务。
6. 精度、量化、LoRA、TP/PP 与 backend 的兼容性。
7. 与可信参考的数值/任务对照、长上下文与批量边界。

只复制一个 PyTorch `forward()`，不等于已经接入生产推理系统。

### LoRA

LoRA 为特定权重增加低秩增量。多 adapter 服务要管理 adapter 身份、加载、批内混合和缓存命名空间。相同 token 使用不同 adapter，KV 不一定兼容；adapter 更新也应使旧身份失效。

### 多模态

图片/音频可能先经过 encoder，再与文本表示组合；带来预处理、encoder compute budget、特征缓存和不同位置编码。相同占位 token 不等于相同视觉内容，缓存身份不能只看文本占位符。

### 结构化输出

约束生成常需在采样时应用合法 token mask，并维护语法状态。异步执行时，下一步 grammar 可能依赖尚未处理的输出，影响调度和采样时序。

### Pooling 与生成

embedding/classification 等 pooling 任务通常不走长时间自回归 Decode。吞吐指标、输出处理和 KV 需求与生成任务不同，不能把全部模型都当成聊天生成模型。

本地资料：[模型接入](../contributing/model/basic.md)、[注册](../contributing/model/registration.md)、[多模态接入](../contributing/model/multimodal.md)、[LoRA](../features/lora.md)、[结构化输出](../features/structured_outputs.md)。

## P10. 服务工程：队列、路由、取消和成本

### Admission 与背压

引擎有有限 GPU/KV 容量，服务入口不能无限接受等待。队列上限、超时、优先级和拒绝策略使系统在过载时保留可预测行为。

重试也会产生新负载。缓存失效后大量请求同时重算，如果没有限制，可能导致第二次过载；应让上层知道失败类型并控制回退并发。

### 缓存感知路由

估计节点代价可以从以下项开始：

```text
排队 + 缓存取回 + 未命中Prefill + 必要通信
```

命中最多的节点可能队列最长。缓存位置索引也可能过期，需要失效、重试和回退；固定粘性路由可能提升命中也可能制造热点。

### 请求取消

逻辑取消要阻止后续调度和输出，物理回收要等待设备与传输不再使用 buffer。取消成功的服务响应不等于底层所有内存必须瞬间归零。

### 容量和成本

稳定系统中 Little 定律 `L=λW` 关联平均在途请求数、实际吞吐和平均停留时间；它不能直接保证 P99，也不适合忽略持续增长的队列。

单位成本可以按：

```text
满足SLO的成功请求成本 = 统计窗口内总资源成本 / 达标成功请求数
```

成本应计入 GPU、CPU/缓存节点、网络、重复计算和低负载浪费。比较分层缓存方案时，只看 GPU 节省可能漏掉重要支出。

### 故障分析模板

| 症状 | 先看什么 | 需要验证的假设 |
| --- | --- | --- |
| TTFT 高、ITL 稳定 | 入口/Prefill排队、缓存miss、输入处理 | 首输出前阶段是否主导？ |
| ITL 周期尖峰 | 长Prefill、collective、IO与CPU事件 | 是否与尖峰时间对齐？ |
| 高命中但更慢 | lookup/取回/H2D/热点排队 | 搬运是否贵于重算？ |
| OOM | 加载、capture、Prefill峰值或KV增长 | 发生在哪个阶段？ |
| GPU 低利用 | CPU/IPC、等待、限流、小kernel | 谁没有及时提供工作？ |
| 加卡反而变慢 | 通信、矩阵形状、拓扑、不均衡 | 通信增长是否抵消计算节省？ |

本章完成后，选择一个症状，提出假设、设计最小对照并写出可能推翻假设的证据。能做到这一点，比记住几十个调参开关更接近推理 Infra 的实际工作。
