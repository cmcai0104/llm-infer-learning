# vLLM 源码教材：跟踪一条请求及其资源

[返回导航](overview.md) · [基础教材](../../../01-foundations/attention-and-inference.md) · [性能教材](../../../06-serving/implementations/vllm/performance-and-parallelism.md)

本章已核对本地源码，阅读基准为提交 `f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`。先沿普通文本、单卡、稠密模型、常规采样阅读，再增加异步、外部 KV、多卡和特殊模型。源码链接指向上游的固定提交；路径相对于 vLLM 源码根目录，符号名用于定位，行号只作为该快照的辅助。

## R1. 先识别系统边界和版本

V1 指引擎架构；**Model Runner V2（MRV2）是 V1 引擎内部的另一套执行器实现，不是“vLLM V2 引擎”**。

当前快照同时存在：

| 层 | 源码 | 主要职责 |
| --- | --- | --- |
| 离线接口 | [LLM](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/entrypoints/llm.py) | 提交批量 prompt，收集完整结果 |
| 离线公用逻辑 | [OfflineInferenceMixin](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/entrypoints/common/offline.py) | 输入接入和驱动引擎循环 |
| 客户端引擎 | [LLMEngine](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/llm_engine.py)、[AsyncLLM](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/async_llm.py) | 接入、输出与异步请求管理 |
| 核心引擎 | [EngineCore](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/core.py) | 调度和执行的核心循环 |
| 调度 | [Scheduler](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/sched/scheduler.py) | 决定每个请求本轮计算量 |
| KV 管理 | [KVCacheManager](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/kv_cache_manager.py)、[BlockPool](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/block_pool.py) | 分配、共享、命中与回收 |
| GPU Worker | [Worker](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/gpu_worker.py) | 设备、执行后端与模型 runner |
| 原 runner 路径 | [gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/gpu_model_runner.py) | 原有 V1 执行路径 |
| MRV2 路径 | [gpu/model_runner.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/gpu/model_runner.py) | 更模块化的输入、状态和执行 |

`Worker._make_model_runner()` 在该快照约 570 行，按配置选择 runner。先看实际配置与日志再决定追哪条路径；不要仅因为文件名带 V1，就断定所有执行都走旧 runner。

配套阅读：[架构概览](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/arch_overview.md)、[MRV2 设计](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/model_runner_v2.md)。设计文档可能记录阶段性目标，实际支持情况仍由代码和选定配置决定。

## R2. 从 `LLM.generate()` 开始读

离线接口适合第一次学习，因为不需要先理解 HTTP 框架。

当前源码的主线：

```text
LLM.generate()
  → OfflineInferenceMixin._run_completion()
      → _add_completion_requests()
      → _run_engine()
          → while llm_engine.has_unfinished_requests()
              → LLMEngine.step()
              → 收集 finished 的 RequestOutput
```

对应入口：

- `LLM.generate`：`vllm/entrypoints/llm.py`，约 417 行。
- `_run_completion`：`vllm/entrypoints/common/offline.py`，约 340 行。
- `_run_engine`：同文件约 589 行。

这里的 `generate()` 会把多个 prompt 交给引擎自动组织批次；调用者给出的 Python list 不直接等于每次 GPU forward 的固定 batch。

阅读时记录三个事实：

1. 输入如何变成引擎请求，哪些参数在每请求层存储？
2. 一次 `step()` 为什么可能返回多个请求的进展？
3. 对外完整结果的排列顺序，与内部按什么顺序完成请求，是两个问题。

**练习**：为什么把 100 个 prompt 放进一次调用，与串行调用 100 次 `generate()` 可能有不同吞吐？  
**答案**：前者给调度器提供更大的可调度集合，能混合不同请求进度；后者可能只暴露单请求，无法形成同样的跨请求批处理。实际收益仍受显存和负载限制。

## R3. 输入、请求对象与进程间通信

### 输入处理

对照 [InputProcessor](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/input_processor.py) 和 [LLMEngine.add_request](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/llm_engine.py)。

概念上要完成：

```text
输入渲染/模板与tokenization
→ 参数和长度等校验
→ EngineCoreRequest
→ 核心引擎中的 Request
→ 调度队列
```

输入处理在 renderer、入口和 engine 间的具体分工随版本演进，不要把所有 tokenization 都归到一个固定函数。

### `Request` 是执行状态，不只是 API 参数

读 [request.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/request.py)：

| 字段/属性 | 应怎样理解 |
| --- | --- |
| prompt / output token IDs | 输入和已经产生的输出历史 |
| `num_computed_tokens` | 调度使用的计算进度；异步路径可能包含乐观推进 |
| `num_tokens_with_spec` | 包含投机候选的待追赶 token 目标 |
| `num_output_placeholders` | 异步情况下尚未拿到具体 token 的占位状态 |
| `num_in_flight_tokens` | 已调度但输出尚未处理的工作 |
| `block_hashes` | 已构建的缓存身份 |
| `status` | 等待、运行、抢占或完成等状态 |

**重要区别**：调度器视角的计数、GPU 真正完成的工作、客户端已收到的输出，不一定同时更新。当前代码明确记录 in-flight 和 stale output 等状态，不能用一个变量覆盖全部语义。

### 通信边界

在线引擎通常用 client/core 通信，见 [core_client.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/core_client.py)。同一个对象名出现在两边，不代表它们共享同一 Python 对象或地址空间。IPC 需要明确序列化、消息顺序、请求身份和失败清理。

## R4. EngineCore 怎样驱动模型

先读 [EngineCore.step](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/core.py) 约 657 行。简化后的职责如下：

```text
检查是否有请求
→ scheduler.schedule(...)
→ executor.execute_model(..., non_block=True)
→ 准备适用的 grammar 信息
→ 获取执行结果，必要时独立 sample_tokens(...)
→ 处理执行期间发生的 abort
→ scheduler.update_from_output(...)
→ 产生 EngineCoreOutputs
```

这段流程有两个值得注意的设计：

1. 模型执行与采样不总是单个同步调用；某些路径的执行结果为空，随后单独采样。
2. 执行过程中也可能发生请求取消，更新输出之前需要处理它，避免把取消请求当作仍活跃请求推进。

### batch queue 与异步路径

随后读同文件 `step_with_batch_queue()`，约 698 行：

- 有空间且有工作时，优先继续调度/提交。
- 不能继续提交时，再等待较早的执行结果。
- 根据结果更新调度状态。

目的在于让 CPU 准备与 GPU 计算重叠，但必须处理在途状态和正确的依赖。不要用只读同步 `step()` 得到的顺序，断言所有配置都严格“提交 → 完整等待 → 再调度”。

**练习**：一个 batch forward 已提交，客户端此时取消请求，为什么不能立即覆盖它的 KV？  
**答案**：设备可能仍在读写旧 buffer。逻辑取消应阻止后续推进，物理回收必须覆盖到真实访问结束。

## R5. Scheduler：从请求集合到每轮 token 数

读 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/sched/scheduler.py) 约 675 行，以及 [SchedulerOutput](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/sched/output.py)。

核心抽象是：

```text
让每请求的已计算进度追赶本轮允许处理的 token 目标
同时满足：token预算、请求数、KV容量、状态、优先级和特殊约束
```

当前 `schedule()` 同时读取 `max_num_scheduled_tokens` 和 `max_num_batched_tokens` 等预算。旧文章只讲一个 token budget，学习时可以用它理解原理，但做配置解释应返回当前实现。

### 一个可手推的例子

假设预算 8，A/B 各需一个 Decode token，C 还有 10 个 prompt token：

```text
轮1：A=1，B=1，C=6
轮2：A=1，B=1，C=4
```

这是教学策略，假设资源充足且优先服务两个 Decode 请求；实际源码有优先级、prefill 节流、公平性、外部 KV、投机等分支，不保证所有配置都产生相同顺序。

### 输出不只是一个 batch list

`SchedulerOutput` 包含：

- 首次被调度的新请求信息。
- 之前已调度请求的增量信息。
- `num_scheduled_tokens` 和总 token 数。
- 结束或抢占的 request IDs。
- 投机、encoder、结构化生成和 connector 元数据。

“新请求全量 + 旧请求增量”的设计降低每轮重复传输和重建的成本。它要求 worker 的持久状态与调度器保持一致，取消/抢占恢复不能漏通知。

### 阅读分支的顺序

先跟踪一条运行请求；再看新请求 admission；然后看 `allocate_slots` 失败时怎样处理；最后看远端 KV 和投机。不要从第一行起试图一次记住所有 if 分支。

在相同文件中继续定位 `update_from_output()`，约 2180 行：它把实际执行和采样结果带回调度状态，连接本轮和下一轮。

## R6. KV Manager 与 BlockPool：三个不同问题

先分清：

1. **可复用多少**：前缀 hash 和模型语义决定。
2. **内存够不够**：物理 blocks、引用、预留与组布局决定。
3. **数据是否 ready**：本地完成或外部传输状态决定。

### `get_computed_blocks`

入口：[kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/kv_cache_manager.py) 约 264 行。

当前快照返回三个值：缓存 blocks、命中 token 数、混合/稀疏状态相关的共享边界信息。旧文章写成两个返回值时，不能直接照抄成当前代码。

源码明确把普通全命中的查找上界限制到 `request.num_tokens - 1`，以便重新处理尾部获得 logits。受完整块对齐影响，有时重算的不止一个 token。

例子：块大小 16、prompt 长度 32、只有完整块复用且遵循此上界，则最大可用的对齐命中可能为 16，而不是 32。prompt 长度 33 时，32-token 命中后处理最后一 token 更容易实现。特殊缓存组/后端需按实际规则分析。

### `allocate_slots`

入口同文件约 371 行。除了新 token，还区分：

- 已有计算进度。
- 新发现的本地缓存命中。
- 外部可复用 token。
- 投机 lookahead。
- 延迟发布缓存、全序列可容纳 admission、保留给其他在途请求的 blocks。

这说明“为请求分配多少内存”不只是 `ceil(new_tokens/block_size)`。已有块、共享引用、模型缓存组和在途工作都影响结果。

### `touch` / `free_blocks`

入口：[block_pool.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/core/block_pool.py)。

- `touch()`：命中块重新被使用时增加引用，必要时从 free queue 移除。
- `free_blocks()`：减少引用，归零后根据是否可缓存等条件归回可重用结构。
- `_maybe_evict_cached_block()`：物理块准备改作他用时移除旧缓存身份。

“在 free queue”“没有请求引用”“没有有效缓存内容”不是同义词。缓存内容可保留到真正驱逐时；设备 pool 也可能始终保留物理内存。

**练习**：共享块 ref=2，A 完成后 ref=1，B 完成后 ref=0，此时一定清零 KV 吗？  
**答案**：不一定，可作为空闲有效缓存继续保留；实际驱逐或复用时再撤销旧身份。异步传输还可能延长安全生命周期。

## R7. ModelRunner 怎样把请求变成 Tensor

调度结果是请求级描述，GPU 需要紧凑的 Tensor 与注意力元数据。

假设 A 已有 5 个 KV token，本轮计算 2 个位置；B 已有 9 个 KV token，本轮 Decode 1 个：

```text
num_scheduled_tokens = [2, 1]
packed input_ids = [A的新token0, A的新token1, B的新token0]
positions = [5, 6, 9]
query_start_loc = [0, 2, 3]
本轮后可访问的序列长度 = [7, 10]
```

`query_start_loc` 是打包 query 的累计边界，不是历史 KV 的完整长度。Attention 还要知道 block table、序列长度和本轮 token 的写入位置。

### slot mapping

在一个简单全注意力组中，block size=C：

```text
logical_block = position // C
offset = position % C
physical_block = block_table[request][logical_block]
slot = physical_block × C + offset
```

例如 C=4，A 的 block table 为 `[7,2]`，position=5，则落到物理 block 2 的 offset 1，slot=9。这个 slot 是概念上的 token 槽索引，具体 K/V 各维物理排布还由 backend 决定。

阅读：[worker/block_table.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/block_table.py)、[MRV2 input_batch.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/gpu/input_batch.py)。

### 为什么要 persistent state

连续两轮大多数请求仍相同。每轮重建完整 block table 和采样参数会增加 CPU 成本，因此 worker 保留状态并应用变化。

MRV2 的设计进一步区分“请求存储位置”和“本轮输入顺序”：为活跃请求分配状态行，再按本轮需要 gather。相比反复重排全部持久状态，这可以减少耦合与 CPU 工作。

## R8. 模型 forward、Attention backend 与 MRV2

先看 [LlamaAttention.forward](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/model_executor/models/llama.py) 约 224 行：

```text
qkv_proj(hidden_states)
→ 按 q_size / kv_size 切出 Q/K/V
→ rotary_emb(positions, Q, K)
→ Attention(Q,K,V)
→ o_proj
```

再看同文件 `LlamaDecoderLayer.forward`，约 313 行：归一化和残差、Attention、归一化和残差、MLP。这里可以把 F2 的数学对应到真实模型层。

Attention 调用没有显式传完整历史 K/V，不表示没有历史缓存；执行上下文、backend 和 KV 张量组织负责连接它们。继续阅读 [Attention backend 设计](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/design/attention_backends.md)。

### backend 不只是一个 kernel 名字

选择 backend 时可能涉及 GPU 架构、dtype、head dimension、KV layout、模型类型、特性兼容等。换 backend 既可能改变性能，也可能改变数值路径和支持能力。

### MRV2 的本轮执行

入口：[gpu/model_runner.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/worker/gpu/model_runner.py) 的 `execute_model()`，约 1742 行。

当前代码先处理请求结束、状态释放、新请求加入和更新，再应用 block table 的 staged writes；如果没有实际 forward 工作，可走 connector 等无 forward 路径。

然后准备 batch 描述、输入、执行模式，运行模型并产生后续采样需要的状态。它并非一个单纯的 `model(input_ids)` 包装。

MRV2 的几个阅读重点：

- 请求状态与本轮输入分离。
- 将部分 metadata 准备放到 GPU。
- 用增量写入维护大 Tensor。
- 尽量避免主循环里的 CPU/GPU 同步。
- 把采样和功能拆入独立模块。

**注意**：CPU 写 pinned buffer 的同时，GPU 异步读取同一块内存，会产生竞争。异步正确性要求稳定源副本或同步/所有权方案，不能只加 `non_blocking=True`。

## R9. 采样、输出处理和停止

模型输出 hidden states/logits 后，需要根据策略选择 token。原 sampler 可从 [v1/sample/sampler.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/sample/sampler.py) 阅读；MRV2 有自己的模块化路径，应沿实际 runner 的 `sample_tokens()` 继续追。

要区分：

1. 模型 forward 生成什么计算结果。
2. 哪些位置需要 logits；中间 Prefill chunk 不一定需要生成输出。
3. 采样怎样处理 penalties、约束与随机性。
4. 哪些 token 交回引擎，以及怎样更新已计算/在途状态。

### 从 token IDs 到用户结果

[OutputProcessor](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/output_processor.py) 管理请求输出状态，配合 [detokenizer.py](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/v1/engine/detokenizer.py) 转成文本。

`LLMEngine.step()` 在约 319 行先获取核心结果，再调用输出处理；stop string 等条件还可能触发对核心请求的 abort。

停止可以来自 EOS、生成长度、停止串、取消或错误。流式返回文本不能随意泄漏未确认的停止串片段；输出层与核心状态需一致。

### 不要把时钟混在一起

GPU step 完成时间、输出处理时间、服务端写出时间、客户端收到时间不同。优化 GPU kernel 后客户端 TTFT 不变，可能是排队或输入处理主导，而不是测试一定有错。

## R10. 外部 KV Connector：语义与同步的接口

入口：[KVConnectorBase_V1](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/vllm/distributed/kv_transfer/kv_connector/v1/base.py)。

| 调度侧 | 要解决的问题 |
| --- | --- |
| `get_num_new_matched_tokens()` | 本地已计算范围之外，可从外部复用多少？查询是否还未完成？ |
| `update_state_after_alloc()` | 目标 blocks 已分配，如何记录传输状态？ |
| `build_connector_meta()` | 给 worker 哪些执行元数据？ |
| `request_finished()` 等完成接口 | 是否需要延迟释放请求 blocks？ |

| 执行侧 | 要解决的问题 |
| --- | --- |
| `start_load_kv()` | 何时发起加载？ |
| `wait_for_layer_load()` | 本层计算前数据是否 ready？ |
| `save_kv_layer()` | 如何保存已经产生的本层 KV？ |
| `wait_for_save()` / `get_finished()` | 哪些操作真正完成、哪些引用可释放？ |

在当前基类约 516 行，`get_num_new_matched_tokens()` 的第一个返回值可为 `None`，表示还不能确定命中数，应之后再查询；另一个值表达是否跨调度步异步加载。它不是单纯的同步 `get(key)`。

学习链路：

```text
兼容身份查询
→ 命中范围
→ 预留目标内存
→ 搬运
→ 完成/失败
→ 更新可计算范围
→ 后续forward
→ 保存与回收
```

参考 [04 专题](../../../03-kv-cache/implementations/vllm/integration.md)理解 LMCache、Mooncake 如何接到这一边界。不要因为某个后端能读写 bytes，就假设它自动理解 token 前缀或所有模型布局。

## 最后一次源码作业

按顺序在 IDE 标出以下 10 个符号，并给每个符号写一句输入、输出与所有权说明：

```text
LLM.generate
OfflineInferenceMixin._run_completion
LLMEngine.add_request
EngineCore.step / step_with_batch_queue
Scheduler.schedule
KVCacheManager.get_computed_blocks / allocate_slots
BlockPool.touch / free_blocks
Worker._make_model_runner
GPUModelRunner.execute_model / sample_tokens
OutputProcessor.process_outputs
```

再追一条失败路径：外部命中后加载失败，或者请求执行期间取消。能讲清失败后哪些状态回退、哪些内存暂时不能释放，才算读通主线。

可继续阅读的真实测试入口：[scheduler tests](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/tests/v1/core/test_scheduler.py)、[async scheduler tests](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/tests/v1/core/test_async_scheduler.py)。阅读测试关注边界条件；本手册未在当前 macOS 环境运行 GPU 测试。
