# 实践教材：运行步骤、源码作业与参考答案

[返回导航](../../../02-runtime/implementations/vllm/overview.md) · [28 天计划](../../../08-career/implementations/vllm/study-plan.md) · [实验设计](experiments.md)

本章把“该做什么”展开成可操作的步骤。CPU 教学实验不依赖 vLLM、PyTorch 或 GPU；GPU 命令供兼容环境执行，不能当作本机已完成的 benchmark。

## L1. 准备环境与自测

### 阅读环境

源码阅读与 GPU 安装命令在 vLLM 上游源码目录执行；CPU 教学脚本在本资料库 `Infra` 根目录执行。先在 vLLM 源码目录确认版本：

```bash
git rev-parse --show-toplevel
git rev-parse HEAD
```

本教材源码快照为 `f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`。以后切换版本时，重新搜索符号，不依赖行号。

vLLM 上游要求使用 uv 和虚拟环境。已有 `.venv` 时直接使用，不需要重建；缺少时按[官方 Python 环境说明](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/getting_started/installation/python_env_setup.inc.md)准备：

```bash
uv venv --python 3.12
.venv/bin/python 07-practice/cpu/inference_lab.py all
```

上面两条在 `Infra` 根目录执行。脚本只有标准库依赖。若已经有虚拟环境，只运行第二条即可。

### 五分钟自测

| 问题 | 基础答案 |
| --- | --- |
| Q 和 K 的长度一定相同吗？ | Decode 中 query 通常只有一个位置，历史 KV 可以很长 |
| 生成出的最后一个 token 一定已经有 KV 吗？ | 通常还要下一次 forward 才建立它的 KV |
| `ref_cnt=0` 必须立刻清零内存吗？ | 不必须，完整兼容缓存可以保留到驱逐 |
| 提交 CUDA/RDMA 操作成功表示完成了吗？ | 不表示，需要完成与可用性语义 |
| 命中更多缓存一定更快吗？ | 不一定，取回、排队和转换可能比重算更贵 |

不熟悉这些问题时，先完成 07 的 F1～F5，不必暂停整个月计划。

## L2. 四个可直接运行的 CPU 实验

脚本：[inference_lab.py](../../cpu/inference_lab.py)。

```bash
.venv/bin/python 07-practice/cpu/inference_lab.py attention
.venv/bin/python 07-practice/cpu/inference_lab.py prefix
.venv/bin/python 07-practice/cpu/inference_lab.py schedule
.venv/bin/python 07-practice/cpu/inference_lab.py capacity
```

### 实验 A：KV 与重算的等价关系

脚本对四个固定二维输入运行一层 Attention：

- 参考路径每轮重新投影全部历史 K/V。
- 缓存路径只追加当前位置的 K/V。
- 每个位置比较得到的输出。

预期每个位置 `cached_matches_full=True`。每个 K 或 V 投影矩阵处理的行数分别是 10 与 4。

**为什么少了 6 行？** 重算路径处理 `1+2+3+4` 个历史位置，缓存路径只处理 4 个新位置。

**这个实验没有证明什么？** 它没有模型生成、RoPE、多层 hidden states、GPU kernel、分页或性能计时，不能据此推断 vLLM 加速倍数。

### 实验 B：链式前缀 hash

输入 A 是 `[1,2,3,4,5,6]`，B 是 `[1,2,9,4,5,6]`，每块 2 token。

预期块匹配结果为：

```text
[True, False, False]
```

第三块 token 都是 `[5,6]`，为什么仍不匹配？因为它们前面的上下文不同，链式父 hash 不同。

更换模型 namespace 后，所有块都失配。教学 key 不覆盖实际系统所有配置，也不实现权限系统。

### 实验 C：按轮 token 调度

初始 A/B 已处于 Decode，C 需 10 个 Prefill token；D 在第二轮到达，需 2 个 Prefill token。预算为 8，简化策略先为当前 Decode 各分一个 token，再填 Prefill：

```text
轮1：A decode 1，B decode 1，C prefill 6
轮2：A decode 1，B decode 1，C prefill 4，D prefill 2
轮3：A decode 1，C decode 1，D decode 1
轮4：C decode 1
```

`decode_steps_left` 表示后续还需执行几次 Decode forward，不是从 prompt 开始的总输出数；末次 Prefill 产生首输出的概念已经分开。

本实验没有 KV 容量、抢占、优先级、外部加载和异步状态，是理解 budget 的模型，不是 vLLM 调度器副本。

**修改练习**：把预算从 8 改成 2，预测 C/D 的 Prefill 会怎样等待，再运行核对。解释为什么高优先级 Decode 持续到来时，需要考虑公平性。

### 实验 D：容量与传输下界

预期关键结果：

```text
每 token KV：128 KiB
8192 tokens KV：1 GiB
17/31 token，block=16：4 blocks，64 reserved slots
1 GiB / 400 Gbit/s：约 21.47 ms
```

增加每元素字节数、KV heads 或层数，观察线性关系。改变上下文和并发后，再加入共享前缀假设，分别计算逻辑与物理占用。

## L3. 六道计算与状态推演题

### 题 1：MHA 与 GQA

32 层、head dimension=128、BF16，Hkv 从 32 减到 8，8192-token 序列 KV 怎样变化？

**答案**：从 4 GiB 降为 1 GiB，忽略量化、共享、分片和其他开销。不能据此说整个推理显存下降 4 倍，因为权重等还在。

### 题 2：block mapping

block size=4，请求的物理 block table 为 `[7,2,9]`。逻辑位置 6 和 8 分别写到哪个 token slot？

**答案**：位置 6 → block table[1]=2，offset=2，slot=10；位置 8 → block table[2]=9，offset=0，slot=36。

### 题 3：packed query

三个请求本轮分别计算 3、1、2 个 token，`query_start_loc` 是什么？

**答案**：`[0,3,4,6]`。这是本轮 query 的累计边界，不是各请求全部历史 KV 长度。

### 题 4：全命中仍重算

普通完整块复用，block=16、prompt=32，当前 `get_computed_blocks` 用 `num_tokens-1` 限制命中，为什么可能只能复用 16？

**答案**：上界 31 向完整块边界对齐为 16，剩余部分重算以获得 logits；特殊缓存组和 backend 按各自实现解释。

### 题 5：延迟与 Goodput

十秒内成功 100 个请求，其中只有 75 个同时满足 TTFT 和 TPOT 约束。吞吐与 Goodput 各多少？

**答案**：成功请求吞吐 10 req/s，Goodput 7.5 req/s。还需报告总提交量、失败和拒绝。

### 题 6：取回与重算

重算可复用前缀需 60 ms；lookup=3 ms、数据路径=20 ms、转换/等待=7 ms，有 8 ms 与必要计算确实重叠。是否有时间收益？

**答案**：简化非重叠成本 `3+20+7-8=22 ms`，比重算少 38 ms。但这不是承诺 TTFT 必降 38 ms；整体排队、其他计算与保存成本可能变化。

## L4. 在 Linux GPU 环境完成最小 vLLM 使用

先按硬件查看[安装指南](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/getting_started/installation/README.md)。下面两种安装路线选择一种：

| 目的 | 路线 |
| --- | --- |
| 先体验稳定版本 | 安装选定 release，保存版本；其源码与本文快照可能不同 |
| 阅读当前 checkout 并实验 | 按仓库贡献指南准备匹配当前源码的环境和预编译产物 |

稳定版本示例，先设置实际存在且与硬件兼容的版本：

```bash
uv pip install "vllm==${STUDY_VLLM_VERSION:?请先设置具体版本}" --torch-backend=auto
```

当前源码的 Python 开发路线可参考：

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

是否存在匹配产物、还需哪些 kernels/平台依赖，以[当前贡献指南](https://github.com/vllm-project/vllm/blob/f3871c3cbf8a608f7ee833ec4c2c5277f77bee41/docs/contributing/README.md)为准。这些安装命令没有在本机执行，不建议为了阅读文档在 macOS 上安装 CUDA 软件栈。

### 最小离线生成

设置一个实际可访问、受支持且显存能容纳的模型目录或仓库 ID：

```bash
export STUDY_MODEL="/absolute/path/to/your/model"
.venv/bin/python - <<'PY'
import os
from vllm import LLM, SamplingParams

llm = LLM(
    model=os.environ["STUDY_MODEL"],
    max_model_len=2048,
    gpu_memory_utilization=0.8,
    seed=0,
)
results = llm.generate(
    ["The purpose of a KV cache is", "Continuous batching means"],
    SamplingParams(temperature=0, max_tokens=64),
)
for result in results:
    print(result.request_id, result.outputs[0].text)
PY
```

这是原始文本 completion 的学习例子。对 instruction/chat 模型，要理解其模板，并使用匹配的消息渲染方式；不要把未应用模板的质量当作模型最终能力。

验收：能加载、两条请求都完成、记录实际输出长度和配置。`temperature=0` 有利于排除采样随机性，但不保证跨所有 batch/backend 的结果逐 token 完全相同。

## L5. 服务与一组可复现的压测

选择支持至少相应上下文长度的小模型；以下参数是学习起点，不是通用最优配置。

终端 A：

```bash
.venv/bin/vllm serve "$STUDY_MODEL" \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8
```

终端 B 使用相同模型标识，先用 CLI help 核对安装版本：

```bash
.venv/bin/vllm bench serve --help
mkdir -p study-results
.venv/bin/vllm bench serve \
    --model "$STUDY_MODEL" \
    --host 127.0.0.1 --port 8000 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --random-range-ratio 0 \
    --random-prefix-len 0 \
    --num-prompts 200 \
    --request-rate 2 \
    --ignore-eos \
    --save-result --result-dir study-results
```

这些参数已在当前 CLI 源码中核对，但未做 GPU 执行验证。200 个请求用于起步检查；严肃分析 P99 需更充分的样本、时间窗口和重复实验。

`--ignore-eos` 用于控制 benchmark 输出长度，会改变停止行为；真实业务实验应另做保留自然停止条件的一组。

随后仅改变：

- 长输入：input=4096、output=128。
- 长输出：input=128、output=1024。
- 再逐步改变到达率，确认实际发送速率和排队。

不要在这一步同时开启所有优化。选择一个假设，例如“长 Prefill 干扰 Decode”，再按 E3 改 token budget。

**无 GPU 替代作业**：用 L3 的计算案例和 `Scheduler.schedule` 阅读结果，写出预期曲线及可能推翻它的因素。把它标注为分析报告，不标注成 benchmark。

## L6. KV 专项的最小对照和源码作业

### 本地前缀复用

构造 token 级共同前缀，而不是只凭肉眼认为文本相同。做冷缓存、热缓存、前缀早期变动、非前缀文档四组，观察实际命中与计算 token。

### LMCache / Mooncake

先只选一种集成路径：

```text
vLLM → LMCache → CPU/L2
或 vLLM → Mooncake Store
或 vLLM P → Mooncake P2P → vLLM D
```

具体配置来自[版本对应资料](../../../08-career/implementations/vllm/references.md)。本教材不提供混用不同 release 的“万能 JSON”。

必须能回答：

1. key 由谁构建，如何隔离模型/adapter？
2. 查到的是本地还是外部前缀？
3. 目标 GPU blocks 由谁分配？
4. 哪个事件表示数据真正可用？
5. 保存和加载中的 buffers 由谁持有？
6. 超时、取消、驱逐后，谁撤销无效命中并决定重算？

### 无分布式硬件时的源码作业

从 `get_num_new_matched_tokens` 追到 `update_state_after_alloc`、worker load 和完成状态，画出四种情况：miss、同步 hit、异步 hit、load 失败。

**参考答案结构**：miss→正常计算；hit→预留+加载+完成后推进；异步 hit→暂缓相关依赖并允许其他工作；失败→撤销无效复用、清理使用权，再按支持路径重算或终止。不要画成“查到 key 就立即执行 Decode”。

## L7. 一道完整系统设计练习

题设：同一模型服务短对话和长文档；希望分别约束 TTFT 与生成间隔。硬件、流量尚未给定。

合格回答先补齐问题：

- 输入/输出 token 分布、峰值和突发度。
- 模型、精度、允许的质量变化和总资源预算。
- TTFT、TPOT/ITL 的具体目标以及失败/拒绝容忍度。
- 是否存在重复前缀，是否允许跨租户共享。
- GPU/NIC 拓扑、可用主存、存储与网络。

再给方案：

1. 估算权重/KV，先建立共置 + Chunked Prefill 基线。
2. 根据实际干扰决定是否分池或 PD。
3. 以预计排队+计算+缓存取回进行路由，而非只看命中。
4. 明确 admission、取消、超时、缓存失败回退和扩缩容冷启动。
5. 记录阶段指标，使用混合负载和故障实验验证 Goodput。

**不合格信号**：一上来指定某 GPU 数量、PD 比例或命中收益，却没有题设和计算依据。

## L8. 月末作业与评分标准

提交一份 3～5 页报告：

```text
1. 模型、硬件/假设、负载和 SLO
2. 架构与请求/KV生命周期
3. 基线或明确标注的手算
4. 一个瓶颈假设和证据
5. 对照结果与正确性/质量
6. 失败路径、代价与无收益区间
7. 当前源码中对应的函数
```

| 项目 | 分值 | 判断标准 |
| --- | --- | --- |
| 原理与计算 | 25 | token、KV、单位、并行假设正确 |
| 源码链路 | 25 | 能从入口追到调度、执行、输出和回收 |
| 性能分析 | 20 | 口径一致，有可证伪假设和对照 |
| 缓存正确性 | 20 | 命中/加载/完成/回收区分明确 |
| 表达与边界 | 10 | 实测与推演分开，不夸大收益 |

没有 GPU 也可以完成原理、源码与设计部分；“性能分析”使用清楚的模型和待验证实验设计评分，不能声称获得实测优化收益。
