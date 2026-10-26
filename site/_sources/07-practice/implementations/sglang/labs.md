# 实验手册：从单机基线到共享缓存与 PD

[返回导航](../../../02-runtime/implementations/sglang/overview.md) · [理论与选型](../../../03-kv-cache/implementations/sglang/tiered-cache.md) · [报告模板](../../../08-career/implementations/sglang/assessment.md)

## 1. 实验边界与准备

本文提供七组实验。命令按本手册记录的源码和官方示例静态核对，**本次资料编写未运行 GPU benchmark，不提供虚构实测结果**。

### 1.1 平台与版本

- GPU 实验：Linux + 支持所选软件版本的 NVIDIA GPU。
- 当前仓库 NVIDIA 安装文档使用 CUDA 13 环境；旧 release 的依赖可能不同。
- 固定 SGLang、PyTorch、Attention backend、LMCache、Mooncake 和 Router 版本。
- 当前 main 里的集成可能尚未包含在你安装的稳定 wheel 中。
- LMCache 必须与 SGLang 的统一 MP connector 匹配。以 [官方 Quickstart](https://docs.lmcache.ai/getting_started/quickstart.html) 的 SGLang 页签核对双方版本要求。
- 小模型选 `Qwen/Qwen2.5-1.5B-Instruct`，使用公开权重，不需要在文档中保存 token。
- 所有启动命令分别在前台终端运行；看到服务就绪后再发送请求。

安装依据：[上游安装文档](https://github.com/sgl-project/sglang/blob/30dc2f94303e339431cbd51489b2d1719678b0c1/docs/docs/get-started/install.mdx)。

不要为了执行示例无条件升级到浮动 `latest`。选择兼容的固定版本或镜像，并记录模型 revision。

### 1.2 环境记录

至少保存：

```text
日期：
SGLang commit / wheel：
PyTorch / CUDA runtime / driver：
LMCache / Mooncake / Router：
模型 ID / revision：
GPU 型号、数量、显存：
CPU / RAM / NUMA：
网络 / RDMA / NIC：
TP / PP / DP / EP / CP：
完整启动命令：
```

实验中的服务仅用于隔离环境。SGLang 和 Router 绑定回环地址；Mooncake Master、元数据与 Store 的监听范围需要检查并通过网络策略保护。

### 1.3 运行位置

启动命令依赖已安装的 Python 模块，可在实验工作目录执行。示例中的 `lmcache_config.yaml` 与 benchmark 输出使用当前工作目录；如果放到其他目录，请改为实际路径。

下文每次变更服务配置，都应先正常停止之前对应终端的服务，避免端口占用。不要清理生产缓存，也不要使用共享生产服务做容量压力测试。

## 2. 对照实验规则

1. 一次只改变一个主要变量。
2. 分清冷缓存、热缓存和模型/JIT/Graph 预热。
3. 固定模型、精度、并行、长度分布和资源预算。
4. 每组重复运行，保留逐请求数据与异常。
5. 缓存实验使用受控共享前缀，随机输入主要用于压力基线。
6. 输出长度若受 EOS 影响，记录实际长度；为了固定长度改变生成规则时也要记录。
7. 不仅测 TTFT，还测 ITL、吞吐、Goodput、资源与质量。
8. 比较 PD 时保持总 GPU 预算一致。

## 3. 实验一：基线、API 与前缀命中

### 3.1 启动单 GPU 服务

TP=1，模型路径可替换为兼容模型 ID 或本地模型目录。内存参数需要结合实际 GPU 调整：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.7 \
  --enable-metrics
```

`max-total-tokens` 限制可用 token 容量；`mem-fraction-static` 不能视为“KV 单独占整张卡的比例”。启动日志中的实际容量是重要依据。

### 3.2 普通与流式 API

普通生成：

```bash
curl http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain what a KV cache stores.","sampling_params":{"temperature":0,"max_new_tokens":64},"stream":false}'
```

流式生成：

```bash
curl -N http://127.0.0.1:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Explain what a KV cache stores.","sampling_params":{"temperature":0,"max_new_tokens":64},"stream":true}'
```

OpenAI-compatible Chat：

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"Explain KV cache briefly."}],"temperature":0,"max_tokens":64,"stream":false}'
```

原生生成输入和 Chat 模板处理后的 token 序列未必相同，不要跨两种 API 直接推断命中结果。

### 3.3 共享前缀探测

以下 Python 代码仅使用标准库：

```python
import json
import time
import urllib.request

url = "http://127.0.0.1:30000/generate"
prefix = (
    "SGLang separates scheduling, model execution, "
    "and KV cache management.\n"
) * 256

for trial in range(3):
    payload = {
        "text": prefix + f"\nQuestion {trial}: What does the KV cache store?",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    meta = result.get("meta_info", {})
    print({
        "trial": trial,
        "e2e_seconds": round(time.perf_counter() - start, 4),
        "prompt_tokens": meta.get("prompt_tokens"),
        "cached_tokens": meta.get("cached_tokens"),
        "cached_tokens_details": meta.get("cached_tokens_details"),
    })
```

这里计时是 E2E，不是 TTFT。字段可能随版本变化，使用服务端日志和指标交叉验证。第一条请求是否为冷缓存取决于服务已有状态，不能只按 trial 编号推断。

### 3.4 压力基准

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 160 \
  --max-concurrency 16 \
  --output-file baseline.jsonl
```

把并发改为 1、4、8、16，保存不同结果文件。请求数至少应足以形成稳态，当前官方指南建议不低于最大并发的 5 倍。

然后用有限 `--request-rate` 扫描到达率。若还保留并发上限，记录客户端反压造成的实际 offered rate 变化。

验收：

- 确认服务、模型和指标可用。
- 证明共享前缀确实被复用，而不是仅观察总耗时下降。
- 区分 JIT/Graph 预热与 KV 复用。
- 保存并发与延迟/吞吐曲线。

## 4. 实验二：调度与执行优化

分别在基线服务上改变：

| 变量 | 示例 | 观测目标 |
|---|---|---|
| Prefill chunk | `--chunked-prefill-size 512`、2048、4096 | 长输入对活跃 Decode 的干扰 |
| CPU/GPU overlap | `--disable-overlap-schedule` | GPU 空隙与 CPU 工作 |
| CUDA Graph | `--disable-cuda-graph` | Kernel launch 开销 |
| 前缀复用 | `--disable-radix-cache` | 重算量与缓存收益 |

上述是分别进行的对照实验，不是一次性追加所有参数。使用普通基线，不与不兼容的 LMCache 等配置混合。

至少准备两种负载：

1. 同分布短请求。
2. 少量长输入混入正在 Decode 的请求。

记录 p50/p95/p99 TTFT、ITL、吞吐、内存与 trace。小 chunk 若改善 ITL 却降低吞吐，应依据 SLO 评价，不简单判定优劣。

## 5. 实验三：HiCache GPU 与 Host 命中

### 5.1 启动

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.7 \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --page-size 64 \
  --hicache-write-policy write_through
```

实际 Host 内存需求要以所选模型、设备 KV 池、并行和日志为准，不要只看 ratio 数值。

### 5.2 步骤

1. 发出目标前缀，产生 KV。
2. 立即重复，验证 GPU 命中。
3. 等待需要的备份完成。
4. 发送其他不同前缀，施加足以挤出目标 GPU KV 的压力。
5. 保证目标仍有 Host 副本，重新请求它。
6. 用分层指标验证 Host 命中和加载，而不是只比较总时间。

不能通过重启 SGLang 保留原来的 HiCache L2。若压力太大连 Host 也被淘汰，应降低压力并重新构造。

比较 `write_through`、`write_through_selective`、`write_back` 时，分别重新建立初始状态，并记录写入量与命中来源。

## 6. 实验四：Mooncake Store 外部复用

### 6.1 前置条件

- 安装与当前 SGLang 兼容的 Mooncake。
- 为便于功能验证，示例使用 TCP。
- TCP 实验不能代表 RDMA 的性能。
- 使用独立 Store 持有数据，避免所有副本随推理进程退出。

### 6.2 终端 1：Master

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_port=8080 \
  --eviction_high_watermark_ratio=0.95
```

### 6.3 终端 2：独立 Store

```bash
MOONCAKE_LOCAL_HOSTNAME="localhost" \
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="tcp" \
MOONCAKE_GLOBAL_SEGMENT_SIZE="4gb" \
MOONCAKE_LOCAL_BUFFER_SIZE=0 \
python -m mooncake.mooncake_store_service --port=8081
```

### 6.4 终端 3：SGLang

```bash
MOONCAKE_LOCAL_HOSTNAME="localhost" \
MOONCAKE_TE_META_DATA_SERVER="http://127.0.0.1:8080/metadata" \
MOONCAKE_MASTER="127.0.0.1:50051" \
MOONCAKE_PROTOCOL="tcp" \
MOONCAKE_GLOBAL_SEGMENT_SIZE=0 \
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.7 \
  --enable-metrics \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --page-size 64 \
  --hicache-mem-layout page_first_direct \
  --hicache-io-backend direct \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake
```

### 6.5 验证

1. 发送共享前缀请求。
2. 通过后端日志/指标确认写入完成，而不是只等 HTTP 返回。
3. 仅正常停止并重启 SGLang，保留 Master 和 Store。
4. 使用相同模型、格式和命名空间重复请求。
5. 验证外部命中、取回字节、实际少算 token 和 TTFT。

跨机器时把回环地址替换成各参与进程可访问的地址，配置网络隔离；不能直接把本地命令当作已完成多节点部署。

如果取回比重算慢，记录对象大小、查询延迟、传输与 H2D 时间。这是有效实验结果。

## 7. 实验五：LMCache 独立服务

### 7.1 配置

在实验工作目录创建 `lmcache_config.yaml`：

```yaml
mp_host: 127.0.0.1
mp_port: 5556
```

### 7.2 终端 1：LMCache

```bash
lmcache server \
  --host 127.0.0.1 \
  --port 5556 \
  --l1-size-gb 4 \
  --eviction-policy LRU
```

### 7.3 终端 2：SGLang

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --mem-fraction-static 0.7 \
  --enable-lmcache \
  --lmcache-config-file lmcache_config.yaml
```

本例不追加 HiCache、PD 或投机解码开关。其他限制见 [06](../../../03-kv-cache/implementations/sglang/tiered-cache.md)。

### 7.4 验证

1. 运行共享前缀探测。
2. 确认外部存储完成。
3. 仅重启 SGLang，保留 LMCache。
4. 检查是否取回已有缓存。
5. 区分推理进程本地 GPU 命中与 LMCache 服务命中。

与 HiCache 比较前，要统一实际 GPU KV 容量、CPU 缓存预算、数据分布与并发。本文两个入门启动示例不保证 CPU 缓存容量天然相同。

可选进阶：按 LMCache 官方文档配置 Mooncake Store 后端。需要相应扩展和版本支持；不要仅凭 SGLang 的 Mooncake 环境变量就认定 LMCache 已接入同一后端。

## 8. 实验六：1P1D 功能与性能

### 8.1 前置条件

- 两张能够容纳模型的 GPU。
- 匹配版本的 Mooncake 与 Router。
- TCP 路径先验证功能，RDMA 单独验证。
- 所有参与服务使用相同模型及兼容版本。

### 8.2 终端 1：Prefill

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30000 \
  --base-gpu-id 0 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake_tcp
```

### 8.3 终端 2：Decode

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 30001 \
  --base-gpu-id 1 \
  --tp 1 \
  --context-length 8192 \
  --max-total-tokens 32768 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake_tcp
```

### 8.4 终端 3：Router

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 127.0.0.1 \
  --port 8000
```

把实验一 benchmark 的 `--base-url` 改为 `http://127.0.0.1:8000`。

### 8.5 对照与验收

比较短输入、长输入、长输出和混合负载。总 GPU 预算相同，固定输出行为与资源设置；不能拿两卡 PD 对单卡服务直接宣称架构提升。

回答：

- Prefill 干扰减少了多少？
- 每请求传输了多少字节？
- KV 传输是否暴露在 TTFT 关键路径？
- P 与 D 是否一侧排队、一侧闲置？
- 预留目标 KV 的时机如何影响容量？
- TCP 结果能否支持关于 RDMA 的结论？

PD 不一定更快。明确收益消失的边界比只展示最好的一组结果更有价值。

## 9. 实验七：分析 trace 并写报告

### 9.1 采集

在已经就绪的普通基线服务上：

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 80 \
  --max-concurrency 16 \
  --profile \
  --profile-output-dir /tmp/sglang-profile
```

该命令依赖当前版本提供 profiling 接口；生成路径需要对服务端可写。实际文件位置以日志为准。

PD 模式中 Prefill 与 Decode 应按官方方法分别采集，不要把普通服务的采集命令当作完整 PD trace。

[Benchmark 与 Profiling 指南](https://github.com/sgl-project/sglang/blob/30dc2f94303e339431cbd51489b2d1719678b0c1/docs/docs/developer_guide/benchmark_and_profiling.mdx)

### 9.2 分析顺序

1. GPU 空隙：CPU 调度、Tokenizer 或同步是否造成饥饿。
2. Kernel：主要时间在 GEMM、Attention、拷贝还是通信。
3. Overlap：拷贝与计算是否真正重叠。
4. Launch：是否有密集小 kernel 和明显 CPU 发射成本。
5. 多 rank：是否有慢 rank 或通信等待。
6. 缓存：加载是否减少了 Prefill，还是仅增加等待。
7. 端到端：改变后 TTFT、ITL、Goodput 是否改善。

Profiler 本身有开销，最终性能结论应使用未开启 profiler 的独立 benchmark 确认。

## 10. 结果表模板

| 场景 | 实际缓存来源 | 少算 token | 传输字节 | TTFT p50/p95/p99 | ITL p95 | 输出 tok/s | Goodput | 错误率 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 冷缓存 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| GPU 命中 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Host 命中 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 外部命中 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| PD | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

“待测”表示需要实际运行，不是缺失的示例成绩。

## 11. 故障与反例实验

只在隔离环境中进行：

- 检查重复输入、部分前缀、修改中间 token 的命中差异。
- 改变兼容命名空间，确认不会错误复用。
- 在请求取消后观察资源是否回收。
- 正常停止外部缓存服务，观察超时、错误与可用降级行为。
- 提高压力，检查 GPU 命中是否转为 Host/外部，是否出现缓存抖动。

不得假定所有后端都自动 fail-open。记录实际行为，并区分产品当前能力与自己提出的恢复设计。

## 12. 最终报告

至少包括环境、workload、配置、结果、trace/缓存证据、正确性、结论与局限。完整模板见 [09](../../../08-career/implementations/sglang/assessment.md)。
