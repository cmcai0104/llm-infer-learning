# 06 · 实验与性能诊断

实验按硬件分层：A/G 可在当前 macOS 运行；B 需要 Linux 和 Mooncake 原生包；C 的 RDMA 部分需要真实网络；D/E/F 的推理部分需要匹配模型的 GPU。以下步骤是学习实验说明，除纯 Python 算例外，不代表已在此工作区完成部署或获得性能数据。

## 实验 A：容量与带宽预算

**目标**：从配置得到可检查的数量级；仅使用 Python 标准库。

```python
import math

layers = 32
kv_heads = 8
head_dim = 128
element_bytes = 2
seq_len = 8192
batch = 64
page_tokens = 16
chunk_tokens = 256
tp = 8  # 本例假设 KV heads 均匀分片，不复制
effective_gib_per_s = 20

bytes_per_token = 2 * layers * kv_heads * head_dim * element_bytes
allocated_tokens = math.ceil(seq_len / page_tokens) * page_tokens
request_bytes = allocated_tokens * bytes_per_token
print("KiB/token:", bytes_per_token / 1024)
print("GiB/request:", request_bytes / 2**30)
print("MiB/request/rank:", request_bytes / tp / 2**20)
print("GiB/batch:", batch * request_bytes / 2**30)
print("MiB/chunk, all ranks:", chunk_tokens * bytes_per_token / 2**20)
print("network lower bound ms:", request_bytes / 2**30 / effective_gib_per_s * 1000)
```

预期输出依次为 `128、1、128、64、32、50`。把 `seq_len` 改成 8193，观察分页容量与有效 token 字节数的差别；把 KV dtype 改成 1 byte 后，记得为量化元数据另计预算。

**交付物**：MHA/GQA/MQA/MLA 各一行容量表，MLA 按实际缓存维度另写公式；不要仅改 `kv_heads` 假装完成 MLA 计算。

## 实验 B：Linux TCP 双进程对象读写

### B1. 环境

按 [Quick Start](../getting_started/quick-start.md) 选择当前支持的 Linux/Python 环境和 wheel。没有 GPU 时选择其 Non-CUDA 路径；当前指南要求 Python 3.10+ 和相应 Linux 动态库。macOS 系统 Python 可以做实验 A，但不按 Linux wheel 路径直接安装原生组件。

在独立学习环境安装：

```bash
python -m pip install mooncake-transfer-engine-non-cuda
```

缺少系统库时按 Quick Start 的依赖列表处理。记录实际安装版本，不同时安装多个 Mooncake wheel 变体。

以下是**直接 Store API 学习实验**的服务启动。真实 vLLM/SGLang 集成应使用各自官方指南的服务与配置。

终端 1：

```bash
mooncake_master \
  --rpc_port=50051 \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=127.0.0.1 \
  --http_metadata_server_port=8080
```

### B2. 保存为 `store_roundtrip.py`

writer 先写入并保持进程存活；reader 随后在另一个进程读取。代码检查实际内容，使用唯一 key 避免与历史对象同键。

```python
import argparse
import hashlib
import os
import time
import uuid

from mooncake.store import MooncakeDistributedStore

parser = argparse.ArgumentParser()
parser.add_argument("role", choices=("writer", "reader"))
parser.add_argument("--key")
args = parser.parse_args()
if args.role == "reader" and not args.key:
    parser.error("reader requires --key from the writer")

store = MooncakeDistributedStore()
try:
    rc = store.setup(
        os.environ.get("LOCAL_HOST", "127.0.0.1"),
        os.environ.get("METADATA_URL", "http://127.0.0.1:8080/metadata"),
        64 * 1024 * 1024,
        16 * 1024 * 1024,
        "tcp",
        "",
        os.environ.get("MASTER_ADDR", "127.0.0.1:50051"),
    )
    if rc != 0:
        raise RuntimeError(f"setup failed: {rc}")

    payload = bytes(range(256)) * 4096  # 1 MiB，确定性内容
    expected = hashlib.sha256(payload).hexdigest()
    if args.role == "writer":
        key = args.key or f"study/{uuid.uuid4().hex}"
        rc = store.put(key, payload)
        if rc != 0:
            raise RuntimeError(f"put failed: {rc}")
        print(f"key={key}", flush=True)
        print(f"bytes={len(payload)} sha256={expected}", flush=True)
        input("Keep writer alive. Press Enter after the reader finishes: ")
    else:
        start = time.perf_counter()
        data = store.get(args.key)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if data != payload:
            raise RuntimeError("missing object or payload mismatch")
        print(f"bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}")
        print(f"get elapsed_ms={elapsed_ms:.3f}")
finally:
    store.close()
```

终端 2 运行 writer，复制打印的 key；终端 3 用实际 key 运行 reader：

```bash
python store_roundtrip.py writer
```

```bash
python store_roundtrip.py reader --key study/REPLACE_WITH_PRINTED_KEY
```

预期：两个进程打印相同的 SHA-256，reader 得到 1048576 bytes。单次 elapsed 只是功能观测，不是可靠的带宽基准。

writer 先于 reader 写入，使初始分配发生在 writer 已贡献的 segment 上；仍要通过日志/指标验证实际数据路径。退出 writer 可能使其内存副本失效，不能把“写完立刻退出”当常驻存储服务。

跨主机扩展时，Master metadata 绑定可达接口，各进程 `LOCAL_HOST` 设成可达地址，`MASTER_ADDR` 和 `METADATA_URL` 指向服务端，并确保实际客户端传输端口互通。`127.0.0.1` 只能用于本机。

### B3. Buffer API 练习

在已经成功 setup 且对象存在的脚本中，可用文档中的 BufferPool：

```python
from mooncake.buffer_pool import BufferPool

pool = BufferPool(store)
try:
    with pool.buffer(1024 * 1024) as lease:
        n = store.get_into(args.key, lease.ptr, lease.size)
        if n != 1024 * 1024:
            raise RuntimeError(f"get_into failed or size mismatch: {n}")
        view = lease.buffer[:n]
        try:
            # 真正消费者可直接使用 view；此处为验证而主动复制。
            if bytes(view) != payload:
                raise RuntimeError("payload mismatch")
        finally:
            del view
finally:
    pool.close()
```

把代码放在 B2 的 reader 分支，`store.close()` 之前。`args.key`、`payload`、`store` 来自 B2。导出的 view 必须先销毁，lease 才能释放；pool 必须先于 store 关闭。不要把 view 保存到函数外继续使用。

补充练习：按 [Python API](../api-reference/python/mooncake-store.md) 用 NumPy 或 pinned tensor 注册 buffer，调用 `put_from/get_into`，检查每个注册/传输/注销结果。不要在在途任务结束前释放张量。

## 实验 C：TE 与 Store 分层测量

**目标**：区分“网络慢”“对象管理慢”“GPU 加载慢”。

先按 [TE bench](../design/transfer-engine/index.md#example-transfer-engine-bench) 在两端运行 `transfer_engine_bench`，再看 [调优指南](../design/transfer-engine/transfer-engine-bench-tuning.md)。用当前二进制 `--help` 核对 flags 和单位，固定 tag/commit。

推荐扫描矩阵：

| 变量 | 样例取值 | 原因 |
|---|---|---|
| block bytes | 4 KiB、64 KiB、1 MiB、16 MiB | 观察固定时延与带宽区间 |
| batch size | 1、8、32、128 | 观察提交开销与队列拥塞 |
| 并发 | 1、2、4、8 个受控 worker | 找饱和点，不只找最高吞吐 |
| 方向 | READ、WRITE | 两侧资源瓶颈可能不同 |
| 协议 | TCP、RDMA（有设备时） | 作为机制对比，报告硬件条件 |
| 路径 | 单 NIC、多 NIC、同/跨 NUMA | 验证拓扑影响 |
| 注册 | 预注册复用 vs 含注册成本 | 防止把 MR 开销混进或漏出结果 |

然后运行 Store 的现有 [存储 benchmark](../performance/mooncake/storage-benchmark.md)，使用相同对象量级；对比 TE payload 带宽与 Store 的查询/分配/复制成本。

有效带宽定义为成功传输的 payload bytes / 时间。复制两份时同时报告逻辑写字节与物理发送字节；多 NIC 聚合需避免双向或多进程重复计数。

**验收**：找到一个受固定开销限制的小对象区间、一个带宽饱和区间，并解释为什么增加并发之后 P99 仍可能恶化。没有 RDMA 时只完成 TCP，不把 TCP 数字外推为 RDMA 预测。

## 实验 D：选择一条推理集成路线

推荐先选择熟悉的引擎，完成一条路径后再比较另一条：

1. [vLLM + 原生 Store](../deployment/integrations/vllm/kv-cache-storage.md)。
2. [vLLM + LMCache MP + Store](../deployment/integrations/lmcache/vllmv1-lmcache-mp-integration.md)。
3. [SGLang HiCache + Store](../deployment/integrations/sglang/hicache-quick-start.md)。

严格复用同一指南的完整启动顺序、模型和兼容版本。先从单节点/小模型正确运行开始，再扩展 PD 或跨主机。MP 的 native 扩展、L1 连续内存和 RDMA 注册尤其要检查。

### 请求集

构造至少四类，每类记录实际 tokenizer 产生的 token 数：

- 固定长 system prefix + 不同短问题；
- 相同长文档 + 不同问题；
- 完全不同 prompt；
- 相同文本但不同 namespace/模型 revision/LoRA，验证隔离。

改变 suffix 的同时确认共享部分 token IDs 完全一致；模板、空格和分隔符都可能改变边界。不要用字符数代替 token 数。

### 对照组

| 组 | 配置 | 用途 |
|---|---|---|
| A | 无跨请求复用，固定其余参数 | 计算基线 |
| B | 仅引擎本地 prefix cache | 本地复用收益 |
| C | 外部缓存冷启动 | 查询/写入附加成本 |
| D | 外部缓存预热，读取端本地冷 | 真正跨实例读取收益 |
| E | 本地和外部都热 | 正常多层工作状态 |

用新的、兼容的读取实例或框架提供的局部缓存控制构造 D，保持资源贡献者存活。不能盲目重启所有进程清空“本地缓存”，同时把唯一 Store segment 也关掉。只清理本实验 namespace，不对共享环境调用 `remove_all`。

### 正确性

同权重、同输入、确定性或固定采样条件下比较输出。底层执行路径变化可能引起浮点差异，所以同时检查 token/logits 的合理容差及任务质量；若使用有损 KV 压缩或 CacheBlend，单独做质量评估。

验收必须确认：缓存确实被保存、从预期层读取、最终跳过了相应 prefill，而不是仅看某个“hit”日志。

## 实验 E：故障、租约与资源压力

只在自己的隔离实验环境执行，先保存配置和基线。每个实验一次只改变一个故障维度。

| 注入 | 观察 | 合格结果 |
|---|---|---|
| writer 在 PutStart 后退出 | 未完成对象、清理超时、后续重试 | 不读到半写入数据，资源最终可回收 |
| 关闭持有唯一副本的客户端 | descriptor 失效、读取失败、重算 | 不把无效缓存当成功，服务可降级 |
| 增加读取耗时或实验性缩短 lease | Get 返回、目标 buffer 使用情况 | 过期失败不能继续消费 buffer |
| 减小 segment、增加唯一对象 | 分配失败、淘汰、pin/lease 占比 | 回收符合规则，错误可解释 |
| 制造热点 key | 节点 NIC、队列、P99 | 能区分热点与整体容量不足 |
| 停止 Master | RPC 错误/暂停；HA 则观察切换 | 行为与已配置模式一致，不假设单 Master 自动 HA |
| 隔离 namespace 或换模型版本 | hit/miss 与输出 | 不发生错误跨组复用 |

PutStart 中途崩溃和租约时间边界不一定能用 Python 高层 API 稳定触发。优先使用已有 C++ 测试的 failpoint/受控时钟或测试夹具；没有这种控制就记录“未复现”，不要把随机杀进程当成已验证精确阶段。

复盘按四栏写：注入了什么、预期什么、实际证据、如何修复/降级。特别记录失败后缓冲区是否仍被访问。

## 实验 F：SLO 与性能报告

对每个配置至少包含：

```text
硬件与版本；模型/输入输出长度；TP/PP；cache dtype/layout
到达模型与 offered load；并发；预热条件；测量持续时间/样本量
成功/错误/拒绝/超时；TTFT/ITL/E2E 的 P50/P95/P99
input/output tokens/s；成功 req/s；SLO goodput
各 tier 的 token/byte 命中；实际成功加载的 prefix
Master RPC 延迟与 CPU；NIC/PCIe/DRAM/GPU 利用率；空间和回收
```

使用开放式请求到达率扫描时，报告计划发送时间与实际发送延迟，避免负载生成器变慢导致 offered load 被隐性降低。闭环固定并发适合饱和探索，但慢响应会自然降低到达率，不能单独代表固定用户流量下的排队行为。

P99 需要足够样本，不能拿几十个请求声称稳定尾延迟。至少做多次重复，报告样本量与波动；不要只取最佳一轮。

诊断顺序：

1. 先看兼容性、正确性和到底命中哪层。
2. 分解 TTFT：queue、lookup、fetch、H2D/layout、prefill、handoff。
3. 检查谁饱和：Master CPU、NIC、PCIe、内存带宽、GPU。
4. 固定其他参数，只改 chunk、batch、并发或预取策略之一。
5. 同时检查 ITL/goodput，确认优化没有把代价转移给 decode。

## 实验 G：没有 GPU 的策略模拟

用本章 A 的 KV 字节模型构建离散事件或 trace 回放，输入至少有：

```text
arrival_time、namespace、prefix_id、prefix_tokens、suffix_tokens
cache_capacity、chunk_size、network_bandwidth、fixed_lookup_cost
prefill_cost_model、queue_policy、prefetch_deadline、eviction_policy
```

比较 LRU、频率准入、按预计节省时间/字节的策略。用顺序扫描、Zipf 热点、突发和模型版本切换四种 workload。

必须建模网络/计算队列和并发资源，不能把每个请求都当独占机器。输出 token 命中率、搬运字节、队列长度和模拟 goodput。

**局限**：模拟可以证明“在某组假设下策略更好”，不能证明 Mooncake 真实实现达到相同带宽、时延或质量。报告中把模拟结果与硬件实测分开。

## 实验记录模板

| 项目 | 填写内容 |
|---|---|
| 假设 | 例如：提高 chunk 可减少 RPC，但会降低部分前缀命中 |
| 配置/版本 | commit、硬件、模型、KV 格式、内存、网卡 |
| 唯一变量 | 本轮仅改变什么 |
| 正确性证据 | 字节校验、兼容性、输出比较、错误码 |
| 原始数据 | trace、日志、指标导出文件位置 |
| 结果 | 效应大小、分位数、样本量和重复运行波动 |
| 限制 | 未覆盖的网络、负载、模型或故障 |
| 下一步 | 下一项能区分两种解释的实验 |
