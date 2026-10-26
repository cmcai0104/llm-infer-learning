实验、基准与验收方法
====================

实验分为 A（仅 Python 标准库）、B（兼容的 Linux/GPU 推理环境）、
C（Mooncake 服务与 native 扩展）和 D（系统设计）。
本文给的是操作与验收步骤，不代表 GPU/RDMA 实验已经在你的机器上运行成功。
不要在生产集群清空缓存、停止服务或做故障注入；所有实验使用独立实例与数据目录。

实验 0：记录版本与边界
----------------------

在仓库根目录执行：

.. code-block:: bash

   git rev-parse HEAD
   python --version
   python -m pip show lmcache vllm torch mooncake-transfer-engine
   lmcache server --help

未安装的包应记为“未安装”，而不是先任意升级全部依赖。
安装路径见 `安装指南 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/getting_started/installation.rst>`__、
`兼容性 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/getting_started/compatibility.rst>`__；优先采用已经验证的 wheel/container 组合。
macOS 无 GPU 的 source-only 安装不等于可以运行 Linux eventfd、CUDA IPC 或 RDMA 路径。

记录模型 revision、tokenizer/chat template、KV dtype、TP/PP、GPU/NIC、
驱动、CPU NUMA、host memory、chunk/block size、transfer mode、adapter 配置。
模型权重和运行时不一致的结果不能直接横向比较。

实验 A1：容量与读取盈亏（无 GPU）
---------------------------------

先运行 `公共基础 <../../../03-kv-cache/semantics-and-costs.rst>`__ 的容量算例，再运行下面的敏感性分析。
假设 8K 的 KV 为 1 GiB，可省 prefill 时间为 120 ms，
固定开销为 3 ms，H2D 有效带宽为 24 GiB/s。
表中的 load 是串行模型，overlap 是理想两阶段流水的下界近似：

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   payload_gib = 1.0
   saved_prefill_ms = 120.0
   fixed_ms = 3.0
   h2d_ms = payload_gib / 24.0 * 1000
   for network_gib_s in (4.0, 8.0, 16.0, 24.0):
       network_ms = payload_gib / network_gib_s * 1000
       serial_ms = fixed_ms + network_ms + h2d_ms
       pipelined_ms = fixed_ms + max(network_ms, h2d_ms)
       print(
           f"{network_gib_s:4.0f} GiB/s: serial={serial_ms:6.1f} ms, "
           f"ideal_pipeline={pipelined_ms:6.1f} ms, "
           f"serial_gain={saved_prefill_ms - serial_ms:6.1f} ms"
       )

验收：4 GiB/s 时串行约 294.7 ms，应选择重算或避免该慢路径；
24 GiB/s 时串行约 86.3 ms，在这些假设下可获益。
进一步把命中量减到 64 MiB、固定延迟改为 10 ms，解释小对象场景为什么不同。
理想流水忽略启动/收尾、争用和分块，是边界分析而非预测实测值。

实验 A2：前缀哈希与缺口（无 GPU）
---------------------------------

这是教学模型，使用 SHA-256 和 JSON 编码，**不与 LMCache 的真实 key 协议兼容**。
它只验证前缀身份、namespace 隔离和连续命中语义：

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   import hashlib
   import json

   def prefix_keys(
       tokens: list[int], chunk_size: int, namespace: str
   ) -> list[str]:
       """Return full-chunk teaching hashes for tokens in a namespace.

       Args:
           tokens: Input token IDs.
           chunk_size: Positive number of tokens in each chunk.
           namespace: Model/version/tenant identity for this exercise.

       Returns:
           Ordered SHA-256 hex digests for complete chunks.

       Raises:
           ValueError: If chunk_size is not positive.
       """
       if chunk_size <= 0:
           raise ValueError("chunk_size must be positive")
       previous = hashlib.sha256(namespace.encode()).hexdigest()
       result: list[str] = []
       for start in range(0, len(tokens) - chunk_size + 1, chunk_size):
           payload = json.dumps(
               [previous, tokens[start : start + chunk_size]],
               separators=(",", ":"),
           ).encode()
           previous = hashlib.sha256(payload).hexdigest()
           result.append(previous)
       return result

   def prefix_hit_count(keys: list[str], stored: set[str]) -> int:
       """Count consecutive available keys from the start.

       Args:
           keys: Ordered chunk identities.
           stored: Available identities in the teaching store.

       Returns:
           Number of consecutive hits before the first miss.
       """
       count = 0
       for key in keys:
           if key not in stored:
               break
           count += 1
       return count

   first = prefix_keys([1, 2, 3, 4, 5, 6], 2, "model-v1/tenant-a")
   same_prefix = prefix_keys([1, 2, 3, 4, 9, 10], 2, "model-v1/tenant-a")
   changed_history = prefix_keys([9, 9, 3, 4, 5, 6], 2, "model-v1/tenant-a")
   other_tenant = prefix_keys([1, 2, 3, 4, 5, 6], 2, "model-v1/tenant-b")
   assert first[:2] == same_prefix[:2]
   assert first[1:] != changed_history[1:]
   assert set(first).isdisjoint(other_tenant)
   assert prefix_hit_count(first, {first[0], first[2]}) == 1
   assert len(prefix_keys([1, 2, 3], 2, "model-v1/tenant-a")) == 1
   print("prefix identity, isolation, gap and tail checks passed")

验收：修改第一块使所有后续 key 改变；删掉中间一块后只能复用连续前缀。
再思考：如果只保存 token chunk 自身的 hash，哪条断言会失去其语义保证？
如果两个进程的序列化格式不同，即使算法同名还能共享吗？

实验 B1：MP 冷写与热读（Linux/GPU）
-----------------------------------

前提：已按兼容性文档安装，模型已可在纯 vLLM 上正常运行。
下面采用仓库 quickstart 的 ``Qwen/Qwen3-8B``，请按 GPU 显存选用已验证的模型。
4 GiB L1 是演示预算，不包括引擎、权重、系统和额外 buffer。

终端一，启动独立 server：

.. code-block:: bash

   lmcache server \
     --host 127.0.0.1 --port 5555 \
     --http-host 127.0.0.1 --http-port 8090 \
     --l1-size-gb 4 --eviction-policy LRU --chunk-size 256

终端二，以下 external connector 配置要求支持该加载方式的 vLLM
（本基线兼容性文档的边界为 vLLM >= 0.20.0）：

.. code-block:: bash

   vllm serve Qwen/Qwen3-8B \
     --host 127.0.0.1 --port 8000 \
     --kv-transfer-config \
     '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":5555}}'

对于更旧的 vLLM 使用其受支持的 bundled connector，并核对 server 协议；
不要只删掉 module path 就声称任何版本都兼容。
端口仅监听本机，跨节点时必须改为可达地址并按服务边界配置访问控制。

终端三，用 Python 标准库发两次相同请求：

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   import json
   import time
   import urllib.request

   prompt = (
       "A cache stores reusable intermediate state. "
       "Correctness requires matching model, tokens and context. "
   ) * 160 + "\nSummarize the main idea in one sentence."
   payload = json.dumps(
       {
           "model": "Qwen/Qwen3-8B",
           "prompt": prompt,
           "max_tokens": 32,
           "temperature": 0,
           "stream": False,
       }
   ).encode()
   for run in range(2):
       request = urllib.request.Request(
           "http://127.0.0.1:8000/v1/completions",
           data=payload,
           headers={"Content-Type": "application/json"},
       )
       start = time.perf_counter()
       with urllib.request.urlopen(request, timeout=180) as response:
           body = json.load(response)
       elapsed = time.perf_counter() - start
       print(run, f"end_to_end_s={elapsed:.3f}", body["usage"])
       print(body["choices"][0]["text"])

这是功能 smoke test，打印的是完整响应延迟，不是 TTFT。
用返回的 prompt token 数确认实际长度大于最终协商的 chunk，
不要用字符数猜 token 数。首次请求应产生可存储 KV；第二次可能由 APC 或
LMCache 命中，必须结合指标/trace 才能归因。

只读检查：

.. code-block:: bash

   curl --fail http://127.0.0.1:8090/healthcheck
   curl --fail http://127.0.0.1:8090/status
   curl --fail http://127.0.0.1:8090/metrics

如果配置了 HTTP 认证，使用自己的合法凭据，不要把凭据写入报告或仓库。
HTTP 200 只按 endpoint 文档解释，不能当作所有后端完整健康的证明。

实验 B2：排除 APC 的假收益
---------------------------

1. 保留 LMCache server，确认首轮 STORE 已完成。
2. 在自己的测试终端优雅停止并重新启动 **仅 vLLM**，保持模型与配置相同。
3. 发相同 prompt。新引擎没有上一进程的 GPU APC，但可从仍存活的 LMCache 加载。
4. 记录 LMCache retrieve/load、有效命中 token 和端到端数据。
5. 改动 prompt 开头，验证普通前缀命中缩短；保留开头只改末尾则应保留可复用前缀。

新进程也有权重/kernel 初始化影响；在正式测量前用不共享前缀的请求预热计算路径。
不要把进程重启后的第一次请求直接和旧进程稳定态比较。
若 restart 导致缓存也消失，应先查 server 生命周期、容量淘汰和 key 兼容性，
不能直接判定“LMCache 不支持复用”。

实验 B3：L2 与性能矩阵
----------------------

先阅读 `文件系统 adapter <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/l2_storage/fs.rst>`__。在专用测试目录配置文件 adapter；
需要确保目录有权限与足够空间，不能指向已有生产数据。
每次实验使用一个新目录，避免通过删除旧数据制造冷缓存。

.. code-block:: bash

   lmcache server \
     --host 127.0.0.1 --port 5555 \
     --http-host 127.0.0.1 --http-port 8090 \
     --l1-size-gb 4 --eviction-policy LRU --chunk-size 256 \
     --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-study-run-001"}'

此命令是 B1 server 的替代配置，不要同时占用相同端口。
等待 L2 写完成，在测试环境优雅重启 LMCache 与引擎，再验证新进程从同一目录读取。
文件系统 page cache 可能仍是热的，必须将“L1 冷/文件页热”和“介质真正冷”分开报告。

正式 TTFT/ITL 使用支持 streaming token 时间戳的 benchmark，
参考 `Benchmarking <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/getting_started/benchmarking.rst>`__；按所安装 vLLM 版本的
``vllm bench serve --help`` 选项运行，而不是复制不匹配版本的脚本。
模型、请求 trace、输出 token 上限和并发保持固定，只改变一个变量。

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 变量
     - 建议取值
     - 需要观察
   * - 前缀共享比例
     - 0%、25%、75%、接近全共享
     - token 命中、实际省下的 prefill
   * - 上下文长度
     - 1K、8K、32K（不超过模型/显存能力）
     - 容量、lookup、网络/H2D 和排队
   * - chunk
     - 128、256、512 的协商后实际值
     - 过取、对象数、尾部、提交开销
   * - 并发
     - 1、4、16，逐级到 SLO 边界
     - goodput、p95/p99、staging 和抢占
   * - 层级
     - APC、L1、文件 L2、远端 L2
     - 哪一级真正服务了命中

结果模板：

.. code-block:: text

   run_id / versions / model / dtype / TP / block / effective_chunk
   request_count / input_distribution / output_distribution / concurrency
   APC_tokens / LMCache_L1_tokens / LMCache_L2_tokens / queried_tokens
   TTFT_p50_p95_p99 / ITL_p50_p95_p99 / goodput / errors
   transferred_bytes / L1_resident / L1_staging / host_peak_memory
   cold_or_warm_state / output_quality_check / raw_logs

实验 C：Mooncake Store 与 LMCache 接入
--------------------------------------

需要兼容的 Linux 环境、Mooncake Master/metadata 服务及 LMCache native
Mooncake 扩展。先用 TCP 验证功能，再用具备 NIC/驱动/权限的 RDMA 环境测性能；
TCP 成功不证明 RDMA 注册或 GPUDirect 路径正确。
服务安装与启动以 Mooncake 当前版本 quickstart/API 文档为准：
`Store API <https://github.com/kvcache-ai/Mooncake/blob/dd1217b6c461491e7c73ff29b4ffad215e0a9ac0/docs/source/api-reference/python/mooncake-store.md>`_。

以下假定本机隔离环境的 metadata 为 8080、Master 为 50051。
先做独立 API smoke test；测试 key 加随机后缀，不覆盖别人的数据：

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   import uuid

   from mooncake.store import MooncakeDistributedStore

   store = MooncakeDistributedStore()
   try:
       status = store.setup(
           "127.0.0.1",
           "http://127.0.0.1:8080/metadata",
           512 * 1024 * 1024,
           128 * 1024 * 1024,
           "tcp",
           "",
           "127.0.0.1:50051",
       )
       if status != 0:
           raise RuntimeError(f"Mooncake setup failed: {status}")
       key = f"lmcache-study-{uuid.uuid4().hex}"
       data = b"isolated KV storage smoke test"
       status = store.put(key, data)
       if status != 0:
           raise RuntimeError(f"Mooncake put failed: {status}")
       restored = store.get(key)
       if restored != data:
           raise RuntimeError("Mooncake read did not return the expected bytes")
       print("Mooncake roundtrip passed")
   finally:
       store.close()

注意返回值不是“一律 0=成功”：``get`` 返回数据，``is_exist`` 返回存在性状态，
buffer API 可能返回字节数，异步 API 返回任务/状态；必须按各自文档处理。

然后按 `Mooncake adapter 指南 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/l2_storage/mooncake_store.rst>`__ 构建 native 扩展，在独立环境用下面配置
替换 B1 server。此示例的 L1、Store segment 与 local buffer 总预算超过 10 GiB，
还未计引擎与系统；应预留足够 RAM，或按模型实际 KV 和文档调整。

.. code-block:: bash

   lmcache server \
     --host 127.0.0.1 --port 5555 \
     --http-host 127.0.0.1 --http-port 8090 \
     --l1-size-gb 4 --eviction-policy LRU --chunk-size 256 \
     --l2-adapter '{"type":"mooncake_store","num_workers":4,"local_hostname":"127.0.0.1","metadata_server":"http://127.0.0.1:8080/metadata","master_server_addr":"127.0.0.1:50051","protocol":"tcp","local_buffer_size":"3221225472","global_segment_size":"3221225472"}'

验证顺序：STORE 完成→远端对象可查→通过另一组兼容引擎/LMCache 实例读取。
远端存储 segment 的所有者必须持续存活；如果唯一副本来自将被停止的 Client，
重启它会丢失对象，不能把这个现象当成共享缓存失效。
做持久共享实验宜使用独立的 Store 贡献节点。

RDMA 阶段改用可达地址、匹配 NIC、``protocol: rdma`` 与正确注册设置，
并按文档考虑 ``--no-l1-use-lazy``。采集 NIC 计数、CPU 使用、注册耗时和有效带宽，
证明确实走目标路径；不能只根据配置字符串宣布使用了 RDMA。

实验 D：故障与结业报告
----------------------

无需硬件也能做桌面推演：L2 返回部分命中、一个 TP rank 缺块、请求取消、
metadata 变陈旧、1 个 Store 节点退出、in-flight load 超预算。
逐项写出有效前缀、buffer 所有者、应释放的资源、回退方式与预期指标。

硬件环境中只在独立测试实例验证这些场景，并在操作前保存日志和配置。
验收不是“程序没崩”：还应验证输出正确、没有引用/锁泄漏、后续请求可以恢复。

最终交付一张架构图、一张时序图、一份版本清单、一张实测或明确标为估算的性能表，
以及一个含失败路径的设计复盘。结合 `系统设计 <../../../08-career/implementations/lmcache/system-design.rst>`__
准备 5 分钟技术讲述。
