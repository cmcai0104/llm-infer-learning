KV Cache 原理与容量推导
=======================

为什么缓存 K/V
--------------

对单层单个 attention head，令当前位置的表示为 x，投影得到 Q、K、V。
生成下一个 token 时，新的 query 要关注历史所有有效位置：

.. math::

   \mathrm{Attention}(Q,K,V)
   = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}} + M\right)V

自回归 causal mask 保证历史 token 不会依赖未来 token。已有历史 K/V 可以保留，
只为新增 token 计算新的投影；历史 Q 在普通自回归解码中不再使用。
缓存省去的是历史前向计算，attention 仍要访问历史 K/V，所以长上下文 decode
仍可能受内存带宽和容量限制。KV 是模型运行时状态，不是原文、embedding、
检索向量或最终回答。

Prefill 同时处理 prompt 的多个位置，通常有更高的计算并行度；
Decode 每条序列每步通常产生一个新 token，在许多配置中更受内存带宽限制。
这不是绝对硬件分类：batch、模型、context 长度、attention kernel 与量化都会改变瓶颈。
连续批处理把不同请求的阶段交错，P/D 分离则把两种阶段部署到不同资源池。

容量公式与例外
--------------

对于各层具有相同 K/V head 数、head dimension 和 dtype 的普通 attention：

.. math::

   M_{\mathrm{KV}} =
   2 \times L \times T \times H_{\mathrm{KV}} \times D \times b

其中 2 表示 K 和 V；L 为层数；T 为有效 token 数；
H_KV 为 KV heads 数；D 为每个 head 的维度；b 为每个元素的字节数。
容量是模型的逻辑 KV payload，不含 allocator 空洞、对齐、量化 scale、
对象元数据、临时 buffer 和副本。多个不同序列应按各自有效长度求和；
共享前缀只在确实共享物理对象时去重。

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - 结构
     - 应如何计算
     - 容易犯的错误
   * - MHA
     - H_KV 通常等于 query heads
     - 把参数量直接当 KV 容量
   * - GQA
     - 多个 query heads 共享一组 KV，使用实际 H_KV
     - 仍用 query heads，估算高出多倍
   * - MQA
     - H_KV 通常为 1
     - 假定任意 TP 度数都能无副本地切分
   * - MLA
     - 读取引擎实际缓存的 latent、RoPE 分量及 dtype 后相加
     - 机械套入 2×heads×D；或把每 rank 缓存当作总量/TP
   * - Hybrid / SWA
     - 按 full-attention、滑窗、递归状态各 group 求和
     - 让所有层都线性乘上下文长度

算例：L=32、H_KV=8、D=128、FP16/BF16（b=2）：

* 每 token：131,072 bytes = 128 KiB。
* 8,192 tokens：1 GiB；256-token chunk：32 MiB。
* 64 条各 8K 且不共享的序列：64 GiB 逻辑 KV。
* 4 路 TP 若恰好均匀切分 KV heads：每 rank 256 MiB/8K 请求。
  如果 KV 被复制、层数被 PP 切分或存在额外状态，应按实际布局重算。

GB 使用 10^9，GiB 使用 2^30。200 Gbit/s 网卡的理论线速为
25 GB/s；传 1 GiB 的纯网络下界约 42.95 ms，不是 5 ms，也不是 1/200 秒。
H2D、协议、排队和布局转换还会增加延迟。

可以直接运行下面的独立算例：

.. code-block:: python

   # SPDX-License-Identifier: Apache-2.0
   layers = 32
   kv_heads = 8
   head_dim = 128
   bytes_per_element = 2
   tokens = 8192
   bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
   request_bytes = bytes_per_token * tokens
   chunk_bytes = bytes_per_token * 256
   network_bytes_per_second = 200e9 / 8
   print(f"KV/token: {bytes_per_token / 1024:.0f} KiB")
   print(f"KV/request: {request_bytes / 2**30:.2f} GiB")
   print(f"KV/chunk: {chunk_bytes / 2**20:.0f} MiB")
   print(f"Network lower bound: {request_bytes / network_bytes_per_second * 1000:.2f} ms")

输出依次为 128 KiB、1.00 GiB、32 MiB、42.95 ms。这里没有测量任何真实硬件。

PagedAttention、APC 和 LMCache 的分工
-------------------------------------

引擎以逻辑 block 管理序列，把它映射到 GPU 物理页，避免为每个请求预留一整段
最大长度的连续显存。分页降低分配浪费，但不消除内部碎片或所有复制。
Automatic Prefix Caching（APC）进一步识别相同前缀，使多个请求复用已有 GPU block。

LMCache 将复用扩展到更大的内存层级、远端后端或独立服务生命周期。
如果请求已经由 APC 完全服务，LMCache 不一定增加读取收益；在某些配置中，
它仍可能参与 offload 或为未来重用保留数据。测试时必须单独记录 APC 和外部 cache。

.. code-block:: text

   请求 A: [共享 system prompt][共享文档][问题 A]
   请求 B: [共享 system prompt][共享文档][问题 B]
                GPU APC / LMCache 都可能复用这一连续前缀

   请求 C: [另一个 system prompt][相同文档][问题 C]
                普通精确前缀缓存不能直接复用后面的文档 KV

block 是推理引擎分配单位；chunk 是 LMCache 哈希、对象与传输的逻辑分块单位。
MP 会进行 chunk size 协商以满足 block/group 对齐要求，启动时的请求值不一定等于
最终采用值。见 `Chunk size 协商 <https://github.com/LMCache/LMCache/blob/ff9185d419e1ce55c9f4f1ad84a4f4f1f06af7be/docs/source/mp/chunk_size_negotiation.rst>`__。

若只缓存完整 chunk，长度 T 可产生 floor(T/C) 个完整 chunk，剩余部分需要按
引擎/connector 语义处理。chunk=256 时，255、256、257 tokens 分别只有 0、1、1
个完整 chunk。不能据此断言最终一定复用 0、256、256 tokens：
模型计算首个输出所需的末位 logits、mask、APC 覆盖和具体 API 约束还可能减少命中。

正确性比“文本相同”更严格
------------------------

第 i 个 chunk 的标识可用教学式表示为：

.. math::

   h_i = H(h_{i-1}, \mathrm{tokens}_i, \mathrm{extra}_i)

前缀链使相同 chunk 内容在不同历史之后得到不同身份。实际实现还涉及初始 hash、
序列化、算法和模型/租户命名空间。不能把 token ID 列表用任意语言的默认
``hash()`` 就当作跨进程稳定协议。

复用必须确保产生 KV 的计算等价，至少检查：

1. 相同模型权重及版本，包括影响前向的 LoRA/adapter。
2. 相同 tokenization、模板展开结果与真实输入 token；多模态还包含对应输入身份。
3. 相同历史上下文、attention mask、位置/RoPE 设置及可见范围。
4. 兼容的 dtype、量化信息、layout、层/group、TP/PP 分片语义。
5. 正确的租户/授权隔离；同一字符串 key 不能替代身份认证。

这些是系统应保证的语义维度，**不是说当前 ObjectKey 自动编码了全部维度**。
部署需要用模型命名空间、版本隔离与兼容性校验补齐实际接口没有表达的内容。
采样 temperature/top-p 通常作用在 logits 之后，不改变固定 prompt 的 KV；
生成的新 token 不同则后续 KV 自然不同。

RoPE 改变位置相关的表示；仅旋转 K 并不能普遍修复“换了前文”的 KV。
因为更深层 hidden state 已经通过 attention 依赖原上下文。
因此 CacheBlend 属于有选择重算和质量评估的高级方法，
不能当作任意文本片段 KV 的无损拼接。

缓存不是持久化数据库
--------------------

KV 通常可由输入和模型重新计算，所以系统可以用 miss 回退换取简单生命周期，
但必须做到“不把未完成或错误数据当命中”。复制提高可用性，
内存副本不能自动提供跨断电持久化。写入完成的含义取决于层级：
GPU copy 完成、L1 可读、L2 写成功、远端副本持久化是不同承诺。

保留原始输入和模型版本是可重算的前提。高并发故障时所有请求一起重算，
可能导致 GPU 和队列过载；缓存可丢失并不意味着故障没有服务成本。

延迟与收益的最小模型
--------------------

.. math::

   TTFT \approx T_{\mathrm{queue}} + T_{\mathrm{lookup}}
   + T_{\mathrm{load,critical}} + T_{\mathrm{prefill,remaining}}
   + T_{\mathrm{first\ output}}

串行加载近似为查询、L2 读取、网络、解码和 H2D 之和；充分流水时更接近瓶颈阶段
加启动/收尾。不能在同一数据路径上既把并行阶段全部相加，又宣称完全 overlap。

可复用前缀的收益条件为：

.. math::

   T_{\mathrm{recompute,saved}} >
   T_{\mathrm{lookup}} + T_{\mathrm{load,exposed}}
   + T_{\mathrm{interference}}

省下的 prefill 时间应通过对照测量，不能直接按命中 token 百分比线性缩放：
attention 复杂度、剩余 query 仍需关注旧 KV、batch 效率和排队都会改变结果。
当缓存命中短、后端 RTT 大或传输争抢 decode 带宽时，重算可能更快。

本章自测
--------

* 将算例 H_KV 改为 32，容量变为多少？答案：每 token 512 KiB，8K 为 4 GiB。
* 为什么“把 KV 存 SSD 就能无限上下文”不成立？仍需计算、搬运及引擎可用工作集。
* 为什么精确 key 命中但 layout 不兼容仍可能产生错误答案？内容身份没有定义物理解释。
* 为什么请求数命中率 90% 不一定省掉 90% prefill？每次命中长度与代价不同。
