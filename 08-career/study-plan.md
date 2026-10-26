# 统一 28 天学习计划：基础只学一次，后面按岗位分流

[返回主题](README.md) · [上一篇：岗位路线](roles.md) · [下一篇：面试与项目](interview-and-design.md)

这是按技术主题推进的主计划：每天先回答一个系统问题，再选框架验证。原有四份 28 天计划放在本主题的实现资料中，作为深读备选，不要求逐份执行。

**专家岗位定位**：这 28 天是查漏和整理作品的节奏，不保证从零达到 P7/P8。先按[专家能力总纲](expert-interview-guide.md)自评；已有经验者整合设计、实验与复盘证据，缺少子系统实践者应延长相应阶段。第一周配合[设计图册](../01-foundations/design-atlas.md)，KV 路线重点深读[存储教材](../04-storage/kv-cache-storage.md)和[联合实现](../04-storage/implementations/lmcache/mooncake-walkthrough.md)。

默认具备 Python、基本 C++、Linux 进程/线程和网络基础。工作日 2 小时、周末 4 小时，共 `4 × (5 × 2 + 2 × 4) = 72` 小时。它用于建立主线和完成一个有边界的项目，不保证一个月精通四套系统。

每天约 30 分钟读概念、60 分钟读代码或实验、30 分钟闭卷输出。周末增加实验与复盘。明显缺乏基础时延长计划，不为赶日程跳过正确性。

## 第一周：所有方向共同完成

| 天 | 阅读与任务 | 当天交付 |
|---|---|---|
| D1 / 2h | [总体介绍](../01-foundations/overview.md)：从请求看系统分层 | 一张数据流图，标出计算、缓存、存储与传输 |
| D2 / 2h | [公共基础](../01-foundations/concepts.md)第 1–3 节 | 从 K/V 解释复用，独立算 8K 请求容量 |
| D3 / 2h | [运行时](../02-runtime/README.md)：请求、调度、预算；选择一个引擎追踪 | 请求路径图、一个 token 预算例子 |
| D4 / 2h | [KV 管理](../03-kv-cache/README.md)：分页、前缀、分层与发布 | 共享块图、两个错误命中反例 |
| D5 / 2h | [分布式存储](../04-storage/README.md)：对象、副本、租约与回收 | Put/Get 状态图，标出数据和目录所有者 |
| D6 / 4h | [数据传输](../05-transfer/README.md)，运行 CPU 练习 | lookup→load→ready 图、传输下界与安全释放条件 |
| D7 / 4h | [Serving 与性能](../06-serving/README.md)，做闭卷检查 | 分段延迟模型、主方向、错题表和项目问题 |

**验收**：算出 128 KiB/token、8,192 token 为 1 GiB；解释 TP 除法前提；区分查询命中与 GPU ready；说清 PD 与共享缓存的区别。

CPU 教学脚本在 [07-practice/cpu/inference_lab.py](../07-practice/cpu/inference_lab.py)，包含 attention/KV、前缀 hash、调度和容量练习。需要 Python 3.9+，建议使用 Python 3.12 虚拟环境；确认下面的 `python` 指向该环境，再从资料库根目录执行：

```bash
python --version
python 07-practice/cpu/inference_lab.py all
```

它是教学模型，不代表某个框架的真实调度策略，也不替代 GPU 正确性验证。

## 第二周：沿一个方向追通数据与状态

每天选择自己方向所在列，**不用把四列全部完成**。

| 天 | 引擎方向 | KV 存储方向 | 通信/GPU 方向 | 平台方向 |
|---|---|---|---|---|
| D8 / 2h | 请求对象与状态 | Connector 的调度/worker 契约 | 源/目标 buffer 与设备依赖 | 请求时间线与指标口径 |
| D9 / 2h | Scheduler 预算和队列 | LMCache key、group 与连续前缀 | 注册内存/传输完成，或 kernel shape | 到达率、并发、队列与服务率 |
| D10 / 2h | KV 分配、引用与回收 | L1 staging、发布、读保留 | CPU/GPU/NIC 拓扑或 Roofline | 本地/host/远端缓存资源预算 |
| D11 / 2h | Runner 与模型输入布局 | Store PutStart→写入→PutEnd | TE 批量路径或 attention backend | 路由：缓存热度与排队代价 |
| D12 / 2h | 取消、抢占与异步执行 | Get、lease、失败 buffer 和回收 | 异步取消、同步与安全释放 | 重试、限流、缓存失效回源 |
| D13 / 4h | 追一条真实代码链；有 GPU 做执行基线 | TCP 对象实验；无 Linux 做状态与失败推演 | 有硬件做传输/前向基线；否则做成本模型 | 准备工作负载与压测方案 |
| D14 / 4h | 讲解请求全过程 | 讲解一条加载和一条写入全过程 | 讲解瓶颈、测量边界与依赖 | 讲解容量和故障场景 |

**统一产物**：一份“函数/状态—资源所有者—何时可用—何时释放—失败怎么办”的表。只画正常路径不通过验收。

对应阅读：

- 引擎：[vLLM 源码教材](../02-runtime/implementations/vllm/request-walkthrough.md)或 [SGLang Runtime](../02-runtime/implementations/sglang/runtime.md)。
- KV：[LMCache 内核](../03-kv-cache/implementations/lmcache/internals.rst)、[Mooncake 内核](../04-storage/implementations/mooncake/internals.md)。
- 通信/GPU：[性能教材](../06-serving/implementations/vllm/performance-and-parallelism.md)、[TE 与 Store 分层测量](../07-practice/implementations/mooncake/labs.md)。
- 平台：[性能运维](../06-serving/implementations/lmcache/observability.rst)、[系统设计](implementations/mooncake/system-design.md)。

## 第三周：做对照，而不是堆功能

本周时间仍是工作日 2h、周末 4h。所有方向采用相同实验步骤，改变被研究的对象。

| 天 | 通用任务 | KV 存储方向的具体例子 |
|---|---|---|
| D15 | 写一个能被证伪的假设，冻结版本与变量 | 长共享前缀下，远端加载关键路径低于省下的 Prefill |
| D16 | 选择一个最小可跑路径，核对格式与资源 | 只选原生 Store、HiCache 或 LMCache 中一条 |
| D17 | 正确性与冷基线 | 改前缀、改 namespace、部分缺失时不错误复用 |
| D18 | 热负载对照 | 分清本地命中与远端实际加载，记录省下的 token |
| D19 | 低收益反例 | 短输入、低复用、热点切换，观察额外查询/写入 |
| D20 / 4h | 扫描一个关键变量，保留原始数据 | chunk、并发或预取预算三选一 |
| D21 / 4h | 解释曲线，确认瓶颈和收益边界 | 命中率、搬运量、TTFT/ITL、goodput 联合分析 |

其他方向选题：引擎研究长短请求混合下的 token 预算；通信研究对象大小与并发；GPU 研究代表性 shape 下的一个瓶颈；平台研究固定到达率与突发流量下的 SLO。

**验收**：至少有一个正向结果、一个不收益或退化的条件，并能解释。没有真实硬件时产出推导/模拟结果，标出未测指标，不填写虚构曲线。

## 第四周：故障、系统设计与面试

| 天 | 时间 | 任务 | 交付 |
|---|---:|---|---|
| D22 | 2h | 做容量、网络、对象数与在途预算 | 一页计算表，单位和假设齐全 |
| D23 | 2h | 测/推演部分失败、超时和取消 | 安全回收、重算与拒绝消费的规则 |
| D24 | 2h | 测/推演节点退出、热点和回源压力 | 故障矩阵与降级顺序 |
| D25 | 2h | 完成一个系统设计题 | 需求→估算→数据路径→故障→验证 |
| D26 | 2h | 整理项目材料 | 配置、版本、基线、结果、边界、未验证项 |
| D27 | 4h | 模拟面试；重点追问不确定处 | 30 分钟录音、错题与补证据清单 |
| D28 | 4h | 按[验收标准](interview-and-design.md)自评 | 5 分钟项目讲述、10 分钟系统设计、后续专题 |

## 只有两周怎么裁剪

保留第一周公共主线，用第二周选择一个方向，至少完成一次路径追踪、一个算例、一份失败表和项目提纲。优先删高级后端、MoE、非前缀复用等选修，不能删缓存正确性和实验条件。

## 不同硬件能完成什么

| 环境 | 可以完成 | 不能据此声称 |
|---|---|---|
| macOS / 普通 CPU | 概念、容量、代码阅读、教学脚本、策略模拟 | GPU 推理或 RDMA 的实测收益 |
| Linux CPU/TCP | 适用的对象协议实验、字节正确性、部分失败 | GPU 布局、GDR、真实 TTFT 改善 |
| Linux 单 GPU | 推理、host 加载、局部性能与正确性 | 跨机吞吐、真实网络故障域 |
| 多 GPU / 多节点 RDMA | 并行一致性、远端共享、拓扑与故障实验 | 未测模型/负载上的普遍最优 |

每阶段都可以有有效产物，但产物的结论必须和证据匹配。
