# 07 · 实验与验证：用证据回答系统问题

[总目录](../README.md) · [前置：Serving 与性能](../06-serving/README.md) · [下一主题：岗位与面试](../08-career/README.md)

先决定要验证的问题、观察量和失败条件，再选实现及硬件。每个实验应能支持或推翻一个明确假设，而不只是证明服务启动成功。

## 1. 按问题安排实验

| 顺序 | 要验证的问题 | 最低环境 | 交付物 |
|---|---|---|---|
| A | 缓存是否等价、前缀改变会怎样 | 普通 CPU | attention 等价和 hash 失配结果 |
| B | token/内存预算怎样影响执行 | 普通 CPU | 调度推演和容量表 |
| C | 对象何时完整、失败后如何处理 | 适用的 Linux CPU/TCP 环境，或明确标注的状态推演 | 读写校验、失败表和所有权图 |
| D | 加载能否替代 Prefill | 兼容的 GPU 环境 | 冷/热、短/长前缀对照与正确性证据 |
| E | 传输瓶颈在哪里 | 对应网络和设备 | 对象大小/并发曲线与分段时间 |
| F | 故障与过载是否放大影响 | 相应多进程或多机环境 | 退出、超时、取消、回源和恢复报告 |

## 2. CPU 起步

使用 Python 3.9+，建议 Python 3.12 虚拟环境；从 `Infra` 根目录运行：

```bash
python --version
python 07-practice/cpu/inference_lab.py all
```

[脚本](cpu/inference_lab.py)只依赖标准库，包含 attention/KV 等价、链式前缀 hash、调度和容量四个教学实验。也可把 `all` 换成 `attention`、`prefix`、`schedule` 或 `capacity`。

预期包括：每个 attention 位置缓存与重算一致；改变中间前缀后后续块也不匹配；调度不超过预算；示例模型为 128 KiB/token，8K token 为 1 GiB。脚本不实现真实引擎，也不测 GPU/RDMA 性能。

## 3. 再选择实现手册

| 主题任务 | 实现手册 |
|---|---|
| 调度、GPU 基线、缓存、PD | [vLLM 实验设计](implementations/vllm/experiments.md)、[步骤与参考答案](implementations/vllm/walkthrough.md) |
| 前缀缓存、分层和运行时 profiling | [SGLang 实验](implementations/sglang/labs.md) |
| 缓存服务、后端和生命周期 | [LMCache 实验](implementations/lmcache/labs.rst) |
| Store 对象、TCP、RDMA、集成与故障 | [Mooncake 实验](implementations/mooncake/labs.md) |

不需要把四份手册全部跑一遍。选一条最小链路，核对[版本与环境](../08-career/references.md)，完成正确性和基线，再扩展变量。

## 4. 每份报告都应包含

1. 问题与假设：预期省掉哪段计算或等待。
2. 环境与负载：版本、模型、layout、并行方式、硬件、长度和到达模式。
3. 基线与单一变量：冷/热状态、容量、并发和其他开关保持可比。
4. 正确性与失败：失配、部分命中、超时或取消时如何避免错误消费。
5. 原始证据：运行命令、日志、指标、trace、重复次数和波动。
6. 结论与边界：在哪些条件成立，哪里退化，还有哪些未测。

没有相应硬件时，容量计算、状态推演和实验设计仍是有效产物；结论注明“推导/模拟/未测”。项目验收和面试讲述见[面试与系统设计](../08-career/interview-and-design.md)。
