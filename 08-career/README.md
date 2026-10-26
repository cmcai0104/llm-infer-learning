# 08 · 岗位、学习计划与面试：按能力组织成果

[总目录](../README.md) · [实验主题](../07-practice/README.md)

面试准备先证明能分析一类系统问题，再用具体项目的源码和实验说明自己做到了什么。框架用法是证据的一部分；容量估算、正确性、并发与资源管理、失败处理和性能归因是更稳定的能力主线。

## 1. 四个入口

| 资料 | 使用方式 |
|---|---|
| [岗位与掌握标准](roles.md) | 选择引擎、KV 存储、通信/GPU 或 Serving 平台方向，确定 P0/P1/P2 |
| [统一 28 天计划](study-plan.md) | 第一周按技术链路建立共同基础，随后沿岗位深入 |
| [面试与系统设计](interview-and-design.md) | 闭卷解释、容量/带宽计算、生命周期、故障与项目验收 |
| [资料与版本索引](references.md) | 按问题找深读材料，核对源码版本、实验条件和论文 |

## 2. 按主题组织面试证据

| 主题 | 必须拿得出的证据 | 实现题库入口 |
|---|---|---|
| 请求与调度 | 请求状态机、token 预算、取消/抢占路径 | [vLLM](implementations/vllm/questions.md)、[SGLang](implementations/sglang/questions.md) |
| KV 语义与管理 | 容量公式、失配反例、页引用与加载完成条件 | [运行时题](implementations/vllm/questions.md)、[缓存管理题](implementations/lmcache/questions.rst) |
| 存储与生命周期 | 完整写入发布、读保护、副本、淘汰和回收 | [缓存服务题](implementations/lmcache/questions.rst)、[Store 题](implementations/mooncake/questions.md) |
| 传输与性能 | 字节路径、理论下界、拓扑、完成和安全释放 | [通信题](implementations/mooncake/questions.md)、[性能题](implementations/sglang/questions.md) |
| 系统设计 | 需求、资源预算、容量与故障域、降级、观测 | [设计主线](interview-and-design.md)、[缓存服务设计](implementations/lmcache/system-design.rst)、[共享存储设计](implementations/mooncake/system-design.md) |

题库保留了原教材的题号，方便定位。相同基础题只做一份主答，其他实现用于比较边界；不要把题目数量当作进度。

## 3. 专项计划是补充材料

只有需要持续深入某个实现时才使用：[vLLM 专项](implementations/vllm/study-plan.md)、[SGLang 专项](implementations/sglang/study-plan.md)、[LMCache 专项](implementations/lmcache/study-plan.rst)、[Mooncake 专项](implementations/mooncake/study-plan.md)。

原教材导航与补充版本材料也保留在实现层：[LMCache 目录](implementations/lmcache/index.rst)、[Mooncake 目录](implementations/mooncake/index.md)、[vLLM 参考资料](implementations/vllm/references.md)、[SGLang 报告与验收](implementations/sglang/assessment.md)。其中的旧章节编号仅用于对应原教材，全库顺序以主题入口和统一计划为准。

## 4. 完成学习的标准

能不看文档讲出一条请求的数据和状态路径；能算容量与传输下界；能解释一个失败场景如何安全处理；能用源码或实验支持一次设计取舍；能区分已验证的事实与还需验证的假设。

KV 存储方向尤其要能连续回答：**为什么能复用 → 谁拥有数据 → 何时可读 → 何时可释放 → 加载是否比重算划算 → 失败后怎样恢复服务。**
