# 8-bit 量化项目参考：精度预期、方法收益与成本权衡

资料日期：2026-10-17。面向接收项目的开发者与 Agent，可单独复制本文件使用。

**定位：外部证据与选型参考，不是项目实施计划、性能承诺或验收规格。** 模型、硬件、业务负载、引擎与预算由目标项目确认；项目既有约束优先，不要求实现本文所有方法或评测所有 benchmark。

## 先读这一页

| 项目关心的问题 | 现有证据支持到哪里 |
|---|---|
| W8A8KV8 能不能接近高精度？ | **可以。** 优化后的 INT8 学术结果、FP8 实际引擎结果，均有平均准确率损失不到 1 个绝对百分点的实例 |
| 能否保证每项都不掉超过 1 点？ | **不能。** 即使使用先进方法，GPQA、MATH、ARC、AIME 仍有超过 1 点的公开反例 |
| 8-bit 权重有必要默认上 GPTQ 吗？ | **没有普遍必要性的证据。** 同配置配对中，GPTQ 相对 RTN 的平均收益经常只有零点几；需看目标模型与任务 |
| 能否拿 GPTQ 的 4-bit 收益推断 8-bit？ | **不能。** Qwen3 的 MMLU 配对中，W8 平均收益约 0.03 点，W4 约 2.74 点，属于不同问题 |
| 有高精度论文结果，是否就能高效部署？ | **不等价。** fake quantization 证明数值方案，不证明生产 kernel、实际加速或显存收益 |
| 哪条路线一定最好？ | 无统一赢家。格式、scale 粒度、异常值处理、校准与 kernel 支持共同决定结果 |
| 这份资料能否直接给出本项目耗时预算？ | **不能。** 下文提供有出处的成本量级；本项目的 W8 配置、机器和实现仍需实测 |

可建立的信心：**8-bit 联合量化不是只能接受几个点损失的勉强方案；低于 1 点的平均损失是有公开依据的现实目标。** 但能否达到、是否值得额外校准、部署能否加速，要由目标项目的数据决定。

## 1. 口径：先保证讨论的是同一个问题

### 1.1 W8A8KV8 的含义

- **W8**：参与量化的线性层权重为 8-bit。
- **A8**：相应矩阵乘法输入激活采用 8-bit 表示；不意味着所有中间张量永久以 8-bit 保存。
- **KV8**：attention 的 K/V cache 采用 8-bit 存储；不是只量化权重后顺便减少了一些显存。
- W、A、KV 可以使用不同的 8-bit 格式，如 INT8 或 FP8；应逐项记录，不能只写“8-bit 模型”。
- 常见实现仍保留高精度 Norm、Softmax、残差、累加，可能排除 `lm_head` 或部分敏感层；需公开量化层范围。
- **KV8 不等于 FP8 attention 计算。** cache 的存储格式与 QK/PV 的计算格式是不同选择。
- 混合 attention/递归架构中，KV8 不代表其他递归状态也变成 8-bit。

### 1.2 “掉 1 点”与平均值

`掉点 = 高精度准确率 − 量化准确率`，单位为绝对百分点 pp：80% → 79% 是掉 1 点。

| 应报告的指标 | 用途 |
|---|---|
| 各 benchmark 的高精度分、量化分、掉点 | 防止平均值掩盖关键任务退化 |
| 预先约定任务集合上的平均掉点 | 回答整体精度预期；等权或业务权重要说明 |
| 最大单项掉点、超过阈值的任务数 | 判断尾部风险 |
| `mean(max(0, 掉点))`，可选 | 防止某项涨点抵消其他任务的损失 |
| 多次评测或不确定性，条件允许时 | 区分小分差与随机采样波动 |

不要把不同论文、模型、题集的分数混成一个“行业平均”。PPL、F1、相对百分比变化也不能直接当准确率 pp 合并。

## 2. 完整 W8A8KV8：能达到什么精度

以下是**各自实验范围内**的平均值，不是所有模型的保证。学术模拟与真实部署分开解释。

### 2.1 INT8 学术结果

| 方法与实验范围 | 平均掉点 | 单项超过 1 点 | 最大单项掉点 | 来源 |
|---|---:|---:|---:|---|
| SmoothQuant，5 模型×5 项常识任务 | 0.30 | 3/25 | 2.45 | [S1] Table 18 |
| QuaRot，同一组模型与任务 | 0.08 | 1/25 | 1.66 | [S1] Table 18 |
| PrefixQuant（不微调），同一组模型与任务 | 0.38 | 5/25 | 3.41 | [S1] Table 18 |
| FlatQuant，8 模型×4 项推理任务 | -0.25 | 5/32 | 2.00 | [S2] |

- [S1] 固定引用 **2024 年 v1**：Llama-2 7B/13B/70B、Llama-3 8B/70B；WinoGrande、HellaSwag、ARC-C、ARC-E、PIQA。
- [S2] 固定引用 **2025 年 v2**：R1-Distill-Qwen/Llama、QwQ、Qwen3 等八个模型；AIME-120、MATH-500、GSM8K、GPQA-Diamond。作者公开仓库提供 fake quantization；不能据此承诺生产 INT8 kernel 的吞吐。
- FlatQuant 的负均值受到若干涨点影响；将负损失截为零后，平均损失为 **0.38 点**。不是“量化稳定提高能力”。
- 不从这几行给方法排全局名次：模型、校准、版本和任务都会改变排序。

典型反例：

| 方法 / 模型 / 任务 | 高精度 → W8A8KV8 | 掉点 |
|---|---:|---:|
| FlatQuant / R1-Distill-Qwen-7B / GPQA-Diamond | 51.0 → 49.3 | **1.7** |
| FlatQuant / R1-Distill-Llama-8B / MATH-500 | 91.0 → 89.0 | **2.0** |
| PrefixQuant / Llama-3-70B / ARC-E | 85.90 → 82.49 | **3.41** |

另有 2026 年预印本 [S3] 报告同为 FlatQuant W8A8KV8 的 AIME-120 退化：R1-Distill-Qwen-7B 掉 **5.6 点**，R1-Distill-Llama-8B 掉 **6.6 点**。它不代表该方法的最佳上限，但足以否定“先进算法自动保证每项 ≤1 点”。

### 2.2 FP8 实际引擎：不是只看 KV 的增量损失

vLLM 官方评测 [S4]：H100、FlashAttention 3，比较原始 BF16 与 **FP8 W/A/KV，并开启 FP8 attention** 的完整配置。

| 模型 | Benchmark | BF16 → 联合 FP8 | 掉点 |
|---|---|---:|---:|
| Qwen3-30B-A3B-Thinking-2507 | AIME25 | 86.3 → 85.7 | 0.6 |
| 同上 | GPQA-Diamond | 72.8 → 72.3 | 0.5 |
| 同上 | MATH-500 | 92.8 → 92.0 | 0.8 |
| Qwen3.5-27B | AIME25 | 92.7 → 93.3 | -0.6 |
| 同上 | GPQA-Diamond | 86.9 → 86.5 | 0.4 |
| 同上 | MATH-500 | 85.4 → 85.4 | 0.0 |

- 两个模型的三项平均掉点分别为 **0.63**、**-0.07**。
- 同图 Qwen3-30B-A3B 的 LiveCodeBench-v6 从 65.4 → 63.5，掉 **1.9 点**；加入代码后四项平均掉 **0.95 点**。
- Qwen3.5 为混合架构；该结果不是所有模型状态都量化为 8-bit 的证明。

TensorRT-LLM 的独立工程测评 [S5]：Llama-3.1-8B-Instruct、H100 PCIe 80GB、v0.13.0、MMLU 5-shot：

| 配置 | MMLU | 对 BF16 掉点 |
|---|---:|---:|
| BF16 | 68.30 | — |
| FP8 W/A + FP8 KV | 67.90 | **0.40** |
| 再开启 FP8 attention | 67.50 | **0.80** |

这证明了特定条件下的可部署低损失，不是对当前所有引擎版本、硬件或模型的保证。

### 2.3 哪些 benchmark 还存在证据缺口

| 任务 | 本资料能提供的判断 |
|---|---|
| MMLU、GSM8K、MATH-500、GPQA、ARC、HellaSwag、PIQA、WinoGrande | 有严格联合量化或明确配对结果，但覆盖模型有限 |
| AIME | 有结果，但 AIME-120、AIME24、AIME25、Avg@16 等协议不能混同；单年 30 题的一题即 3.33 点 |
| CMMLU、BoolQ | 本轮严格 W8A8KV8 配对证据不足；后文 weight-only 的结果不能补成联合量化保证 |

中文、业务领域、长上下文与代码任务，不应仅凭英文常识或 MMLU 的平均值推断安全。

## 3. 方法地图：各自解决什么问题

这些方法分属不同层次，**不是要求全部叠加的清单**。

| 选择 | 解决的问题 | 可能的收益 | 需要付出的成本 / 限制 |
|---|---|---|---|
| 合理粒度的 RTN 权重量化 | 将权重映射到目标网格 | W8 中常已接近高精度；强基线 | 不利用校准激活补偿层输出；不自动解决 A/KV 异常值 |
| GPTQ | 利用校准激活与近似二阶信息补偿权重量化误差 | W4 常有明显收益；W8 增量经常很小 | 额外校准计算、内存与超参数；不保证下游单调改善 |
| SmoothQuant | 将部分 activation outlier 压力迁到权重侧 | 让 INT8 W/A 更容易量化 | 需校准统计和适合的 smoothing；小型推理模型可能仍敏感 |
| QuaRot | 正交 / Hadamard 旋转，分散异常值 | 改善 W/A/KV 的可量化性 | 部分变换可融合，部分仍有运行时开销；需相应实现 |
| FlatQuant | 学习仿射变换、clipping、channel scaling | 推理模型联合 INT8 有高精度证据 | 迭代校准与专用变换 / kernel 集成，不能默认零成本 |
| PrefixQuant | 利用前缀 token 处理异常值，支持静态量化 | 减少静态激活量化的困难 | 前缀、初始化及可选微调；需验证缓存、模板和业务行为 |
| KV scale 校准 / 粒度调整 | 改善 KV 的动态范围与异常值处理 | 降低 KV8 引入的额外误差 | scale 元数据、校准与 kernel 支持；动态 scale 还有在线计算 |
| FP8 / MXFP8 格式与粒度 | 改变数值范围、分辨率与共享 scale 方式 | 原生支持时可兼顾精度与性能 | 不同格式不能互换；设备与引擎需支持具体 layout / scale |

注意：**W8 的 RTN 已很好，不等于整套 W8A8KV8 可以不处理异常值。** 权重误差很小时，主要损失可能来自 A 或 KV，此时追加 GPTQ 不一定对症。

## 4. GPTQ 对 RTN：8-bit 应期待多少增量收益

### 4.1 固定 A/KV，只替换权重量化算法

QuaRot [S6] Table 9 / 13 保持同一旋转、per-channel symmetric 权重量化，以及同一 A/KV 配置；只比较权重 RTN 与 GPTQ。六项任务包含 PIQA、WinoGrande、HellaSwag、ARC-E/C、LAMBADA。

| 模型 | W8 RTN 均分 | W8 GPTQ 均分 | GPTQ 收益 |
|---|---:|---:|---:|
| Phi-3-mini，约 3.8B | 73.02 | 73.18 | **0.16** |
| Llama-2-7B | 69.65 | 69.81 | **0.16** |
| Llama-2-13B | 72.46 | 72.49 | **0.03** |
| Llama-2-70B | 77.17 | 77.28 | **0.11** |

同一固定位宽内 A/KV 相同，差值才是权重算法的收益；不能把 BF16 到整套联合量化的损失全归因于权重。

### 4.2 纯 W8A16 的多规模对比

Qwen3 [S7] Table 1 / 2，固定 per-channel、A16、weight-only。十个有效 Base/非 Base 模型版本的 MMLU：

| 指标 | 结果 |
|---|---:|
| RTN 平均掉点 | **0.13 pp** |
| GPTQ 平均掉点 | **0.10 pp** |
| GPTQ 平均收益 | **0.03 pp** |
| GPTQ 胜 / 平 / 负 | **4 / 4 / 2** |
| 单个版本的最大配对分差 | **0.2 pp** |

- 六项常识任务的重算平均收益为 **-0.04 点**，没有稳定优势。
- 单项仍可能差超过 1 点：Qwen3-0.6B 的 GPTQ 比 RTN，BoolQ 高 **1.4 点**，ARC-C 低 **1.1 点**。
- 数据清理：Qwen3-4B GPTQ-W8 整行与 1.7B 重复，经原始 PDF 核实也相同，疑似原表错误。保留并标记、主均值排除；若纳入，11 版本 MMLU 平均收益为 **-0.87 点**。未获得作者更正或复现确认，不能静默把原值“修好”。
- 本表不能外推为完整 W8A8KV8 的掉点；它只回答 W8 权重的 RTN/GPTQ 差异。

### 4.3 反例与模型大小

OpenPangu [S8] 固定 symmetric group-128、A16，7B 模型的 W8 MMLU 为 RTN 68.20、GPTQ 67.88；GSM8K 为 66.03、62.85。GPTQ 分别低 **0.32**、**3.18 点**。这是特定校准与实现的结果，不是 GPTQ 的最优上限。

规模方面：

- 从约 0.6B 到 70B，已有 W8 配对中未观察到“某个规模必须 GPTQ”的统一规律。
- 小模型在 W4 下通常更敏感，GPTQ 也可能无法救回精度；不能把这条 W4 经验直接搬到 W8。
- Qwen3 的 W4A16 MMLU：11 个版本 GPTQ 全胜，平均挽回 **2.74 点**；它与 W8 的 **0.03 点**不是同一量级。
- 大模型也不保证方法收益小：同一模型在 MMLU 与常识任务上的收益可能差很多。

这些是描述统计。没有充分重复实验时，不宣称“统计显著改善”或“统计等效”。

## 5. 时间与资源：把三种成本分开

### 5.1 成本不只是量化脚本跑多久

| 成本 | 应记录什么 | 为什么影响选择 |
|---|---|---|
| 离线准备与校准 | 数据准备、前向统计、Hessian / 变换学习、导出耗时；GPU 数、型号、峰值内存 | 影响迭代速度、能否在预算内反复更新模型 |
| 在线推理 | prefill / decode、TTFT、每 token 延迟、吞吐、峰值显存 | 复杂方法可能带额外变换；也可能被高效 kernel 融合 |
| 集成与维护 | 格式转换、引擎导入、专用 kernel、分布式支持、版本升级与回退 | 数值上更好，不代表工程上值得引入 |

校准 GPU-hours 可按“实际使用 GPU 数×wall time”记录；跨异构 GPU 的 GPU-hours 仍不可直接等价。首次编译、数据下载、模型加载、导出与评测是否计入，应单独注明。

### 5.2 有原始出处的时间量级

**这些不是同硬件、同位宽、同校准量的横向跑分，更不是本项目 W8 的预算。** 原论文常主攻低比特；数字用于展示不同步骤的成本结构。

| 来源与步骤 | 原实验条件 | 公开耗时 / 资源 | 能说明什么 |
|---|---|---|---|
| QuaRot 旋转处理 [S6] | Llama-2-70B，单张 NVIDIA A100 | 约 **5 分钟** | 预处理本身可相对便宜 |
| 在上述处理中额外做 GPTQ [S6] | 同模型、同单卡；128×2048 WikiText-2 校准 | 再约 **2 小时** | 二阶权重量化可能远贵于变换本身；不是 W8 专项计时 |
| FlatQuant 默认校准 [S9] | Llama-3-8B，单 GPU；128×2048 WikiText-2、15 epochs、batch 4 | 约 **0.9 小时**、约 **26 GB**；FP32 recipe 约 2.2 小时 | 变换学习有迭代成本，AMP / recipe 影响明显；引用段未确认 GPU 型号，不用它和 A100 数据排名 |
| PrefixQuant：前缀识别 / 网格初始化 [S10] | Llama-3-8B，单 A100-80GB | **0.2 / 0.7 分钟** | 不做可选微调时，部分准备步骤很便宜 |
| PrefixQuant：可选微调 [S10] | 同一 8B、同单卡 | **2.2 小时** | 微调是另一档成本，不应与前缀识别混为一谈 |
| PrefixQuant：前缀识别 / 初始化 / 可选微调 [S10] | Llama-3-70B，单 A100-80GB | **1 分钟 / 12 分钟 / 17 小时** | 模型规模、是否微调可能极大改变预算 |

- PrefixQuant 表是 **v2 Appendix C / Table 6**；前文 W8 精度表来自 v1，不能拼成一次“相同实验”的性价比跑分。
- 上述步骤未必包含全部 I/O、导出、转换和评测，不能把步骤相加后冒充端到端测时。
- 纯 RTN 权重量化本身无需校准激活，但完整 A8/KV8 部署仍可能需要校准统计、scale 或其他预处理。
- GPTQ 增加离线成本，不意味着同位宽、同 layout、同 kernel 的最终模型必然推理更慢。
- 不把“8-bit GPTQ 常只提高 0.几”与“某低比特实验 GPTQ 花 2 小时”拼成精确的每小时收益比；需要同项目配对测量。

## 6. 方法可用性：从“有代码”到“能部署”还差几层

| 路线 | 现成入口 | 项目仍需确认 |
|---|---|---|
| FP8 W/A + FP8 KV | vLLM、SGLang、TensorRT-LLM 的相关功能；LLM Compressor 量化示例 [S11–S14] | 设备原生支持、格式 / scale 粒度、attention backend、模型层覆盖、联合开启方式 |
| INT8 W/A + KV8 | SmoothQuant 相关工具链；TensorRT-LLM、LMDeploy 等的能力 [S13、S15] | W8A8 与 KV8 在目标版本是否可组合；KV 的 INT8/FP8 格式、kernel 与硬件限制 |
| GPTQ / RTN 权重量化 | QuaRot、Qwen3 作者代码；其他量化工具需核对具体模式 | 不能只因工具支持 INT4 GPTQ 就假定支持所需 INT8/FP8、布局与激活格式 |
| QuaRot / FlatQuant / PrefixQuant | 公开论文与作者实现 | 有没有生产低比特 kernel；在线变换如何实现；导出格式能否被目标引擎接收 |
| MXFP8 | 专门的 microscaling 工具与相应后端 [S16] | 不等于普通 per-tensor FP8；block-32 共享 scale 及设备支持要匹配 |

判断可用性的顺序：

1. 数值模拟是否支持该模型与量化配置。
2. 是否能导出可保存、可重载的量化产物。
3. 目标引擎是否接受该格式与附加变换。
4. 日志 / profiling 是否显示实际走到预期的低精度 kernel，而不是退回高精度。
5. 实际负载的精度、显存与速度是否达到项目目标。

W8 和 KV8 的原始数据体积相对 16-bit 理论上约减半；**整机峰值显存不会因此必然减半，吞吐也不会因此必然翻倍**。scale、未量化层、workspace、并发与缓存布局都会影响结果。

## 7. 给项目 Agent 的决策框架

以下用于减少无效实验，不是强制执行顺序。目标项目可以依据已有证据直接跳过已解决的问题。

### 7.1 先确定自己的问题，而不是先挑算法

| 需要确认 | 具体内容 |
|---|---|
| 目标模型 | checkpoint / revision、Base/Instruct、dense/MoE、attention/混合架构、上下文范围 |
| 硬件与引擎 | GPU/NPU 型号、SDK、引擎版本、支持的 INT8/FP8 格式与 kernel |
| 业务负载 | prefill/decode 比例、输入/输出长度、batch、并发、长上下文、中文 / 数学 / 代码占比 |
| 验收条件 | 平均掉点、关键任务单项门槛、显存、延迟 / 吞吐；由项目明确，本文不代设 |
| 资源预算 | 允许的校准时长、设备内存、GPU-hours、依赖与维护成本 |

### 7.2 最有信息量的消融

- **高精度基线 → 只量化 W → 加 A8 → 加 KV8**：尽可能定位主要误差来自哪里；不把所有退化都归给权重算法。
- **RTN 对 GPTQ**：固定 W 格式、位宽、分组、对称性、clipping、预处理、量化层范围、A/KV 和评测。
- 若主要是 A8/KV8 损失，优先比较对应的 scale、粒度和异常值处理，不默认追加 GPTQ 就能修复。
- 若尝试旋转、仿射或前缀方法，先确认生产集成可行；将其独立增量收益和成本记下来。
- 固定采样、模板、shot、答案提取、最大生成长度；长 CoT 还需记录输出 token 数，避免把“答得更长”误当同负载的速度变化。

这些分解能定位问题，但误差不一定线性可加；最终仍看完整部署组合。

### 7.3 什么时候简单方案就够了

| 观察 | 可考虑的判断 |
|---|---|
| 原生 FP8 或同粒度 INT8 RTN 基线已经满足精度与性能 | 不必默认增加 GPTQ、变换学习或微调 |
| W8 只有零点几损失，但 A8/KV8 明显退化 | 将预算投到激活 / KV，而不是权重二阶补偿 |
| GPTQ 的收益小于评测波动，且校准 / 维护成本增加 | 保留简单方案，或扩大评测确认差异 |
| 某方法能跨过关键任务验收线，离线成本可接受 | 即使均值只改善零点几，也可能值得采用 |
| fake quantization 精度好，但目标引擎无对应高效实现 | 作为算法上限参考，不宣称已具备部署方案 |
| 为追求文献最优分数而引入大量复杂度，业务并不受益 | 可停止升级，不必穷举所有方法 |

### 7.4 最终交付应能回答的几句话

1. 选了什么 W/A/KV 格式、粒度和异常值处理，哪些层仍保留高精度。
2. 相对同协议高精度基线，平均与最差任务各掉多少；中文、业务和长上下文是否覆盖。
3. 方法相对最简单可用基线，额外改善了多少，离线花多少时间与内存。
4. 在真实引擎上，显存、TTFT、decode 延迟和吞吐实际改善多少；有无高精度回退。
5. 为什么选它而没有选更复杂 / 更简单的候选，模型更新后需要重做什么。

## 8. 已知误读与证据边界

- **W8A8-only、KV8-only 不是联合 W8A8KV8。** 两边分别近乎无损，也不能凭加法认定组合一定无损。
- 《Give Me BF16 or Give Me Death?》[S17] 明确没有覆盖 KV 压缩；其 W8A8 成绩不能直接当联合量化证据。
- FP8 / INT8 模型名称不保证所有关键层已量化；模型卡需要检查排除层和实际配置。
- 同位宽但分组、zero-point、clipping、模板或 checkpoint 不同的结果，是部署方案比较，不是单独的 GPTQ/RTN 算法消融。
- 公开 GPTQ/RTN 配对中的“接近”不是所有场景的等效性保证；OpenPangu 等有明确反例。
- 本资料未运行目标项目模型、未做本地 GPU 复现；时间表来自不同论文，不能横向排统一速度名次。
- 资料引用的是指定版本与日期，尤其有历史引擎和论文 v1；不是对所有最新模型、硬件或当下 SOTA 的穷尽评测。

## 来源索引

所有关键结论的原始出处均为外部 URL，本文件不依赖调研机器的目录、记忆工具或附件即可阅读。补充底稿名称列于末尾，按需传递。

| 编号 | 原始来源与定位 |
|---|---|
| S1 | [PrefixQuant v1，2410.05265v1](https://arxiv.org/html/2410.05265v1#A8.T18)，Table 18：明确 W8A8KV8 的常识任务配对 |
| S2 | [Quantization Hurts Reasoning? v2，2504.04823v2](https://arxiv.org/html/2504.04823v2)，Table 1/3/4/5；[作者代码](https://github.com/ruikangliu/Quantized-Reasoning-Models)：含 fake quantization |
| S3 | [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not，2606.00206v1](https://arxiv.org/html/2606.00206v1#A2.T2)，Appendix B.6、Table 2 |
| S4 | [vLLM：The State of FP8 KV-Cache and Attention Quantization，2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)，Figure 7/8、Accuracy Benchmarking |
| S5 | [SqueezeBits：vLLM vs TensorRT-LLM #8，2024-11-18](https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079)，Table 1、FP8 attention 后的准确率说明 |
| S6 | [QuaRot v2，2404.00456v2](https://arxiv.org/html/2404.00456v2)，Table 9/13；实现段：70B 旋转与 GPTQ 耗时；[作者代码](https://github.com/spcl/QuaRot) |
| S7 | [An Empirical Study of Qwen3 Quantization，2505.02214v1](https://arxiv.org/html/2505.02214v1)，Table 1/2；[作者代码](https://github.com/AI-Efficiency/Qwen3-Quantization-Toolkit) |
| S8 | [An Empirical Study of OpenPangu Quantization on Ascend NPUs，2606.21257v2](https://arxiv.org/html/2606.21257v2)，Table 2–5、Quantization protocol |
| S9 | [FlatQuant v2，2410.09426v2](https://arxiv.org/html/2410.09426v2)，校准设置、Table 7 的训练 recipe 成本；[作者代码](https://github.com/ruikangliu/FlatQuant) |
| S10 | [PrefixQuant v2，2410.05265v2](https://arxiv.org/html/2410.05265v2#A3.T6)，Appendix C、Table 6：步骤耗时、单 A100-80GB |
| S11 | [LLM Compressor 0.9.0：FP8 Weight, Activation, and KV Cache Quantization](https://docs.vllm.ai/projects/llm-compressor/en/0.9.0/examples/quantization_kv_cache/)，流程能力；不凭单个量化后分数推断掉点 |
| S12 | [vLLM：Quantized KV Cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)，cache 配置、scale 与后端限制 |
| S13 | [TensorRT-LLM：Quantization](https://nvidia.github.io/TensorRT-LLM/blogs/quantization-in-TRT-LLM.html)；[FP8 tuning guide](https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html)，W/A 与 KV 配置需独立核查 |
| S14 | [SGLang server arguments](https://docs.sglang.ai/advanced_features/server_arguments.html)，具体模式、硬件与当前版本参数需复核 |
| S15 | [LMDeploy：INT4/INT8 KV Cache](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)，KV 能力不自动证明完整 W8A8KV8 的精度 |
| S16 | [Benchmarking PTQ under Microscaling Floating Point Formats，2601.09555v1](https://arxiv.org/html/2601.09555v1)，MXFP8 格式与配对；主表不等于联合 KV8 |
| S17 | [Give Me BF16 or Give Me Death? v4，2411.02355v4](https://arxiv.org/html/2411.02355v4)，Limitations：KV 压缩不在本研究范围 |

可选底稿，不是阅读本文的前置条件：

- `w8a8kv8-accuracy-survey-2026-10-15.md`：完整联合量化调查；`assets/w8a8kv8-2026-10-15/paired-accuracy.csv` 共 220 条配对准确率。
- `gptq-vs-rtn-weight-ablation-2026-10-17.md`：权重算法消融；`assets/gptq-rtn-2026-10-17/paired-comparison.csv` 共 565 条准确率 / PPL 配对，含重算均值与异常标记；不能全文件直接混算平均。

## 可随文档发给项目 Agent 的说明

> 请将本文件作为外部研究资料，而不是已决定的技术方案。结合当前项目的模型、硬件、引擎、已有实现与业务负载，判断哪些证据可迁移、哪些方法值得做消融。重点评估精度收益、离线时间 / 内存、在线性能与维护成本；不要求实现所有候选。先明确你采用的精度目标和验收口径，再按项目流程推进。公开结果、推测与本项目实测请分开报告；若结论与本文不同，保留实测并解释实验条件，不为迎合文献而修改结论。
