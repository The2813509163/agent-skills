# 外部证据：精度、schema与成本的经验锚点

整理于2026-10-20；主体来自随包两份用户委托调查，含论文表与固定源码链接。本项目没有复现这些外部模型。新增NVFP4格式核对来自官方正文；不给缺失字段补默认值。

## 1. 低掉点有依据，但不是逐任务保证

`drop=baseline−quant`，pp为绝对百分点。各行只在其论文/实验内部汇总。

| 来源/范围 | mean(drop) | mean(max(0,drop)) | 最大单项drop | 说明 |
|---|---:|---:|---:|---|
| PrefixQuant v1 Table18中的SQ，5模型×5任务 | .304 | .376 | 2.45 | INT8 W8A8KV8，非裸RTN |
| 同表QuaRot | .079 | .180 | 1.66 | 旋转＋相应粒度 |
| 同表PrefixQuant无微调 | .382 | .601 | 3.41 | 后续高精度前缀KV边界见下 |
| Quantization Hurts Reasoning v2的FlatQuant，8模型×4任务 | −.247 | .381 | 2.00 | fake quant，Q16/无P8；负均值不能当稳定能力提升 |
| TRT-LLM FP8 W/A/KV，Llama-3.1-8B MMLU | .40 | .40 | .40 | H100 PCIe80GB、v0.13.0 |
| 上行再开FP8 attention | .80 | .80 | .80 | 比固定KV8路线额外掉.40pp |
| vLLM Qwen3-30B-A3B，3项推理 | .633 | .633 | .80 | 加代码后均值.95、最差1.9 |
| vLLM Qwen3.5-27B，4项 | .000 | .150 | .40 | KV不包括所有递归状态 |

- SQ/QuaRot/PrefixQuant：Llama-2 7/13/70B、Llama-3 8/70B；WinoGrande、HellaSwag、ARC-C/E、PIQA。
- FlatQuant：AIME-120、MATH-500、GSM8K、GPQA-Diamond。另有2026预印本报告FlatQuant的AIME掉5.6/6.6pp，足以否定“先进方法每项必≤1”。
- 中文、业务域、长上下文、生成推理不由英文常识均分保证；AIME单年30题、AIME-120和Avg@16不是同口径。
- 参考：随包原资料§2、来源S1–S5；增补§5.1。不要跨表造一个“行业平均掉点”。

## 2. 学术低掉点通常用了什么

| 配方 | W | A | KV | 其他条件 |
|---|---|---|---|---|
| Table18 SmoothQuant | 对称per-output-channel | 动态对称per-token | 静态对称per-tensor | Pile校准、平滑 |
| 同表QuaRot | 对称per-output-channel | 动态对称per-token | 动态非对称group128 | 配对正交/Hadamard；Q16，无P8 |
| 推理模型FlatQuant公开入口 | 对称per-output；部分层受TP影响 | 动态非对称per-token | 动态非对称，D=128时每token/head | 学习变换、LWC/LAC、对角scale；Q16，无P8 |
| PrefixQuant W8 | 对称per-output | 静态对称per-tensor | 普通KV静态per-tensor | 前缀、变换、MSE初始化；后续保留高精度前缀KV |

注意：原论文并未逐个公开完整W8 run manifest。当前脚本默认GPTQ/RTN选项不能反推历史论文那一格。

FlatQuant公开推理脚本的例子：WikiText2、128样本、batch4、15epochs、lr=.005；1.5/7B distill Qwen长度4096，其余部分模型2048；TP由调用方传入。TP改变o/down的A/W分组，不只是速度参数。论文原版对称A与这个非对称A脚本也不是同配方。

PrefixQuant v1旧分数保留；v2/代码高精度前缀KV使其不能保证每个cache元素均8-bit。无微调MSE初始化8样本、前缀/统计64×512等是不同步骤，不能拼成一个训练预算。

## 3. 工业FP8常见的是多种layout

| 实例 | W/A | KV | 解释边界 |
|---|---|---|---|
| TRT-LLM v0.13.0标准FP8 QDQ | 每逻辑矩阵/节点静态scalar | 每层一个scalar，K/V共用 | `FP8RowLinear`的Row是并行层名，不是量化per-row |
| Qwen3-30B-A3B-Thinking-2507-FP8历史官方配置 | W128×128，A动态token-group128 | 另由引擎配置 | W块有16384个值，A组只有128个 |
| Qwen3.5-27B-FP8历史官方配置 | 同上 | attention cache单独定义 | 有GDN等豁免层，不是所有状态/模块FP8 |
| vLLM所引FP8联合评测 | W/A FP8＋FP8 attention | per-tensor，未校准scale1.0 | 历史官方checkpoint已核验，但未与博客run revision逐一绑定 |

官方历史revision：Qwen3 `60d80c83c53c3b611c642dbb8c942b3f90c5948a`；Qwen3.5 `2e1b21350ce589fcaafbb3c7d7eac526a7aed582`。不能把W128×128/A128称为MXFP8，不能让其KV自动“继承”group128。

格式选择与部署检查分开：原生支持、packing/scale layout、量化节点、豁免模块、加载后实际dispatch都要核对。MXFP8/MXFP4/NVFP4的定义见格式文档；本地未做这三者的GLM整模验收。

## 4. GPTQ：8-bit应期待多少额外收益

| 同协议配对 | RTN→GPTQ收益 | 限制 |
|---|---:|---|
| QuaRot W8，Phi-3-mini | +.16pp | 固定旋转/粒度/A/KV |
| 同上Llama-2-7B/13B/70B | +.16/+.03/+.11pp | 不是BF16到完整量化的总收益 |
| Qwen3，10个有效版本W8A16 MMLU | 平均+.03pp | RTN掉.13、GPTQ掉.10；4胜4平2负 |
| 同Qwen3的W4A16 | 平均+2.74pp | 低比特问题，不能搬给W8 |
| OpenPangu7B W8A16，MMLU/GSM8K | −.32/−3.18pp | 特定校准/实现反例，不是GPTQ上限 |

Qwen3原表4B GPTQ-W8与1.7B整行重复，调查核对PDF后标为疑似表错误，主均值排除；没有作者更正前不修造数值。weight-only结果不替完整W8A8KV8背书。

## 5. Q/P计算与KV存储：性能不能混归因

- LMDeploy标准路径可以KV8存储、QK/PV解量化后16-bit计算；SageAttention存在INT8 QK＋FP16 PV；FA3/XQA存在更低精度两次乘法路径。
- 接口Q16不证明内核Q16：某些Hopper XQA内部先转Q8。具体分派受head_dim、q_seq_len、滑窗、ragged等约束。
- FA3局部指数权重有放大和高精度分母，所查版本常数256；Hopper XQA所查分支448。普通全局P fake-quant不能等价模拟它们。
- 实际使用的KV格式可能prefill/decode不同；首次prefill旁路原KV与decode读量化cache必须分开确认。

### 可引用的速度量级

| 证据 | 实测 | 不能外推 |
|---|---|---|
| SqueezeBits，H100 PCIe80GB、TRT-LLM .13，Llama3.1-8B | 固定FP8模型＋KV后再开context FMHA，2048/128与128/2048两负载未增吞吐 | 旧版本/单卡；不是当前所有attention路线 |
| NVIDIA H200 kernel，64Q/4KV头、D128，16k/64k | prefill1.43–1.44×；decode1.87–1.88× | KV与计算同时变化，非固定KV8净收益 |
| 同表B200 | prefill1.48–1.56×；decode1.55–1.60× | kernel微基准，不是整服务倍率 |
| vLLM，H100/FA3、Llama3.1-8B、约20k/2k、并发8 | 输出吞吐450.3→517.5tok/s，+14.9% | KV与attention共同变化 |
| 同vLLM gpt-oss-20b | 831.6→853.0tok/s，+2.6% | 模型与负载不同，不能复用14.9% |

同一方案也可能decode快、prefill慢。TTFT、ITL/TPOT、吞吐、纯kernel时间和容量收益分别报告。

## 6. 离线成本有多大

| 来源步骤 | 条件 | 公开量级 |
|---|---|---|
| QuaRot旋转 | Llama2-70B，单A100 | 约5min |
| 上例额外GPTQ | 128×2048校准 | 约2h；不是W8专项计时 |
| FlatQuant校准 | Llama3-8B，128×2048、15epochs、batch4 | 约.9h/26GB；FP32 recipe约2.2h，所引段GPU型号未确认 |
| PrefixQuant前缀/初始化 | 8B，单A100-80GB | .2/.7min |
| PrefixQuant可选微调 | 同8B/70B、单A100 | 2.2h/17h |

不同位宽、卡、数据与步骤不可做统一速度榜。复杂权重离线准备不意味着同layout的推理必慢；在线Hadamard/变换则要另计。

## 7. NVFP4：补充的4-bit边界

[NVIDIA官方正文](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)核对于2026-10-20：E2M1数据、16值E4M3 scale＋tensor FP32总尺度。相较MXFP4的32值E8M0，尺度更细且非幂次。

- 厂商图6：DeepSeek-R1-0528，**原FP8→NVFP4**，MMLU-Pro约85→84、GPQA81→80、HLE15→14、LiveCodeBench77→76、SciCode40→40、Math50098→98、AIME2024 89→91%。这些是图示整数近似，不是原始精确数表。
- 不能写成“所有模型4-bit掉点≤1pp”，更不能当BF16→严格W4A4KV4的证据；该段完整KV配置未锁定。
- 约4.5bit/value的表示成本可解释相对16bit约3.5×的理想数据压缩量级；整模型峰值、workspace、KV与未量化层另算。
- Blackwell整系统/能效对H100的倍数包含硬件和系统变化，不能单独归因NVFP4，不能移给本地Ascend。

## 8. 随包原资料的地位

- `external-reference-2026-10-17.md`：总体精度预期、W8 GPTQ配对、算法与成本、全部S1–S17来源。
- `external-supplement-2026-10-19.md`：Q/P与KV解耦、真实kernel分派、细粒度官方配置、batch轴、全部固定源码URL。
- 两份原文按字节保留；源hash见`assets/provenance.json`。本摘要不是重新进行所有外部实验核验，原调查未锁定的字段继续标未知。
