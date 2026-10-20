# 8 位量化项目：Attention、低掉点配方与 batch 粒度增量补充

资料截止：2026-10-19。

**接收对象：已经读过《8 位量化项目参考》的开发 Agent。** 本文汇集旧总报告之后的新增核验及解释修正，不重讲 GPTQ/RTN 消融和量化方法概论。核心内容与外部来源均在本文件内，单独转交即可；不依赖原会话、本机记忆系统或临时克隆目录。

用途：提供证据与配置边界，帮助目标项目自行判断；不是已批准的开发路线、验收标准或实验计划。讨论性能时固定“持久 KV cache 采用 8-bit”为问题前提，但引用实验若没有固定 KV8，会明确标出。

核验方式：论文、固定提交源码、官方历史配置、公开数表交叉核对。未在目标项目运行 GPU、模型或服务；不能据本文认定其复现失败的根因。

## 1. 先更新旧报告中的理解边界

| 主题 | 应采用的解释 |
|---|---|
| W8A8KV8 | 通常规定线性层 W/A 与持久 KV 的位宽；不自动规定 attention 的 Q/P 位宽 |
| 低掉点 INT8 结果 | 所查 QuaRot / 推理模型 FlatQuant 默认 Q16、无 P8 节点，不能拿来保证 QK/PV 全 8-bit 的精度 |
| KV8 与计算 | 存储 8-bit 可以搭配 8-bit、16-bit，甚至 FP32 SIMT 算术；接口 dtype 也不等于内核乘法输入 dtype |
| Prefill / decode | 可以独立分派；不存在通用的“prefill 必须 8、decode 必须 16”规则 |
| 性能归因 | KV16→8 与 attention 计算16→8一起变化的收益，不等于已有 KV8 后追加 Q8/P8 的收益 |
| PrefixQuant | v1 原分数保留，但后续论文/代码保留高精度前缀 KV；不能据 v1 标签保证每个 cache 元素均为 8-bit |
| FP8 粒度 | Qwen 官方 W128×128 / A128 不是 MXFP8；也不使 KV 自动变成组量化 |
| Per-token 与 batch | 动态激活 scale 按每个实际 `(b,t)` 独立计算，不跨 batch 合并相同序列位置 |
| 低于 0.5 / 1 点 | 实验均值、每项上限、涨点能否抵消损失是不同口径；原均值不能当逐项保证 |

## 2. Attention 里究竟量化了谁

### 2.1 先拆开线性层 A、Q、KV、PV 左操作数

```text
X → Q/K/V 投影 → 位置变换等 → QKᵀ → Softmax → PV → O → 输出投影
```

- 投影 W8A8：量化 X 与投影权重；投影输出 Q/K/V 仍可为 FP16/BF16。
- KV8：量化持久缓存中的 K/V；不自动量化 Q 或 P。
- Q8：在 QK 前额外把 Q 转到 8-bit 网格；“在线转换”不等于 scale 必须在线重估，scale 也可以静态校准。
- P8：PV 左操作数进入 8-bit 网格；高性能实现可能量化未归一化指数权重，而不是完整的全局 P。
- 输出投影 A8：量化的是 O，不是 P。

### 2.2 三类计算图都真实存在

| 路线 | QK 乘法输入 | PV 乘法输入 | 所查实例 |
|---|---|---|---|
| KV8 解量化计算 | Q16、解量化 K16 | P16、解量化 V16 | LMDeploy 标准 PyTorch/Triton；部分 FlashInfer FA2 / XQA 通用路径 |
| 只压低 QK | Q8、K8 | P16、V16 | SageAttention `sageattn_qk_int8_pv_fp16_triton` |
| 两次乘法均低精度 | Q8、K8 | 低精度指数/概率权重、V8 | vLLM 普通 FA3 FP8；符合条件的 Hopper XQA FP8；部分 SageAttention API |

“乘法输入 8-bit”不要求累加、score、mask、Softmax、分母和残差也为 8-bit。SageAttention 的单次函数也不自动负责持久 KV8 cache，不能单靠该 API 宣称整模型 W8A8KV8。[A1][A2][A3]

### 2.3 FA3 的 P8 不能用普通全局 P fake quant 代替

下式用单块、可分离的标量 Q/K/V scale 示意；组量化必须按实际轴恢复 scale，不能把任意分组 scale 都移到矩阵乘法外。

```text
S = sQ*sK*(Q8 @ K8ᵀ)/sqrt(d) + mask
U = exp(S - rowmax(S))
E = 256*U
E8 = round_FP8(E)
O ≈ sV*(E8 @ V8)/rowsum(E)
```

关键点：

- 所查 vLLM FA3 fork 使用指数放大常数 **256**。
- 分母来自转换 FP8 之前的指数权重，不是 `rowsum(E8)`。
- `round_FP8(256U)/(256ΣU)` 通常不等于 `round_FP8(softmax(S))`。
- 多块处理会更新最大值、分母和输出；块划分与局部舍入可能影响误差。

对应 online Softmax 的逻辑递推为：

```text
m_t = max(m_(t-1), rowmax(S_t))
alpha_t = exp(m_(t-1) - m_t)
E_t = 256*exp(S_t - m_t)
l_t = alpha_t*l_(t-1) + rowsum(E_t)
N_t = alpha_t*N_(t-1) + round_FP8(E_t) @ (sV*V8_t)
O = N_final/l_final
```

`m/l/N` 的统计与累加保持高精度。公式表达量化点与归一化，不试图复刻硬件累加的全部舍入细节。所查 Hopper XQA E4M3 分支使用 **448** 而非 256；两者均可称低精度 PV，但不能直接共用一个不带区别的 fake-quant 模拟。[A4][A5]

## 3. Prefill 和 decode 会怎样切换

### 3.1 形状与 cache 边界

省略 batch/head 轴：`QK=[M,d]×[d,N]`，`PV=[M,N]×[N,dv]`。

| 场景 | M | N |
|---|---:|---:|
| 完整 prefill、无历史 | prompt 长度 | prompt 长度，配 causal mask |
| Chunked prefill、有历史 H | 当前 chunk 长度 | 通常 H+M |
| 普通 decode | 1 | 通常 H+1，随生成增长 |
| 推测验证等多 token 查询 | 可大于 1 | 取决于历史与当前查询 |

**持久 KV8 不保证 prefill 的新 K/V 一定先经过该量化 cache。** 有些路径直接用新 K16/V16 计算 prefill，同时另行写入 KV8。因此，“缓存格式一样”也不保证实际消费的 K/V 量化误差一样。

### 3.2 已核验的组合，不是全框架默认值

每格表示 QK / PV 的乘法输入精度；16 包括 FP16/BF16，8 的具体格式须另记。

| 指定路线 | Prefill | Decode |
|---|---|---|
| vLLM 普通 FA3 FP8 | 8 / 8 | 8 / 8 |
| FlashInfer Hopper FA3 prefill + 受支持的 FP8 XQA decode | 8 / 8 | 8 / 8 |
| 明确选择 FA3 prefill、FA2 tensor-core decode | 8 / 8 | 16 / 16 |
| 明确选择 FA2 prefill、Hopper FP8 XQA decode | 16 / 16 | 8 / 8 |
| LMDeploy 所查标准 PyTorch/Triton KV8 路径 | 16 / 16 | 16 / 16 |

中间两行是现有后端/API可形成的组合，不是默认分派保证，也未声称其完整服务性能已经有四组合对照。FlashInfer native SIMT decode 还存在转换成 `float` 后用 FP32 算术的路径，不能把所有非 FP8 路线都写成 FP16。

### 3.3 最易误判：接口 Q16，内核仍可能 Q8

所查 FlashInfer Hopper XQA 在 SM90、E4M3 KV、`head_dim<=256` 等支持条件下，接收 Q16 后由 `F16QToF8Converter` 在内部转 FP8，QK/PV 使用 FP8 输入。

但在部分 `q_seq_len>1` 的 sinks、ragged Q、滑动窗口、指定形状等条件下，会显式关闭该 Hopper 路径并进入通用 16-bit 路径。更大的 head_dim 也不能直接沿用该 FP8 kernel 的结论。[A5][A6]

因此：

- `prefill=FP8, decode=BF16` 日志可能只描述接口 dtype，不证明 decode 内部没有 Q8。
- `disable_flashinfer_q_quantization` 等前端开关也不能单独证明内核内部没有 Q8。
- TensorRT-LLM 的固定版本同时有 FP8 QGMMA 与 16-bit HMMA XQA；不是 generation 只剩一种算法。[A7]

## 4. 性能：公开数据能支持到哪一步

**本轮未找到同时满足“固定 KV8、四种阶段组合、多卡、实际服务指标”的完整公开矩阵；不等于证明不存在。** 下面区分增量实验、联合方案、内核微基准。

### 4.1 最接近固定 KV8 的增量实验：没有额外吞吐收益

SqueezeBits，2024-11-18：[P1]。

- Llama-3.1-8B-Instruct；H100 PCIe 80GB。
- TensorRT-LLM v0.13.0，Triton Server v2.50.0。
- 1024 个请求，最大 batch 256。
- 固定 FP8 模型＋FP8 KV，再开启 `use_fp8_context_fmha`。

| 负载 | 输入 / 输出 tokens | 作者报告 |
|---|---:|---|
| Prefill-heavy | 2048 / 128 | 未增加吞吐 |
| Decode-heavy | 128 / 2048 | 未增加吞吐 |

同实验 MMLU：68.30（原精度）→67.90（FP8 W/A/KV）→67.50（再开 FP8 attention）。额外损失 0.40 pp。

限制必须保留：

- 软件较旧、单卡，不能概括为当前 FP8 attention 都不值得。
- 该开关还影响 QKV buffer、attention 输出 dtype/scale，不是其余数值节点全固定的纯乘法位宽消融；持久 KV8 相同也不保证 prefill 使用同一份量化值。[P2]
- 未测完整四组合。原图无原始数表、两个面板柱高相同，只引用作者“无改善”的文字，不读精确吞吐百分比。

### 4.2 NVIDIA H200 / B200：分阶段 kernel 数据

NVIDIA TensorRT-LLM 1.3.0rc27 版本化文档，取 **不开启 Skip Softmax** 的基线。[P3]

条件：`q_heads=64, kv_heads=4, head_dim=128`；prefill batch=1、decode batch=64。单位 µs。

| GPU / 长度 | Prefill BF16→FP8 | 加速比 | Decode BF16→FP8 | 加速比 |
|---|---:|---:|---:|---:|
| H200 / 16k | 7403→5157 | 1.44× | 498→266 | 1.87× |
| H200 / 64k | 115301→80550 | 1.43× | 1962→1045 | 1.88× |
| B200 / 16k | 4273→2886 | 1.48× | 303→196 | 1.55× |
| B200 / 64k | 67775→43399 | 1.56× | 1209→755 | 1.60× |

这证明对应 FP8 kernel 路线有收益，但**不是固定 KV8 的计算净收益**：decode 原表的带宽×耗时给出的数据量口径约为 2:1，例如 H200/16k 约 2.15/1.07 GB。也不能把后面的 Skip Softmax 端到端 TTFT/TPOT 表移过来当作 FP8 单变量服务对照。

### 4.3 vLLM：实际服务有收益，但 KV 与计算同时改变

vLLM，2026-04-22：[P4]。单 H100 / FA3，150 请求、并发 8，约 20k 输入 / 2k 输出。

| 模型 | Median TTFT，ms | Median ITL，ms | 输出吞吐，tok/s |
|---|---:|---:|---:|
| Llama-3.1-8B | 763.6→742.8 | 15.18→12.93 | 450.3→517.5 |
| gpt-oss-20b，full FP8 | 468.9→451.7 | 8.09→7.90 | 831.6→853.0 |

Llama 输出吞吐约 +14.9%、ITL约 −14.8%；gpt-oss 吞吐只约 +2.6%。这是 BF16 cache+attention 对 FP8 cache+attention，不能承诺在已有 KV8 后追加 Q8/P8 还能取得该收益。

单请求、输出 128 tokens、扫描上下文并拟合 `ITL=slope×input_len+intercept`：

| GPU / 后端 / 模型 | FP8/BF16 的 ITL 斜率 |
|---|---:|
| H100 / FA3 / Llama-3.1-8B | 54.2% |
| B200 / FlashInfer / Llama-3.1-8B | 54.0% |
| H100 / FA3 / gpt-oss-20b，full FP8 | 79.9% |
| B200 / FlashInfer / gpt-oss-20b | 57.9% |

这些是上下文相关拟合项，不是总 ITL 的通用倍率；硬件和后端也一起变化，不能把差异全归给卡。

反例：同报告 Gemma-4-E2B、H100/FA3、`head_dim=256`，FP8 decode 的 ITL 斜率为 BF16 的约 68%，但 prefill 的 TTFT 二次项系数约为 1.6 倍。**同一方案可能 decode 更快、prefill 更慢**；不证明某个混搭已被测为最优。

### 4.4 其他论文/硬件数据的用途

- FlashInfer 2024-02-02 有 H100、A100、RTX Ada 6000、RTX 4090 的 decode kernel 比较；比较 cache 精度路线，不能都解释成原生 FP8 乘法或服务净收益。[P5]
- FA3 / SageAttention 的大矩阵吞吐与数值误差，不能直接代替 `M=1` decode 的 TTFT/ITL。[A3][P6]
- QServe、TurboAttention 等同时改变 KV 位宽、Softmax 或其他系统设计，不能拼成固定 KV8 的 Q/P 净收益。

## 5. 低掉点结果：阈值与配方必须一起读取

### 5.1 重新限定统计口径

单位为绝对百分点 pp。`Δ=原精度−量化精度`；负数表示量化分数更高。第二个均值为 `mean(max(0,Δ))`，不允许涨点抵消损失。各行在各自实验内部统计，不能再跨论文合成行业平均。[R1][R2][P1][P4]

| 实验范围 | mean(Δ) | mean(max(0,Δ)) | 最大单项 Δ | 重要边界 |
|---|---:|---:|---:|---|
| PrefixQuant v1 Table18：SQ，5模型×5任务 | 0.304 | 0.376 | 2.45 | 对应论文的 SQ 配置 |
| 同表：QuaRot | 0.079 | 0.180 | 1.66 | 有旋转，不是裸 RTN |
| 同表：PrefixQuant，不微调 | 0.382 | 0.601 | 3.41 | 高精度前缀边界，见下文 |
| Quantization Hurts Reasoning? v2：FlatQuant，8模型×4任务 | −0.247 | 0.381 | 2.00 | fake quant，Q16、无 P8 节点 |
| TRT-LLM FP8 W/A/KV：MMLU 一项 | 0.400 | 0.400 | 0.40 | 没有开启 FP8 attention |
| 同实验再开 FP8 attention | 0.800 | 0.800 | 0.80 | ≤1，但不是 ≤0.5 |
| vLLM Qwen3-30B-A3B：3项推理任务 | 0.633 | 0.633 | 0.80 | 加 LiveCodeBench 后均值0.95、最大1.9 |
| vLLM Qwen3.5-27B：4项任务 | 0.000 | 0.150 | 0.40 | KV8 只指 attention cache，非所有递归状态 |

SQ/QuaRot/PrefixQuant 范围：Llama-2 7B/13B/70B、Llama-3 8B/70B；WinoGrande、HellaSwag、ARC-C/E、PIQA。FlatQuant 范围：AIME-120、MATH-500、GSM8K、GPQA-Diamond，不混入代码任务。

Qwen3.5 四项为 AIME25、GPQA、LiveCodeBench、MATH-500，Δ分别为 −0.6、0.4、0.2、0.0；与只取三项任务的均值不是同一口径。小差值还受评测噪声影响，不能据此宣称统计等价。

### 5.2 INT8：粒度、变换、非对称必须一起对齐

| 配方 | W | 线性层 A | K/V | 额外条件 |
|---|---|---|---|---|
| Table18 对应 SmoothQuant | 对称 per-output-channel | 动态、对称 per-token | 静态、对称 per-tensor | 平滑、异常值迁移；Pile 校准 |
| 同表 QuaRot | 对称 per-output-channel | 动态、对称 per-token | 动态、非对称 group-128 | 正交/Hadamard 旋转；Pile 校准 |
| 推理模型 FlatQuant 脚本 | 对称 per-output-channel；部分层受 TP 分组影响 | 动态、非对称 per-token；部分层另有 TP 分组 | 动态、非对称；原 D=128 时每 token/head 一个范围 | 学习变换、对角 scaling、LWC/LAC；Q16、无 P8 |
| PrefixQuant W8 | 对称 per-output-channel | 静态、对称 per-tensor | 普通 KV 静态、对称 per-tensor | 前缀、变换、MSE 初始化；高精度前缀例外 |

SQ/QuaRot 的论文只锁定到配置表和复现来源，未逐个提供原 W8 run manifest。不能将当前默认 GPTQ/RTN 选项冒充原次实验已确认设置。[R1]

**FlatQuant 的公开复现入口，不是每个已发表 run 的完整 manifest：**

```text
w_bits=8, w_asym=false
a_bits=8, a_asym=true
k_bits=8, k_asym=true, k_groupsize=128
v_bits=8, v_asym=true, v_groupsize=128
q_bits=16（默认）, gptq=false（脚本未传 --gptq）
lwc=true, lac=true, cali_trans=true, add_diag=true
cali_dataset=wikitext2, nsamples=128, cali_bsz=4
epochs=15, flat_lr=5e-3, deactive_amp=true, direct_inv=true
seqlen=4096：R1-Distill-Qwen-1.5B / 7B
seqlen=2048：该脚本其他模型
tp：调用者传入
```

- `o_proj/down_proj` 的 W 输入轴与 A 特征轴按 `C/tp` 分组；TP 改变数值配方，不只是并行速度。[R3][R4][R5]
- 所查 Qwen 校准 wrapper 将 KV 组织为 `[B,H,T,D]`，用 `groupsize=-1` 沿 D 求范围。D=128 才恰好等于 group-128；D=256 应另查是否真的拆成两组。
- K 量化位于 RoPE 之后、写 cache 之前，伴随 Q/K 变换。只改 CLI 的 `k_groupsize` 不证明校准、导出、运行三处都已改变。[R6]
- FlatQuant 原论文的对称 A 与这里显式 `a_asym` 的推理模型脚本不是同一配置。各分数的 TP、完整运行环境与 checkpoint 哈希仍未逐项披露。

**PrefixQuant 的特殊边界：**

- v1 §4.1：W8 的 KV 为静态 per-tensor，W4 才是静态 per-head；不能只套总览表。[R1]
- 后续 v2 和公开代码保留高精度 prefixed KV。原 v1 分数不删除，但在缺少原 run manifest 时，不作为“所有 cache 元素严格 8-bit”的保证。[R7][R8]
- 无微调时公开 MSE 初始化默认 8 样本，论文长度1024；前缀识别/统计另有64×512路径，跨批统计为EMA 0.99而非全校准集绝对最大值。当前公开代码不等于原2024 run已锁定。[R7][R9]

### 5.3 FP8：粗粒度 QDQ、block FP8 与 MXFP8 分开

**TRT-LLM v0.13.0 标准 FP8 QDQ：** W/A 为每个逻辑线性矩阵/节点的静态张量级 scalar；KV 每层一个 scalar，K/V 共用。`FP8RowLinear` 的 Row 是并行层类型，不代表 per-row 量化。[R10][R11]

这与 SqueezeBits 连续比较的 FP8 Context FMHA 约束相符，但作者未公开完整量化文件、scale 数值、校准数据和排除层；不能声称已完全还原其配置。

**同名 Qwen 官方历史 FP8 checkpoint：**

| 模型 | 固定 revision | 官方 W/A 配方 |
|---|---|---|
| Qwen3-30B-A3B-Thinking-2507-FP8 | `60d80c83c53c3b611c642dbb8c942b3f90c5948a` | W128×128；A 动态 token-group128 |
| Qwen3.5-27B-FP8 | `2e1b21350ce589fcaafbb3c7d7eac526a7aed582` | W128×128；A 动态 token-group128 |

- W 的128×128是一组 **16384个权重**；A 的group128是一组 **128个激活**。两个“128”不是同一种分组。[R12][R13][R14]
- 原 vLLM 评测的 KV 则明确为 **per-tensor、未校准 scale=1.0**，不是随 W/A 一起变成 group128。[P4]
- Qwen3 配置含145条忽略规则，包括 router、norm、`lm_head`；Qwen3.5 含259条，包括部分GDN投影、conv、视觉/embedding等。规则数不等于线性层数；全量量化这些敏感模块不是同一配置。
- **历史官方配方已核验，但未与原博客跑分 revision 一一绑定。** 它们是已验证的候选配方，不能写成原低掉点运行已完全复现。[R12][R13]

**MXFP8：** 32个值一组、E8M0 scale；分组方向和转换规则也要匹配，不是把普通FP8的group_size改32即可。W128×128/A128不等于MXFP8，W/A为MXFP8也不使KV自动为MXFP8。本轮没有找到完整锁定“联合KV8、≤0.5掉点、部署配方”的干净MXFP8实例；W8A8-only或训练后BF16评测不能补成该证据。[R15]

## 6. 动态 per-token 与 batch：已核对归约轴

### 6.1 每个实际 token 独立，不共享相同位置

`X[B,T,C]` 中 token 指实际出现的 `(b,t)`，不是词表ID，也不是所有batch共同的序列位置。

以无clipping的对称INT8为例，省略全零保护：

```python
amax = X.abs().amax(dim=-1, keepdim=True)  # [B,T,1]
scale = amax / 127                       # B*T 个独立 scale
```

同一位置 t 的样本A最大值1、样本B最大值100，分别采用1/127、100/127；B的异常值不会扩大A的scale。独立计算不意味着数值必须各不相同，也可能碰巧相等。

`reshape(B*T,C)` 后逐行求范围仍然是独立per-token。部署框架用packed `[N_tokens,C]` 时也是每个实际token一行。所查实现：QuaRot逐行 `min(1)/max(1)`；FlatQuant保留独立行/组后 `amax(-1)/amin(-1)`；vLLM参考实现per-token用 `amax(dim=-1)`，per-tensor才用全局 `.max()`。[R4][B1][B2]

### 6.2 激活 group128 同样逐样本、逐token独立

C可整除128时：

```text
X：[B,T,C]
分组：[B,T,C/128,128]
scale：[B,T,C/128,1]
```

不同样本的同一组号不共享scale。组边界可以相同，但scale来自各自的数据；权重无batch/token轴，是另一种情况。learned clipping系数可以共享，不意味着最终动态scale共享。

| 对象/方式 | 一个scale覆盖范围 | 逻辑scale索引 |
|---|---|---|
| W per-output-channel，W[O,I] | 一行全部I | s[o] |
| A动态per-token，A[B,T,C] | 一个实际token全部C | s[b,t] |
| A动态token-group-g | 一个token内的g个特征 | s[b,t,c//g] |
| KV动态token/head，KV[B,T,H,D] | 一个token、一个head的D | s[b,t,h] |
| KV动态token/head/group-g | 上一行再沿D切组 | s[b,t,h,d//g] |
| KV静态per-head | 对应head跨token/batch共享 | s[h] |

### 6.3 怎样才会发生担忧中的batch污染

```python
amax = X.abs().amax(dim=(0,2), keepdim=True)  # [1,T,1]
```

这会让不同batch相同t共用范围，不是上面核验的动态per-token。普通decode的T=1时，它会退化成整批per-tensor；prefill的T>1时仍保留逐位置区别，不完全等于整tensor一个scale。

复现核对点：**实际张量轴、reshape、amax/amin的reduce轴、scale广播轴**。有些fake-quant会把一个scale展开复制成输入同形状，不能只看最终scale张量大小判断粒度。

## 7. 接收Agent应保留的最小配置记录

这是一份对照字段，不要求据此新增功能或补齐所有后端组合。没有原始证据的字段标“未知”，不要用当前默认值代替。

| 类别 | 应记录的信息 |
|---|---|
| 模型与基线 | 模型ID/revision、原始dtype、tokenizer、chat template、评测任务/shot/采样和seed |
| W/A/K/V | 数值格式、对称性、静态/动态、逻辑轴/组大小、scale与zero-point定义、clipping |
| Q/P | 是否新增量化；Q量化位置与scale；PV是归一化P还是局部指数权重；Softmax/累加精度 |
| 算法与豁免 | 平滑/旋转/学习变换、LWC/LAC、前缀、router/GDN/lm_head等排除项 |
| 标定/导出 | 数据、长度、样本数、epochs、TP、校准与运行实际归约轴是否一致 |
| 阶段与硬件 | Prefill/decode实际backend、设备、版本、head_dim/GQA、普通decode或多token验证 |
| 性能 | 输入/输出长度、并发、TTFT、ITL/TPOT、输出吞吐；变动是否包含KV格式/调度/输出dtype |
| 精度 | 单项Δ、mean(Δ)、mean(max(0,Δ))、最差项和评测噪声；区分fake quant与真实kernel |

判断原则：**粒度可能解释复现差距，但不是已确认根因。固定KV8时，16-bit attention是有效的精度/部署基线，Q8/P8是否值得取决于目标卡、模型和负载。** 原公开数据不能替目标项目决定路线，也不能从不同实验中扣减/拼接一个“净计算收益”。

## 8. 外部来源与固定版本

这些链接提供独立核查入口。源码快照用于解释特定数值路径，不是目标项目的依赖锁；在线论文/博客的完整原始run文件未必公开。

[A1]: https://github.com/InternLM/lmdeploy/blob/0a101134baaa0db32fe5d13d3069d1b9e90ca4a7/lmdeploy/pytorch/kernels/cuda/pagedattention.py#L444-L548 "LMDeploy KV解量化及QK/PV"
[A2]: https://github.com/vllm-project/vllm/blob/072c0886d53d933a7e1d3da7e8a3a91670cf181d/vllm/model_executor/layers/attention/attention.py#L467-L521 "vLLM query在线量化"
[A3]: https://github.com/thu-ml/SageAttention/blob/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5/sageattention/triton/attn_qk_int8_per_block.py#L41-L68 "SageAttention INT8 QK与FP16 PV"
[A4]: https://github.com/vllm-project/flash-attention/blob/b273a02621d3c94bc747df67204f591b67f38fae/hopper/softmax.h#L63-L153 "FA3 指数缩放、分母及输出归一化"
[A5]: https://github.com/flashinfer-ai/flashinfer/blob/5b0f0ca978ef28d05d71ee78637c3e445f32ddca/csrc/xqa/mha_sm90.cu#L1657-L1688 "Hopper XQA内部Q16转Q8；同文件含FP8计算"
[A6]: https://github.com/flashinfer-ai/flashinfer/blob/5b0f0ca978ef28d05d71ee78637c3e445f32ddca/flashinfer/xqa.py#L387-L430 "XQA条件分派"
[A7]: https://github.com/NVIDIA/TensorRT-LLM/tree/3163e66b2c5b8e5de2995f87dafb06ef62148446/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention "TensorRT-LLM XQA固定源码目录"
[P1]: https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079 "SqueezeBits，2024-11-18，精度与Figure8吞吐"
[P2]: https://github.com/NVIDIA/TensorRT-LLM/blob/201135e58aa525af7e523d091d4c9584229524bc/cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp#L268-L271 "TRT-LLM v0.13.0 FP8 Context FMHA输出dtype"
[P3]: https://nvidia.github.io/TensorRT-LLM/1.3.0rc27/blogs/tech_blog/blog16_Accelerating_Long_Context_Inference_with_Skip_Softmax_Attention.html#kernel-performance "NVIDIA H200/B200无Skip Softmax基线"
[P4]: https://vllm.ai/blog/2026-04-22-fp8-kvcache "vLLM FP8 KV与attention：准确率、H100/B200性能"
[P5]: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html#low-precision-attention "FlashInfer多卡低精度decode实验"
[P6]: https://arxiv.org/html/2407.08608v2#S4 "FlashAttention-3算子实验"
[R1]: https://arxiv.org/html/2410.05265v1 "PrefixQuant v1：§4.1、Table7、Table18"
[R2]: https://arxiv.org/html/2504.04823v2 "Quantization Hurts Reasoning? v2"
[R3]: https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/scripts/quantization/flatquant.sh#L9-L29 "FlatQuant公开复现脚本"
[R4]: https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/flatquant/flatquant/quant_utils.py#L49-L118 "FlatQuant动态量化函数"
[R5]: https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/flatquant/flatquant/flat_linear.py#L9-L46 "FlatQuant TP分组、LWC"
[R6]: https://github.com/ruikangliu/Quantized-Reasoning-Models/blob/bf947e29f52e3f666e3263efac149dae0ac18d00/methods/flatquant/flatquant/model_tools/qwen_utils.py#L133-L148 "FlatQuant Qwen KV校准wrapper"
[R7]: https://github.com/ChenMnZ/PrefixQuant/blob/60b74237cf2cf20e3354569d998ab708ff598ce3/main.py#L212-L315 "PrefixQuant前缀与MSE初始化"
[R8]: https://arxiv.org/html/2410.05265v2 "PrefixQuant v2高精度前缀说明"
[R9]: https://github.com/ChenMnZ/PrefixQuant/blob/60b74237cf2cf20e3354569d998ab708ff598ce3/utils/quant_utils.py#L20-L88 "PrefixQuant EMA统计"
[R10]: https://github.com/NVIDIA/TensorRT-LLM/blob/201135e58aa525af7e523d091d4c9584229524bc/tensorrt_llm/quantization/layers.py#L1189-L1242 "TRT-LLM FP8Linear张量级scale"
[R11]: https://github.com/NVIDIA/TensorRT-LLM/blob/201135e58aa525af7e523d091d4c9584229524bc/tensorrt_llm/quantization/quantize.py#L262-L267 "TRT-LLM每层KV scalar"
[R12]: https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8/blob/60d80c83c53c3b611c642dbb8c942b3f90c5948a/config.json "Qwen3官方历史配置"
[R13]: https://huggingface.co/Qwen/Qwen3.5-27B-FP8/blob/2e1b21350ce589fcaafbb3c7d7eac526a7aed582/config.json "Qwen3.5官方历史配置"
[R14]: https://huggingface.co/docs/transformers/v4.57.0/en/quantization/finegrained_fp8 "细粒度FP8的W块与激活组定义"
[R15]: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html "Transformer Engine MXFP8分组与E8M0"
[B1]: https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/fake_quant/quant_utils.py#L121-L178 "QuaRot per-token/group的归约轴"
[B2]: https://github.com/vllm-project/vllm/blob/072c0886d53d933a7e1d3da7e8a3a91670cf181d/vllm/model_executor/layers/quantization/input_quant_fp8.py#L202-L263 "vLLM参考实现per-token、per-tensor、per-group"

- 算法路径：[A1]–[A7]；性能：[P1]–[P6]；低掉点配方：[R1]–[R15]；batch归约：[B1]、[B2]。
- 更细的本地专题可按需附送，但本文不以其存在为前提：`attention-quantization-equivalent-algorithms-2026-10-18.md`、`attention-precision-performance-evidence-2026-10-18.md`、`w8a8kv8-granularity-recipes-2026-10-19.md`。

## 9. 可随文转交的一段话

> 你已读过旧的8位量化参考。本文件是之后的增量证据和解释修正，请更新现有理解，尤其注意：W8A8KV8不保证Q/P为8-bit；per-token不跨batch共享动态scale；PrefixQuant有高精度前缀边界；Qwen官方历史配置尚未与原博客跑分revision绑定；KV压缩与低精度计算的联合收益不能当作额外Q8/P8的净收益。请按目标项目的实际配置与源码分派判断适用性，不把未知字段补成默认值，也不把这份资料当作已批准的实现计划。
