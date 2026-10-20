# 从廉价基线到有效变换

## 1. 把四件事分开

```text
预处理 → 定标组/scale/center → 舍入到码本 → 还原/实际低精度kernel
```

- RTN是舍入规则，不等于只允许Amax选scale。可以先MSE选scale再RTN。
- 对称浮点QDQ：`x̂=Δ·R_F(x/Δ)`；带中心：`x̂=c+Δ·R_F((x−c)/Δ)`。
- 浮点c与整数zero-point不同；均值中心可能使原来的零不能精确还原。不要直接套INT8 zero-point语义。
- FP8/HiF8原码本不是概率分布；KL需要同一坐标、分箱和重建定义，不能对“码点列表”与直方图直接求KL。

## 2. Amax、minmax、clip与scale

| 方法 | 定义/作用 | 判断 |
|---|---|---|
| Amax | `Δ=max(abs(x))/Fmax` | 极便宜、零中心、覆盖峰值；必须保留为对照 |
| minmax | 用实际上下界推导范围/中心 | 适合某些偏斜分布；跨零、zero-point和c定义要明确 |
| 百分位/clip | 比如`τ=r·amax`，超出τ的值饱和 | 牺牲尾部换主体精度；先验证尾部贡献 |
| MSE选scale | 扫Δ，计算真实还原误差，argmin | 静态W离线做；A可离线学习快速规则 |
| 固定σ规则 | `Δ=α·std(x)`，中心0或mean | 无须假设高斯；α、中心及组轴共同构成配方 |

**clip搜索与scale搜索仅在定义一致时等价。** 固定对称码本、中心、饱和与舍入时，`τ=Fmax·Δ`一一对应。若只搜`r∈[.5,1]`，就只覆盖Amax尺度的0.5–1倍；直接搜scale可以大于Amax，此时没有裁剪原峰值，却改变浮点网格位置。

- INT8可从100%、95%、…、50%的11个clip候选做便宜起点；不是所有模型的完整最优区间。
- HiF8不能直接照搬这个范围。实测两张真实W切片的最优Δ/Amax-Δ中位数约2221，属于把分布移进精细中段，并非只“裁掉上半段”。它是局部例子，不是通用乘数。
- 先覆盖合理的对数尺度范围，再少量细化；边界最优、平坦目标或退步时检查含义，不自动锁3σ、q99或1/8。
- Amax不是“最偷懒所以一定最差”：本地E4M3 Amax代理常胜过迁移的q99规则；HiF8离线W-MSE胜了局部误差却输任务分。

## 3. W离线搜索与A动态规则

| 项目 | W | A |
|---|---|---|
| 是否随输入改变 | 冻结模型后不变 | 每个实际输入改变 |
| 可用预算 | 离线多候选/center搜索 | 在线少量归约＋逐元素运算 |
| 目标 | L_W=mean((W−Wq)^2)，或L_Y=mean((XWᵀ−XWqᵀ)^2) | 校准输入上选α、中心、Amax/σ；再检查输出误差 |
| 推理阶段 | 直接用已量化W/固定参数 | 当前组计算统计，套固定规则 |
| 失效条件 | checkpoint、变换、group、格式、舍入等变化 | 上游量化/变换改变，训练输入分布或权重绑定变化 |

- L_W不需要X；L_Y要用训练校准输入。两者不同，不把“权重重建更准”当“任务必涨”。
- 分组L_Y搜索有组间输出耦合：更新每组须对**完整输出残差**评估，不能把独立局部X_gW_g误差之和冒充全投影目标。
- A规则可以逐层/投影选择，但不是每次推理逐token扫130候选。共享规则与逐组oracle分开报告。
- 先在fit集选，再用不重叠training transfer集检查；最终任务集不参与选参。float-upstream校准和quantized-upstream回放是不同证据。
- 稀有专家行数不足应补采样并记录；不能悄悄忽略专家或降低事先门槛。行数门槛是实验预算，不是通用32行定理。

### 快速实现：批候选而不是逐group Python循环

```text
W_groups [N,G]
candidate scales [N,K,1]
broadcast W [N,1,G] → QDQ [N,K,G]
沿G还原MSE → loss[N,K] → 沿K argmin
```

- 按`N_chunk×K×G`及中间张量dtype估显存，分大块；避免硬限每批64组。
- 保留原候选、tie-break、dtype、还原域和损失归约，先验证选中scale/center/各loss逐位一致，再谈同口径提速。
- 本地把L_W分块改为32Mi候选元素预算，130候选/三盆地细化不变，额外allocated约0.32–0.40GiB。不是所有卡都固定用32Mi。
- L_Y有X和投影临时量，不能直接沿用L_W显存公式。cache重载查真实W/rows/算法版本摘要，缺失模块才续算。

## 4. SmoothQuant：沿inner dimension平衡W/A

`s_i>0`，`X′=X/s`，`W′=W·s`，所以`X′W′ᵀ=XWᵀ`。常用`s_i=a_i^α/w_i^(1−α)`，a_i/w_i来自校准与权重对应输入channel统计。

- A per-token保留t，W per-output保留o；SQ在二者共同的inner dimension i改变各行内部范围，把activation峰值压力部分移向W。
- α越大一般越倾向压A、扩大W压力；不是保证每个W/A误差同时下降。α=.5只是本地本轮固定选择。
- 能把`1/s`融合进上游Norm时才称融合SQ；每Linear hook在线X/s是另一种实现和性能口径。
- 一份Norm输出的**所有消费者**必须补偿同一个s，包括未量化router、共享/路由expert、并行投影。只改选中专家会改变浮点函数。
- 先查浮点等价和覆盖，再重采变换后训练输入、重搜W和A。不得用未融合W/rows生成的缓存。
- 动态A per-channel若每channel采用齐次Amax/σ，且中心/舍入规则同样缩放，正数s的缩放会被该channel重新定标抵消。此时SQ对A QDQ的直接收益可能消失；W误差和非理想路径仍可变化。
- **旧SQ六候选是W/A per-tensor**，不能用上述per-channel抵消推导解释其全部失利。

本地GLM：188共享Norm组、3226 Linear＋46router；o/down无法按该Norm路径融合的部分不强行处理。融合logits NMSE约4.73e−5属于浮点路径验证，不是量化收益。

## 5. QuaRot及其他方法

| 方法 | 核心机制 | 易漏条件 |
|---|---|---|
| QuaRot | 正交/符号Hadamard分散离群坐标，配对补偿保持浮点函数 | 某些旋转在线；非线性、RoPE、head结构决定能融合哪里 |
| GPTQ | 校准驱动、近似二阶的顺序权重量化误差补偿 | 不是单纯clip搜索；W4效果不能外推W8 |
| AWQ | 激活感知的权重缩放/保护 | 多为weight-only路线，不自动解决A8/KV8 |
| FlatQuant | 学习变换、对角scale与clipping | 有训练校准成本，论文/脚本的A对称性可能不同 |
| PrefixQuant | 前缀与异常值处理，支持静态量化 | 后续实现可保留高精度前缀KV；不是所有cache值严格8-bit |

- 旋转可让坐标能量更分散，不保证所有样本都均匀或任务分提高。
- 旋转要配对**同一矩阵**，维度相等不够。本地GLM o输入5120：`H40⊗H128`不等于V/o实际配对的`H20⊗H256`；盲换的双精度相对MSE约2，正确配对约1.24e−30。
- RoPE后旋转Q和K以保持点积，不代表Q也必须量化。P无需默认增加低精度节点。
- 最终选择看同协议整模分、最差任务、离线成本和线上kernel支持；不按算法名字排序。

## 6. KV与Attention的真实消费路径

- 新KV量化一次后写入；历史读出不重新估scale。token/head、channel、跨token静态head不是同一分组。
- 首次prefill直接消费原KV而只在旁路写低精度cache，可能使likelihood评测完全漏测KV误差；检查实际用于QK/PV的张量。
- 短prefill＋decode需查历史bitwise不变和新写值确已QDQ；动态A跨prefill/decode块统计会改变时，不强求整模跨块等价。
- Softmax mask不参与范围。QK/PV计算可以BF16/FP16，norm/Softmax/累加保持高精度。
- 完整Softmax P的fake-quant不等于FA3的块内指数权重量化。FA3/XQA的放大常数、分母来源、online softmax与kernel分派须按固定版本查。
