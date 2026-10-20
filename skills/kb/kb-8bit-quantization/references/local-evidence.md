# 本地实测账本：配置与掉点一起读

证据期：2026-10-17～20。模型GLM-4.7-Flash；Ascend 910B2、CANN9.0.0，PyTorch研究路径。W/A/KV均为伪量化，GEMM与cache实际存储仍BF16；不代表原生8-bit部署成绩。

## 1. 共用评分协议

- HellaSwag validation、ARC-Easy test、ARC-Challenge test各512，seed42固定随机同题；zero-shot、无chat template、max_length1024。
- 指标：三任务acc_norm等权均值，单位%；掉点=`BF16−候选`，单位pp。
- 唯一BF16基线：417/419/292题正确，81.4453/81.8359/57.0313%，均值73.4375%。后续复用；核对真实题目三哈希，不按曾错标的旧doc_id盲连。
- 512是反复使用的研发集，所有相关正式验收仍为null。128是其中子集；n4只是链路。
- 每512配置1536题、6141候选评分请求。新路径去重5680实际输入、279579有效token、763个无padding forward；batch上限8。

## 2. 旧W/A per-tensor：六候选的完整条件

权重：每Linear/专家矩阵对称Amax RTN。A：动态per-tensor，专家/投影独立。KV：逐token/head，写入/读出均QDQ，历史不重定标；Q/P BF16。batch1、旧torch grouped后端。

σ配方：A均值中心、Δ=√2σ；KV均值中心、Δ=.625σ。Amax配方零中心。**换σ也换了中心，不是仅统计量单变量消融。**

| 方法 | A/KV定标 | 均值 | BF16掉点 |
|---|---|---:|---:|
| RTN | Amax | 69.0755 | 4.3620 |
| RTN | σ | 72.3958 | 1.0417 |
| 真融合SQ，α=.5 | Amax | 70.1823 | 3.2552 |
| 真融合SQ，α=.5 | σ | 70.7031 | 2.7344 |
| 配对QuaRot，seed42 | Amax | 69.6615 | 3.7760 |
| 配对QuaRot，seed42 | σ | 71.4844 | 1.9531 |

- RTN σ比Amax高3.3203pp，93得42失，描述性95%区间[1.8229,4.8177]。
- QuaRot σ比自身Amax高1.8229pp，67得39失，区间[.5859,3.125]。
- SQ σ比RTN σ低1.6927pp；不能据此否定其他粒度的SQ。本轮不是A per-channel，不能用“逐channel缩放抵消”解释所有结果。
- SQ训练校准64×256，188Norm共享组，补偿所有消费者含router；没有在线X/s。
- QuaRot特殊GLM head/低秩布局：95Norm、3179消费者、2991down；V/o使用H20⊗H256，RoPE后完整256维QK配对。

来源：`deployable-w8a8kv8.final.report.md`，发布33a267d；逐任务CSV随知识库保存为比较组`legacy_tensor_methods`。

## 3. 旧QKVP诊断与A per-channel，不能混成同一组

### 旧六组诊断：W per-output-channel、batch1

| 配方 | 均值 | 解释 |
|---|---:|---|
| Linear Amax | 70.7031 | 只量化Linear W/A，Attention不额外QDQ |
| Linear σ | 72.3958 | A per-tensor均值σ |
| Linear minmax浮点中心 | 27.2135 | 旧中心偏移实现的真实大幅退化，不是所有非对称INT8的结果 |
| Linear σ＋QKVP σ | 72.1354 | 跨token整head Q/K/V及query-row P；不是token/head KV8 |
| Linear σ＋QKVP Amax | 69.0755 | 同上，额外QKVP节点换Amax |

### 后续RTN per-channel：W仍per-tensor

A每序列×输入channel，MoE每expert×序列×channel；KV固定逐token/head .625σ，Q/P开关。batch8、逐样本统计、无padding。

| Q/P | 正确数HS/AE/AC | 均值 | BF16掉点 |
|---|---|---:|---:|
| BF16 | 405/411/296 | 72.3958 | 1.0417 |
| HIF8 QDQ，仍BF16乘法 | 403/408/287 | 71.4844 | 1.9531 |

额外QP8净少14题：21得35失，候选降低.9115pp；区间触零。这支持本轮保留Q/P BF16，不是原生INT8/FP8累加性能实验。

**三个72.3958%不是同一次实验**：旧Linear σ不额外量化Attention；RTN σ是per-tensor A＋KV8；channel_kv8是per-channel A＋KV8。平均数相同不表示配置或逐题答案相同。

来源：`hif8-std-e2e.stage2.report.md`、`perchannel-qp.stage1.report.md`（c1f1eb6）。

## 4. 精度优先：W per-output、A per-token并不自动提点

128/任务，W为对称Amax，A均值√2σ，KV .625σ、QP BF16、batch8/reference。

| W | A | 正确/384 | 均值 |
|---|---|---:|---:|
| per-tensor | per-channel | 275 | 71.6146 |
| per-tensor | per-token | 274 | 71.3542 |
| per-output-channel | per-channel | 273 | 71.0938 |
| per-output-channel | per-token | 275 | 71.6146 |

四格未显示稳定粒度收益；8题smoke曾多1题，不能据此下结论。新轴更符合部署/理论条件，不等于这个子集必涨。

### 权重目标与A规则：128筛选

W per-output、A per-token。A0为全局mean√2σ；A1为各投影训练校准规则，需按实际量化W重拟合。L_W是权重还原MSE，L_Y是训练输入投影输出MSE。

| W选参 | A0均值 | A1均值 |
|---|---:|---:|
| Amax，对称 | 71.6146 | 72.1354 |
| L_W，对称 | 70.0521 | 71.0938 |
| L_Y，对称 | 71.6146 | 70.8333 |
| L_W，scale＋浮点center | 72.1354 | 70.5729 |

- 对称L_W真实切片重建MSE只剩Amax的约5.6%–6.4%，整模A0却低1.5625pp。局部指标不能替任务验收。
- 对称L_Y相对L_W，训练输出误差的模块中位比约.54，W重建误差约1.10；不同目标不能简单排名。
- A1在不同W上正负效果不同；没有稳定交互证据。规则在float-upstream训练输入拟合，不用未测的上游漂移替失败找借口。
- 非对称L_Y未完成并按新路线停止，不补成已完成数据。非对称L_W模块重建MSE比对称版中位约.978，也不是整模收益保证。

### 512扩样，旧torch grouped串行、Triton

| W选参 | A规则 | 正确/1536 | 均值 | BF16掉点 |
|---|---|---:|---:|---:|
| Amax | A0 | 1110 | 72.2656 | 1.1719 |
| Amax | A1 | 1116 | 72.6563 | .7813 |
| L_W对称 | A1 | 1109 | 72.2005 | 1.2370 |
| L_Y对称 | A1 | 1113 | 72.4609 | .9766 |

Amax+A1比A0为34得28失、+.3906pp，区间跨零；收益主要在ARC-Easy，ARC-Challenge净少8题。W离线优化两候选没超过Amax+A1。

来源：`hif8-precision-first.parent-stage0.report.md`、`hif8-precision-first.core.report.md/.csv`（数值39fc4f1，后续批量实现682239b）。

## 5. 新后端＋联合group＋融合SQ：本轮最终选择

固定原生`npu_grouped_matmul`＋threaded双卡完整batch流水线，Triton QDQ、无padding batch≤8。KV固定token/head .625σ，QP BF16。

| 配方 | W | A | HS/AE/AC正确数 | 均值 | BF16掉点 | 入口分钟 |
|---|---|---|---|---:|---:|---:|
| 新后端原现任 | per-output Amax | per-token A1 | 411/421/289 | 72.9818 | .4557 | 15.25 |
| 联合group128 | row-group128，L_W对称搜索 | token-group128，新A1 | 408/414/298 | 72.9167 | .5208 | 16.34 |
| 联合group64 | row-group64，L_W对称搜索 | token-group64，新A1 | 410/410/294 | 72.5260 | .9115 | 17.45 |
| 融合SQ＋group128 | 同融合后的W重新L_W搜索 | SQ后rows与新W重新A1 | 414/413/299 | **73.3073** | **.1302** | **16.88** |

- 后端切换同配方由72.6563→72.9818%，13得8失；不是量化算法收益。此后比较固定新后端。
- group128比group64高.3906pp，但区间跨零；据点估计选128继续，不宣布普遍优于64。
- 分组候选同时改变W粒度/选参与A规则；不是单因素group消融。
- SQ后188组Norm、3226 Linear＋46router融合；6264个变换后W摘要逐项核对；使用新rows重做W/A。
- 最终A规则6264条：4877 mean-σ、1377 zero-σ、10 zero-Amax；每投影共享规则，实际scale仍按token×group动态算。
- 训练数据WikiText2 raw train：fit/transfer各64×256、seed42；每投影最多64行，稀有专家额外采样到fit≥32/transfer≥16。SQ统计来自64×256的fit窗口，新rows来自融合后的模型；两套数据不得混用。
- A1初始67条规则：Amax零中心＋α=2^(k/4)、k=−12…20的零/均值σ；先8行筛选前2条，加入Amax与mean√2σ保底，再全fit比较相对融合浮点W输出。边界命中可扩展，不从任务分倒推α。
- W初始97个对数候选：Δ=(amax/32768)×2^(k/4)、k=−32…64；3个间隔盆地，1/16与1/256 octave两轮细化。当前对称group正常路径共130次候选评估；触边扩展、非对称或L_Y的预算不同。
- 覆盖6264 Linear、46原生GMM、47 KV层；n4与512规则文件完全相同。W缓存和A规则均绑定变换后实际W。

| SQ＋group128相对 | 得/失 | 提升pp | 描述性95%区间pp |
|---|---:|---:|---|
| 未融合group128 | 35/29 | +.3906 | [−.5859,+1.4323] |
| 新后端原现任 | 53/48 | +.3255 | [−.9766,+1.6276] |
| BF16参考 | 25/27 | −.1302 | [−1.0417,+.7813] |

**最差任务没有同样接近BF16**：HS掉.5859pp，ARC-Easy掉1.1719pp，ARC-Challenge涨1.3672pp；均值允许涨点抵消。正式非劣/独立确认未做，不把“区间跨零”解释为等价。

用户对这个研发结果满意，停止追加QuaRot/旧矩阵。它是有希望的HiF8配方，不是“HiF8普遍比E4M3好”的证明，也不是生产量化导出。

来源：`hif8-joint-group-sq.report.md`（16d3719），源682239b；工件标识`20261020_fused_sq_g128_v1`、`20261020_joint_group_sq_v2`。完整可迁移配置在模板和本节，原始模型/张量/缓存不随skill打包。

## 6. Proxy给过哪些正确与错误方向

| 观察 | 实测 | 可迁移教训 |
|---|---|---|
| HIF8固定共享σ规则 | 扩展Linear激活only尾部NMSE .02758→.003546；W8A8 .05448→.02933 | 便宜共享规则可以明显超过本格式Amax；代理不等于整模 |
| q99规则 | 早期HIF8 A-only .009692→.001519，W8A8 .020198→.012113 | 有效候选，但仍输该组E4M3 Amax；不能按HIF8赢否筛样本 |
| Attention联合QKVP | 尾部NMSE约.204→.0127；E4M3 Amax约.0114 | HIF8改进大不等于比E4M3更好；此处不是部署KV8 |
| σ/KL共享10σ | Linear尾部.0131，落后q99 .00765；输出误差选k达.00345 | 选参损失决定规则；KL主体可看不见关键尾部 |
| 细网格34目标、20728候选 | P/logits输出MSE中位降64%/87.5%；各有5目标最坏行退步 | oracle仍有空间，不能在线逐张量搜索或称共享规则 |
| 固定1/8锚点 | 34目标仅2个接近已测最好 | 1/8是初始化直觉，不是最佳尺度规律 |

矩阵NMSE与单行NMSE、A-only与W8A8、训练与验证、mean与worst10%不能混在同一列排名；这些数值只对应各自报告的代理集合。

来源：`activation-scale-proxy.md`、`attention-scale-proxy.md`、`hif8-kl-std-scale.stage1.report.md`、`hif8-scale-refine.probe.report.md`。极端P行、KL并列与层组恢复的详细反例见诊断文档。

## 7. 提速经验：不是量化能力涨分

| 工作 | 实测 | 限制 |
|---|---|---|
| per-channel B1→B8，同1536题 | 均值72.4609→72.3958%；入口207.2→34.2min，6.065× | 35得36失、71正确性翻转、90改答案；净少1题≠只变1题 |
| 旧Triton，96题 | 评分270→230s，1.17× | 旧B1/per-tensor倍率不可搬用 |
| 新per-channel batch8核，96题 | 264.42→204.95s，1.29× | 同候选LL一致；非512倍率，AICore仍低 |
| per-token核，48题 | 263.13→219.94s，1.196× | 评分口径；入口1.17×，不与上行连乘 |
| 原生GMM＋流水线，48题 | 评分约221→73s | 原生GMM改变8题LL、未改答案；原生串行/流水线逐项一致 |
| 同配方新后端512 | 入口40.37→15.25min，2.65× | backend改变了少量答案，不能视作数学逐位等价 |
| L_W搜索批量化 | 热态完整投影2–13s→.16–.34s，首形状约4s | 130候选全保留；新增allocated .323–.401GiB |
| SQ变换后6264W搜索，4片 | 每片约480–493s | 包含各片加载/融合重放；不是单卡时间 |

旧权重搜索group128剩1731模块续算238s、group64剩3662模块491s；**不是从零6264模块耗时**。SQ准备约17.6min也不是单次评测时间。

两个910B2上BF16去重参数约55.77GiB，没有第二份完整BF16备份；grouped buffers共享Linear storage。allocated与reserved要分开，不能把绑定切片重复计内存。

来源：`glm-eval-memory-batch.probe.report.md`、`fast-eval-error-localization.final.report.md`、`hif8-batch8-specialization.report.md`、`hif8-precision-fast-eval.report.md`、最终联合group报告。
