# 格式、尺度与归约轴

更新：2026-10-20。这里的数值是**未乘scale的码本**；张量落在哪一段，由`z=(x−c)/Δ`决定。

## 1. 数值格式：范围不是精度排名

| 格式/约定 | 最大有限正值 | 最小正值 | 正常区精度 | 主要取舍 |
|---|---:|---:|---|---|
| 对称INT8，使用−127…127 | 127 | 1 | 固定间距1；还原域间距Δ | 范围被峰值撑大时，小值一起变粗 |
| E4M3FN | 448 | 2^-9 | 3显式尾数位；最小normal为2^-6 | 常用幅值分辨率较好，指数覆盖小于E5M2 |
| E5M2 | 57344 | 2^-16 | 2显式尾数位；最小normal为2^-14 | 覆盖更广，同指数段的相对分辨率比E4M3粗 |
| HiF8，本地官方实现 | 32768 | 2^-22 | 可变尾数；[1/8,16)有3位 | 把更多精度留给中段，同时保留较宽尾部 |

- INT8有符号存储范围是−128…127；对称实现常只用−127…127，不能混淆。
- “E4M3”存在FN/FNUZ等变体；bias、NaN/Inf、signed zero、最小值/最大值可能不同。本表是常见E4M3FN，不覆盖所有FP8。
- 正常浮点区的**相对**间距大致固定，绝对间距随指数变大。E5M2大数粗，不等于它的小数相对精度更高。
- INT8硬件成熟，但FP8/MX/FP4是否更快取决于卡、内核、layout、scale与工作负载；不能称INT8在所有硬件上最优。

### HiF8的准确含义

令`e=floor(log2(abs(z)))`，本地官方参考的有效尾数：

| 指数区域 | 显式尾数位 |
|---|---:|
| abs(e)≤3 | 3 |
| 4≤abs(e)≤7 | 2 |
| 8≤abs(e)≤15 | 1 |
| 更小的有限值区域，最小2^-22 | 0 |

abs表示绝对值；有限上端截止32768。它不是“E4M3与E5M2所有优点相加”：总码点仍有限，扩大某段必然牺牲另一段。

- 正区间`[1/8,16)`：HiF8与E4M3各56个码点，E5M2为28。HiF8在此与E4M3相当，不比E4M3更密。
- 正区间`(0,1/8)`：上述码本枚举，HiF8 39、E4M3FN 31、E5M2 47；不含0与1/8。**码点个数仍不等于误差大小。**
- 官方HiF8模拟采用ties-away；普通FP8常用ties-even。后端对齐要明确舍入、溢出、subnormal与输出dtype。
- 官方HiF8参考在`abs(z)≥40960`给Inf，项目fake-quant先限到有限范围。不能把项目的饱和行为当所有HiF8实现规定。
- 范围和中段布局是格式事实；“更适合这个模型的激活”必须由相同定标预算的实测支持。

来源：Luo等，[Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/abs/2409.16626)；[官方实现](https://github.com/global-computing-consortium/HiFloat8/tree/7799d8b656fe5235ea555c38bce669d5caff60d6)，`hif8_npu/toF8.py`与`hif8_quant_op.h`。本库保留的定义对应该revision；没有用其名称推断原生GEMM支持。

## 2. 近零、稀疏与高斯：哪些直觉可用

**常见观察**：很多权重直方图中心在零附近，激活也有大量小幅值。可以用来提出假设，不能替代测量。

- 权重不同层/投影可偏斜、重尾、多峰；训练后的权重不保证高斯。混合不同channel的整体直方图也会掩盖局部分布。
- Transformer的SiLU/GELU输出通常不是大量精确零。幅值集中近零不等于结构稀疏，更不等于稀疏kernel可提速。
- 单个小值舍入通常带来较小绝对误差；许多小值相加、相消、乘大权重、进入router/Softmax或影响很小的选项margin时，仍可能重要。
- 看`Δ·R(z)`还原后的误差。移进“3尾数位区”并不自动减少原单位误差：同一个10.4可在Δ=1和16下都还原为10。
- 高频中值、少量峰值、近零长尾哪个重要，应分区累计**信号能量、误差能量和实际输出误差**，不能按元素占比定结论。

## 3. per-token/per-channel：保留与归约分开写

`Y=XWᵀ`，`X[B,T,I]`、`W[O,I]`。将实际token展平为M=B×T不改变语义。

| 对象/粒度 | 保留索引 | 归约轴 | scale形状 |
|---|---|---|---|
| A per-token | b,t | I | [B,T,1] |
| A per-channel，每序列独立 | b,i | T，仅有效token | [B,1,I] |
| A per-tensor，每序列独立 | b | T,I | [B,1,1] |
| A per-tensor，整batch共享 | 无 | B,T,I | scalar |
| W per-output-channel | o | I，即dim=1 | [O,1] |
| W per-input-channel | i | O，即dim=0 | [1,I] |
| A token-group-G | b,t,g | 组内G个I | [B,T,ceil(I/G),1] |
| W row-group-G | o,g | 组内G个I | [O,ceil(I/G),1] |
| KV token/head | b,h,t | D | [B,H,T,1] |

**“W per-channel=axis0”通常指保留输出轴，不是对axis0求amax。**

```python
# PyTorch Linear的存储布局
a_max = X.abs().amax(dim=-1, keepdim=True)  # A per-token
w_max = W.abs().amax(dim=-1, keepdim=True)  # W per-output-channel
```

外侧两个维度各有scale，统计时却都是沿inner dimension I归约。不要写成“对外侧轴归约”。A per-channel不是数学错误，只是不同配方，部署代价和SQ作用也不同。

G不整除I：最后一组只统计真实元素；即使QDQ打包需要padding，也不能把补零加入均值/方差/分位数。MoE按真实路由token取组，不把expert padding当数据。

## 4. group、block、MX：都叫分组但不能互换

| 配方 | 数据值 | 共享尺度 | 分组实例 |
|---|---|---|---|
| row-group128 | INT8/FP8/HiF8均可 | 常为FP16/FP32，具体由实现定 | 每输出行128个W；每token128个A |
| W block128×128 | 例如E4M3 | 一块一尺度 | 每组16384个权重，不是128个 |
| MXFP8 | E4M3或E5M2 | E8M0，幂次尺度 | 通常沿指定方向32值一组 |
| MXFP4 | E2M1 | E8M0 | 32值一组 |
| NVFP4 | E2M1 | 每16值E4M3尺度＋每tensor FP32总尺度 | 16值micro-block |

- MX是OCP microscaling标准家族，不是所有block FP8的别名；NVIDIA提供相应硬件/软件实现。
- NVFP4基础非负码点为`0,.5,1,1.5,2,3,4,6`，加符号；还原值还要乘两级尺度。
- NVFP4的名义平均成本约4.5 bit/value再加tensor总尺度；MXFP4约4.25，MXFP8约8.25，未含对齐、元数据与未量化层。
- 更细组减少共享范围压力，却增加scale/packing开销；更细不保证实际整模更准或更快。
- transpose、GEMM K轴、TP切分、二维权重block、scale layout都影响真实计算。不能只改`group_size`就声称实现MXFP8。
- NVIDIA的NVFP4训练recipe可能额外包含旋转、随机舍入等；这些算法不等于格式本身。原生支持通常依赖Blackwell路径，不能外推其他卡。
- 生产支持分别查W/A/KV；W/A为MXFP8或NVFP4，不代表KV也如此。

来源：[NVIDIA NVFP4介绍](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)（2026-10-20核对正文）；MXFP8与Qwen block配方的固定来源见随包2026-10-19增补§5.3。该增补的旧Transformer Engine页面此次已返回404，保留历史引用，不伪称链接当前可用。
