# 案例：MoE评测的执行组织、流水线与收口

> 真实项目记录，数字只适用于该模型、设备、样本和协议；用来说明方法，不是通用倍数承诺。

## 2026-10-17：起点

- 模型：GLM-4.7-Flash（46 组专家，6264 个 Linear）与 Kimi Linear（26 组，20257 个 Linear）。
- 评测：HellaSwag 前 4 题 / 16 个候选，batch 1，max_length 1024；模型已常驻，不含加载和权重量化。
- 路径：eager 逐专家前向 + 伪量化 hook；计算仍是原始浮点 dtype。
- 硬件：2×64GB NPU（Ascend 910B2），BF16 计算。

## 根因

- GLM eager 每个活跃专家开头 `torch.where(mask[index])`（单参数动态索引）→ 等待同流已排队任务完成，逐专家打断异步流水线。
- 每层还有 `mask.sum(...) > 0` 后接 `nonzero().flatten().tolist()` 取活跃专家；逐专家小 GEMM，设备吃不满。
- Kimi 已排序分组，只在专家循环前 `counts.cpu().numpy()` 一次；不能把 GLM 的逐专家同步结论套到 Kimi。
- 先用定向探针确认，不靠算子数量推测。

## 改法

1. 在线路径按专家排序 token，计算 offsets，一次 `torch._grouped_mm`。
2. 权重首次 stack 成 packed tensor；每个原 Linear 的 `weight.data` 指向 packed 切片，共享 storage，不复制第二份。
3. 激活伪量化按专家分组、批量舍入；设备 reduce 回退 CPU，就用补零分组 + amax/amin，避免逐专家同步。
4. 校准/权重量化仍走 eager + hook；只在评测上下文切换 forward，退出恢复。
5. 各专家、各投影的权重量化范围和 SmoothQuant 尺度保持独立，统计口径不变。

## 结果

| 模型 | eval 总时间 eager→grouped | 评分循环 eager→grouped | 覆盖 |
|---|---:|---:|---|
| GLM-4.7-Flash | 186.52→69.10 s，2.70× | 151.95→37.09 s，4.1× | 6264 Linear，46 组 |
| Kimi Linear | 310.59→84.42 s，3.68× | ~272→49 s，5.5× | 20257 Linear，26 组 |

- 单层合成输入：FP8 174.52→16.17 ms；HIF8 per-channel 非对称 160.64→5.48 ms；单层不外推整模。
- 同权重、同 4 题/16 候选，acc/acc_norm 一致；仅链路验收，不是格式结论。

## 第二步：内存与 batch

- 去重权重 29,943,390,976 个 BF16 参数 = 55.774 GiB；原地量化 + packed 切片共享 storage，没有第二份原权重。
- 两卡各剩 27–31 GiB；batch 1 不是容量限制。
- 纯打分 `use_cache=True` 的 cache 每次 forward 丢弃；改 `use_cache=False` 后最长输入 logits 逐位一致。KV 本身只有 159.73 MiB（batch1/174 token）。
- 全词表 logits/log-probs 在 batch8/174 token 时各 411.21 MiB；词表 154,880。先切片再 log_softmax 是后续优化。
- 真实 16 候选：batch1 37.11s → batch8 6.05s（6.13×），峰值仅比基线多 851/437 MiB。
- 但 batch8 相对 batch1 的最大对数似然绝对差 4.0（batch2 为 5.0）：动态定标混入跨样本和 padding。容量可行 ≠ 原协议数值不变。
- 实际输入长度：最短 9、中位 37、P90 98、P99 119、最长 174 token；配置上限 1024 不是实际最长输入。

## 第一轮边界

- 没有为提速改量化数值代码或自研核。
- 没有把单层 10× 外推整模。
- 没有在活动 512 评测上直接改 batch；先做独立探针，确认协议影响。
- 没有把 `reserved` 高当成泄漏，也没有把参数 + buffer 相加当权重。
- 没有把 batch8 的 6.13× 当成完整 512 的加速倍数。

## 第一轮可迁移的结论

- 该轮最大杠杆是去掉逐专家Python循环与同步；后续仍需查真实底层实现，不能据此认定单算子不值得优化。
- 权重去重和 `use_cache=False` 先腾空间/减工作，再把 batch 拉起来。
- 每个优化都同输入对照；batch 改变定标时单独标注，不混入精度结论。
- 先用 PyTorch/厂商现成的 grouped GEMM；没有自研 kernel。
- 定向探针比算子计数可靠：同一个模型里的同步点，不能想当然套到另一个模型。

## 2026-10-19/20：grouped接口不等于原生融合

- 相同设备的PyTorch 2.9／torch-npu 2.9.0.post2，`torch._grouped_mm`无专用设备实现，走通用展开；两次真实前向184次调用，内部共11776次`aten::mm`
- 换`torch_npu.npu_grouped_matmul`，trace确认184次原生GMM替代展开。GLM同384题、等长无padding、batch上限8，BF16评分320.99→122.09s；HIF8 W8A8KV8另验567.32→419.79s
- 两种后端有整模数值漂移：BF16 6题改选项、HIF8 7题改选项，各2得2失；总分相同不等于逐题一致。按项目约定接受浮点算子差异，后续方法比较固定后端

## 2026-10-20：双卡流水线的独立收益

- `balanced`把前后层放两卡，原先每批顺序走完；改为两线程、两个完整batch在途，前段处理下一批时后段处理上一批
- 固定原生GMM、同一阶段切分、384题／231批，串行→重叠→串行评分358.88／224.94／361.93s，纯调度1.60倍；全部1536个选项响应相同，trace确认设备任务重叠
- logits留输出卡评分等整理也有收益；不能把567→225s的2.52倍全部称流水线收益。与其他分支Triton收益尚未统一验收，不相乘独立倍数

## 2026-10-20：大矩阵压测与真实文本规模

- 单卡连续8192×8192 BF16 GEMM的AICore约98%，验证设备／监控可用；这个理想负载不代表模型运行应满载
- GLM双卡native GMM＋流水线、固定batch8，两个真实WikiText batch的嵌套前缀预热后约45秒重放；含prefill和全词表log_softmax/gather，不是harness任务吞吐

| 长度 | BF16 token/s | BF16 AICore：两卡 | reference HIF8 token/s |
|---|---:|---:|---:|
| 128 | 5,709 | 15.6／16.7% | 1,527 |
| 512 | 18,851 | 32.6／38.1% | 3,101 |
| 2048 | 9,817 | 23.9／23.5% | 2,730 |

- 专家GMM平均行数64→256→1024，支持小矩阵影响；更长却不总更快。HIF8长输入AICore约5%，AIVector约76–82%，不能把低AICore当整卡空闲或单纯矩阵小
- HIF8 2048统计补零缓冲区曾OOM；新进程使用`PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`后跑通，512锚点评分相同、吞吐差−0.40%。该后端与形状历史的经验不是通用分配器保证
- 2048 HIF8仅9批，填充／排空比例更高；不把窗口吞吐下降全当稳态退步。测试的是reference QDQ，不外推Triton组合
- 另修持久打包权重误继承inference属性：在`inference_mode(False)`下创建并绑定共享切片，恢复后可继续no_grad原位量化；无需复制第二份模型

## 为什么在此收口

已经覆盖冗余、内存、批次、真实算子、可复用融合核与跨卡调度等层面，但它们不是每个项目的必修顺序。正确性与复现通过、等待已支持研究节奏后，采用可靠版本回算法主线；“还能更快”不构成继续投入的义务。新工作负载或实际等待重新成为阻塞时再定位。

来源项目报告：`docs/experiments/npu67-eval-efficiency.probe.report.md`、`docs/experiments/npu67-long-input.probe.report.md`；融合核的独立证据见`kernel-decision.md`。路径仅作案例溯源，不是使用本skill的依赖。
