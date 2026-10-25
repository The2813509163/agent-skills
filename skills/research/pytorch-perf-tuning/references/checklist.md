# PyTorch 性能排查清单

按当前热点查对应项，不把本页当成每次必须跑完的工作流。复用已验证配方，少量尝试得到可靠收益后回到 PyTorch 算法研究；完整工具调研另开任务。

## 1. 症状 → 先查什么

| 症状 | 先查 | 常见处置 |
|---|---|---|
| 设备利用率低，时间线有大段空白 | host sync、Python 循环 | 去 `.item()/nonzero()`，批量/分组 |
| 单步很多小 kernel | 逐 expert/head/segment 循环 | grouped/batched/fused |
| batch 1 就 OOM 或余量很小 | 权重重复、全量 logits、KV | 去重、`use_cache=False`、切片 |
| batch 增大吞吐不涨 | 已饱和、CPU 瓶颈、padding | 测 tokens/s、分桶、profiler |
| 首次慢、后续快 | 编译/warmup/缓存 | 预热、缓存、分开计时 |
| 多卡扩展差 | 真实放置、阶段顺序、共同空白与传输 | 按证据分片／流水线；分层驻留不等于并行调度 |
| 大GEMM能满载、模型不满 | 实际batch/长度、专家行数、向量与访存 | 必要时做有界规模对照，不把压测当整模目标 |
| 只有 eval 慢，训练正常 | eval 路径/hook/指标 | 关 hook、离线准备、批量指标 |

## 2. 同步点和循环 grep

```bash
rg -n '\.item\(\)|\.cpu\(\)|\.tolist\(\)|\.numpy\(\)|nonzero\(\)|synchronize\(\)|empty_cache\(\)' src
rg -n 'for .* in .*expert|for .* in .*head|for .* in .*layer|for .* in .*segment' src
```

- 单参数 `torch.where(mask)` 是动态索引，通常触发同步；三参数 `torch.where(cond, a, b)` 不是。
- `mask.sum(...) > 0` 后接 `.nonzero().tolist()` 是典型逐层/逐专家同步。
- 日志、assert、tqdm 里的 tensor 取值也算。
- 先判断在不在热循环；不在热路径的 `.cpu()` 不用管。

## 3. 普通Event计时示例：首尾标记包围一批调用

以下是plain微基准起点，默认100次可按任务调整；先复用项目可信计时器。专业dummy覆盖、profiler交叉验证和平台计量按独立领域库处理，不把此示例当唯一验收合同。

```python
import statistics
import torch

def bench(fn, backend, warmup=10, repeat=5, iterations=100):
    for _ in range(warmup):
        fn()
    backend.synchronize()
    times = []
    for _ in range(repeat):
        start = backend.Event(enable_timing=True)
        end = backend.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / iterations)
    return statistics.median(times), times  # milliseconds per call
```

- `backend=torch.cuda` 或 `torch.npu`，先选择实际设备；当前流上记录事件。多流工作须显式汇合依赖后记录 end。
- 计时循环内不插 Event、同步、日志或 `.item()`；编译、数据搬运和正确性校验在外面。除以`iterations`得批内摊销均值，不是单次延迟分布。
- 小核的 Event 区间可能含 Python 喂不满队列的空隙；这不是纯 kernel 时长。可把 100 次调用捕获成图，预热后用一对 Event 包围一次 replay，再除以 100。
- 图捕获、编译不计入稳态，但保留冷启动完整成本；核对图输出仍与参考一致。eager/plain与graph分别比较，禁止混算加速比。
- 训练 step 若有权重更新，应准备等价状态再重放，不能直接套固定输入的纯前向模板。

## 4. 显存模板

```python
torch.cuda.reset_peak_memory_stats()   # NPU: torch.npu.reset_peak_memory_stats()
# run
print("allocated", torch.cuda.memory_allocated() / 2**30)
print("peak     ", torch.cuda.max_memory_allocated() / 2**30)
print("reserved ", torch.cuda.memory_reserved() / 2**30)
```

- `allocated` 是活张量；`reserved` 是缓存池；`max_allocated` 是峰值。
- 权重去重：按 `(device, untyped_storage().data_ptr())` 计一次 `storage.nbytes()`；tensor 切片偏移不是新 allocation。
- 会再次原位量化的持久packed张量在`inference_mode(False)`下创建、绑定；共享切片不能意外成为仅限inference上下文写入的tensor。
- 阶段之间按需释放无用引用／`empty_cache()`，再测下一阶段峰值；不把清缓存当每批必做项。
- `reserved - allocated` 不能解释成另一份模型或全部临时张量。

## 5. Profiler

仅在定向计时不足以定位时使用；已能决定下一步就不扩大采集。

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True, profile_memory=True) as prof:
    fn()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))
```

- NPU 用厂商 profiler 或 `torch_npu.profiler`；活动名按实际后端。
- 先看 `self_*_time_total` 和前几个 kernel；再看时间线空白（host 等待）。
- `with_stack=True` 很重，只在定位 Python 调用时开。
- profiler 本身会同步/加开销；不要用它产出的总时间当加速结论。

## 6. 少量 Batch 候选

```python
for batch in (1, 2, 4, 8, 16):
    try:
        ms, _ = bench(lambda: step(batch), backend=torch.npu)
        print(batch, batch * 1000 / ms, peak_memory())
    except torch.OutOfMemoryError as e:
        print(batch, "OOM", e)
        break
```

- 示例候选按实际预算缩减，不要求全跑；吞吐收益够用即停。`step`、`peak_memory` 为工作负载自己的函数；只捕获明确的 OOM，其他错误原样抛出。
- 指标用有效 tokens/s 或 samples/s，不把 padding token 计入吞吐。
- 变长输入固定同一组样本，按长度排序后再切 batch。
- 记录数值变化：同一样本在不同 batch 下的 logits/指标差异。

## 7. 无用张量审计

按阶段列出活张量，逐项问“下一步还需要吗”：

- 权重：原权重、转换/量化权重、FP32 master、打包副本、转置副本。
- 优化器：eval 时的 optimizer state、梯度。
- 输入：padding 到 max_length、未截断的长样本、重复 tokenize。
- 中间：KV cache、全词表 logits、hidden states、attention 矩阵、mask。
- 输出：持有 cache 的 output 对象、逐样本大张量、profiler trace。
- 缓存：response cache、dataset encoding、编译产物。

处置优先级：不创建 > 复用 > 原地覆盖 > 共享 storage > 阶段边界释放。

## 8. 串行/并行审计

- 画依赖图：哪些计算只依赖输入，不依赖前一步？
- 可合并：同一输入的 QKV、gate_up、多个 head 的投影。
- 可分组：逐 expert/segment 的 GEMM → grouped GEMM。
- 可批处理：独立请求/样本 → 一个 batch。
- 可重叠：数据加载/预处理→计算、独立分支→多stream、不同完整batch→跨卡阶段流水线。
- 同一batch的层有严格顺序，不代表不同batch不能重叠；小算子并行可能被下发成本抵消，按端到端结果取舍。
- 线程的设备／inference／AMP／样本上下文、跨流依赖、tensor寿命和在途数量须正确；同阶段切分串行对照＋真实trace见`multi-device-pipeline.md`。

## 9. Batch 语义检查

- 动态量化/定标是否按 batch 统计？padding 是否进入统计？
- BatchNorm 训练态用 batch 统计；eval 态用 running stats。
- 归约顺序变化是否改变浮点结果？是否在容差内？
- dropout/RNG 是否在 eval 关闭？数据顺序是否影响结果？
- 响应缓存 key 是否包含 batch/协议/源码版本？

## 10. 现成 fast path 清单

- `torch.nn.functional.scaled_dot_product_attention` / flash attention。
- `torch._grouped_mm`（按实际 PyTorch 版本确认）；厂商 grouped matmul。
- `torch.compile` / graph mode；厂商图模式。
- fused optimizer / fused layernorm / rmsnorm。
- varlen attention / sequence packing。
- 上游模型的 optimized forward（例如 HF 的 grouped MoE 实现）。
- 先查版本、实际设备dispatch与实现，再在目标设备测数值/性能；grouped/fused接口可能仍逐项展开，不能只凭函数名或导入成功。

## 11. 报告模板

| 项 | 值 |
|---|---|
| 设备 / 版本 | ... |
| 输入 / batch / dtype | ... |
| 物理卡 / PID / 观测阶段与窗口 | ...；其他作业及缺失字段明确标注 |
| 设备忙碌度 / 功率W / 温度或限制 | 同阶段均值／范围；保留字段定义 |
| 有效 tokens/s 或 samples/s | ...；不计 padding |
| load | ... |
| offline prepare | ... |
| warmup | ... |
| scoring loop | ... |
| total | ... |
| 峰值 allocated / reserved | ... |
| 设备内存 / 必要驻留 / 可确认冗余 | 按 storage 去重；不把缓存池全算死内存 |
| 数值对齐 | 逐位 / 容差 / 指标一致 |
| 协议变化 | 无 / batch 定标变化 / ... |
| 判断与证据 | 可避免浪费／当前口径合理受限／证据不足；下一步只选最小必要动作 |
