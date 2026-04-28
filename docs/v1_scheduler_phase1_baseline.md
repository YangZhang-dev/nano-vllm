# V1 调度器第一阶段基线

## 当前请求生命周期

1. `LLM.add_request(prompt, sampling_params)` 会对 prompt 进行分词，创建一个 `Sequence`，将其推入 `Scheduler.waiting`，并返回 `seq_id`。
2. `LLM.step()` 调用旧版调度器：
   - `Scheduler.schedule()` 返回 `(seqs, is_prefill)`。
   - `ModelRunner.run(seqs, is_prefill)` 执行 prefill 路径或 decode 路径。
   - `Scheduler.postprocess(seqs, token_ids, is_prefill)` 更新已计算 token，在允许时追加采样 token，并释放已完成序列的 KV block。
3. `LLM.generate()` 仍然是围绕 `add_request()` 和 `step()` 的离线循环。

在这一阶段，调度器有意保持 phase-first 模式。如果调度了任何等待中的 prefill，则整个 step 都是 prefill，并且该 step 会跳过 decode。

## 兼容性接口

第一阶段在不改变调度器语义的前提下，新增了一套兼容性数据模型：

- `ScheduledItem`：某个 step 中调度的单个序列，包含 `seq`、`num_scheduled_tokens` 和 `kind`。
- `ScheduledBatch`：调度项列表，以及 `prefill_tokens`、`decode_tokens`、`is_mixed`、`num_scheduled_tokens` 等聚合属性。
- `StepMetrics`：从 `ScheduledBatch` 派生的逐 step 计数器，以及调度器抢占次数。
- `StepOutput`：`LLM.step()` 新增的结构化返回值。

`StepOutput` 仍然兼容旧的调用模式：

```python
outputs, num_tokens, seqs = llm.step()
```

新代码应优先使用：

```python
result = llm.step()
outputs = result.outputs
metrics = result.metrics
scheduled_batch = result.scheduled_batch
```

## 需要保留的后处理规则

- Prefill chunk 未完成：更新 `num_cached_tokens`，不要追加采样 token。
- Prefill 完成：追加采样 token，并让序列进入正常的完成检查流程。
- Decode：只追加一个采样 token。
- 已完成序列：释放所有 KV block，并将其从 `running` 中移除。
- Prefix cache 哈希只覆盖已经完整计算的 block。

## 混合批处理边界

第一阶段不实现混合 prefill/decode 执行。

计划中的 MVP 边界如下：

- 纯 prefill batch 继续使用现有的 varlen prefill 路径。
- 纯 decode batch 继续使用现有 decode 路径，并在启用时使用 CUDA graph replay。
- 未来的混合 batch 将使用 eager varlen 路径。
- 混合 CUDA graph capture 不在范围内。

当前阻塞点是全局 `Context.is_prefill` 标志，以及 `Attention.forward()` 和 `ParallelLMHead.forward()` 中的 prefill/decode 分支。

## 基准测试基线

`bench_latency.py` 记录：

- TTFT
- TPOT
- 端到端请求延迟
- 输出吞吐量
- prefill 和 decode 吞吐量
- 每个 step 的平均调度 token 数
- 抢占次数
- prefill/decode/mixed step 计数

该基准测试不再读取 `llm.scheduler.waiting[-1]` 来获取请求 ID，而是使用 `LLM.add_request()` 返回的 `seq_id`。
