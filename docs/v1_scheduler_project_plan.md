# Nano-vLLM V1 Scheduler 项目计划

## 目标

在 nano-vLLM 基础上完成一个适合面试展示的项目：

> 实现 vLLM V1 风格的 scheduler，支持 online continuous batching、decode-first token-budget scheduling、chunked prefill，并补充 benchmark 分析。

这个项目的目标不是向原仓库贡献 PR，而是快速做出一个能讲清楚、能运行、能展示指标的系统项目。

## 当前问题

nano-vLLM 现在已经有一些重要基础：

- `waiting` / `running` 队列
- KV cache block manager
- prefix caching
- CUDA graph decode 路径
- 初步 chunked prefill 支持
- offline `generate(prompts, sampling_params)` API

但当前 scheduler 仍然偏阶段式：

```text
优先调度 prefill
只要本轮有 prefill，整轮就是 prefill
只有 waiting 队列为空时才进入 decode
```

这会导致长 prompt prefill 阻塞 decode 请求。在 online serving 场景下，这会伤害 TTFT、TPOT/ITL 和端到端请求延迟。

vLLM V1 的核心思路是：不再严格按 prefill/decode phase 调度，而是按每个请求的 token 进度调度：

```text
每个 request 记录已经计算了多少 token
每个 step 有固定 token budget
优先调度 decode 请求
剩余 budget 用于 prefill chunk
长 prompt 拆成多个 step 计算
```

## 分支策略

建议创建独立分支，方便后续做干净的 benchmark 对比。

推荐结构：

```text
main 或 upstream baseline
  -> codex/bench-baseline     # 只添加 benchmark 和文档
      -> codex/v1-scheduler   # 实现 scheduler 改动
```

这样做的原因：

- `codex/bench-baseline` 可以用同一套 benchmark 跑原始 scheduler。
- `codex/v1-scheduler` 可以在相同 benchmark 上对比新 scheduler。
- 避免用“有 benchmark 的新分支”和“没有 benchmark 的旧分支”做不干净的对比。

建议命令：

```powershell
git status
git checkout -b codex/bench-baseline
# 只添加 benchmark/docs，然后提交
git checkout -b codex/v1-scheduler
# 在这里实现 scheduler
```

如果时间很紧，可以简化成：

```powershell
git checkout -b codex/v1-scheduler
```

然后保持原始分支不动，后续小心记录 benchmark 命令和结果。不过两分支方案更干净。

## 目标架构

### Sequence / Request 状态

把请求进度统一成 token 计数：

```python
num_tokens              # 当前已知 token 总数，prompt + generated
num_prompt_tokens       # 原始 prompt 长度
num_computed_tokens     # 已经计算进 KV cache 的 token 数
num_scheduled_tokens    # 当前 step 调度的 token 数
status                  # waiting / running / finished
```

当前代码里的 `num_cached_tokens` 可以先暂时视作 `num_computed_tokens`，MVP 阶段不一定要立刻重命名。

### Scheduler

把 scheduler 输出从：

```python
seqs, is_prefill = schedule()
```

改成：

```python
scheduled_batch = schedule()
```

`scheduled_batch` 需要记录：

```text
sequence
本轮需要计算多少 token
这个任务是 decode 还是 prefill chunk
```

调度策略：

```text
1. token_budget = max_num_batched_tokens
2. 优先调度 running decode 请求，通常每个请求 1 token
3. 用剩余 budget 调度 partial prefill 和 waiting 请求
4. 长 prompt 超过剩余 budget 时自动切 chunk
5. KV cache 不足时使用 recompute preemption
```

### ModelRunner

新增基于 scheduled batch 的执行路径：

```python
run_scheduled(scheduled_batch)
```

MVP 阶段分三种执行模式：

```text
纯 prefill batch -> 复用现有 prefill 路径
纯 decode batch  -> 复用现有 decode + CUDA graph 路径
mixed batch      -> 新增 eager mixed path
```

对于 mixed batch，可以把 decode 当成 `q_len = 1`，把 prefill chunk 当成 `q_len = chunk_len`，统一构造成 varlen attention batch。

### BlockManager

MVP 阶段先保守处理：

```text
request 进入 running 时分配 block
decode 跨 block 时追加新 block
finish 或 preemption 时释放所有 block
只有完整且已经 computed 的 block 才写入 prefix cache hash
```

后续优化再考虑：

```text
根据 scheduled tokens 增量分配 KV block
```

### LLMEngine

新增 online continuous batching API：

```python
request_id = llm.add_request(prompt, sampling_params)
outputs = llm.step()
```

同时保留原 offline API：

```python
outputs = llm.generate(prompts, sampling_params)
```

## 实现路线

### 第一步：阅读当前代码

重点读：

```text
nanovllm/engine/scheduler.py
nanovllm/engine/model_runner.py
nanovllm/engine/block_manager.py
nanovllm/engine/sequence.py
nanovllm/layers/attention.py
nanovllm/utils/context.py
bench.py
bench_latency.py
```

修改代码前，先总结当前 request lifecycle。

### 第二步：研究参考实现

阅读 `nano-vllm-v1` fork 和上游 PR/issue。不要盲目照搬，只提取这个项目需要的最小设计。

重点问题：

- 它是否按 token progress 调度？
- 它是否支持真正的 online request arrival？
- 它如何处理 mixed prefill/decode？
- 它如何管理 KV block 和 prefix cache？
- 它记录了哪些 benchmark 指标？

### 第三步：先添加 benchmark

先添加或改造一套能同时跑 baseline 和新 scheduler 的 benchmark。

必须记录：

```text
throughput
TTFT
TPOT / ITL
端到端 request latency
preemption 次数
每个 step 平均 scheduled tokens
```

必须覆盖：

```text
offline batch
online request arrival
长短 prompt 混合
不同 max_num_batched_tokens
```

### 第四步：添加 online API

让请求可以在生成过程中动态加入。

保持接口简单：

```python
llm.add_request(...)
llm.step()
llm.is_finished()
```

### 第五步：实现 decode-first token-budget scheduler

把 phase-first 调度改成 token-budget 调度：

```text
running decode 优先
partial prefill 其次
waiting prefill 最后
长 prompt 自动 chunk
KV cache 满时使用 recompute preemption
```

### 第六步：添加 mixed batch 执行路径

纯 batch 继续走旧路径。

只有当 scheduled batch 同时包含 decode 和 prefill chunk 时，才走新的 eager mixed path。

### 第七步：更新 postprocess

每个 step 执行后：

```text
更新 computed token 数
只有 prompt computation 完成后才 append sampled token
EOS 或 max_tokens 时结束请求
释放 finished request 的 KV block
```

### 第八步：验证正确性

至少验证：

```text
greedy 输出和 baseline 一致
长 prompt chunking 正确
online arrival 正确
prefix cache hit 不崩
preemption 后能恢复
```

### 第九步：跑 benchmark 并写结果

对比：

```text
codex/bench-baseline
codex/v1-scheduler
```

使用相同模型、相同 prompt、相同随机种子、相同 benchmark 脚本。

## 重要风险

### Mixed Batch Attention

当前 attention 使用全局 `context.is_prefill`。mixed batch 不能简单复用旧的 prefill/decode 二分逻辑。

MVP 方案：

```text
mixed batch 使用 eager varlen attention
纯 decode 保留 CUDA graph
```

### Prefix Cache 和 Chunked Prefill

这里很容易出错。

MVP 规则：

```text
request 第一次进入 running 时查询 prefix cache
只 hash 完整 computed block
不 hash partial block
```

### CUDA Graph

第一版不要强行支持 mixed batch 的 CUDA graph。动态 mixed batch 和 graph capture 天然冲突。先只保留纯 decode 的 CUDA graph。

### Benchmark 设计

offline throughput 不一定能体现收益。主要收益应该出现在 online mixed workload，尤其是长 prompt 和 decode-heavy 请求混合时。

### 控制范围

scheduler 项目完成前，不要同时做 speculative decoding、LoRA、model registry、KV quantization 或 disaggregated prefill。

## 参考资料

vLLM：

- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide.html
- vLLM optimization and chunked prefill: https://docs.vllm.ai/en/stable/configuration/optimization.html
- vLLM V1 scheduler API: https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/

nano-vLLM：

- Issue #165，nano-vLLM-v1 fork 讨论: https://github.com/GeeeekExplorer/nano-vllm/issues/165
- nano-vLLM-v1 fork: https://github.com/slwang-ustc/nano-vllm-v1/tree/main
- PR #176，CUDA graph decode allocation 优化: https://github.com/GeeeekExplorer/nano-vllm/pull/176
- PR #193，PyTorch profiler: https://github.com/GeeeekExplorer/nano-vllm/pull/193
- PR #199 / #210 / #211，prefix cache 和 scheduler bugfix，可作为排坑参考

## 面试叙事

可以按这条线讲：

```text
原始 nano-vLLM 已经有基础 batching 和 KV cache，但调度仍然偏 phase-based。
我参考 vLLM V1，把调度重构为基于 token progress。
每个 step 使用固定 token budget。
优先调度 decode，降低 TPOT/ITL。
剩余 budget 用于 chunked prefill。
我补充 online continuous batching benchmark，用 TTFT、TPOT、throughput 和 latency 展示 tradeoff。
```

预期结果：

```text
online mixed workload 下 TPOT/ITL 和 request latency 应该改善
offline throughput 可能只小幅改善或基本持平
TTFT 和 TPOT 会随着 max_num_batched_tokens 产生取舍
```
