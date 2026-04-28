# V1 调度器 Legacy Baseline 结果

## 测试环境与配置

- 分支：`codex/bench-baseline`
- 模型：`Qwen3-0.6B`
- 请求数：256
- 最大输入长度：1024
- 最大输出长度：1024
- `max_num_batched_tokens`：16384
- 调度器：legacy phase-first scheduler

说明：该结果用于后续和 V1 scheduler 做性能对比。当前 legacy scheduler 不支持 mixed prefill/decode，同一 step 只会是 prefill 或 decode，因此 `mixed=0` 是预期行为。

## Offline Batch

命令：

```bash
python bench.py
```

结果：

| 指标 | 数值 |
| --- | ---: |
| 生成 token 数 | 133966 |
| 总耗时 | 91.17 s |
| 吞吐量 | 1469.33 tok/s |

## Online Burst Arrival

命令：

```bash
python bench_latency.py --request-rate inf --no-tqdm
```

整体结果：

| 指标 | 数值 |
| --- | ---: |
| 请求数 | 256 |
| 请求到达率 | inf req/s |
| 生成 token 数 | 133966 |
| 总耗时 | 90.26 s |
| 完成速率 | 2.84 req/s |
| 输出吞吐量 | 1484.26 tok/s |
| Prefill 吞吐量 | 17987.20 tok/s |
| Decode 吞吐量 | 1718.81 tok/s |
| 平均每 step 调度 token 数 | 82.52 |
| 抢占次数 | 168 |
| step 总数 | 4204 |
| prefill step 数 | 202 |
| decode step 数 | 4002 |
| mixed step 数 | 0 |

延迟结果：

| 指标 | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 33944.13 ms | 33433.18 ms | 68597.05 ms | 78078.96 ms |
| TTFT queue delay | 33725.89 ms | 33392.48 ms | 68525.40 ms | 78004.65 ms |
| TTFT scheduling delay | 0.21 ms | 0.06 ms | 0.94 ms | 0.94 ms |
| TTFT compute delay | 218.03 ms | 66.87 ms | 722.15 ms | 722.15 ms |
| TPOT | 27.34 ms/token | 25.96 ms/token | 29.14 ms/token | 50.26 ms/token |
| E2E | 47586.13 ms | 48869.40 ms | 81978.60 ms | 86821.25 ms |

## Online Poisson Arrival

命令：

```bash
python bench_latency.py --request-rate 4 --no-tqdm
```

整体结果：

| 指标 | 数值 |
| --- | ---: |
| 请求数 | 256 |
| 请求到达率 | 4.00 req/s |
| 生成 token 数 | 133966 |
| 总耗时 | 93.50 s |
| 完成速率 | 2.74 req/s |
| 输出吞吐量 | 1432.76 tok/s |
| Prefill 吞吐量 | 16304.49 tok/s |
| Decode 吞吐量 | 1655.15 tok/s |
| 平均每 step 调度 token 数 | 68.91 |
| 抢占次数 | 146 |
| step 总数 | 4787 |
| prefill step 数 | 246 |
| decode step 数 | 4541 |
| mixed step 数 | 0 |

延迟结果：

| 指标 | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 4756.61 ms | 4762.77 ms | 9224.61 ms | 13975.82 ms |
| TTFT queue delay | 4702.38 ms | 4719.02 ms | 9178.53 ms | 13929.71 ms |
| TTFT scheduling delay | 0.05 ms | 0.05 ms | 0.08 ms | 0.14 ms |
| TTFT compute delay | 54.18 ms | 45.89 ms | 78.72 ms | 98.63 ms |
| TPOT | 24.70 ms/token | 25.72 ms/token | 28.01 ms/token | 32.94 ms/token |
| E2E | 17486.04 ms | 17098.46 ms | 28618.98 ms | 32999.59 ms |

## 基线观察

- Offline batch 吞吐量约为 1469 tok/s，online burst 场景输出吞吐量约为 1484 tok/s，两者接近。
- 两个 online 场景的 `mixed step 数` 都是 0，说明 legacy scheduler 仍然严格按 prefill/decode phase 执行。
- Burst 场景 TTFT 很高，且 TTFT 主要由 queue delay 构成，说明大量请求在进入 prefill 前排队等待。
- Poisson 场景的 TTFT 明显低于 burst，但仍存在秒级 queue delay。
- 后续 V1 scheduler 的主要对比目标是降低 online mixed workload 下的 TTFT、TPOT/ITL 和端到端延迟，同时观察吞吐量是否保持稳定。
