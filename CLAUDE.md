# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install:**
```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
# or locally:
pip install -e .
```

**Run inference:**
```bash
python example.py
```

**Benchmark:**
```bash
python bench.py
```

There is no formal test suite or lint configuration.

## Architecture

Nano-vLLM is a ~1,200-line from-scratch LLM inference engine. The request lifecycle flows through these layers:

```
LLM (llm.py)                    — thin public API wrapper
  └─ LLMEngine (engine/llm_engine.py)   — orchestrator: tokenizer, schedule loop
        ├─ Scheduler (engine/scheduler.py)       — two-phase scheduling
        ├─ BlockManager (engine/block_manager.py) — KV cache allocation + prefix caching
        └─ ModelRunner (engine/model_runner.py)  — GPU execution, CUDA graphs, IPC
              └─ Qwen3ForCausalLM (models/qwen3.py)
                    └─ layers/ (attention, linear, sampler, ...)
```

### Key subsystems

**Two-phase scheduling** (`engine/scheduler.py`): The scheduler alternates between a *prefill* phase (process full new sequences, variable-length tokens) and a *decode* phase (one new token per in-flight sequence). Sequences move through WAITING → RUNNING → FINISHED states and can be preempted back to WAITING if memory is tight.

**Prefix caching** (`engine/block_manager.py`): KV cache is split into fixed-size blocks (256 tokens). Blocks are content-addressed via xxhash so sequences sharing a common prefix reuse the same physical GPU memory. Reference counting handles shared blocks.

**Attention context** (`utils/context.py`): A thread-local `Context` object carries per-batch data (block tables, slot mappings, sequence lengths) from the engine down to the attention layer without passing it through every model call. Set before forward pass, read inside `layers/attention.py`.

**CUDA graph capture** (`engine/model_runner.py`): During warmup, decode-phase execution is recorded as CUDA graphs for batch sizes `[1, 2, 4, 8, 16, 32, ..., max_num_seqs]`. At runtime, the closest graph is replayed for near-zero kernel-launch overhead. Prefill and oversized batches fall back to eager mode. Disable with `enforce_eager=True`.

**Tensor parallelism** (`layers/linear.py`, `models/qwen3.py`): For `tensor_parallel_size > 1`, separate processes are spawned (via `multiprocessing`) for non-primary GPUs. The primary GPU orchestrates and exchanges sequence data via `SharedMemory`. Linear layers are sharded as `ColumnParallelLinear` / `RowParallelLinear` using NCCL all-reduce.

**Flash attention** (`layers/attention.py`): Uses `flash_attn_varlen_func` for prefill (variable-length) and `flash_attn_with_kvcache` for decode. KV values are written into the cache via a custom Triton kernel `store_kvcache_kernel`.

### Adding a new model

Follow [nanovllm/models/qwen3.py](nanovllm/models/qwen3.py) as the template:
- Implement attention using `layers/attention.py` (handles KV cache automatically via context)
- Use `layers/linear.py` parallel linear classes for tensor-parallel support
- Use `utils/loader.py` `load_weights` to load safetensors shards
- Register the model in `engine/model_runner.py` where the model class is instantiated
