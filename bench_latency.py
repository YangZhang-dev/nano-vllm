from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from random import expovariate, randint, seed
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from nanovllm import LLM


@dataclass
class LatencyObserver:
    queue_enter_time: dict[int, float] = field(default_factory=dict)
    schedule_time: dict[int, float] = field(default_factory=dict)
    prefill_start_time: dict[int, float] = field(default_factory=dict)
    first_token_time: dict[int, float] = field(default_factory=dict)
    finish_time: dict[int, float] = field(default_factory=dict)
    output_lens: dict[int, int] = field(default_factory=dict)

    prefill_tokens: int = 0
    decode_tokens: int = 0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    last_decode_start: float | None = None
    last_decode_interval: float = 0.0

    def add_request(self, seq_id: int, max_tokens: int, queue_enter_time: float):
        self.queue_enter_time[seq_id] = queue_enter_time
        self.output_lens[seq_id] = max_tokens

    def observe_step(
        self,
        seqs,
        is_prefill: bool,
        num_tokens: int,
        schedule_time: float,
        prefill_start_time: float,
        step_end: float,
    ):
        step_time = step_end - prefill_start_time
        if is_prefill:
            self.prefill_tokens += num_tokens
            self.prefill_time += step_time
        else:
            self.decode_tokens += len(seqs)
            self.decode_time += step_time
            if self.last_decode_start is not None:
                self.last_decode_interval = prefill_start_time - self.last_decode_start
            self.last_decode_start = prefill_start_time

        for seq in seqs:
            if is_prefill and seq.seq_id not in self.schedule_time:
                self.schedule_time[seq.seq_id] = schedule_time
                self.prefill_start_time[seq.seq_id] = prefill_start_time
            if seq.num_completion_tokens >= 1 and seq.seq_id not in self.first_token_time:
                self.first_token_time[seq.seq_id] = step_end
            if seq.is_finished:
                self.finish_time[seq.seq_id] = step_end
                self.output_lens[seq.seq_id] = seq.num_completion_tokens

    def ttft(self) -> list[float]:
        return [
            self.first_token_time[seq_id] - self.queue_enter_time[seq_id]
            for seq_id in self.first_token_time
        ]

    def queue_delay(self) -> list[float]:
        return [
            self.schedule_time[seq_id] - self.queue_enter_time[seq_id]
            for seq_id in self.first_token_time
        ]

    def scheduling_delay(self) -> list[float]:
        return [
            self.prefill_start_time[seq_id] - self.schedule_time[seq_id]
            for seq_id in self.first_token_time
        ]

    def compute_delay(self) -> list[float]:
        return [
            self.first_token_time[seq_id] - self.prefill_start_time[seq_id]
            for seq_id in self.first_token_time
        ]

    def tpot(self) -> list[float]:
        values = []
        for seq_id, finish in self.finish_time.items():
            output_len = self.output_lens[seq_id]
            if output_len > 1:
                values.append((finish - self.first_token_time[seq_id]) / (output_len - 1))
        return values

    def e2e(self) -> list[float]:
        return [
            self.finish_time[seq_id] - self.queue_enter_time[seq_id]
            for seq_id in self.finish_time
        ]


def percentile(values: list[float], p: int) -> float:
    values = sorted(values)
    index = round((len(values) - 1) * p / 100)
    return values[index]


def summarize_ms(name: str, values: list[float], per_token: bool = False):
    if not values:
        print(f"{name}: n/a")
        return
    suffix = " ms/token" if per_token else " ms"
    scale = 1000
    print(f"{name} mean: {sum(values) / len(values) * scale:.2f}{suffix}")
    print(f"{name} p50:  {percentile(values, 50) * scale:.2f}{suffix}")
    print(f"{name} p90:  {percentile(values, 90) * scale:.2f}{suffix}")
    print(f"{name} p99:  {percentile(values, 99) * scale:.2f}{suffix}")


def timed_step(llm: LLM, observer: LatencyObserver):
    schedule_time = time.perf_counter()
    seqs, is_prefill = llm.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)

    prefill_start_time = time.perf_counter()
    token_ids = llm.model_runner.call("run", seqs, is_prefill)
    llm.scheduler.postprocess(seqs, token_ids, is_prefill)
    step_end = time.perf_counter()

    observer.observe_step(seqs, is_prefill, abs(num_tokens), schedule_time, prefill_start_time, step_end)
    return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished], num_tokens


def parse_request_rate(value: str) -> float:
    if value.lower() == "inf":
        return math.inf
    rate = float(value)
    if rate <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return rate


def build_arrival_offsets(num_requests: int, request_rate: float) -> list[float]:
    if math.isinf(request_rate):
        return [0.0] * num_requests

    offsets = []
    current = 0.0
    for i in range(num_requests):
        if i > 0:
            current += expovariate(request_rate)
        offsets.append(current)
    return offsets


def parse_args():
    parser = argparse.ArgumentParser(description="Online latency benchmark for TTFT/TPOT/E2E.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"))
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument(
        "--request-rate",
        type=parse_request_rate,
        default=4.0,
        help="Target request arrival rate in requests/second. Use 'inf' for a burst.",
    )
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from nanovllm import LLM, SamplingParams

    seed(args.seed)

    llm = LLM(
        args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    observer = LatencyObserver()
    arrival_offsets = build_arrival_offsets(args.num_seqs, args.request_rate)
    outputs = {}
    next_request = 0

    start = time.perf_counter()

    pbar = tqdm(
        total=args.num_seqs,
        desc="Generating",
        dynamic_ncols=True,
        disable=args.no_tqdm,
    )

    while len(outputs) < args.num_seqs:
        now = time.perf_counter()
        elapsed = now - start
        while next_request < args.num_seqs and arrival_offsets[next_request] <= elapsed:
            scheduled_arrival = start + arrival_offsets[next_request]
            llm.add_request(prompt_token_ids[next_request], sampling_params[next_request])
            observer.add_request(llm.scheduler.waiting[-1].seq_id, sampling_params[next_request].max_tokens, scheduled_arrival)
            next_request += 1

        if llm.is_finished():
            if next_request < args.num_seqs:
                next_arrival = start + arrival_offsets[next_request]
                sleep_time = max(0.0, next_arrival - time.perf_counter())
                time.sleep(sleep_time)
                continue
            break

        output, num_tokens = timed_step(llm, observer)
        pbar.set_postfix({
            "Arrived": next_request,
            "TTFT": f"{percentile(observer.ttft(), 50) * 1000:.1f}ms" if observer.ttft() else "n/a",
            "Queue": f"{percentile(observer.queue_delay(), 50) * 1000:.1f}ms" if observer.ttft() else "n/a",
            "TPOT": f"{percentile(observer.tpot(), 50) * 1000:.1f}ms" if observer.tpot() else "n/a",
            "Step": f"{observer.last_decode_interval * 1000:.1f}ms",
        })
        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids
            pbar.update(1)

    end = time.perf_counter()
    pbar.close()

    total_tokens = sum(len(token_ids) for token_ids in outputs.values())
    total_time = end - start

    print(f"Requests: {args.num_seqs}")
    print(f"Request rate: {'inf' if math.isinf(args.request_rate) else f'{args.request_rate:.2f}'} req/s")
    print(f"Generated tokens: {total_tokens}")
    print(f"Elapsed time: {total_time:.2f} s")
    print(f"Completed rate: {args.num_seqs / total_time:.2f} req/s")
    print()
    summarize_ms("TTFT", observer.ttft())
    print()
    summarize_ms("TTFT queue delay", observer.queue_delay())
    print()
    summarize_ms("TTFT scheduling delay", observer.scheduling_delay())
    print()
    summarize_ms("TTFT compute delay", observer.compute_delay())
    print()
    summarize_ms("TPOT", observer.tpot(), per_token=True)
    print()
    summarize_ms("E2E", observer.e2e())


if __name__ == "__main__":
    main()

# 现在的TTFT compute delay中如果出现了chunk，也会被计算在内