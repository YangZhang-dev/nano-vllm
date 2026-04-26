import os
import time
import statistics
from random import randint, seed

from nanovllm import LLM, SamplingParams


def pct(xs, p):
    xs = sorted(xs)
    i = int((len(xs) - 1) * p / 100)
    return xs[i]


def main():
    seed(0)

    num_seqs = 1
    max_input_len = 1024
    max_output_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len))
        for _ in range(num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)

    arrival = {}
    first = {}
    finish = {}
    output_lens = {}

    start = time.perf_counter()

    for prompt, sp in zip(prompt_token_ids, sampling_params):
        llm.add_request(prompt, sp)

    # 这些 seq_id 是 add_request 后自动生成的；直接从 scheduler 队列里取
    for seq in list(llm.scheduler.waiting):
        arrival[seq.seq_id] = start
        output_lens[seq.seq_id] = seq.max_tokens

    while not llm.is_finished():
        _, _, seqs = llm.step()
        now = time.perf_counter()

        for seq in seqs:
            if seq.num_completion_tokens >= 1 and seq.seq_id not in first:
                first[seq.seq_id] = now
            if seq.is_finished:
                finish[seq.seq_id] = now

    end = time.perf_counter()

    ttft = [first[i] - arrival[i] for i in first]
    tpot = [
        (finish[i] - first[i]) / (output_lens[i] - 1)
        for i in finish
        if output_lens[i] > 1
    ]

    total_tokens = sum(output_lens.values())
    throughput = total_tokens / (end - start)

    print(f"Total: {total_tokens} tok")
    print(f"Time: {end - start:.2f} s")
    print(f"Throughput: {throughput:.2f} tok/s")
    print()
    print(f"TTFT mean: {statistics.mean(ttft) * 1000:.2f} ms")
    print(f"TTFT p50:  {pct(ttft, 50) * 1000:.2f} ms")
    print(f"TTFT p90:  {pct(ttft, 90) * 1000:.2f} ms")
    print(f"TTFT p99:  {pct(ttft, 99) * 1000:.2f} ms")
    print()
    print(f"TPOT mean: {statistics.mean(tpot) * 1000:.2f} ms/token")
    print(f"TPOT p50:  {pct(tpot, 50) * 1000:.2f} ms/token")
    print(f"TPOT p90:  {pct(tpot, 90) * 1000:.2f} ms/token")
    print(f"TPOT p99:  {pct(tpot, 99) * 1000:.2f} ms/token")


if __name__ == "__main__":
    main()
