import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

# 端到端离线批处理的吞吐量
def main():
    seed(0)
    # 并发压力
    num_seqs = 256
    # Prefill 首轮计算压力
    max_input_len = 1024
    # Decode KV Cache压力
    max_ouput_len = 1024

    # max_num_batched_tokens: int = 16384
    # max_num_seqs: int = 512
    # max_model_len: int = 4096

    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, max_num_batched_tokens=max_num_batched_tokens, max_num_seqs=max_num_seqs, max_model_len=max_model_len)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
