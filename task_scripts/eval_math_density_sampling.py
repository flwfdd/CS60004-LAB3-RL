#!/usr/bin/env python3

import os
import re
from pathlib import Path
from statistics import mean

from common import EvalResult, GenerateConfig, Sample, eval_batch, load_samples
from dotenv import load_dotenv
from llm import OpenAILLM

load_dotenv()


# OpenAI-compatible model config
MODEL_SIZE = "0P6B"
MODEL_NAME = os.getenv(f"MODEL_NAME_{MODEL_SIZE}", "")
BASE_URL = os.getenv(f"BASE_URL_{MODEL_SIZE}", "")
API_KEY = os.getenv("API_KEY", "EMPTY")

# Data / sampling config
DATA_PATH = os.getenv("VAL_DATA_PATH", "")
MAX_SAMPLES = 32
REPEAT_TIMES = 256
BATCH_SIZE = 256

# Generation config
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


def compute_math_density_from_result(result: EvalResult) -> float:
    """与 GRPO 奖励保持一致：只统计最后一条 assistant 的 <think> 中数学字符的占比。"""
    if not result.messages:
        return 0.0

    text = result.messages[-1].get("content", "")
    if not text:
        return 0.0

    think_match = re.search(
        r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not think_match:
        return 0.0

    think_text = think_match.group(1)
    if not think_text.strip():
        return 0.0

    math_chars = re.findall(r"[-0-9+*/()]", think_text)
    return len(math_chars) / len(think_text)


def summarize_values(values: list[float]) -> str:
    if not values:
        return "count=0 avg=0.0000 min=0.0000 max=0.0000"
    return (
        f"count={len(values)} "
        f"avg={mean(values):.4f} "
        f"min={min(values):.4f} "
        f"max={max(values):.4f}"
    )


def sample_many(
    llm: OpenAILLM, sample: Sample, generate_config: GenerateConfig
) -> list[EvalResult]:
    repeated_samples = [sample] * REPEAT_TIMES
    results: list[EvalResult] = []
    for start in range(0, len(repeated_samples), BATCH_SIZE):
        results.extend(
            eval_batch(
                llm, repeated_samples[start : start + BATCH_SIZE], generate_config
            )
        )
    return results


def print_sample_report(idx: int, sample: Sample, results: list[EvalResult]) -> None:
    correct = [r for r in results if r.ok]
    wrong = [r for r in results if not r.ok]

    correct_lengths = [float(r.output_len) for r in correct]
    wrong_lengths = [float(r.output_len) for r in wrong]
    correct_density = [compute_math_density_from_result(r) for r in correct]
    wrong_density = [compute_math_density_from_result(r) for r in wrong]

    print("=" * 80)
    print(f"[sample {idx}] nums={sample.nums} target={sample.target}")
    print(f"correct_count={len(correct)} wrong_count={len(wrong)} total={len(results)}")
    print(f"correct_output_len: {summarize_values(correct_lengths)}")
    print(f"wrong_output_len:   {summarize_values(wrong_lengths)}")
    print(f"correct_density:    {summarize_values(correct_density)}")
    print(f"wrong_density:      {summarize_values(wrong_density)}")


def main() -> None:
    if not DATA_PATH:
        raise ValueError("VAL_DATA_PATH is required")
    if not MODEL_NAME or not BASE_URL:
        raise ValueError(
            f"MODEL_NAME_{MODEL_SIZE} and BASE_URL_{MODEL_SIZE} are required"
        )

    samples = load_samples(Path(DATA_PATH), max_samples=MAX_SAMPLES)
    generate_config = GenerateConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    )

    print(
        f"[config] data_path={DATA_PATH} max_samples={MAX_SAMPLES} "
        f"repeat_times={REPEAT_TIMES} batch_size={BATCH_SIZE} "
        f"temperature={TEMPERATURE}"
    )

    llm = OpenAILLM(base_url=BASE_URL, api_key=API_KEY, model_name=MODEL_NAME)
    try:
        for idx, sample in enumerate(samples):
            results = sample_many(llm, sample, generate_config)
            print_sample_report(idx, sample, results)
    finally:
        llm.close()


if __name__ == "__main__":
    main()
