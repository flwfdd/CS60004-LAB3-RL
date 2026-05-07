# Task 4: 推理脚本示例（请替换为实际代码）
# 加载最优 checkpoint，对 raw_test.json 进行推理
# 输出 task_4_test.jsonl，格式：{"id": 1, "prediction": "<think>...</think> <answer>...</answer>"}

import json
import os
from pathlib import Path

from common import (
    LLM,
    BenchmarkResult,
    GenerateConfig,
    eval_batch,
    load_samples,
    run_benchmark,
    summarize_eval_results,
    summarize_eval_results_by_num_count,
)
from dotenv import load_dotenv
from llm import OpenAILLM, TransformersLLM, VllmLLM
from tqdm import tqdm

load_dotenv()


def save_benchmark_result(result: BenchmarkResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in result.details:
            record = {
                "nums": item.sample.nums,
                "target": item.sample.target,
                "ok": item.ok,
                "ans": item.ans,
                "format_ok": item.format_ok,
                "output_len": item.output_len,
                "messages": item.messages,
            }
            if item.sample.id is not None:
                record["id"] = item.sample.id
                # 保留完整模型输出，不只截取 <answer> 内的表达式
                record["prediction"] = item.messages[-1].get("content", "")
            repeat_accuracy = getattr(item, "repeat_accuracy", None)
            if repeat_accuracy is not None:
                record["repeat_accuracy"] = repeat_accuracy
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "summary": {
            "total": result.summary.total,
            "correct": result.summary.correct,
            "accuracy": result.summary.accuracy,
            "format_correct": result.summary.format_correct,
            "format_accuracy": result.summary.format_accuracy,
            "avg_output_len": result.summary.avg_output_len,
            "avg_output_len_correct": result.summary.avg_output_len_correct,
            "avg_output_len_format_ok_wrong": result.summary.avg_output_len_format_ok_wrong,
            "avg_output_len_format_wrong": result.summary.avg_output_len_format_wrong,
        },
        "by_num_count": {
            str(num_count): {
                "total": stats.total,
                "correct": stats.correct,
                "accuracy": stats.accuracy,
                "format_correct": stats.format_correct,
                "format_accuracy": stats.format_accuracy,
                "avg_output_len": stats.avg_output_len,
                "avg_output_len_correct": stats.avg_output_len_correct,
                "avg_output_len_format_ok_wrong": stats.avg_output_len_format_ok_wrong,
                "avg_output_len_format_wrong": stats.avg_output_len_format_wrong,
            }
            for num_count, stats in result.by_num_count.items()
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved benchmark details to {output_path}")
    print(f"Saved benchmark summary to {summary_path}")


def run_benchmark_with_repeat(
    llm: LLM,
    data_path: str,
    *,
    max_samples: int | None = None,
    batch_size: int = 1,
    repeat_times: int = 1,
    generate_config: GenerateConfig | None = None,
) -> BenchmarkResult:
    """
    每个样本重复采样 repeat_times 次。
    对于同一个任务，只要任意一次 ok 就算通过，并保存其中一条通过的 prediction；
    如果都不通过，则回退保存第一条 prediction。
    """
    if repeat_times <= 0:
        raise ValueError("repeat_times must be positive")

    if repeat_times == 1:
        return run_benchmark(
            llm,
            data_path,
            max_samples=max_samples,
            batch_size=batch_size,
            generate_config=generate_config,
        )

    samples = load_samples(Path(data_path), max_samples=max_samples)
    if not samples:
        return BenchmarkResult(
            summary=summarize_eval_results([]),
            details=[],
            by_num_count={},
        )
    if generate_config is None:
        generate_config = GenerateConfig()

    repeated_samples = [sample for sample in samples for _ in range(repeat_times)]
    repeated_results = []
    selected_results = []
    progress = tqdm(total=len(repeated_samples), desc="Evaluating")
    for start in range(0, len(repeated_samples), batch_size):
        batch_samples = repeated_samples[start : start + batch_size]
        repeated_results.extend(eval_batch(llm, batch_samples, generate_config))
        progress.update(len(batch_samples))
    progress.close()

    assert len(repeated_results) == len(samples) * repeat_times
    for i in range(len(samples)):
        group = repeated_results[i * repeat_times : (i + 1) * repeat_times]
        passed = [result for result in group if result.ok]
        # 任意一次通过就记为通过并保存其中一条通过 prediction，否则保存第一条失败 prediction
        selected = passed[0] if passed else group[0]
        setattr(selected, "repeat_accuracy", len(passed) / repeat_times)
        selected_results.append(selected)

    return BenchmarkResult(
        summary=summarize_eval_results(selected_results),
        details=selected_results,
        by_num_count=summarize_eval_results_by_num_count(selected_results),
    )


def main() -> None:
    # local / openai / vllm
    backend = "openai"

    # 0p6b / 8b
    model_size = "0p6b"
    # model_size = "8b"

    # 从 .env 读取模型配置
    model_path = os.getenv(f"MODEL_PATH_{model_size.upper()}", "")
    model_name = os.getenv(f"MODEL_NAME_{model_size.upper()}", "")
    base_url = os.getenv(f"BASE_URL_{model_size.upper()}", "")
    api_key = os.getenv("API_KEY", "EMPTY")

    # 数据路径
    val_data_path = os.getenv("VAL_DATA_PATH", "")
    val_data_path = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits/raw_test.jsonl"
    # val_data_path = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits/raw_test_low_repeat_accuracy_v2_tmp.jsonl"
    output_path = ""
    output_path = Path("data/benchmark/tmp.jsonl")

    # VLLM 配置
    vllm_gpu_memory_utilization = 0.85

    # 评测规模与生成配置
    max_samples = 1000
    print_samples = 1
    repeat_times = 1
    batch_size = 256
    max_new_tokens = 1024
    temperature = 0.6
    top_p = 0.95
    top_k = 20

    llm: LLM
    if backend == "local":
        llm = TransformersLLM(model_path)
    elif backend == "openai":
        llm = OpenAILLM(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
        )
    elif backend == "vllm":
        llm = VllmLLM(
            model_path,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
    else:
        raise ValueError(f"No backend: {backend}")

    generate_config = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    try:
        benchmark_result = run_benchmark_with_repeat(
            llm,
            val_data_path,
            max_samples=max_samples,
            batch_size=batch_size,
            repeat_times=repeat_times,
            generate_config=generate_config,
        )

        for result in benchmark_result.details[:print_samples]:
            print(f"nums={result.sample.nums}, target={result.sample.target}")
            print(f"assistant={result.messages[-1]['content']}")
            print(f"answer={result.ans}")
            print(f"ok={result.ok}")
            print("=" * 60)

        summary = benchmark_result.summary
        print(f"Evaluated: {summary.total}")
        print(f"Correct: {summary.correct} ({summary.accuracy:.4f})")
        print(
            f"Format Correct: {summary.format_correct} ({summary.format_accuracy:.4f})"
        )
        print(
            "Avg output len (tokens): "
            f"all={summary.avg_output_len:.1f}, "
            f"correct={summary.avg_output_len_correct:.1f}, "
            f"format_ok_wrong={summary.avg_output_len_format_ok_wrong:.1f}, "
            f"format_wrong={summary.avg_output_len_format_wrong:.1f}"
        )
        if benchmark_result.by_num_count:
            print("By nums count:")
            for num_count, stats in benchmark_result.by_num_count.items():
                print(
                    f"  nums={num_count}: total={stats.total}, "
                    f"correct={stats.correct} ({stats.accuracy:.4f}), "
                    f"format_correct={stats.format_correct} ({stats.format_accuracy:.4f}), "
                    "avg_output_len="
                    f"{stats.avg_output_len:.1f}, "
                    "avg_output_len_correct="
                    f"{stats.avg_output_len_correct:.1f}, "
                    "avg_output_len_format_ok_wrong="
                    f"{stats.avg_output_len_format_ok_wrong:.1f}, "
                    "avg_output_len_format_wrong="
                    f"{stats.avg_output_len_format_wrong:.1f}"
                )
        if output_path:
            save_benchmark_result(benchmark_result, output_path)

    finally:
        llm.close()


if __name__ == "__main__":
    main()
