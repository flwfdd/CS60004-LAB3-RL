import json
import os
from pathlib import Path

from common import LLM, BenchmarkResult, GenerateConfig, run_benchmark
from dotenv import load_dotenv
from llm import OpenAILLM, TransformersLLM, VllmLLM

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
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc = result.correct / result.total if result.total else 0.0
    format_rate = result.format_correct / result.total if result.total else 0.0
    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "total": result.total,
        "correct": result.correct,
        "accuracy": acc,
        "format_correct": result.format_correct,
        "format_accuracy": format_rate,
        "avg_output_len": result.avg_output_len,
        "avg_output_len_correct": result.avg_output_len_correct,
        "avg_output_len_format_ok_wrong": result.avg_output_len_format_ok_wrong,
        "avg_output_len_format_wrong": result.avg_output_len_format_wrong,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved benchmark details to {output_path}")
    print(f"Saved benchmark summary to {summary_path}")


def main() -> None:
    # local / openai / vllm
    backend = "openai"

    # 0p6b / 8b
    # model_size = "0p6b"
    model_size = "8b"

    # 从 .env 读取模型配置
    model_path = os.getenv(f"MODEL_PATH_{model_size.upper()}", "")
    model_name = os.getenv(f"MODEL_NAME_{model_size.upper()}", "")
    base_url = os.getenv(f"BASE_URL_{model_size.upper()}", "")
    api_key = os.getenv("API_KEY", "EMPTY")

    # 数据路径
    val_data_path = os.getenv("TRAIN_DATA_PATH", "")
    output_path = Path("data/benchmark/8b_4096/")

    # VLLM 配置
    vllm_gpu_memory_utilization = 0.85

    # 评测规模与生成配置
    max_samples = 10000
    print_samples = 1
    batch_size = 256
    max_new_tokens = 4096
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
        benchmark_result = run_benchmark(
            llm,
            val_data_path,
            max_samples=max_samples,
            batch_size=batch_size,
            generate_config=generate_config,
        )

        for result in benchmark_result.details[:print_samples]:
            print(f"nums={result.sample.nums}, target={result.sample.target}")
            print(f"assistant={result.messages[-1]['content']}")
            print(f"answer={result.ans}")
            print(f"ok={result.ok}")
            print("=" * 60)

        acc = (
            benchmark_result.correct / benchmark_result.total
            if benchmark_result.total
            else 0.0
        )
        format_rate = (
            benchmark_result.format_correct / benchmark_result.total
            if benchmark_result.total
            else 0.0
        )
        print(f"Evaluated: {benchmark_result.total}")
        print(f"Correct: {benchmark_result.correct} ({acc:.4f})")
        print(f"Format Correct: {benchmark_result.format_correct} ({format_rate:.4f})")
        print(
            "Avg output len (tokens): "
            f"all={benchmark_result.avg_output_len:.1f}, "
            f"correct={benchmark_result.avg_output_len_correct:.1f}, "
            f"format_ok_wrong={benchmark_result.avg_output_len_format_ok_wrong:.1f}, "
            f"format_wrong={benchmark_result.avg_output_len_format_wrong:.1f}"
        )
        if output_path:
            save_benchmark_result(benchmark_result, output_path)

    finally:
        llm.close()


if __name__ == "__main__":
    main()
