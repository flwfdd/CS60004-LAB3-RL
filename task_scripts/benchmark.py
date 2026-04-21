from dataclasses import dataclass

from eval_common import run_benchmark
from llm import LLM, GenerateConfig, OpenAILLM, TransformersLLM, VllmLLM


@dataclass(frozen=True)
class Config:
    # 后端类型：local / openai / vllm
    backend: str = "openai"

    # 模型与数据路径
    model_path: str = "/root/autodl-tmp/models/Qwen3-0.6B"
    # model_name: str = "Qwen3-0.6B"
    model_name: str = "Qwen3-8B"
    data_path: str = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits/val.jsonl"

    # OpenAI 接口配置
    # base_url: str = "http://127.0.0.1:8006/v1"
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "EMPTY"

    # VLLM 配置
    vllm_gpu_memory_utilization: float = 0.85

    # 评测规模与生成配置
    max_samples: int = 100
    print_samples: int = 1
    batch_size: int = 128
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


CFG = Config()


def build_llm(cfg: Config) -> LLM:
    if cfg.backend == "local":
        return TransformersLLM(
            cfg.model_path,
        )
    if cfg.backend == "openai":
        return OpenAILLM(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            model_name=cfg.model_name,
        )
    if cfg.backend == "vllm":
        return VllmLLM(
            cfg.model_path,
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
        )
    raise ValueError(f"No backend: {cfg.backend}")


def main(cfg: Config = CFG) -> None:
    llm = build_llm(cfg)
    generate_config = GenerateConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
    )
    try:
        benchmark_result = run_benchmark(
            llm,
            cfg.data_path,
            max_samples=cfg.max_samples,
            batch_size=cfg.batch_size,
            generate_config=generate_config,
        )

        for result in benchmark_result.details[: cfg.print_samples]:
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
        print(f"Evaluated: {benchmark_result.total}")
        print(f"Correct:   {benchmark_result.correct}")
        print(f"Accuracy:  {acc:.4f}")
    finally:
        llm.close()


if __name__ == "__main__":
    main()
