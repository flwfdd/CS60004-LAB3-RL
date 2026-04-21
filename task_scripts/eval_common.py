import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from llm import LLM, GenerateConfig
from tqdm import tqdm


@dataclass
class Sample:
    nums: List[int]
    target: int


@dataclass
class EvalResult:
    ok: bool
    ans: str  # 如果不为空说明格式正确
    sample: Sample
    messages: List[Dict[str, str]]


@dataclass
class BenchmarkResult:
    total: int
    correct: int
    details: List[EvalResult]


def build_messages(sample: Sample) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Using the numbers {sample.nums}, create an equation that equals {sample.target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.",
        },
    ]


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def extract_answer_text(model_text: str) -> Optional[str]:
    # 截取最后一个</think>后的内容作为输出 再从中提取
    think_end_pos = model_text.rfind("</think>")
    if think_end_pos == -1:
        return None
    model_text = model_text[think_end_pos + len("</think>") :].strip()
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", model_text, flags=re.DOTALL | re.IGNORECASE
    )
    return matches[-1].strip() if matches else None


def safe_eval(expr: str) -> float | None:
    # 只保留整数、括号、空格和四则运算符
    expr = re.sub(r"[^0-9\+\-\*/\(\)\s]", "", expr)
    try:
        return float(eval(expr))
    except Exception:
        return None


def check_uses_numbers_once(text: str, nums: List[int]) -> bool:
    # 判断是否每个数字用且仅用一次
    found = re.findall(r"-?\d+", text)
    if not found:
        return False
    used = [int(x) for x in found]
    cnt_used = Counter(used)
    cnt_nums = Counter(nums)
    for k, v in cnt_used.items():
        if cnt_nums.get(k, 0) != v:
            return False
    return True


def eval_batch(
    llm: LLM,
    batch_samples: List[Sample],
    generate_config: GenerateConfig,
) -> List[EvalResult]:
    batch_messages = [build_messages(sample) for sample in batch_samples]
    batch_text = llm.generate_batch(batch_messages, generate_config)

    results = []
    for sample, messages, gen_text in zip(batch_samples, batch_messages, batch_text):
        messages = messages + [{"role": "assistant", "content": gen_text}]
        ans_text = extract_answer_text(gen_text) or ""
        value = safe_eval(ans_text)
        ok_value = value and (abs(value - float(sample.target)) < 1e-6)
        ok_nums = check_uses_numbers_once(ans_text, sample.nums)
        ok = bool(ok_value and ok_nums)
        results.append((EvalResult(ok, ans_text, sample, messages)))
    return results


def load_samples(path: Path, max_samples: Optional[int] = None) -> List[Sample]:
    samples = []
    for item in iter_jsonl(path):
        nums = [int(x) for x in item["nums"]]
        target = int(item["target"])
        samples.append(Sample(nums=nums, target=target))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def run_benchmark(
    llm: LLM,
    data_path: str,
    *,
    max_samples: Optional[int] = None,
    batch_size: int = 1,
    generate_config: Optional[GenerateConfig] = None,
) -> BenchmarkResult:
    samples = load_samples(Path(data_path), max_samples=max_samples)
    if not samples:
        return BenchmarkResult(total=0, correct=0, details=[])
    if generate_config is None:
        generate_config = GenerateConfig()

    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
    results: List[EvalResult] = []
    progress = tqdm(total=len(samples), desc="Evaluating")
    for batch_samples in batches:
        results.extend(eval_batch(llm, batch_samples, generate_config))
        progress.update(len(batch_samples))
    progress.close()

    total = len(results)
    correct = sum(int(result.ok) for result in results)
    return BenchmarkResult(total=total, correct=correct, details=results)
