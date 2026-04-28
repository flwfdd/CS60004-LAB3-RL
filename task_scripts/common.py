import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from tqdm import tqdm
from transformers import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class GenerateConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


@dataclass
class Generation:
    text: str
    completion_tokens: int


class LLM(ABC):
    @abstractmethod
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[Generation]:
        pass

    def generate(
        self, messages: List[Dict[str, str]], config: GenerateConfig
    ) -> Generation:
        return self.generate_batch([messages], config)[0]

    def generate_text(
        self, messages: List[Dict[str, str]], config: GenerateConfig
    ) -> str:
        return self.generate(messages, config).text

    def close(self) -> None:
        pass


@dataclass
class Sample:
    nums: List[int]
    target: int
    id: Any | None = None


@dataclass
class EvalResult:
    ok: bool
    ans: str  # 格式不正确为空
    format_ok: bool
    output_len: int
    sample: Sample
    messages: List[Dict[str, str]]


@dataclass
class BenchmarkStats:
    total: int
    correct: int
    format_correct: int
    accuracy: float
    format_accuracy: float
    avg_output_len: float
    avg_output_len_correct: float
    avg_output_len_format_ok_wrong: float
    avg_output_len_format_wrong: float


@dataclass
class BenchmarkResult:
    summary: BenchmarkStats
    details: List[EvalResult]
    by_num_count: Dict[int, BenchmarkStats]


def build_messages(sample: Sample) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                f"Using the numbers {sample.nums}, create an equation that equals {sample.target}. "
                "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
                "Show your work in <think> </think> tags, and return the final answer in <answer> </answer> tags. "
                "For example <answer> (1 + 2) / 3 </answer>. "
                "You only have limited tokens budget, so do not think too much."
            ),
        },
    ]


def build_model_inputs(
    tokenizer: PreTrainedTokenizerBase,
    batch_messages: List[List[Dict[str, str]]],
    *,
    add_generation_prompt: bool,
    padding: bool = True,
    return_tensors: str = "pt",
    device: Any | None = None,
) -> BatchEncoding:
    """把messages转换为模型输入 包含input_ids, attention_mask"""
    input_texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        for messages in batch_messages
    ]
    model_inputs = tokenizer(
        input_texts,  # type: ignore
        return_tensors=return_tensors,
        padding=padding,
    )
    if device is not None:
        model_inputs = model_inputs.to(device)
    return model_inputs


def build_training_inputs(
    tokenizer: PreTrainedTokenizerBase,
    batch_messages: List[List[Dict[str, str]]],
    *,
    padding: bool = True,
    return_tensors: str = "pt",
    device: Any | None = None,
) -> BatchEncoding:
    """把messages转换为训练输入 包含input_ids, attention_mask, assistant_mask"""
    model_inputs = build_model_inputs(
        tokenizer,
        batch_messages,
        add_generation_prompt=False,
        padding=padding,
        return_tensors=return_tensors,
        device=device,
    )
    prompt_inputs = build_model_inputs(
        tokenizer,
        [messages[:-1] for messages in batch_messages],
        add_generation_prompt=True,  # 这样会包含 assistant 开头的东西 不参与训练
        padding=padding,
        return_tensors=return_tensors,
        device=device,
    )

    input_ids: torch.Tensor = model_inputs["input_ids"]  # type: ignore
    full_attention_mask: torch.Tensor = model_inputs["attention_mask"]  # type: ignore
    prompt_attention_mask: torch.Tensor = prompt_inputs["attention_mask"]  # type: ignore

    assistant_mask = torch.zeros_like(full_attention_mask, dtype=torch.bool)

    # 组成为 [PAD system user assistant]
    seq_len = input_ids.shape[1]  # 整个长度
    for i in range(input_ids.shape[0]):
        full_len = full_attention_mask[i].sum().item()  # 不包含 padding 长度
        prompt_len = (
            prompt_attention_mask[i].sum().item()
        )  # system 直到 assistant 开头长度
        assistant_start = seq_len - full_len + prompt_len
        assistant_mask[i, assistant_start:seq_len] = True

    model_inputs["assistant_mask"] = assistant_mask
    return model_inputs


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def extract_output_text(model_text: str) -> Optional[str]:
    # 截取最后一个</think>后的内容作为输出
    think_end_pos = model_text.rfind("</think>")
    if think_end_pos == -1:
        return None
    return model_text[think_end_pos + len("</think>") :].strip()


def extract_answer_text(model_text: str) -> Optional[str]:
    output_text = extract_output_text(model_text)
    if output_text is None:
        return None
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", output_text, flags=re.DOTALL | re.IGNORECASE
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
    return cnt_used == cnt_nums


def eval_batch(
    llm: LLM,
    batch_samples: List[Sample],
    generate_config: GenerateConfig,
) -> List[EvalResult]:
    batch_messages = [build_messages(sample) for sample in batch_samples]
    batch_gens = llm.generate_batch(batch_messages, generate_config)

    results = []
    for sample, messages, gen in zip(batch_samples, batch_messages, batch_gens):
        gen_text = gen.text
        messages = messages + [{"role": "assistant", "content": gen_text}]
        ans_text = extract_answer_text(gen_text) or ""
        format_ok = bool(ans_text)
        output_len = gen.completion_tokens
        value = safe_eval(ans_text)
        ok_value = value and (abs(value - float(sample.target)) < 1e-6)
        ok_nums = check_uses_numbers_once(ans_text, sample.nums)
        ok = bool(ok_value and ok_nums)
        results.append(
            (EvalResult(ok, ans_text, format_ok, output_len, sample, messages))
        )
    return results


def load_samples(path: Path, max_samples: Optional[int] = None) -> List[Sample]:
    samples = []
    for item in iter_jsonl(path):
        nums = [int(x) for x in item["nums"]]
        target = int(item["target"])
        samples.append(Sample(nums=nums, target=target, id=item.get("id")))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def summarize_eval_results(results: List[EvalResult]) -> BenchmarkStats:
    total = len(results)
    correct = sum(int(result.ok) for result in results)
    format_correct = sum(int(result.format_ok) for result in results)
    output_lens = [result.output_len for result in results]
    correct_output_lens = [result.output_len for result in results if result.ok]
    format_ok_wrong_output_lens = [
        result.output_len for result in results if result.format_ok and not result.ok
    ]
    format_wrong_output_lens = [
        result.output_len for result in results if not result.format_ok
    ]

    def _avg(xs: List[int]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    return BenchmarkStats(
        total=total,
        correct=correct,
        format_correct=format_correct,
        accuracy=(correct / total) if total else 0.0,
        format_accuracy=(format_correct / total) if total else 0.0,
        avg_output_len=_avg(output_lens),
        avg_output_len_correct=_avg(correct_output_lens),
        avg_output_len_format_ok_wrong=_avg(format_ok_wrong_output_lens),
        avg_output_len_format_wrong=_avg(format_wrong_output_lens),
    )


def summarize_eval_results_by_num_count(
    results: List[EvalResult],
) -> Dict[int, BenchmarkStats]:
    buckets: Dict[int, List[EvalResult]] = defaultdict(list)
    for result in results:
        buckets[len(result.sample.nums)].append(result)
    return {
        num_count: summarize_eval_results(items)
        for num_count, items in sorted(buckets.items(), key=lambda kv: kv[0])
    }


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
        return BenchmarkResult(
            summary=BenchmarkStats(
                total=0,
                correct=0,
                format_correct=0,
                accuracy=0.0,
                format_accuracy=0.0,
                avg_output_len=0.0,
                avg_output_len_correct=0.0,
                avg_output_len_format_ok_wrong=0.0,
                avg_output_len_format_wrong=0.0,
            ),
            details=[],
            by_num_count={},
        )
    if generate_config is None:
        generate_config = GenerateConfig()

    batches = [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]
    results: List[EvalResult] = []
    progress = tqdm(total=len(samples), desc="Evaluating")
    for batch_samples in batches:
        results.extend(eval_batch(llm, batch_samples, generate_config))
        progress.update(len(batch_samples))
    progress.close()

    return BenchmarkResult(
        summary=summarize_eval_results(results),
        details=results,
        by_num_count=summarize_eval_results_by_num_count(results),
    )
