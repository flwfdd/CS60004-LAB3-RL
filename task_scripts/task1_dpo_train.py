import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from common import (
    LLM,
    BenchmarkResult,
    GenerateConfig,
    Sample,
    build_training_inputs,
    eval_batch,
    extract_output_text,
    load_samples,
    run_benchmark,
)
from dotenv import load_dotenv
from llm import OpenAILLM, TransformersLLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

load_dotenv()


@dataclass
class DPOSample:
    chosen_messages: List[Dict[str, str]]
    rejected_messages: List[Dict[str, str]]


def evaluate_checkpoint(
    actor_model: Any,
    tokenizer: Any,
    *,
    val_data_path: str,
    eval_samples: int,
    eval_batch_size: int,
    eval_generate_config: GenerateConfig,
) -> BenchmarkResult:
    was_training = actor_model.training
    actor_model.eval()
    llm = TransformersLLM(model=actor_model, tokenizer=tokenizer)
    try:
        return run_benchmark(
            llm,
            val_data_path,
            max_samples=eval_samples,
            batch_size=eval_batch_size,
            generate_config=eval_generate_config,
        )
    finally:
        if was_training:
            actor_model.train()


def build_preference_pairs(
    samples: list[Sample],
    llm1: LLM,
    llm2: LLM,
    generate_config1: GenerateConfig,
    generate_config2: GenerateConfig,
    batch_size: int = 8,
) -> list[DPOSample]:
    """对同一批样本用两个 LLM 各推理一次，一对一错时构成偏好对"""
    batch_size = min(batch_size, len(samples))
    pairs: list[DPOSample] = []
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        results1 = eval_batch(llm1, batch, generate_config1)
        results2 = eval_batch(llm2, batch, generate_config2)
        for result1, result2 in zip(results1, results2):
            if result1.ok and not result2.ok:
                pairs.append(
                    DPOSample(
                        chosen_messages=result1.messages,
                        rejected_messages=result2.messages,
                    )
                )
            elif not result1.ok and result2.ok:
                pairs.append(
                    DPOSample(
                        chosen_messages=result2.messages,
                        rejected_messages=result1.messages,
                    )
                )
    print(f"Generated {len(pairs)} preference pairs out of {len(samples)}")
    return pairs


def build_rejection_samples(
    llm: LLM,
    samples: list[Sample],
    generate_config: GenerateConfig,
    try_num: int = 1,
    batch_size: int = 8,
    acc_threshold: float = 1.0,
) -> list[DPOSample]:
    """对每条样本采样 try_num 次，正确率低于阈值时取 最短正确 vs 最短错误 构成偏好对。"""
    sample_batch_size = max(1, batch_size // try_num)
    pairs: list[DPOSample] = []
    total_trials = 0
    total_correct = 0
    skipped_no_pair = 0

    for i in tqdm(range(0, len(samples), sample_batch_size), desc="Rejection Sampling"):
        batch = samples[i : i + sample_batch_size]

        # 每条样本重复 try_num 次
        repeated_samples: list[Sample] = []
        for s in batch:
            repeated_samples.extend([s] * try_num)

        results = eval_batch(llm, repeated_samples, generate_config)
        for j, s in enumerate(batch):
            group = results[j * try_num : (j + 1) * try_num]
            total_trials += try_num
            num_correct = sum(int(r.ok) for r in group)
            total_correct += num_correct

            # 只对正确率低于阈值的样本构造偏好对
            print(f"id={i+j} acc={num_correct / try_num:.4f}({num_correct}/{try_num})")
            if (num_correct / try_num) >= acc_threshold:
                continue

            correct = [r for r in group if r.ok]
            wrong = [r for r in group if not r.ok]

            if not correct or not wrong:
                skipped_no_pair += 1
                continue

            # chosen: 正确里长度最短
            chosen = min(correct, key=lambda r: r.output_len)
            # rejected: 错误里长度最短
            rejected = min(wrong, key=lambda r: r.output_len)

            pairs.append(
                DPOSample(
                    chosen_messages=chosen.messages,
                    rejected_messages=rejected.messages,
                )
            )

    overall_acc = total_correct / total_trials if total_trials else 0.0
    print(
        f"[rejection_sampling] trials={total_trials} overall_accuracy={overall_acc:.4f} "
        f"pairs={len(pairs)} skipped_no_pair={skipped_no_pair}"
    )
    return pairs


def save_pairs(pairs: list[DPOSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            record = {
                "chosen_messages": pair.chosen_messages,
                "rejected_messages": pair.rejected_messages,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(pairs)} pairs to {path}")


def read_pairs(path: Path) -> list[DPOSample]:
    with path.open("r", encoding="utf-8") as f:
        pairs: list[DPOSample] = []
        for line in f:
            record = json.loads(line)
            pairs.append(DPOSample(**record))
        return pairs


def _extract_tag_content(text: str, tag: str) -> str | None:
    m = re.findall(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.DOTALL | re.IGNORECASE
    )
    return m[-1].strip() if m else None


def rewrite_think_batch(
    texts: list[str],
    llm: LLM,
    generate_config: GenerateConfig,
    batch_size: int = 8,
) -> list[str]:
    """只重写 <think>，保留 <answer> 不变；重写失败的样本返回空字符串。"""
    if not texts:
        return []

    prompts: list[list[dict[str, str]]] = []
    meta: list[tuple[int, str]] = []  # (index, answer)
    out = ["" for _ in texts]

    for idx, t in enumerate(texts):
        answer = _extract_tag_content(t, "answer")
        if answer is None:
            continue

        prompt = f"""Rewrite this math reasoning more concisely.

Rules:
- Keep ALL steps and reasoning logic (including failed attempts) in the original order
- Remove ALL filler
- Keep full arithmetic calculations
- Do not output final answer at beginning

Original reasoning: {t.replace("<think>", "").replace("</think>", "")}

Plan how to rewrite in <think> and output the full compact reasoning process."""
        prompts.append(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        meta.append((idx, answer))

    # 按 batch_size 分块请求，避免一次性发太多并发/请求体过大
    gens = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc="Rewrite Think",
    ):
        gens.extend(
            llm.generate_batch(prompts[start : start + batch_size], generate_config)
        )

    for gen, (idx, answer) in zip(gens, meta):
        rewrite_think = extract_output_text(gen.text)
        if rewrite_think:
            out[idx] = (
                f"<think>\n{rewrite_think}\n</think>\n<answer> {answer} </answer>"
            )

    return out


def generate_pairs_pipeline() -> None:
    # 模型规模: 0p6b / 8b
    model_size1 = "0p6b"
    model_size2 = "8b"

    # preference: 两次采样；rejection: 多次采样，筛选正确率低于阈值的
    pair_mode: str = "rejection"

    # 从 .env 读取模型配置
    model_name1 = os.getenv(f"MODEL_NAME_{model_size1.upper()}", "")
    base_url1 = os.getenv(f"BASE_URL_{model_size1.upper()}", "")
    model_name2 = os.getenv(f"MODEL_NAME_{model_size2.upper()}", "")
    base_url2 = os.getenv(f"BASE_URL_{model_size2.upper()}", "")

    api_key = os.getenv("API_KEY", "EMPTY")

    # train_data_path = os.getenv("TRAIN_DATA_PATH", "")
    train_data_path = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits/raw_test.jsonl"

    max_samples = 1024
    batch_size = 512
    try_num = 128
    acc_threshold = 0.9

    output_path = Path("data/dpo/0p6b_test_rejection.jsonl")
    rewritten_output_path = Path("data/dpo/0p6b_test_rewritten.jsonl")

    generate_config1 = GenerateConfig(
        max_new_tokens=1024,
        temperature=0.6,
    )
    generate_config2 = GenerateConfig(
        max_new_tokens=4096,
        temperature=0.6,
    )

    llm1 = OpenAILLM(base_url=base_url1, api_key=api_key, model_name=model_name1)
    llm2 = OpenAILLM(base_url=base_url2, api_key=api_key, model_name=model_name2)
    samples = load_samples(Path(train_data_path), max_samples=max_samples)
    if pair_mode == "rejection":
        try:
            pairs = build_rejection_samples(
                llm1,
                samples,
                generate_config1,
                try_num=try_num,
                batch_size=batch_size,
                acc_threshold=acc_threshold,
            )
            save_pairs(pairs, output_path)
        finally:
            llm1.close()
    elif pair_mode == "preference":
        # 同一个模型采样两次来构造偏好对
        try:
            pairs = build_preference_pairs(
                samples,
                llm1,
                llm2,
                generate_config1,
                generate_config2,
                batch_size=batch_size,
            )
            save_pairs(pairs, output_path)
            # pairs = read_pairs(output_path)
            # 重写为精简版
            if pairs:
                rewritten_chosen_texts = rewrite_think_batch(
                    [pair.chosen_messages[-1].get("content", "") for pair in pairs],
                    llm2,
                    generate_config2,
                    batch_size=batch_size,
                )
                rewritten_pairs: list[DPOSample] = []
                for pair, rewritten_text in zip(pairs, rewritten_chosen_texts):
                    chosen_messages = [dict(m) for m in pair.chosen_messages]
                    if not rewritten_text:
                        continue
                    chosen_messages[-1]["content"] = rewritten_text
                    rewritten_pairs.append(
                        DPOSample(
                            chosen_messages=chosen_messages,
                            rejected_messages=pair.rejected_messages,
                        )
                    )
                save_pairs(rewritten_pairs, rewritten_output_path)
        finally:
            llm1.close()
            llm2.close()


def compute_logps(
    logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    mask: torch.Tensor,  # (batch_size, seq_len)
) -> torch.Tensor:
    # 错位
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_mask = mask[:, 1:]
    # 计算 log probs
    all_logps = F.log_softmax(
        shifted_logits, dim=-1
    )  # (batch_size, seq_len-1, vocab_size)
    # 取目标位置的 log probs
    logps = all_logps.gather(dim=-1, index=shifted_input_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (batch_size, seq_len-1)
    # 应用掩码
    masked_logps = logps * shifted_mask
    return masked_logps.sum(dim=-1)


def dpo_loss(
    actor_chosen_logps: torch.Tensor,  # (batch_size,)
    actor_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """计算 DPO 损失"""
    chosen_ratio = actor_chosen_logps - ref_chosen_logps
    rejected_ratio = actor_rejected_logps - ref_rejected_logps
    diff = beta * (chosen_ratio - rejected_ratio)
    return -F.logsigmoid(diff).mean()


def train_dpo() -> None:
    # 训练配置
    micro_batch_size = 1
    train_batch_size = 16
    train_samples = 20480
    lr = 1e-6
    alpha = 0.999  # nll 权重
    beta = 0.1  # dpo 公式内
    eval_every_train_steps = 10000
    eval_samples = 0
    eval_batch_size = 32
    eval_max_new_tokens = 1024
    # model_path = os.getenv(f"MODEL_PATH_0P6B", "")
    model_path = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/ckpt/grpo_trainbs64_minibs64_gs8_test/best"
    pairs_path = Path("data/dpo/0p6b_test_rejection_repeat10.jsonl")
    val_data_path = os.getenv("VAL_DATA_PATH", "")
    val_data_path = "/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits/raw_test.jsonl"
    exp_name = "dpo_0p6b_test_rejection_0p999nll_ep10"
    ckpt_path = Path(f"data/ckpt/{exp_name}")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    pairs = read_pairs(pairs_path)[:train_samples]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    # 初始化
    actor_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  # type: ignore
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  # type: ignore
    actor_model.train()
    ref_model.eval()
    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=lr)
    total_train_steps = math.ceil(len(pairs) / train_batch_size)
    wandb_run = wandb.init(
        project="cs60004-lab3-rl",
        name=exp_name,
        config={
            "model_path": model_path,
            "pairs_path": str(pairs_path),
            "ckpt_path": str(ckpt_path),
            "total_samples": len(pairs),
            "train_batch_size": train_batch_size,
            "micro_batch_size": micro_batch_size,
            "total_train_steps": total_train_steps,
            "lr": lr,
            "beta": beta,
            "val_data_path": val_data_path,
            "eval_every_train_steps": eval_every_train_steps,
            "eval_samples": eval_samples,
            "eval_batch_size": eval_batch_size,
            "eval_max_new_tokens": eval_max_new_tokens,
            "device": str(device),
        },
    )
    optimizer.zero_grad()
    train_step = 0
    seen_samples = 0
    accum_loss = 0.0
    accum_loss_dpo = 0.0
    accum_loss_nll = 0.0
    accum_actor_chosen_prob = 0.0
    accum_actor_rejected_prob = 0.0
    accum_ref_chosen_prob = 0.0
    accum_ref_rejected_prob = 0.0
    accum_reward_margin = 0.0
    accum_preference_prob = 0.0
    accum_chosen_tokens = 0.0
    accum_rejected_tokens = 0.0
    accum_samples = 0
    eval_generate_config = GenerateConfig(max_new_tokens=eval_max_new_tokens)

    # 主训练循环
    progress = tqdm(total=len(pairs), desc="Training DPO")
    for i in range(0, len(pairs), micro_batch_size):
        batch = pairs[i : i + micro_batch_size]
        batch_samples = len(batch)
        seen_samples += batch_samples
        chosen_model_inputs = build_training_inputs(
            tokenizer,
            [pair.chosen_messages for pair in batch],
            device=device,
        )
        rejected_model_inputs = build_training_inputs(
            tokenizer,
            [pair.rejected_messages for pair in batch],
            device=device,
        )
        chosen_forward_inputs = {
            "input_ids": chosen_model_inputs["input_ids"],
            "attention_mask": chosen_model_inputs["attention_mask"],
        }
        rejected_forward_inputs = {
            "input_ids": rejected_model_inputs["input_ids"],
            "attention_mask": rejected_model_inputs["attention_mask"],
        }
        actor_chosen_logits = actor_model(**chosen_forward_inputs).logits
        actor_rejected_logits = actor_model(**rejected_forward_inputs).logits
        actor_chosen_logps = compute_logps(
            actor_chosen_logits,
            chosen_model_inputs["input_ids"],  # type: ignore
            chosen_model_inputs["assistant_mask"],  # type: ignore
        )
        actor_rejected_logps = compute_logps(
            actor_rejected_logits,
            rejected_model_inputs["input_ids"],  # type: ignore
            rejected_model_inputs["assistant_mask"],  # type: ignore
        )
        with torch.no_grad():
            ref_chosen_logits = ref_model(**chosen_forward_inputs).logits
            ref_rejected_logits = ref_model(**rejected_forward_inputs).logits
            ref_chosen_logps = compute_logps(
                ref_chosen_logits,
                chosen_model_inputs["input_ids"],  # type: ignore
                chosen_model_inputs["assistant_mask"],  # type: ignore
            )
            ref_rejected_logps = compute_logps(
                ref_rejected_logits,
                rejected_model_inputs["input_ids"],  # type: ignore
                rejected_model_inputs["assistant_mask"],  # type: ignore
            )

        chosen_token_count = (
            chosen_model_inputs["assistant_mask"].sum(dim=1).float().clamp_min(1)  # type: ignore
        )
        rejected_token_count = (
            rejected_model_inputs["assistant_mask"].sum(dim=1).float().clamp_min(1)  # type: ignore
        )

        loss_dpo = dpo_loss(
            actor_chosen_logps,
            actor_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
        )
        loss_nll = -(actor_chosen_logps / chosen_token_count).mean()
        loss = alpha * loss_nll + (1 - alpha) * loss_dpo

        scaled_loss = loss * (batch_samples / train_batch_size)
        scaled_loss.backward()

        progress.update(batch_samples)
        should_step = (
            accum_samples + batch_samples >= train_batch_size
            or seen_samples == len(pairs)
        )
        if should_step:
            optimizer.step()
            optimizer.zero_grad()
            train_step += 1

        reward_margin = (
            (actor_chosen_logps - actor_rejected_logps)
            - (ref_chosen_logps - ref_rejected_logps)
        ).mean()
        preference_prob = torch.sigmoid(beta * reward_margin)
        actor_chosen_prob = (actor_chosen_logps / chosen_token_count).exp().mean()
        actor_rejected_prob = (actor_rejected_logps / rejected_token_count).exp().mean()
        ref_chosen_prob = (ref_chosen_logps / chosen_token_count).exp().mean()
        ref_rejected_prob = (ref_rejected_logps / rejected_token_count).exp().mean()
        chosen_tokens = (
            chosen_model_inputs["assistant_mask"].sum(dim=1).float().mean().item()  # type: ignore
        )
        rejected_tokens = (
            rejected_model_inputs["assistant_mask"].sum(dim=1).float().mean().item()  # type: ignore
        )
        accum_loss += loss.item() * batch_samples
        accum_loss_dpo += loss_dpo.item() * batch_samples
        accum_loss_nll += loss_nll.item() * batch_samples
        accum_actor_chosen_prob += actor_chosen_prob.item() * batch_samples
        accum_actor_rejected_prob += actor_rejected_prob.item() * batch_samples
        accum_ref_chosen_prob += ref_chosen_prob.item() * batch_samples
        accum_ref_rejected_prob += ref_rejected_prob.item() * batch_samples
        accum_reward_margin += reward_margin.item() * batch_samples
        accum_preference_prob += preference_prob.item() * batch_samples
        accum_chosen_tokens += chosen_tokens * batch_samples
        accum_rejected_tokens += rejected_tokens * batch_samples
        accum_samples += batch_samples

        if should_step:
            log_data = {
                "train/loss": accum_loss / accum_samples,
                "train/loss_dpo": accum_loss_dpo / accum_samples,
                "train/loss_nll": accum_loss_nll / accum_samples,
                "train/actor_chosen_prob": accum_actor_chosen_prob / accum_samples,
                "train/actor_rejected_prob": accum_actor_rejected_prob / accum_samples,
                "train/ref_chosen_prob": accum_ref_chosen_prob / accum_samples,
                "train/ref_rejected_prob": accum_ref_rejected_prob / accum_samples,
                "train/reward_margin": accum_reward_margin / accum_samples,
                "train/preference_prob": accum_preference_prob / accum_samples,
                "train/chosen_tokens": accum_chosen_tokens / accum_samples,
                "train/rejected_tokens": accum_rejected_tokens / accum_samples,
                "train/samples": seen_samples,
                "train/train_step": train_step,
            }
            should_eval = (
                eval_every_train_steps > 0
                and val_data_path
                and (
                    train_step % eval_every_train_steps == 0
                    or train_step == total_train_steps
                )
            )
            if should_eval:
                benchmark_result = evaluate_checkpoint(
                    actor_model,
                    tokenizer,
                    val_data_path=val_data_path,
                    eval_samples=eval_samples,
                    eval_batch_size=eval_batch_size,
                    eval_generate_config=eval_generate_config,
                )
                summary = benchmark_result.summary
                eval_accuracy = summary.accuracy
                log_data.update(
                    {
                        "eval/total": summary.total,
                        "eval/correct": summary.correct,
                        "eval/accuracy": eval_accuracy,
                        "eval/format_correct": summary.format_correct,
                        "eval/format_accuracy": summary.format_accuracy,
                        "eval/avg_output_len": summary.avg_output_len,
                        "eval/avg_output_len_correct": summary.avg_output_len_correct,
                        "eval/avg_output_len_format_ok_wrong": summary.avg_output_len_format_ok_wrong,
                        "eval/avg_output_len_format_wrong": summary.avg_output_len_format_wrong,
                    }
                )
                print(
                    f"[benchmark] train_step={train_step}/{total_train_steps} "
                    f"eval_samples={summary.total} "
                    f"accuracy={eval_accuracy:.4f} "
                )
            wandb.log(log_data, step=seen_samples)

            accum_loss = 0.0
            accum_loss_dpo = 0.0
            accum_loss_nll = 0.0
            accum_actor_chosen_prob = 0.0
            accum_actor_rejected_prob = 0.0
            accum_ref_chosen_prob = 0.0
            accum_ref_rejected_prob = 0.0
            accum_reward_margin = 0.0
            accum_preference_prob = 0.0
            accum_chosen_tokens = 0.0
            accum_rejected_tokens = 0.0
            accum_samples = 0

    actor_model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    wandb_run.config.update({"saved_checkpoint": str(ckpt_path)})
    wandb.finish()
    progress.close()
    print(f"[train_dpo] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    generate_pairs_pipeline()
    # train_dpo()
