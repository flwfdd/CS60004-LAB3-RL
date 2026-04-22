import json
import math
import os
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
    llm1: LLM,
    llm2: LLM,
    samples: list[Sample],
    generate_config: GenerateConfig,
    batch_size: int = 8,
) -> list[DPOSample]:
    """对同一批样本用两个 LLM 各推理一次，一对一错时构成偏好对"""
    batch_size = min(batch_size, len(samples))
    pairs: list[DPOSample] = []
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        results1 = eval_batch(llm1, batch, generate_config)
        results2 = eval_batch(llm2, batch, generate_config)
        for result1, result2 in zip(results1, results2):
            if (result1.ok and not result2.ok) or (
                result1.ok and result2.ok and (result1.output_len < result2.output_len)
            ):
                pairs.append(
                    DPOSample(
                        chosen_messages=result1.messages,
                        rejected_messages=result2.messages,
                    )
                )
            elif (
                not result1.ok
                and result2.ok
                or (
                    result1.ok
                    and result2.ok
                    and (result1.output_len > result2.output_len)
                )
            ):
                pairs.append(
                    DPOSample(
                        chosen_messages=result2.messages,
                        rejected_messages=result1.messages,
                    )
                )
    print(f"Generated {len(pairs)} preference pairs out of {len(samples)}")
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


def generate_pairs_pipeline() -> None:
    # 模型规模: 0p6b / 8b
    model_size = "0p6b"

    # 从 .env 读取模型配置
    model_name = os.getenv(f"MODEL_NAME_{model_size.upper()}", "")
    base_url = os.getenv(f"BASE_URL_{model_size.upper()}", "")
    api_key = os.getenv("API_KEY", "EMPTY")

    # 数据路径
    train_data_path = os.getenv("TRAIN_DATA_PATH", "")

    # 评测规模与生成配置
    max_samples = 10000
    batch_size = 1024
    max_new_tokens = 1024

    # 输出路径
    output_path = Path("data/dpo/0p6b_vs_0p6b_shorter.jsonl")

    # 同一个模型采样两次（不同温度）来构造偏好对
    llm1 = OpenAILLM(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )
    llm2 = OpenAILLM(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )

    generate_config = GenerateConfig(
        max_new_tokens=max_new_tokens,
    )

    try:
        samples = load_samples(Path(train_data_path), max_samples=max_samples)
        pairs = build_preference_pairs(
            llm1, llm2, samples, generate_config, batch_size=batch_size
        )
        save_pairs(pairs, output_path)
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
    micro_batch_size = 2
    train_batch_size = 16
    train_samples = 1000
    lr = 1e-6
    alpha = 1.0
    beta = 0.1
    eval_every_train_steps = 10
    eval_samples = 100
    eval_batch_size = 64
    eval_max_new_tokens = 1024
    model_path = os.getenv(f"MODEL_PATH_0P6B", "")
    pairs_path = Path("data/dpo/0p6b_vs_0p6b.jsonl")
    val_data_path = os.getenv("VAL_DATA_PATH", "")
    exp_name = "dpo_0p6b_vs_0p6b_1000_bs16_nll"
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
        loss = alpha * loss_nll + loss_dpo

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
                eval_accuracy = (
                    benchmark_result.correct / benchmark_result.total
                    if benchmark_result.total
                    else 0.0
                )
                log_data.update(
                    {
                        "eval/total": benchmark_result.total,
                        "eval/correct": benchmark_result.correct,
                        "eval/accuracy": eval_accuracy,
                        "eval/format_correct": benchmark_result.format_correct,
                        "eval/format_accuracy": (
                            benchmark_result.format_correct / benchmark_result.total
                            if benchmark_result.total
                            else 0.0
                        ),
                        "eval/avg_output_len": benchmark_result.avg_output_len,
                        "eval/avg_output_len_correct": benchmark_result.avg_output_len_correct,
                        "eval/avg_output_len_format_ok_wrong": benchmark_result.avg_output_len_format_ok_wrong,
                        "eval/avg_output_len_format_wrong": benchmark_result.avg_output_len_format_wrong,
                    }
                )
                print(
                    f"[benchmark] train_step={train_step}/{total_train_steps} "
                    f"eval_samples={benchmark_result.total} "
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
    # generate_pairs_pipeline()
    train_dpo()
