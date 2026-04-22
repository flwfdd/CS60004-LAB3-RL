import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
import torch.nn.functional as F
from common import (
    BenchmarkResult,
    EvalResult,
    GenerateConfig,
    Sample,
    build_training_inputs,
    eval_batch,
    load_samples,
    run_benchmark,
)
from dotenv import load_dotenv
from llm import TransformersLLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

load_dotenv()


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


def collect_group_rollouts(
    actor_model: Any,
    tokenizer: Any,
    batch_samples: List[Sample],
    *,
    group_size: int,
    rollout_generate_config: GenerateConfig,
) -> List[List[EvalResult]]:
    if not batch_samples:
        return []

    # 同一个sample连续group_size次
    rollout_samples = [sample for sample in batch_samples for _ in range(group_size)]
    llm = TransformersLLM(model=actor_model, tokenizer=tokenizer)
    was_training = actor_model.training
    actor_model.eval()
    try:
        results = eval_batch(llm, rollout_samples, rollout_generate_config)
    finally:
        if was_training:
            actor_model.train()

    return [
        results[i : i + group_size] for i in range(0, len(rollout_samples), group_size)
    ]


@dataclass
class RolloutResult:
    result: EvalResult
    reward: float
    reward_std: float
    advantage: float
    old_logp: float
    ref_logp: float
    response_tokens: float


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


def compute_mean_entropy(
    logits: torch.Tensor,  # (batch_size, seq_len, vocab_size)
    mask: torch.Tensor,  # (batch_size, seq_len)
    chunk_size: int = 32,
) -> torch.Tensor:
    # 错位
    shifted_logits = logits[:, :-1, :]  # (batch_size, seq_len-1, vocab_size)
    shifted_mask = mask[:, 1:].float()  # (batch_size, seq_len-1)
    token_count = shifted_mask.sum(-1).clamp_min(1.0)  # (batch_size,)
    seq_entropy = torch.zeros_like(token_count)

    # 按 seq 维分块，避免为整段 logits 同时构造 full-vocab 中间张量而 OOM
    for start in range(0, shifted_logits.shape[1], chunk_size):
        end = min(start + chunk_size, shifted_logits.shape[1])
        chunk_logits = shifted_logits[:, start:end, :]
        chunk_mask = shifted_mask[:, start:end]
        log_probs = F.log_softmax(chunk_logits, dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (batch_size, chunk_len)
        seq_entropy += (token_entropy * chunk_mask).sum(dim=-1)

    seq_entropy = seq_entropy / token_count
    return seq_entropy.mean()


def compute_group_advantages(
    rewards: torch.Tensor,  # (num_prompts * group_size,)
    group_size: int,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    grouped_rewards = rewards.view(-1, group_size)
    reward_std = grouped_rewards.std(dim=-1, keepdim=True)
    advantages = (grouped_rewards - grouped_rewards.mean(dim=-1, keepdim=True)) / (
        reward_std + eps
    )
    return advantages.reshape(-1), reward_std.expand_as(grouped_rewards).reshape(-1)


def grpo_loss(
    actor_logps: torch.Tensor,  # (batch_size,)
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float = 0.2,
    beta: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ratios = torch.exp(actor_logps - old_logps)
    clipped_ratios = torch.clamp(ratios, min=1 - epsilon, max=1 + epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()
    kl_loss = (torch.exp(ref_logps - actor_logps) - ref_logps + actor_logps - 1).mean()
    total_loss = policy_loss + beta * kl_loss
    return total_loss, policy_loss, kl_loss


def build_rollout_batch(
    actor_model: Any,
    ref_model: Any,
    tokenizer: Any,
    rollout_results: List[EvalResult],
    *,
    group_size: int,
    batch_size: int,
    device: torch.device,
    eps: float = 1e-8,
) -> List[RolloutResult]:
    rollout_messages = [result.messages for result in rollout_results]
    rewards = torch.tensor(
        [reward_fn(result) for result in rollout_results],
        dtype=torch.float32,
        device=device,
    )
    advantages, reward_std = compute_group_advantages(rewards, group_size, eps=eps)

    was_training = actor_model.training
    actor_model.eval()

    old_logps_chunks: list[torch.Tensor] = []
    ref_logps_chunks: list[torch.Tensor] = []
    token_count_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rollout_messages), batch_size):
            end = min(start + batch_size, len(rollout_messages))
            model_inputs = build_training_inputs(
                tokenizer,
                rollout_messages[start:end],
                device=device,
            )
            input_ids: torch.Tensor = model_inputs["input_ids"]  # type: ignore
            assistant_mask: torch.Tensor = model_inputs["assistant_mask"]  # type: ignore
            token_count = assistant_mask.sum(dim=1).float().clamp_min(1)

            old_logits = actor_model(**model_inputs).logits
            old_logps_chunks.append(
                compute_logps(old_logits, input_ids, assistant_mask) / token_count
            )
            ref_logits = ref_model(**model_inputs).logits
            ref_logps_chunks.append(
                compute_logps(ref_logits, input_ids, assistant_mask) / token_count
            )
            token_count_chunks.append(token_count)

    if was_training:
        actor_model.train()

    old_logps = torch.cat(old_logps_chunks, dim=0)
    ref_logps = torch.cat(ref_logps_chunks, dim=0)
    token_count = torch.cat(token_count_chunks, dim=0)

    rollout_batch: List[RolloutResult] = []
    for i, result in enumerate(rollout_results):
        rollout_batch.append(
            RolloutResult(
                result=result,
                reward=float(rewards[i].item()),
                reward_std=float(reward_std[i].item()),
                advantage=float(advantages[i].item()),
                old_logp=float(old_logps[i].item()),
                ref_logp=float(ref_logps[i].item()),
                response_tokens=float(token_count[i].item()),
            )
        )
    return rollout_batch


def reward_fn(result: EvalResult) -> float:
    if result.ok:
        return 1.0
    if result.format_ok:
        return 0.2
    return 0.0


def train_grpo() -> None:
    """
    每轮取prompt_batch_size个prompt，每个prompt采样group_size个回答
    共train_batch_size=prompt_batch_size*group_size个样本一次性rollout
    然后按micro_batch_size分批次计算累积梯度到mini_batch_size更新参数
    按照mini_batch_size也就是参数更新来统计step
    """
    shuffle_rollout = True
    prompt_batch_size = 8
    group_size = 4
    train_batch_size = prompt_batch_size * group_size
    mini_batch_size = 16
    micro_batch_size = 4
    train_samples = 1000
    lr = 1e-6
    epsilon = 0.2
    beta = 0.01
    eval_every_train_steps = 10
    eval_samples = 100
    eval_batch_size = 64
    rollout_max_new_tokens = 1024
    eval_max_new_tokens = 1024

    model_path = os.getenv("MODEL_PATH_0P6B", "")
    train_data_path = os.getenv("TRAIN_DATA_PATH", "")
    val_data_path = os.getenv("VAL_DATA_PATH", "")
    exp_name = f"grpo_gs{group_size}"
    ckpt_path = Path(f"data/ckpt/{exp_name}")
    ckpt_path.mkdir(parents=True, exist_ok=True)

    samples = load_samples(Path(train_data_path), max_samples=train_samples)
    assert train_batch_size % group_size == 0
    assert mini_batch_size % group_size == 0
    assert train_batch_size % mini_batch_size == 0
    assert mini_batch_size % micro_batch_size == 0
    assert len(samples) % prompt_batch_size == 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    actor_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  # type: ignore
    ref_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)  # type: ignore
    actor_model.train()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=lr)
    total_train_steps = (len(samples) * group_size) // mini_batch_size
    wandb_run = wandb.init(
        project="cs60004-lab3-rl",
        name=exp_name,
        config={
            "model_path": model_path,
            "train_data_path": train_data_path,
            "ckpt_path": str(ckpt_path),
            "total_samples": len(samples),
            "shuffle_rollout": shuffle_rollout,
            "prompt_batch_size": prompt_batch_size,
            "train_batch_size": train_batch_size,
            "mini_batch_size": mini_batch_size,
            "micro_batch_size": micro_batch_size,
            "group_size": group_size,
            "total_train_steps": total_train_steps,
            "lr": lr,
            "epsilon": epsilon,
            "beta": beta,
            "val_data_path": val_data_path,
            "eval_every_train_steps": eval_every_train_steps,
            "eval_samples": eval_samples,
            "eval_batch_size": eval_batch_size,
            "rollout_max_new_tokens": rollout_max_new_tokens,
            "eval_max_new_tokens": eval_max_new_tokens,
            "device": str(device),
        },
    )
    optimizer.zero_grad()
    train_step = 0
    seen_prompts = 0
    seen_samples = 0
    rollout_generate_config = GenerateConfig(
        max_new_tokens=rollout_max_new_tokens,
    )
    eval_generate_config = GenerateConfig(max_new_tokens=eval_max_new_tokens)

    progress = tqdm(total=len(samples) * group_size, desc="Training GRPO")
    for i in range(0, len(samples), prompt_batch_size):
        prompt_batch = samples[i : i + prompt_batch_size]
        prompt_count = len(prompt_batch)
        seen_prompts += prompt_count

        rollout_start_time = time.perf_counter()
        rollout_groups = collect_group_rollouts(
            actor_model,
            tokenizer,
            prompt_batch,
            group_size=group_size,
            rollout_generate_config=rollout_generate_config,
        )
        rollout_time = time.perf_counter() - rollout_start_time
        # 平铺开 每个组的在一起
        flat_rollout_results = [result for group in rollout_groups for result in group]
        rollout_results = build_rollout_batch(
            actor_model,
            ref_model,
            tokenizer,
            flat_rollout_results,
            group_size=group_size,
            batch_size=micro_batch_size,
            device=device,
        )

        assert len(rollout_results) == train_batch_size
        if shuffle_rollout:
            shuffle_indices = torch.randperm(train_batch_size).tolist()
            rollout_results = [rollout_results[idx] for idx in shuffle_indices]

        num_mini_batches = train_batch_size // mini_batch_size
        rollout_time_per_mini = rollout_time / num_mini_batches

        for mini_start in range(0, train_batch_size, mini_batch_size):
            mini_batch = rollout_results[mini_start : mini_start + mini_batch_size]
            assert len(mini_batch) == mini_batch_size
            mini_start_time = time.perf_counter()
            optimizer.zero_grad()

            accum_loss = 0.0
            accum_policy_loss = 0.0
            accum_kl_loss = 0.0
            accum_reward = 0.0
            accum_reward_std = 0.0
            accum_accuracy = 0.0
            accum_format_accuracy = 0.0
            accum_advantage_abs = 0.0
            accum_entropy = 0.0
            accum_response_tokens = 0.0

            for micro_start in range(0, mini_batch_size, micro_batch_size):
                micro_batch = mini_batch[micro_start : micro_start + micro_batch_size]
                assert len(micro_batch) == micro_batch_size
                micro_messages = [item.result.messages for item in micro_batch]

                model_inputs = build_training_inputs(
                    tokenizer,
                    micro_messages,
                    device=device,
                )
                input_ids: torch.Tensor = model_inputs["input_ids"]  # type: ignore
                assistant_mask: torch.Tensor = model_inputs["assistant_mask"]  # type: ignore
                token_count = assistant_mask.sum(dim=1).float().clamp_min(1)

                actor_logits = actor_model(**model_inputs).logits
                actor_logps = (
                    compute_logps(actor_logits, input_ids, assistant_mask) / token_count
                )
                entropy = compute_mean_entropy(actor_logits.detach(), assistant_mask)

                old_logps = torch.tensor(
                    [result.old_logp for result in micro_batch],
                    dtype=torch.float32,
                    device=device,
                )
                ref_logps = torch.tensor(
                    [result.ref_logp for result in micro_batch],
                    dtype=torch.float32,
                    device=device,
                )
                advantages = torch.tensor(
                    [result.advantage for result in micro_batch],
                    dtype=torch.float32,
                    device=device,
                )

                loss, policy_loss, kl_loss = grpo_loss(
                    actor_logps,
                    old_logps,
                    ref_logps,
                    advantages,
                    epsilon=epsilon,
                    beta=beta,
                )

                scaled_loss = loss * (micro_batch_size / mini_batch_size)
                scaled_loss.backward()

                micro_accuracy = sum(int(item.result.ok) for item in micro_batch) / len(
                    micro_batch
                )
                micro_format_accuracy = sum(
                    int(item.result.format_ok) for item in micro_batch
                ) / len(micro_batch)
                micro_reward = sum(item.reward for item in micro_batch) / len(
                    micro_batch
                )
                micro_reward_std = sum(item.reward_std for item in micro_batch) / len(
                    micro_batch
                )
                micro_advantage_abs = advantages.abs().mean().item()
                micro_entropy = entropy.item()
                micro_response_tokens = sum(
                    item.response_tokens for item in micro_batch
                ) / len(micro_batch)
                accum_loss += loss.item() * micro_batch_size
                accum_policy_loss += policy_loss.item() * micro_batch_size
                accum_kl_loss += kl_loss.item() * micro_batch_size
                accum_reward += micro_reward * micro_batch_size
                accum_reward_std += micro_reward_std * micro_batch_size
                accum_accuracy += micro_accuracy * micro_batch_size
                accum_format_accuracy += micro_format_accuracy * micro_batch_size
                accum_advantage_abs += micro_advantage_abs * micro_batch_size
                accum_entropy += micro_entropy * micro_batch_size
                accum_response_tokens += micro_response_tokens * micro_batch_size
                progress.update(micro_batch_size)

            optimizer.step()
            optimizer.zero_grad()
            train_step += 1
            seen_samples += mini_batch_size
            total_time = rollout_time_per_mini + (time.perf_counter() - mini_start_time)
            log_data = {
                "train/loss": accum_loss / mini_batch_size,
                "train/policy_loss": accum_policy_loss / mini_batch_size,
                "train/kl_loss": accum_kl_loss / mini_batch_size,
                "train/reward": accum_reward / mini_batch_size,
                "train/reward_std": accum_reward_std / mini_batch_size,
                "train/accuracy": accum_accuracy / mini_batch_size,
                "train/format_accuracy": accum_format_accuracy / mini_batch_size,
                "train/advantage_abs": accum_advantage_abs / mini_batch_size,
                "train/entropy": accum_entropy / mini_batch_size,
                "train/response_tokens": accum_response_tokens / mini_batch_size,
                "time/rollout_sec": rollout_time_per_mini,
                "time/total_sec": total_time,
                "train/prompts": seen_prompts,
                "train/samples": seen_samples,
                "train/train_step": train_step,
            }
            print(log_data)
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
            wandb.log(log_data, step=train_step)

    actor_model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    wandb_run.config.update({"saved_checkpoint": str(ckpt_path)})
    wandb.finish()
    progress.close()
    print(f"[train_grpo] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_grpo()
