import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal

import torch
import torch.nn.functional as F
from common import (
    LLM,
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
from llm import TransformersLLM, VllmHttpIPCLLM, VllmLLM
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import wandb

load_dotenv()


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

    # 只取目标 token 的 logprob，避免显式构造完整 [B, T, V] 的 log_softmax 结果导致 OOM
    selected_logits = shifted_logits.gather(
        dim=-1, index=shifted_input_ids.unsqueeze(-1)
    ).squeeze(
        -1
    )  # (batch_size, seq_len-1)
    log_denom = torch.logsumexp(shifted_logits, dim=-1)  # (batch_size, seq_len-1)
    logps = selected_logits - log_denom

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
            del old_logits
            ref_logits = ref_model(**model_inputs).logits
            ref_logps_chunks.append(
                compute_logps(ref_logits, input_ids, assistant_mask) / token_count
            )
            del ref_logits
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


def compute_math_density(
    messages: List[dict[str, str]],
) -> float:
    """
    计算数学密度：提取最后一条 assistant 的 <think>，统计其中数字和 +-*/() 的占比。
    """
    if not messages:
        return 0.0

    text = messages[-1].get("content", "")
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
    density = len(math_chars) / len(think_text)
    return density


def reward_fn(result: EvalResult) -> float:
    max_output_len = 1024
    length_bonus = (1024 - result.output_len) / 1024  # 0-1
    density_threshold = 0.3  # 封顶数学密度 超过就是刷分了
    density = compute_math_density(result.messages)  # 0-1
    density_bonus = min(density / density_threshold, 1.0)
    if result.ok:
        return 1.0 + 0.2 * length_bonus  # + 0.1 * len(result.sample.nums)
    if result.format_ok:
        # 格式对但答案错
        return 0.1
    # if result.output_len >= max_output_len:
    #     # 被截断 鼓励数学密度
    #     return 0.1 * density_bonus
    # 没被截断还不会格式的
    return 0.0


@dataclass
class ReadyRollout:
    groups: List[List[EvalResult]]
    rollout_time: float  # 生成 rollout 的耗时
    wait_time: float = 0.0  # 等待后台 rollout future 完成的耗时


class RolloutManager:
    """
    一个“单槽”rollout 管理器：把异步、同步频率、以及 eval 的安全点统一封装起来。

    状态机只有 3 个状态：
    - idle:   没有在跑的任务，也没有缓存结果
    - running:后台线程正在生成 rollout
    - ready:  rollout 已完成，但结果尚未被 flush 消费
    """

    def __init__(
        self,
        *,
        async_rollout: bool,
        rollout_backend: str,
        rollout_sync_freq: int,
        group_size: int,
        rollout_generate_config: GenerateConfig,
        tokenizer: Any,
        rollout_model_path: Path,
        rollout_base_url: str,
        rollout_model_name: str,
        rollout_api_key: str,
        vllm_gpu_memory_utilization: float,
        vllm_enforce_eager: bool,
        vllm_max_model_len: int | None,
    ) -> None:
        # transformers 直接复用训练中的 actor_model 不能异步
        if async_rollout and rollout_backend == "transformers":
            raise ValueError(
                "async_rollout=True is not supported when rollout_backend='transformers'"
            )

        self._async = async_rollout
        self._backend = rollout_backend
        self._sync_freq = rollout_sync_freq
        self._group_size = group_size
        self._gen_cfg = rollout_generate_config
        self._tokenizer = tokenizer
        self._rollout_model_path = rollout_model_path
        self._base_url = rollout_base_url
        self._model_name = rollout_model_name
        self._api_key = rollout_api_key
        self._vllm_gpu_mem_util = vllm_gpu_memory_utilization
        self._vllm_enforce_eager = vllm_enforce_eager
        self._vllm_max_model_len = vllm_max_model_len

        self._rollout_llm: LLM | None = None
        self._rollout_count = 0

        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1) if self._async else None
        )
        self._future: Future[ReadyRollout] | None = None
        self._ready: ReadyRollout | None = None

    @property
    def state(self) -> Literal["idle", "running", "ready"]:
        # ready 优先于 running 因为是先设置 ready 后清除 future
        if self._ready is not None:
            return "ready"
        if self._future is not None:
            return "running"
        return "idle"

    def _wait_running_to_ready(self) -> float:
        """
        如果有任务正在跑就等待其结束，并把结果转存到 ready（但不 flush 消费）
        """
        wait_start = time.perf_counter()
        if self._ready is None and self._future is not None:
            # 状态转换：running -> ready
            wait_start = time.perf_counter()
            self._ready = self._future.result()
            self._future = None
        return time.perf_counter() - wait_start

    def sync(self, actor_model: Any) -> None:
        """
        同步权重
        running 时先等待任务结束并缓存到 ready，再进行 sync
        """
        self._wait_running_to_ready()

        if self._backend == "transformers":
            self._rollout_llm = TransformersLLM(
                model=actor_model, tokenizer=self._tokenizer
            )
            return
        if self._backend == "vllm":
            if self._rollout_llm is not None:
                self._rollout_llm.close()
            actor_model.save_pretrained(self._rollout_model_path)
            self._tokenizer.save_pretrained(self._rollout_model_path)
            self._rollout_llm = VllmLLM(
                str(self._rollout_model_path),
                gpu_memory_utilization=self._vllm_gpu_mem_util,
                enforce_eager=self._vllm_enforce_eager,
                max_model_len=self._vllm_max_model_len,
            )
            return
        if self._backend == "vllm_http_ipc":
            if self._rollout_llm is None:
                self._rollout_llm = VllmHttpIPCLLM(
                    base_url=self._base_url,
                    api_key=self._api_key,
                    model_name=self._model_name,
                )
            assert isinstance(self._rollout_llm, VllmHttpIPCLLM)
            self._rollout_llm.sync_from_actor(actor_model)
            return

        raise ValueError(f"unsupported rollout_backend: {self._backend}")

    def submit(self, batch_samples: List[Sample], actor_model: Any) -> None:
        """
        提交一个 rollout 任务
        只能在 idle 状态 否则会覆盖上一个任务
        """
        assert self.state == "idle", "submit() must be called in idle state"
        assert len(batch_samples) > 0, "submit() batch_samples cannot be empty"

        # 首次 rollout 必须同步，之后只有 vllm / vllm_http_ipc 按频率同步
        should_sync = self._rollout_llm is None or (
            self._backend in {"vllm", "vllm_http_ipc"}
            and self._sync_freq > 0
            and (self._rollout_count % self._sync_freq == 0)
        )
        if should_sync:
            self.sync(actor_model)
        assert self._rollout_llm is not None, "submit() requires backend initialization"

        start_time = time.perf_counter()

        def _do_rollout() -> ReadyRollout:
            assert self._rollout_llm is not None
            # 每个 prompt 重复 group_size 次
            rollout_samples = [
                sample for sample in batch_samples for _ in range(self._group_size)
            ]
            if self._backend == "transformers":
                was_training = actor_model.training
                actor_model.eval()
                try:
                    results = eval_batch(
                        self._rollout_llm, rollout_samples, self._gen_cfg
                    )
                finally:
                    if was_training:
                        actor_model.train()
            else:
                results = eval_batch(self._rollout_llm, rollout_samples, self._gen_cfg)
            groups = [
                results[i : i + self._group_size]
                for i in range(0, len(rollout_samples), self._group_size)
            ]
            return ReadyRollout(
                groups=groups,
                rollout_time=time.perf_counter() - start_time,
            )

        if self._async:
            assert self._executor is not None
            self._future = self._executor.submit(_do_rollout)
        else:
            self._ready = _do_rollout()

        self._rollout_count += 1

    def flush(self) -> ReadyRollout:
        """
        消费上一批提交的任务结果
        idle 状态直接报错 执行完会回到 idle 状态
        """
        assert self.state != "idle", "flush() requires a submitted task"

        # 如果还在跑，就先等待结束进入 ready
        wait_time = self._wait_running_to_ready()

        assert self._ready is not None
        ready = self._ready
        # 状态转换：ready -> idle
        self._ready = None
        ready.wait_time = wait_time
        return ready

    def eval(
        self,
        *,
        actor_model: Any,
        tokenizer: Any,
        eval_backend: str,
        val_data_path: str,
        eval_samples: int,
        eval_batch_size: int,
        eval_generate_config: GenerateConfig,
    ) -> BenchmarkResult:
        """
        做评测，并保证不破坏 rollout 暂存状态：
        - 如果 running：等待任务结束，把结果缓存到 ready，但不 flush
        - 然后在安全点（无生成在飞）做一次 sync，再进行评测
        """
        self._wait_running_to_ready()

        if eval_backend == "transformers":
            # 和你原来的逻辑保持一致：评测期间切 eval，结束后恢复 train
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

        if eval_backend == "vllm_http_ipc":
            # 复用同一个 HTTP vLLM server；上面已经 sync 过权重
            self.sync(actor_model)
            assert (
                self._rollout_llm is not None
            ), "eval_backend=vllm_http_ipc requires backend initialization"
            assert isinstance(self._rollout_llm, VllmHttpIPCLLM)
            return run_benchmark(
                self._rollout_llm,
                val_data_path,
                max_samples=eval_samples,
                batch_size=eval_batch_size,
                generate_config=eval_generate_config,
            )

        raise ValueError(f"unsupported eval_backend: {eval_backend}")

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        if self._rollout_llm is not None:
            self._rollout_llm.close()


def train_grpo() -> None:
    """
    每轮取prompt_batch_size个prompt，每个prompt采样group_size个回答
    共train_batch_size=prompt_batch_size*group_size个样本一次性rollout
    然后按micro_batch_size分批次计算累积梯度到mini_batch_size更新参数
    按照mini_batch_size也就是参数更新来统计step
    """
    rollout_backend = "vllm_http_ipc"
    eval_backend = rollout_backend
    rollout_sync_freq = 1
    rollout_base_url = os.getenv("ROLLOUT_BASE_URL", "http://localhost:8006")
    rollout_model_name = os.getenv("MODEL_NAME_0P6B", "Qwen3-0.6B")
    rollout_api_key = os.getenv("ROLLOUT_API_KEY", "EMPTY")
    vllm_gpu_memory_utilization = 0.3
    vllm_enforce_eager = True
    vllm_max_model_len = 8192
    shuffle_rollout = True
    async_rollout = True
    prompt_batch_size = 8
    group_size = 8
    train_batch_size = prompt_batch_size * group_size
    mini_batch_size = 16
    micro_batch_size = 2
    rollout_logp_micro_batch = 8
    train_sample_start = 0
    train_samples = 2048
    lr = 2e-6
    scheduler_type = "constant"
    warmup_ratio = 0.1
    epsilon = 0.2
    beta = 1e-4  # for kl loss
    eval_every_train_steps = 10
    eval_samples = 500
    eval_batch_size = 256
    rollout_max_new_tokens = 1024
    rollout_temperature = 1.0
    eval_temperature = 0.6
    eval_max_new_tokens = 1024

    model_path = os.getenv("MODEL_PATH_0P6B", "")
    model_path = "/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/ckpt/grpo_trainbs64_minibs16_gs8_test_basedpoep10/best"
    train_data_path = os.getenv("TRAIN_DATA_PATH", "")
    train_data_path = "data/splits/raw_test_low_repeat_accuracy_v2_repeat.jsonl"
    val_data_path = os.getenv("VAL_DATA_PATH", "")
    val_data_path = "data/splits/raw_test.jsonl"
    exp_name = f"grpo_trainbs{train_batch_size}_minibs{mini_batch_size}_gs{group_size}_test_basedpoep10_v2"
    ckpt_path = Path(f"data/ckpt/{exp_name}")
    ckpt_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = Path(f"data/ckpt/{exp_name}/best")
    rollout_model_path = ckpt_path / "rollout_model"
    rollout_model_path.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        Path(train_data_path), max_samples=train_sample_start + train_samples
    )[train_sample_start:]
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
    warmup_steps = int(total_train_steps * warmup_ratio)
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )

    wandb_run = wandb.init(
        project="cs60004-lab3-rl",
        name=exp_name,
        config={
            "model_path": model_path,
            "train_data_path": train_data_path,
            "ckpt_path": str(ckpt_path),
            "best_ckpt_path": str(best_ckpt_path),
            "rollout_backend": rollout_backend,
            "eval_backend": eval_backend,
            "rollout_sync_freq": rollout_sync_freq,
            "rollout_model_path": str(rollout_model_path),
            "rollout_base_url": rollout_base_url,
            "rollout_model_name": rollout_model_name,
            "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
            "vllm_enforce_eager": vllm_enforce_eager,
            "vllm_max_model_len": vllm_max_model_len,
            "total_samples": len(samples),
            "shuffle_rollout": shuffle_rollout,
            "prompt_batch_size": prompt_batch_size,
            "train_batch_size": train_batch_size,
            "mini_batch_size": mini_batch_size,
            "micro_batch_size": micro_batch_size,
            "group_size": group_size,
            "total_train_steps": total_train_steps,
            "lr": lr,
            "scheduler_type": scheduler_type,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "epsilon": epsilon,
            "beta": beta,
            "val_data_path": val_data_path,
            "eval_every_train_steps": eval_every_train_steps,
            "eval_samples": eval_samples,
            "eval_batch_size": eval_batch_size,
            "rollout_max_new_tokens": rollout_max_new_tokens,
            "rollout_temperature": rollout_temperature,
            "eval_temperature": eval_temperature,
            "eval_max_new_tokens": eval_max_new_tokens,
            "device": str(device),
        },
    )
    optimizer.zero_grad()
    train_step = 0
    seen_prompts = 0
    seen_samples = 0
    best_eval_accuracy = float("-inf")
    best_eval_step = -1
    rollout_generate_config = GenerateConfig(
        max_new_tokens=rollout_max_new_tokens,
        temperature=rollout_temperature,
    )
    rollout_manager = RolloutManager(
        async_rollout=async_rollout,
        rollout_backend=rollout_backend,
        rollout_sync_freq=rollout_sync_freq,
        group_size=group_size,
        rollout_generate_config=rollout_generate_config,
        tokenizer=tokenizer,
        rollout_model_path=rollout_model_path,
        rollout_base_url=rollout_base_url,
        rollout_model_name=rollout_model_name,
        rollout_api_key=rollout_api_key,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_enforce_eager=vllm_enforce_eager,
        vllm_max_model_len=vllm_max_model_len,
    )
    eval_generate_config = GenerateConfig(
        max_new_tokens=eval_max_new_tokens,
        temperature=eval_temperature,
    )

    progress = tqdm(total=len(samples) * group_size, desc="Training GRPO")
    prompt_batches = [
        samples[i : i + prompt_batch_size]
        for i in range(0, len(samples), prompt_batch_size)
    ]
    if not prompt_batches:
        return

    def train_on_ready(ready_rollout: ReadyRollout) -> None:
        nonlocal train_step, seen_samples, best_eval_accuracy, best_eval_step
        rollout_groups = ready_rollout.groups
        rollout_time = ready_rollout.rollout_time
        rollout_wait_time = ready_rollout.wait_time
        # 平铺开 每个组的在一起
        flat_rollout_results = [result for group in rollout_groups for result in group]
        # 这一步会跑 actor/ref 两次 forward 来预计算 old/ref logp，可能很耗时
        precompute_start = time.perf_counter()
        rollout_results = build_rollout_batch(
            actor_model,
            ref_model,
            tokenizer,
            flat_rollout_results,
            group_size=group_size,
            batch_size=rollout_logp_micro_batch,
            device=device,
        )
        precompute_time = time.perf_counter() - precompute_start

        assert len(rollout_results) == train_batch_size
        if shuffle_rollout:
            shuffle_indices = torch.randperm(train_batch_size).tolist()
            rollout_results = [rollout_results[idx] for idx in shuffle_indices]

        num_mini_batches = train_batch_size // mini_batch_size
        rollout_time_per_mini = rollout_time / num_mini_batches
        rollout_wait_time_per_mini = rollout_wait_time / num_mini_batches
        precompute_time_per_mini = precompute_time / num_mini_batches

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

                # 省点显存
                del actor_logits
                del actor_logps
                del old_logps
                del ref_logps
                del advantages
                del entropy
                del loss
                del policy_loss
                del kl_loss
                del scaled_loss
                del input_ids
                del assistant_mask
                del token_count
                del model_inputs

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_step += 1
            seen_samples += mini_batch_size
            compute_time = time.perf_counter() - mini_start_time
            # total_time 是真实时间成本 包含等待异步 rollout 时间和训练时间
            total_time = (
                compute_time + rollout_wait_time_per_mini + precompute_time_per_mini
            )
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
                "time/rollout_wait_sec": rollout_wait_time_per_mini,
                "time/precompute_sec": precompute_time_per_mini,
                "time/compute_sec": compute_time,
                "time/total_sec": total_time,
                "train/prompts": seen_prompts,
                "train/samples": seen_samples,
                "train/train_step": train_step,
                "train/lr": scheduler.get_last_lr()[0],
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
                # eval 会等到当前 rollout 任务结束，并在安全点 sync 后再推理，不影响暂存的 ready rollout
                benchmark_result = rollout_manager.eval(
                    actor_model=actor_model,
                    tokenizer=tokenizer,
                    eval_backend=eval_backend,
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
                        "eval/accuracy": eval_accuracy,
                        "eval/format_accuracy": summary.format_accuracy,
                        "eval/avg_output_len": summary.avg_output_len,
                        "eval/avg_output_len_correct": summary.avg_output_len_correct,
                        "eval/avg_output_len_format_ok_wrong": summary.avg_output_len_format_ok_wrong,
                        "eval/avg_output_len_format_wrong": summary.avg_output_len_format_wrong,
                    }
                )
                # 只有评测精度创新高时才额外保存一份 best ckpt，不影响最终 ckpt 正常落盘。
                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    best_eval_step = train_step
                    best_ckpt_path.mkdir(parents=True, exist_ok=True)
                    actor_model.save_pretrained(best_ckpt_path)
                    tokenizer.save_pretrained(best_ckpt_path)
                    print(
                        f"[best_ckpt] saved to {best_ckpt_path} "
                        f"(step={best_eval_step}, accuracy={best_eval_accuracy:.4f})"
                    )
                for num_count, stats in sorted(benchmark_result.by_num_count.items()):
                    prefix = f"eval/{num_count}"
                    log_data.update(
                        {
                            f"{prefix}/total": stats.total,
                            f"{prefix}/accuracy": stats.accuracy,
                            f"{prefix}/format_accuracy": stats.format_accuracy,
                            f"{prefix}/avg_output_len": stats.avg_output_len,
                            f"{prefix}/avg_output_len_correct": stats.avg_output_len_correct,
                            f"{prefix}/avg_output_len_format_ok_wrong": stats.avg_output_len_format_ok_wrong,
                            f"{prefix}/avg_output_len_format_wrong": stats.avg_output_len_format_wrong,
                        }
                    )
                print(
                    f"[benchmark] train_step={train_step}/{total_train_steps} "
                    f"eval_samples={summary.total} "
                    f"accuracy={eval_accuracy:.4f} "
                )
            wandb.log(log_data, step=train_step)

    # 预热：第一批先提交并 flush（没有上一批可训练）
    first_batch = prompt_batches[0]
    seen_prompts += len(first_batch)
    rollout_manager.submit(first_batch, actor_model)
    ready = rollout_manager.flush()

    # 提交rollout任务后用上一批的结果训练 同步进行 训练完成后取回这一批rollout结果用于下次训练
    for next_batch in prompt_batches[1:]:
        seen_prompts += len(next_batch)
        rollout_manager.submit(next_batch, actor_model)
        train_on_ready(ready)
        ready = rollout_manager.flush()

    # 最后一批
    train_on_ready(ready)

    actor_model.save_pretrained(ckpt_path)
    tokenizer.save_pretrained(ckpt_path)
    rollout_manager.close()
    wandb_run.config.update({"saved_checkpoint": str(ckpt_path)})
    if best_eval_step >= 0:
        wandb_run.summary["saved_best_checkpoint"] = str(best_ckpt_path)
        wandb_run.summary["best_eval_accuracy"] = best_eval_accuracy
        wandb_run.summary["best_eval_step"] = best_eval_step
    wandb.finish()
    progress.close()
    print(f"[train_grpo] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train_grpo()
