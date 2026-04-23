import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import requests
import torch
from common import LLM, GenerateConfig, Generation, build_model_inputs
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as VLLMEngine
from vllm import SamplingParams
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)


class TransformersLLM(LLM):
    def __init__(
        self,
        model_path: str | None = None,
        *,
        device: str = "auto",
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if model is not None and tokenizer is not None:
            self.model_path = model_path or ""
            self._model = model
            self._tokenizer = tokenizer
            self._tokenizer.padding_side = "left"
            return

        if model_path is None:
            raise ValueError("require model_path or model")

        self.model_path = model_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
        )
        self._model.eval()

    @torch.no_grad()
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[Generation]:
        model_inputs = build_model_inputs(
            self._tokenizer,
            batch_messages,
            add_generation_prompt=True,
            device=self._model.device,
        )
        prompt_len = int(model_inputs["input_ids"].shape[1])  # type: ignore
        out = self._model.generate(
            **model_inputs,  # type: ignore
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        generations: List[Generation] = []
        for i in range(len(batch_messages)):
            output_ids_tensor = out[i][prompt_len:]
            completion_tokens = int(
                (output_ids_tensor != self._tokenizer.pad_token_id).sum().item()
            )
            output_ids = output_ids_tensor[:completion_tokens].tolist()
            generations.append(
                Generation(
                    text=self._tokenizer.decode(output_ids, skip_special_tokens=True),
                    completion_tokens=completion_tokens,
                )
            )
        return generations


class OpenAILLM(LLM):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def generate(
        self, messages: List[Dict[str, str]], config: GenerateConfig
    ) -> Generation:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={"top_k": config.top_k},
        )
        text = resp.choices[0].message.content or ""
        completion_tokens = 0
        usage = getattr(resp, "usage", None)
        if usage is not None and getattr(usage, "completion_tokens", None) is not None:
            completion_tokens = int(usage.completion_tokens)
        return Generation(text=text, completion_tokens=completion_tokens)

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[Generation]:
        if not batch_messages:
            return []

        with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
            return list(
                executor.map(lambda m: self.generate(m, config), batch_messages)
            )


class VllmLLM(LLM):
    def __init__(
        self,
        model_path: str,
        *,
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        max_model_len: int | None = None,
    ) -> None:
        self.model_path = model_path
        self.enforce_eager = enforce_eager
        self.max_model_len = max_model_len
        self._llm = VLLMEngine(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=max_model_len,
        )

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[Generation]:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        outputs = self._llm.chat(
            batch_messages,  # type: ignore
            sampling_params=sampling_params,
            use_tqdm=False,
            add_generation_prompt=True,
        )
        generations: List[Generation] = []
        for output in outputs:
            text = output.outputs[0].text
            completion_tokens = 0
            token_ids = getattr(output.outputs[0], "token_ids", None)
            if token_ids is not None:
                completion_tokens = len(token_ids)
            generations.append(
                Generation(text=text, completion_tokens=completion_tokens)
            )
        return generations


class VllmHttpIPCLLM(OpenAILLM):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        sync_timeout: float = 60.0,
    ) -> None:
        self._control_base_url = base_url.rstrip("/")
        self._sync_timeout = sync_timeout
        self._weight_transfer_initialized = False
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        super().__init__(
            base_url=f"{self._control_base_url}/v1",
            api_key=api_key,
            model_name=model_name,
        )

    def _post(
        self,
        path: str,
        *,
        json_payload: Dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> None:
        response = requests.post(
            f"{self._control_base_url}{path}",
            json=json_payload,
            timeout=timeout or self._sync_timeout,
        )
        response.raise_for_status()

    def init_weight_transfer_engine(self) -> None:
        if self._weight_transfer_initialized:
            return
        self._post("/init_weight_transfer_engine", json_payload={"init_info": dict()})
        self._weight_transfer_initialized = True

    def pause_generation(self) -> None:
        self._post("/pause")

    def resume_generation(self) -> None:
        self._post("/resume")

    def sync_from_actor(self, actor_model: Any) -> None:
        self.init_weight_transfer_engine()
        self.pause_generation()
        try:
            trainer_args = IPCTrainerSendWeightsArgs(
                mode="http",
                url=self._control_base_url,
            )
            IPCWeightTransferEngine.trainer_send_weights(
                iterator=actor_model.named_parameters(),
                trainer_args=trainer_args,
            )
        finally:
            self.resume_generation()
