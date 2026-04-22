from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import torch
from common import LLM, GenerateConfig, Generation, build_model_inputs
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as VLLMEngine
from vllm import SamplingParams


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
    ) -> None:
        self.model_path = model_path
        self._llm = VLLMEngine(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
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
