from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM as VLLMEngine
from vllm import SamplingParams


@dataclass
class GenerateConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20


class LLM(ABC):
    @abstractmethod
    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[str]:
        pass

    def generate(self, messages: List[Dict[str, str]], config: GenerateConfig) -> str:
        return self.generate_batch([messages], config)[0]

    def close(self) -> None:
        pass


class TransformersLLM(LLM):
    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
    ) -> None:
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
    ) -> List[str]:
        input_texts = []
        for messages in batch_messages:
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_texts.append(input_text)

        model_inputs = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
        ).to(self._model.device)
        prompt_len = int(model_inputs["input_ids"].shape[1])
        out = self._model.generate(  # type: ignore
            **model_inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=config.temperature > 0,
            top_p=config.top_p,
            top_k=config.top_k,
        )

        output_texts = []
        for i in range(len(batch_messages)):
            output_ids = out[i][prompt_len:].tolist()
            output_texts.append(
                self._tokenizer.decode(output_ids, skip_special_tokens=True)
            )
        return output_texts


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

    def generate(self, messages: List[Dict[str, str]], config: GenerateConfig) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=cast(Any, messages),
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            extra_body={"top_k": config.top_k},
        )
        return resp.choices[0].message.content or ""

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        config: GenerateConfig,
    ) -> List[str]:
        if not batch_messages:
            return []

        with ThreadPoolExecutor(max_workers=len(batch_messages)) as executor:
            return list(
                executor.map(
                    lambda messages: self.generate(messages, config),
                    batch_messages,
                )
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
    ) -> List[str]:
        sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
        )
        outputs = self._llm.chat(
            cast(Any, batch_messages),
            sampling_params=sampling_params,
            use_tqdm=False,
            add_generation_prompt=True,
        )
        return [output.outputs[0].text for output in outputs]
