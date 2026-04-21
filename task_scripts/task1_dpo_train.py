from dataclasses import dataclass

from eval_common import GenerateConfig, Sample, eval_batch
from llm import LLM, OpenAILLM
from tqdm import tqdm


@dataclass
class DPOSample:
    sample: Sample
    chosen: str
    rejected: str


def generate_dpo_samples(
    llm1: LLM,
    llm2: LLM,
    samples: list[Sample],
    generate_config: GenerateConfig,
    batch_size: int = 8,
) -> list[DPOSample]:
    """批量推理两遍，如果一对一错就作为样本"""
    batch_size = min(batch_size, len(samples))
    dpo_samples = []
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        results1 = eval_batch(llm1, batch, generate_config)
        results2 = eval_batch(llm2, batch, generate_config)
        for sample, result1, result2 in zip(batch, results1, results2):
            if result1.ok and not result2.ok:
                dpo_samples.append(
                    DPOSample(
                        sample,
                        result1.messages[-1]["content"],
                        result2.messages[-1]["content"],
                    )
                )
            elif not result1.ok and result2.ok:
                dpo_samples.append(
                    DPOSample(
                        sample,
                        result2.messages[-1]["content"],
                        result1.messages[-1]["content"],
                    )
                )
    print(f"Generated {len(dpo_samples)} DPO samples out of {len(samples)}")
    return dpo_samples
