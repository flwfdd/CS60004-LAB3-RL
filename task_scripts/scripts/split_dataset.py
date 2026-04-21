#!/usr/bin/env python3

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd

INPUT_PATH = Path("/root/autodl-tmp/code/CS60004-LAB3-RL/data/raw_train.parquet")
OUTPUT_DIR = Path("/root/autodl-tmp/code/CS60004-LAB3-RL/data/splits")

SAMPLE_COUNT = 10000

SPLIT_NAMES = ["train", "val"]
SPLIT_RATIOS = [0.8, 0.2]

SEED = 42


def sample_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if SAMPLE_COUNT > len(df):
        print("Sampling Mode: full dataset")
        return df.reset_index(drop=True)
    sampled_df = df.sample(n=SAMPLE_COUNT, random_state=SEED)
    print(f"Sampling Mode: count={SAMPLE_COUNT}")
    return sampled_df.reset_index(drop=True)


def compute_split_sizes(total: int, ratios: Iterable[float]) -> List[int]:
    ratios = list(ratios)
    s = sum(ratios)
    sizes = [int(total * r / s) for r in ratios]
    sizes[-1] = total - sum(sizes[:-1])  # make sure sum(sizes) == total
    return sizes


def save_splits(
    df: pd.DataFrame,
    output_dir: Path,
    split_names: List[str],
    split_sizes: List[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    start = 0
    for name, size in zip(split_names, split_sizes):
        end = start + size
        split_df = df.iloc[start:end].reset_index(drop=True)
        output_path = output_dir / f"{name}.jsonl"
        with output_path.open("w", encoding="utf-8") as f:
            for row in split_df.to_dict(orient="records"):
                nums = list(row.get("nums", []))
                row["nums"] = [int(v) for v in nums]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split_df)} rows -> {output_path}")
        start = end


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    sampled_df = sample_dataframe(df)
    print(f"Rows After Sampling: {len(sampled_df)}")

    split_sizes = compute_split_sizes(len(sampled_df), SPLIT_RATIOS)
    save_splits(sampled_df, OUTPUT_DIR, SPLIT_NAMES, split_sizes)


if __name__ == "__main__":
    main()
