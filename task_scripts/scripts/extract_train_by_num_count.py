#!/usr/bin/env python3

import json
from pathlib import Path

INPUT_PATH = Path(
    "/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/splits/train.jsonl"
)
OUTPUT_DIR = Path(
    "/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/splits"
)


def main() -> None:
    output_paths = {
        "train_3": OUTPUT_DIR / "train_3.jsonl",
        "train_4": OUTPUT_DIR / "train_4.jsonl",
        "train_5": OUTPUT_DIR / "train_5.jsonl",
        "train_45": OUTPUT_DIR / "train_45.jsonl",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    writers = {
        name: path.open("w", encoding="utf-8") for name, path in output_paths.items()
    }
    counts = {name: 0 for name in output_paths}

    try:
        with INPUT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                num_count = len(item.get("nums", []))

                if num_count == 3:
                    writers["train_3"].write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts["train_3"] += 1
                if num_count == 4:
                    writers["train_4"].write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts["train_4"] += 1
                if num_count == 5:
                    writers["train_5"].write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts["train_5"] += 1
                if num_count in (4, 5):
                    writers["train_45"].write(json.dumps(item, ensure_ascii=False) + "\n")
                    counts["train_45"] += 1
    finally:
        for w in writers.values():
            w.close()

    for name, path in output_paths.items():
        print(f"{name}: {counts[name]} -> {path}")


if __name__ == "__main__":
    main()
