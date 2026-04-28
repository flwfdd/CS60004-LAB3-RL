#!/usr/bin/env python3

import json
import random
from pathlib import Path


def main() -> None:
    # 输入：benchmark 输出的 jsonl（每行一个 record，含 repeat_accuracy 字段）
    input_path = Path(
        "/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/benchmark/v2_tmp.jsonl"
    )
    # 输出：与 data/splits/raw_test.json 相同格式（JSON 数组）
    output_path = Path(
        "/mlx/users/fanliwen.2333/playground/code/CS60004-LAB3-RL/data/splits/raw_test_low_repeat_accuracy_v2_tmp.jsonl"
    )

    threshold = 1e-4
    seed = 42

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    selected = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            ra = record.get("repeat_accuracy", None)
            if ra is None:
                continue
            try:
                ra_val = float(ra)
            except Exception:
                continue

            if ra_val >= threshold:
                continue

            # 目标格式：{"id": ..., "target": ..., "nums": [...]}
            out_item = {
                "id": record.get("id"),
                "target": record.get("target"),
                "nums": record.get("nums"),
            }
            selected.append(out_item)

    random.Random(seed).shuffle(selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(
        f"selected={len(selected)} threshold={threshold} seed={seed} -> {output_path}"
    )


if __name__ == "__main__":
    main()
