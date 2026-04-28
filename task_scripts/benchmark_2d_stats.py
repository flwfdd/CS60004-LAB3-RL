import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class Record:
    nums: List[int]
    ok: bool
    format_ok: bool
    output_len: int


@dataclass
class Stats:
    total: int
    correct: int
    format_correct: int
    accuracy: float
    format_accuracy: float
    avg_output_len: float


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_records(path: Path) -> List[Record]:
    records: List[Record] = []
    for item in iter_jsonl(path):
        records.append(
            Record(
                nums=[int(x) for x in item["nums"]],
                ok=bool(item["ok"]),
                format_ok=bool(item["format_ok"]),
                output_len=int(item["output_len"]),
            )
        )
    return records


def summarize(records: List[Record], trunc_len: int) -> Stats:
    total = len(records)
    if total == 0:
        return Stats(
            total=0,
            correct=0,
            format_correct=0,
            accuracy=0.0,
            format_accuracy=0.0,
            avg_output_len=0.0,
        )

    # 近似策略：若输出 token 数超过截断长度，则认为在该截断长度下会被截断，
    # 导致格式与正确性都失败；未超截断长度则沿用原始评测结果。
    correct = sum(1 for r in records if r.output_len <= trunc_len and r.ok)
    format_correct = sum(1 for r in records if r.output_len <= trunc_len and r.format_ok)
    avg_output_len = sum(min(r.output_len, trunc_len) for r in records) / total

    return Stats(
        total=total,
        correct=correct,
        format_correct=format_correct,
        accuracy=correct / total,
        format_accuracy=format_correct / total,
        avg_output_len=avg_output_len,
    )


def build_2d_stats(
    records: List[Record],
    trunc_lens: List[int],
) -> Dict[int, Dict[int, Stats]]:
    by_num_count: Dict[int, List[Record]] = {}
    for r in records:
        by_num_count.setdefault(len(r.nums), []).append(r)

    table: Dict[int, Dict[int, Stats]] = {}
    for trunc_len in trunc_lens:
        table[trunc_len] = {}
        for num_count in sorted(by_num_count.keys()):
            table[trunc_len][num_count] = summarize(by_num_count[num_count], trunc_len)
    return table


def print_table(table: Dict[int, Dict[int, Stats]]) -> None:
    print("\n=== 2D Metrics Table (trunc_len x num_count) ===")
    for trunc_len in sorted(table.keys()):
        print(f"\n[trunc_len={trunc_len}]")
        print(
            "num_count\ttotal\tcorrect\taccuracy\tformat_correct\tformat_accuracy\tavg_output_len"
        )
        for num_count in sorted(table[trunc_len].keys()):
            s = table[trunc_len][num_count]
            print(
                f"{num_count}\t{s.total}\t{s.correct}\t{s.accuracy:.4f}\t"
                f"{s.format_correct}\t{s.format_accuracy:.4f}\t{s.avg_output_len:.1f}"
            )


def dump_json(table: Dict[int, Dict[int, Stats]], output_path: Path) -> None:
    payload: Dict[str, Dict[str, dict]] = {}
    for trunc_len, row in table.items():
        payload[str(trunc_len)] = {}
        for num_count, stats in row.items():
            payload[str(trunc_len)][str(num_count)] = {
                "total": stats.total,
                "correct": stats.correct,
                "accuracy": stats.accuracy,
                "format_correct": stats.format_correct,
                "format_accuracy": stats.format_accuracy,
                "avg_output_len": stats.avg_output_len,
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 benchmark 结果在不同截断长度和不同数字数量(num_count)下的二维指标表"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/8b_4096_long.jsonl"),
        help="输入 benchmark 结果 jsonl 文件路径",
    )
    parser.add_argument(
        "--trunc-lens",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096],
        help="要统计的截断长度列表",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/benchmark/8b_4096_long_2d_stats.json"),
        help="二维统计结果输出 json 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    table = build_2d_stats(records, sorted(args.trunc_lens))

    print(f"Loaded {len(records)} records from {args.input}")
    print_table(table)
    dump_json(table, args.output_json)
    print(f"\nSaved 2D stats json to {args.output_json}")


if __name__ == "__main__":
    main()
