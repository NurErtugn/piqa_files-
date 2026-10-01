import argparse
import json
from pathlib import Path


def count_jsonl(path):
    if not path.exists():
        return 0

    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--precision-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--recall-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--precision-gpu-hours",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--recall-gpu-hours",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--accepted",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    precision_n = sum(
        count_jsonl(p)
        for p in args.precision_dir.glob("*.jsonl")
    )

    recall_n = 0

    if args.recall_dir:
        recall_n = sum(
            count_jsonl(p)
            for p in args.recall_dir.glob("*.jsonl")
        )

    total_calls = precision_n + recall_n

    gpu_hours = (
        args.precision_gpu_hours
        + args.recall_gpu_hours
    )

    docs_per_gpu_hour = (
        total_calls / gpu_hours
        if gpu_hours > 0
        else float("nan")
    )

    second_pass_rate = (
        recall_n / precision_n
        if precision_n
        else float("nan")
    )

    output = {
        "precision_documents": precision_n,
        "recall_documents": recall_n,
        "total_llm_calls": total_calls,
        "second_pass_rate": second_pass_rate,
        "gpu_hours": gpu_hours,
        "calls_per_gpu_hour": docs_per_gpu_hour,
    }

    if args.accepted:
        output["gpu_hours_per_accepted_doc"] = (
            gpu_hours / args.accepted
        )

    print(json.dumps(output, indent=2))

    print("\nProjected GPU-hours")

    calls_per_source = 1 + second_pass_rate

    for n in [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ]:
        expected_calls = n * calls_per_source
        expected_gpu_hours = (
            expected_calls / docs_per_gpu_hour
        )

        print(
            f"{n:>12,} docs: "
            f"{expected_gpu_hours:,.1f} GPU-h"
        )


if __name__ == "__main__":
    main()
