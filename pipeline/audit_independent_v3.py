"""Check accounting and judgment consistency for the full V3 Mistral audit."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/capstor/scratch/cscs/nertug/fineweb-gems/production10m")
INPUT = ROOT / "independent-input"
OUTPUT = ROOT / "independent-v3"


def records(path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error


def main():
    manifest = json.loads((INPUT / "manifest.json").read_text())
    expected = manifest["accepted_records"]
    shard_count = manifest["output_shards"]
    all_input_ids = set()
    all_output_ids = set()
    decisions = Counter()
    categories = Counter()
    positive_risks = Counter()
    validation_flags = Counter()
    missing = []
    unresolved_errors = []

    for shard in range(shard_count):
        source_path = INPUT / f"part-{shard}.jsonl"
        result_path = OUTPUT / f"part-{shard}.jsonl"
        error_path = OUTPUT / f"part-{shard}.errors.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        source_ids = [row["id"] for row in records(source_path)]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"Duplicate input IDs in shard {shard}")
        if all_input_ids.intersection(source_ids):
            raise ValueError(f"Input IDs repeated across shards at {shard}")
        all_input_ids.update(source_ids)

        if result_path.exists():
            result_rows = list(records(result_path))
        else:
            result_rows = []
        result_ids = [row["id"] for row in result_rows]
        if len(result_ids) != len(set(result_ids)):
            raise ValueError(f"Duplicate output IDs in shard {shard}")
        if not set(result_ids).issubset(source_ids):
            raise ValueError(f"Unexpected output ID in shard {shard}")
        if all_output_ids.intersection(result_ids):
            raise ValueError(f"Output IDs repeated across shards at {shard}")
        all_output_ids.update(result_ids)

        for row in result_rows:
            judge = row["independent_judge"]
            decision = judge["decision"]
            decisions[decision] += 1
            categories[judge["category"]] += 1
            validation_flags.update(judge.get("validation_flags", []))
            if decision == "useful":
                scores = (
                    judge["physical_relevance"],
                    judge["transferability"],
                    judge["reliability"],
                )
                if min(scores) < 2 or not judge["useful_segments"]:
                    raise ValueError(f"Inconsistent positive: {row['id']}")
                positive_risks.update(
                    key for key, active in judge["risk_flags"].items() if active
                )

        missing_ids = set(source_ids) - set(result_ids)
        if missing_ids:
            error_ids = (
                {row["id"] for row in records(error_path)}
                if error_path.exists() else set()
            )
            missing.append({
                "shard": shard,
                "input": len(source_ids),
                "output": len(result_ids),
                "unresolved": len(missing_ids),
                "unresolved_with_error_log": len(missing_ids & error_ids),
            })
            unresolved_errors.extend((shard, record_id) for record_id in missing_ids)

        if shard % 25 == 24 or shard == shard_count - 1:
            print(f"Checked {shard + 1}/{shard_count} shards", flush=True)

    if len(all_input_ids) != expected:
        raise ValueError(f"Expected {expected} unique inputs; got {len(all_input_ids)}")

    report = {
        "expected_inputs": expected,
        "unique_inputs": len(all_input_ids),
        "unique_outputs": len(all_output_ids),
        "unresolved": len(unresolved_errors),
        "incomplete_shards": missing,
        "decisions": dict(decisions),
        "categories": dict(categories),
        "positive_risk_flags": dict(positive_risks),
        "validation_flags": dict(validation_flags),
        "note": "Judgments are candidates; unsupported claims require segment review.",
    }
    print(json.dumps(report, indent=2), flush=True)
    report_path = OUTPUT / "audit-summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if unresolved_errors:
        raise SystemExit(f"{len(unresolved_errors)} records remain unresolved")


if __name__ == "__main__":
    main()
