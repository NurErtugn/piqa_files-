"""Summarize the new 125-document Mistral pilot against the original pilot."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/capstor/scratch/cscs/nertug/fineweb-gems/production10m")
PILOT = ROOT / "independent-v2-pilot"


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def old_group(judge):
    decision = judge["decision"]
    scores = (
        judge["physical_relevance"],
        judge["transferability"],
        judge["reliability"],
    )
    if decision == "useful" and scores == (0, 0, 0):
        return "useful_zero"
    if decision == "useful" and scores == (1, 1, 1):
        return "useful_one"
    if decision == "useful" and min(scores) >= 2:
        return "useful_high"
    if decision == "not_useful" and scores == (0, 0, 0):
        return "negative_zero"
    if decision == "uncertain":
        return "uncertain"
    raise RuntimeError(f"Unexpected old pilot judgment: {judge}")


def main():
    inputs = read_rows(PILOT / "input/part-0.jsonl")
    new_path = PILOT / "output/part-0.jsonl"
    new_rows = read_rows(new_path)
    error_path = PILOT / "output/part-0.errors.jsonl"
    errors = read_rows(error_path) if error_path.exists() else []

    input_ids = [row["id"] for row in inputs]
    new_ids = [row["id"] for row in new_rows]
    error_ids = [row["id"] for row in errors]
    if len(input_ids) != 125 or len(set(input_ids)) != 125:
        raise RuntimeError("Expected 125 unique pilot inputs")
    if len(new_ids) != len(set(new_ids)):
        raise RuntimeError("Duplicate V2 outputs")
    if set(new_ids) & set(error_ids):
        raise RuntimeError("An ID appears in both output and errors")
    if set(input_ids) != set(new_ids) | set(error_ids):
        raise RuntimeError("Pilot outputs and errors do not account for all IDs")

    input_id_set = set(input_ids)
    old_by_id = {
        row["id"]: row["independent_judge"]
        for row in read_rows(ROOT / "independent/part-0.jsonl")
        if row["id"] in input_id_set
    }
    if len(old_by_id) != 125:
        raise RuntimeError("Could not find all pilot IDs in original judgments")

    old_decisions = Counter()
    new_decisions = Counter()
    new_categories = Counter()
    new_scores = Counter()
    flags = Counter()
    transitions = Counter()
    group_transitions = Counter()
    raw_mismatches = 0

    for row in new_rows:
        old = old_by_id[row["id"]]
        new = row["independent_judge"]
        old_decisions[old["decision"]] += 1
        new_decisions[new["decision"]] += 1
        new_categories[new["category"]] += 1
        new_scores[(
            new["physical_relevance"],
            new["transferability"],
            new["reliability"],
        )] += 1
        flags.update(new["validation_flags"])
        transitions[(old["decision"], new["decision"])] += 1
        group_transitions[(old_group(old), new["decision"])] += 1
        if new["model_decision"] != new["decision"]:
            raw_mismatches += 1

    report = {
        "input_records": len(inputs),
        "new_records": len(new_rows),
        "technical_errors": len(errors),
        "old_decisions": dict(old_decisions),
        "new_decisions": dict(new_decisions),
        "new_categories": dict(new_categories),
        "new_score_triples": [
            {"scores": list(scores), "count": count}
            for scores, count in new_scores.most_common()
        ],
        "validation_flags": dict(flags),
        "old_to_new_decisions": [
            {"old": old, "new": new, "count": count}
            for (old, new), count in transitions.most_common()
        ],
        "stratum_to_new_decisions": [
            {"stratum": group, "new": new, "count": count}
            for (group, new), count in sorted(group_transitions.items())
        ],
        "model_vs_validated_decision_mismatches": raw_mismatches,
    }
    print(json.dumps(report, indent=2))
    (PILOT / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
