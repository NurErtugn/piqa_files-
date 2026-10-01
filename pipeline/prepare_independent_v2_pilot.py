"""Create a blinded, stratified re-judgment sample from the first 10M shard."""

import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("/capstor/scratch/cscs/nertug/fineweb-gems/production10m")
TARGET = ROOT / "independent-v2-pilot"
STAGE = ROOT / "independent-v2-pilot-staging"
N_PER_GROUP = 25


def group_for(judge):
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
    return None


def main():
    if TARGET.exists() or STAGE.exists():
        raise RuntimeError("Pilot or staging directory already exists; refusing to overwrite")

    groups = defaultdict(list)
    judged = ROOT / "independent/part-0.jsonl"
    with judged.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group = group_for(row["independent_judge"])
            if group:
                groups[group].append(row["id"])

    expected_groups = (
        "useful_zero",
        "useful_one",
        "useful_high",
        "negative_zero",
        "uncertain",
    )
    rng = random.Random(20260922)
    selected = {}
    for group in expected_groups:
        available = groups[group]
        if len(available) < N_PER_GROUP:
            raise RuntimeError(f"Only {len(available)} rows in group {group}")
        for doc_id in rng.sample(available, N_PER_GROUP):
            if doc_id in selected:
                raise RuntimeError(f"Duplicate selected ID: {doc_id}")
            selected[doc_id] = group

    STAGE.mkdir()
    input_dir = STAGE / "input"
    input_dir.mkdir()
    output_path = input_dir / "part-0.jsonl"
    written = 0
    source = ROOT / "independent-input/part-0.jsonl"
    with source.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dest:
        for line in src:
            row = json.loads(line)
            if row["id"] in selected:
                # Use the original Qwen-accepted record. The old Mistral answer
                # must not appear in the new judge's input or prompt.
                if "independent_judge" in row:
                    raise RuntimeError("Source unexpectedly contains old Mistral judgment")
                dest.write(line)
                written += 1

    if written != len(selected):
        raise RuntimeError(f"Selected {len(selected)} IDs but found {written} inputs")

    manifest = {
        "source_judgment": str(judged),
        "source_input": str(source),
        "seed": 20260922,
        "per_group": N_PER_GROUP,
        "groups": {group: len(groups[group]) for group in expected_groups},
        "pilot_records": written,
    }
    (STAGE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    STAGE.rename(TARGET)
    print(json.dumps(manifest, indent=2))
    print("READY:", TARGET)


if __name__ == "__main__":
    main()
