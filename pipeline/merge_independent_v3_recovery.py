"""Verify and append recovered judgments to their original V3 shards."""

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/capstor/scratch/cscs/nertug/fineweb-gems/production10m")
RECOVERY = ROOT / "independent-v3-recovery"
DESTINATION = ROOT / "independent-v3"


def read(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    manifest = json.loads((RECOVERY / "manifest.json").read_text())
    shard_by_id = manifest["source_shard_by_id"]
    recovered_path = RECOVERY / "output/part-0.jsonl"
    recovered = list(read(recovered_path))
    recovered_ids = [row["id"] for row in recovered]
    if len(recovered_ids) != len(set(recovered_ids)):
        raise RuntimeError("Duplicate IDs in recovery output")
    if set(recovered_ids) != set(shard_by_id):
        raise RuntimeError(
            f"Expected {len(shard_by_id)} recovered IDs; got "
            f"{len(recovered_ids)}. Inspect the recovery errors before merging."
        )
    for row in recovered:
        judge = row["independent_judge"]
        if judge["decision"] == "useful" and (
            min(
                judge["physical_relevance"],
                judge["transferability"],
                judge["reliability"],
            ) < 2
            or not judge["useful_segments"]
        ):
            raise RuntimeError(f"Inconsistent positive: {row['id']}")

    to_append = defaultdict(list)
    for row in recovered:
        to_append[shard_by_id[row["id"]]].append(row)

    # Validate every destination before changing any of them.
    pending = {}
    for shard, rows in to_append.items():
        path = DESTINATION / f"part-{shard}.jsonl"
        existing = {row["id"]: row for row in read(path)}
        if len(existing) != sum(1 for _ in read(path)):
            raise RuntimeError(f"Duplicate existing IDs in shard {shard}")
        pending[shard] = [row for row in rows if row["id"] not in existing]
        for row in rows:
            if row["id"] in existing and row != existing[row["id"]]:
                raise RuntimeError(f"Conflicting existing output: {row['id']}")

    added = 0
    for shard, rows in sorted(pending.items()):
        path = DESTINATION / f"part-{shard}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                added += 1
    print("Recovered records:", len(recovered))
    print("New records appended:", added)
    print("Source shards updated:", len(pending))


if __name__ == "__main__":
    main()
