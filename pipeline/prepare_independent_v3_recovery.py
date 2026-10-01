"""Collect only unresolved V3 audit records into one recovery shard."""

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/capstor/scratch/cscs/nertug/fineweb-gems/production10m")
SOURCE = ROOT / "independent-input"
OUTPUT = ROOT / "independent-v3"
RECOVERY = ROOT / "independent-v3-recovery"


def read(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main():
    if RECOVERY.exists():
        raise RuntimeError(f"Recovery folder already exists: {RECOVERY}")
    audit = json.loads((OUTPUT / "audit-summary.json").read_text())
    if audit["expected_inputs"] != 420842 or audit["unresolved"] != 38:
        raise RuntimeError("Audit changed; inspect before preparing recovery")
    affected_shards = [entry["shard"] for entry in audit["incomplete_shards"]]
    unresolved = {}
    selected_rows = {}
    for shard in affected_shards:
        input_path = SOURCE / f"part-{shard}.jsonl"
        output_path = OUTPUT / f"part-{shard}.jsonl"
        error_path = OUTPUT / f"part-{shard}.errors.jsonl"
        error_ids = {row["id"] for row in read(error_path)}
        output_ids = (
            {row["id"] for row in read(output_path)}
            if output_path.exists() else set()
        )
        needed_ids = error_ids - output_ids
        found_ids = set()
        for row in read(input_path):
            record_id = row["id"]
            if record_id not in needed_ids:
                continue
            if record_id in unresolved:
                raise RuntimeError(f"Duplicate input ID: {record_id}")
            unresolved[record_id] = shard
            selected_rows[record_id] = row
            found_ids.add(record_id)
        if found_ids != needed_ids:
            raise RuntimeError(f"Could not locate all unresolved IDs in shard {shard}")
        expected_missing = next(
            entry["unresolved"] for entry in audit["incomplete_shards"]
            if entry["shard"] == shard
        )
        if len(needed_ids) != expected_missing:
            raise RuntimeError(f"Unresolved IDs changed in shard {shard}")

    if not unresolved:
        raise RuntimeError("Nothing to recover")
    RECOVERY.mkdir()
    input_dir = RECOVERY / "input"
    input_dir.mkdir()
    with (input_dir / "part-0.jsonl").open("w", encoding="utf-8") as handle:
        for record_id in sorted(selected_rows):
            handle.write(json.dumps(selected_rows[record_id], ensure_ascii=False) + "\n")
    manifest = {
        "records": len(unresolved),
        "source_shards": dict(sorted(Counter(unresolved.values()).items())),
        "source_shard_by_id": unresolved,
    }
    (RECOVERY / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("Recovery records:", len(unresolved))
    print("Affected shards:", len(set(unresolved.values())))
    print("Input:", input_dir / "part-0.jsonl")


if __name__ == "__main__":
    main()
