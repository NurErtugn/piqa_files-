from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from .core import (SimhashIndex, classify, clean_document, heuristic_features, ngrams, piqa_overlap,
                   read_jsonl, segment, simhash, stable_id, token_count, write_jsonl)
from .judging import CompatibleJudge, judge_record


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as handle: return json.load(handle)


def cmd_piqa(args):
    from datasets import load_dataset
    signatures, rows = set(), []
    for split in ("train", "validation", "test"):
        try: ds = load_dataset("lighteval/piqa", split=split)
        except Exception: continue
        for row in ds:
            texts = [str(row.get(k, "")) for k in ("goal", "sol1", "sol2")]
            rows.append({"split": split, **{k: row.get(k) for k in ("goal", "sol1", "sol2")}})
            for text in texts: signatures.update(ngrams(text, args.ngram))
    output = {"ngram": args.ngram, "signatures": sorted(signatures), "examples": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"examples": len(rows), "signatures": len(signatures), "output": args.output}))


def cmd_screen(args):
    from datasets import load_dataset
    cfg = load_cfg(args.config); piqa = json.loads(Path(args.piqa).read_text(encoding="utf-8")); signatures = set(piqa["signatures"])
    ds = load_dataset(cfg["dataset"], cfg.get("dataset_config"), split=cfg["split"], streaming=True, revision=cfg.get("dataset_revision"))
    ds = ds.shuffle(seed=cfg["seed"], buffer_size=cfg["shuffle_buffer"]).shard(num_shards=args.workers, index=args.worker)
    exact = set(); near_index = SimhashIndex(cfg["near_duplicate_hamming"])
    counters = {"seen": 0, "kept": 0, "exact_dup": 0, "near_dup": 0, "piqa": 0, "short_long": 0, "low_score": 0}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in ds:
            counters["seen"] += 1; cleaned = clean_document(row.get("text", "")); text = cleaned["text_clean"]; wc = token_count(text)
            if not cfg["min_clean_words"] <= wc <= cfg["max_clean_words"]: counters["short_long"] += 1; continue
            digest = stable_id(text)
            if digest in exact: counters["exact_dup"] += 1; continue
            fingerprint = simhash(text)
            if near_index.is_near(fingerprint): counters["near_dup"] += 1; continue
            if piqa_overlap(text, signatures, cfg["piqa_ngram"]) >= cfg["piqa_overlap_threshold"]: counters["piqa"] += 1; continue
            features = heuristic_features(text, cleaned["removed_text"], row.get("url", ""))
            if features["physical_score"] < cfg["candidate_floor"] and features["recoverability_score"] < .25: counters["low_score"] += 1; continue
            exact.add(digest); near_index.add(fingerprint)
            record = {"id": row.get("id") or digest, "content_sha256": digest, "simhash": fingerprint,
                      "url": row.get("url", ""), "dump": row.get("dump", ""), "date": row.get("date", ""),
                      **cleaned, "segments": segment(text), "features": features, "screen_worker": args.worker}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); counters["kept"] += 1
            if args.max_candidates and counters["kept"] >= args.max_candidates: break
            if counters["seen"] % 10000 == 0: print(json.dumps(counters), flush=True)
    print(json.dumps(counters))
    # A bounded streaming pilot may leave Hugging Face HTTP threads active.
    # The output file is already safely closed here.
    if args.max_candidates:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def cmd_judge(args):
    client = CompatibleJudge(args.endpoint, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def record_key(record):
        return (
            record.get("content_sha256")
            or record.get("id")
        )

    completed = set()

    if args.resume and output.exists():
        completed = {
            record_key(record)
            for record in read_jsonl([output])
            if record_key(record)
        }

    counts = {
        "read": 0,
        "written": 0,
        "skipped": 0,
    }

    mode = (
        "a"
        if args.resume
        else "w"
    )

    with output.open(mode, encoding="utf-8") as handle:
        paths = [
            Path(path)
            for path in glob.glob(args.input)
        ]

        for record in read_jsonl(paths):
            counts["read"] += 1
            key = record_key(record)

            if not key:
                raise ValueError(
                    "Record has neither content_sha256 nor id"
                )

            if key in completed:
                counts["skipped"] += 1
                continue

            # Recall mode now judges every record supplied to it.
            # The caller controls recall sampling through the input file.
            judged = judge_record(
                record,
                args.mode,
                client,
            )

            handle.write(
                json.dumps(judged, ensure_ascii=False)
                + "\n"
            )
            handle.flush()

            completed.add(key)
            counts["written"] += 1

            if (
                args.limit
                and counts["written"] >= args.limit
            ):
                break

    print(json.dumps(counts))

def cmd_finalize(args):
    cfg = load_cfg(args.config); target = args.target or cfg["target_per_class"]
    records = list(read_jsonl([Path(p) for p in glob.glob(args.input)]))
    recall_by_id = {r["content_sha256"]: r for r in read_jsonl([Path(p) for p in glob.glob(args.recall)])} if args.recall else {}
    records.sort(key=lambda r: r["content_sha256"])
    selected, exact = {k: [] for k in ("hidden_gem", "gem", "bad")}, set()
    near_index = SimhashIndex(cfg["near_duplicate_hamming"])
    for original in records:
        r = recall_by_id.get(original["content_sha256"], original); digest = r["content_sha256"]
        if digest in exact: continue
        fingerprint = int(r.get("simhash", simhash(r["text_clean"])))
        if near_index.is_near(fingerprint): continue
        label = classify(r, cfg)
        if label not in selected or len(selected[label]) >= target: continue
        r["label"] = label; r["training_text"] = r.get("text_extracted") if label == "hidden_gem" else r["text_clean"]
        selected[label].append(r); exact.add(digest); near_index.add(fingerprint)
        if all(len(v) >= target for v in selected.values()): break
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    manifest = {label: write_jsonl(out / f"{label}.jsonl", rows) for label, rows in selected.items()}
    (out / "manifest.json").write_text(json.dumps({"target_per_class": target, "counts": manifest, "config": cfg}, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


def parser():
    p = argparse.ArgumentParser(prog="fineweb-gems"); sub = p.add_subparsers(required=True)
    x = sub.add_parser("build-piqa"); x.add_argument("--output", required=True); x.add_argument("--ngram", type=int, default=8); x.set_defaults(func=cmd_piqa)
    x = sub.add_parser("screen"); x.add_argument("--config", required=True); x.add_argument("--piqa", required=True); x.add_argument("--output", required=True); x.add_argument("--worker", type=int, required=True); x.add_argument("--workers", type=int, required=True); x.add_argument("--max-candidates", type=int, default=0); x.set_defaults(func=cmd_screen)
    x = sub.add_parser("judge"); x.add_argument("--input", required=True); x.add_argument("--output", required=True); x.add_argument("--mode", choices=["precision", "recall"], required=True); x.add_argument("--endpoint"); x.add_argument("--model"); x.add_argument("--limit", type=int, default=0); x.add_argument("--resume", action="store_true"); x.set_defaults(func=cmd_judge)
    x = sub.add_parser("finalize"); x.add_argument("--config", required=True); x.add_argument("--input", required=True); x.add_argument("--recall"); x.add_argument("--output", required=True); x.add_argument("--target", type=int); x.set_defaults(func=cmd_finalize)
    return p


def main():
    args = parser().parse_args(); args.func(args)

if __name__ == "__main__": main()
