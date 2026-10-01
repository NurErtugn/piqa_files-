import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import fasttext
from datasets import load_dataset

from fineweb_gems.core import (
    SimhashIndex,
    clean_document,
    heuristic_features,
    piqa_overlap,
    segment,
    simhash,
    stable_id,
    token_count,
)


def positive_probability(model, text, positive_label):
    text = " ".join(text.split())
    labels, probabilities = model.predict(text, k=2)

    values = {
        label: float(probability)
        for label, probability in zip(labels, probabilities)
    }

    return values.get(positive_label, 0.0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--piqa", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", required=True)
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max-seen", type=int, required=True)

    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    with open(args.piqa, encoding="utf-8") as handle:
        piqa_data = json.load(handle)

    with open(args.thresholds, encoding="utf-8") as handle:
        thresholds = json.load(handle)

    signatures = set(piqa_data["signatures"])

    relevance_config = thresholds["relevance"]
    recoverability_config = thresholds["recoverability"]

    relevance_model = fasttext.load_model(
        relevance_config["model"]
    )
    recoverability_model = fasttext.load_model(
        recoverability_config["model"]
    )

    dataset = load_dataset(
        config["dataset"],
        config.get("dataset_config"),
        split=config["split"],
        streaming=True,
        revision=config.get("dataset_revision"),
    )

    dataset = dataset.shuffle(
        seed=config["seed"],
        buffer_size=config["shuffle_buffer"],
    )

    dataset = dataset.shard(
        num_shards=args.workers,
        index=args.worker,
    )

    output_path = Path(args.output)
    stats_path = Path(args.stats_output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    exact_hashes = set()
    near_index = SimhashIndex(
        config["near_duplicate_hamming"]
    )

    counters = Counter()

    route_combinations = Counter()

    with output_path.open("w", encoding="utf-8") as destination:
        for row in dataset:
            if counters["seen"] >= args.max_seen:
                break

            counters["seen"] += 1

            cleaned = clean_document(row.get("text", ""))
            text = cleaned["text_clean"]
            word_count = token_count(text)

            if not (
                config["min_clean_words"]
                <= word_count
                <= config["max_clean_words"]
            ):
                counters["short_or_long"] += 1
                continue

            counters["length_eligible"] += 1

            digest = stable_id(text)

            if digest in exact_hashes:
                counters["exact_duplicate"] += 1
                continue

            fingerprint = simhash(text)

            if near_index.is_near(fingerprint):
                counters["near_duplicate"] += 1
                continue

            overlap = piqa_overlap(
                text,
                signatures,
                config["piqa_ngram"],
            )

            if overlap >= config["piqa_overlap_threshold"]:
                counters["piqa_contaminated"] += 1
                continue

            counters["piqa_clean"] += 1

            features = heuristic_features(
                text,
                cleaned["removed_text"],
                row.get("url", ""),
            )

            heuristic_pass = (
                features["physical_score"]
                >= config["candidate_floor"]
                or features["recoverability_score"] >= 0.25
            )

            relevance_score = positive_probability(
                relevance_model,
                text,
                relevance_config["positive_label"],
            )

            recoverability_score = positive_probability(
                recoverability_model,
                text,
                recoverability_config["positive_label"],
            )

            relevance_pass = (
                relevance_score
                >= relevance_config["threshold"]
            )

            recoverability_pass = (
                recoverability_score
                >= recoverability_config["threshold"]
            )

            fasttext_pass = (
                relevance_pass or recoverability_pass
            )

            selected = heuristic_pass or fasttext_pass

            if heuristic_pass:
                counters["heuristic_pass"] += 1

            if relevance_pass:
                counters["relevance_pass"] += 1

            if recoverability_pass:
                counters["recoverability_pass"] += 1

            if fasttext_pass:
                counters["fasttext_union_pass"] += 1

            if heuristic_pass and fasttext_pass:
                counters["heuristic_and_fasttext"] += 1

            if heuristic_pass and not fasttext_pass:
                counters["heuristic_only"] += 1

            if fasttext_pass and not heuristic_pass:
                counters["fasttext_only"] += 1

            route_name = (
                f"heuristic={heuristic_pass},"
                f"relevance={relevance_pass},"
                f"recoverability={recoverability_pass}"
            )
            route_combinations[route_name] += 1

            if not selected:
                counters["rejected_by_all_retrievers"] += 1
                continue

            exact_hashes.add(digest)
            near_index.add(fingerprint)

            record = {
                "id": row.get("id") or digest,
                "content_sha256": digest,
                "simhash": fingerprint,
                "url": row.get("url", ""),
                "dump": row.get("dump", ""),
                "date": row.get("date", ""),
                **cleaned,
                "segments": segment(text),
                "features": features,
                "piqa_overlap": overlap,
                "retrieval": {
                    "heuristic_pass": heuristic_pass,
                    "relevance_score": relevance_score,
                    "relevance_pass": relevance_pass,
                    "recoverability_score": recoverability_score,
                    "recoverability_pass": recoverability_pass,
                    "fasttext_pass": fasttext_pass,
                    "selected": selected,
                },
                "screen_worker": args.worker,
            }

            destination.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            counters["kept"] += 1

            if counters["seen"] % 5000 == 0:
                print(
                    json.dumps({
                        "worker": args.worker,
                        **dict(counters),
                    }),
                    flush=True,
                )

    report = {
        "worker": args.worker,
        "workers": args.workers,
        "max_seen": args.max_seen,
        "counters": dict(counters),
        "route_combinations": dict(route_combinations),
        "thresholds": {
            "relevance": relevance_config["threshold"],
            "recoverability":
                recoverability_config["threshold"],
        },
        "output": str(output_path),
    }

    stats_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()

    # Avoid waiting for lingering Hugging Face streaming threads.
    os._exit(0)


if __name__ == "__main__":
    main()
