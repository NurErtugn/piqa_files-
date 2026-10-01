# FineWeb gem mining pipeline

This repository implements the supplied workflow as a resumable, streaming pipeline. It targets **100,000 documents per class** by default:

- `gem`: strong physical knowledge with semantic noise ≤ 0.20.
- `hidden_gem`: strong physical knowledge, 0.30–0.70 semantic noise, and at least 80 useful tokens.
- `bad`: the workflow's hard negatives: nonphysical, unreliable, unsafe, software-only, or low-knowledge documents.

The 0.20–0.30 and 0.70–1.00 gaps remain `uncertain`; they are not silently forced into a class. The targets are quotas, not guarantees: the manifest reports any shortfall.

## 1. Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp config.example.json config.json
```

FineWeb's official `sample-10BT` is about 27.6 GB / 10B GPT-2 tokens. Start there, but expect to scan millions of documents to fill high-precision quotas. Pinning `v1.4.0` makes a run reproducible.

## 2. Build the PIQA contamination index

```bash
fineweb-gems build-piqa --output work/piqa-signatures.json --ngram 8
```

This indexes PIQA goal and answer n-grams. Screening rejects documents exceeding the configured overlap threshold. Exact benchmark contamination should be treated conservatively; audit borderline overlap bins before production.

## 3. Parallel cheap screening

Run one process per worker. Each receives a disjoint streaming shard:

```bash
mkdir -p work/candidates
for i in 0 1 2 3 4 5 6 7; do
  fineweb-gems screen --config config.json --piqa work/piqa-signatures.json \
    --worker "$i" --workers 8 --output "work/candidates/part-$i.jsonl" &
done
wait
```

This stage preserves `text_raw`, `text_clean`, and `removed_text`; removes mechanical boilerplate and exact repeated lines; applies local exact/near deduplication; rejects PIQA overlap; and computes physical relevance, recoverability, procedure, measurement, noise, repetition, quality, and URL features. It is deliberately recall-oriented.

For multiple machines, assign the same `--workers N` and a unique `--worker 0..N-1`. Write each output to durable shared storage. Screening restarts its shard and replaces its output; completed IDs are safely skipped only in the judge stages.

## 4. Parallel precision judge

Expose a model through an OpenAI-compatible chat-completions endpoint. The default model name is `Qwen/Qwen3-32B`; override it to the actual served model. Structured JSON support is required.

```bash
export JUDGE_ENDPOINT=http://localhost:8000/v1/chat/completions
export JUDGE_MODEL=Qwen/Qwen3-32B
export JUDGE_API_KEY=local

mkdir -p work/precision
for i in 0 1 2 3 4 5 6 7; do
  fineweb-gems judge --mode precision --input "work/candidates/part-$i.jsonl" \
    --output "work/precision/part-$i.jsonl" --resume &
done
wait
```

The judge selects numbered segments. Python—not the model—calculates semantic noise from selected/total tokens and creates grounded `text_extracted`. This can be parallelized across files and GPUs. Use one process per endpoint replica, or point workers at a load balancer. Avoid many writers to the same JSONL file.

## 5. Recall judge for uncertain cases

```bash
mkdir -p work/recall
for i in 0 1 2 3 4 5 6 7; do
  fineweb-gems judge --mode recall --input "work/precision/part-$i.jsonl" \
    --output "work/recall/part-$i.jsonl" --resume &
done
wait
```

Only first-pass `uncertain` records are sent. This follows the prior audit: the precision judge was 13/15 correct on hidden gems, while 11/19 uncertain examples were manually useful, so discarding uncertainty would lose substantial recall.

## 6. Global deduplication and exact quotas

```bash
fineweb-gems finalize --config config.json \
  --input 'work/precision/*.jsonl' --recall 'work/recall/*.jsonl' \
  --output outputs/fineweb-100k-each
```

Finalization is intentionally global and single-process: it merges recall decisions, enforces exact and SimHash near-duplicate removal across every worker, classifies deterministically, and stops at 100k per class. Outputs are `gem.jsonl`, `hidden_gem.jsonl`, `bad.jsonl`, and `manifest.json`. Hidden gems use `text_extracted` as `training_text`; gems keep natural `text_clean`. Raw and removed text remain available for audit.

Near-duplicate lookup uses four 16-bit SimHash bands followed by an exact Hamming-distance check. With a distance threshold of three, any true match must share at least one band, avoiding a quadratic all-pairs comparison.

## 7. Required audit before a full run

Label stratified samples from physical-score, recoverability, PIQA-overlap, and judge-confidence bins. Recalibrate `candidate_floor`, PIQA overlap, useful-token minimum, and noise boundaries from those labels. Report precision for each final class and recall against a random FineWeb sample; quotas alone do not establish quality.

Then independently verify extracted outputs. Require atomic precision ≥ 0.95, atomic recall ≥ 0.98–0.99, zero contradictions, no changed negation/causal direction, no added safety instructions, and preservation of numbers, units, named objects, conditions, and exceptions. Extraction is the primary representation until downstream training proves rewriting is better.

## 8. Token-balanced proxy experiment

Build mixtures by unique tokens, never document count, and show every document once before repetition:

1. HQ control: 20M natural HQ FineWeb tokens.
2. Raw HG: 10M HQ + 10M raw hidden-gem tokens.
3. Extracted HG: 10M HQ + 10M extracted hidden-gem tokens.
4. Hybrid/fallback: 10M HQ + 10M independently accepted rewrite/extraction tokens.

Use identical initialization, optimization, schedule, and total tokens for at least three seeds. Select primarily by PIQA accuracy, secondarily by correct-vs-wrong normalized margin; use correct-answer BPB only diagnostically. If extraction wins, compare 25%, 50%, and 70% extracted-token shares. Add HellaSwag, ARC-Easy/Challenge, OpenBookQA, held-out procedure loss, paired question bootstrap intervals, and across-seed intervals.

## What parallelizes safely

| Stage | Parallel unit | Coordination rule |
|---|---|---|
| FineWeb screening | dataset shard / worker ID | fixed worker count and unique ID |
| Precision judging | candidate JSONL shard | one output file per process |
| Recall judging | uncertain JSONL shard | one output file per process |
| Independent faithfulness gate | transformed record | one output file per process |
| Proxy training | condition × seed | identical token budget and setup |
| Final quota/dedup merge | do not naively parallelize | global uniqueness is required |

Run unit tests with `pytest -q`.
