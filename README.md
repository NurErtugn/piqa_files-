# PIQA-oriented FineWeb gem pipeline

Code, Slurm jobs, configuration examples, and aggregate results for finding
physical-world commonsense passages in FineWeb and rewriting them for training.
The scripts in `pipeline/` include historical pilots and the 10M-document V3
production workflow. Many Slurm paths are specific to the CSCS Beverin cluster.

This repository intentionally does **not** contain FineWeb document text,
PIQA benchmark signatures, model weights, caches, logs, or credentials.
Rebuild those resources from their original sources when reproducing a run.

See [RESULTS.md](RESULTS.md) for verified aggregate counts and current status.
The included `pipeline/README.md` describes the original CLI workflow; its
pilot-era quotas and estimates should not be mistaken for final 10M results.

## Current sequence

1. FastText/heuristic screening and PIQA-overlap filtering.
2. Propella annotation and Qwen precision judging.
3. Independent Mistral V3 audit with grounded segment selection.
4. Passage-level Qwen risk review.
5. Grounded common-sense rewriting and independent validation, with verbatim
   passage fallback when generation or validation fails.
6. Final contamination and duplicate checks **still pending**.
7. Matched-token PIQA training/evaluation comparison **not yet run**.

The scripts record candidates and audit evidence; no intermediate result should
be described as a final training dataset before the last checks complete.
