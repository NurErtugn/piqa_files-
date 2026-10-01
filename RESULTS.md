# Aggregate results through 1 October 2026

These counts come from completed Beverin job outputs and full-shard audits.
They are process counts, **not PIQA benchmark accuracy**.

| Stage | Count |
|---|---:|
| FineWeb documents screened | 10,000,000 |
| FastText/heuristic candidates | 1,556,882 |
| Qwen precision accepts sent to independent audit | 420,842 |
| Mistral V3 independently useful candidates | 96,977 |
| Of those, hidden-gem candidates | 93,455 |
| Of those, gem candidates | 3,522 |
| Extracted passage records, unique IDs | 96,977 |
| Passage-review passes | 78,622 |
| Passage-review quarantines | 18,355 |
| Passing hidden-gem passages | 75,456 |
| Passing gem passages | 3,166 |
| Validated common-sense rewrites | 78,036 |
| Grounded fallbacks after failed validation | 568 |
| Grounded fallbacks after generation error | 18 |
| Assembled pre-decontamination candidates | 78,622 |

The V3 independent audit accounted for all 420,842 inputs with zero unresolved
records. The passage review accounted for all 96,977 inputs with zero technical
errors. The rewrite/validation stage accounted for all 78,622 approved inputs:
78,604 generated rewrites were validated, and the remaining 18 generation
errors used the grounded passage as fallback. The assembled set has 78,622
unique IDs. Its final PIQA contamination and duplicate checks remain pending.

Document-level risk flags were preserved for review; a risk flag on a noisy
source page did not automatically reject a selected passage. The quarantined
set remains separate and may contain useful false positives; it has not been
silently added to the rewritten set.

## What has not been measured

The earlier 1,316-document rewriting pilot measured faithfulness and fallback
behavior, not PIQA task improvement. There is currently **no measured PIQA
accuracy uplift for this rewriting style**. A valid comparison should fine-tune
the same base model on matched IDs using grounded passages versus their
validated rewrites, hold the training-token budget and settings fixed, and
evaluate on decontaminated PIQA validation questions.

Raw FineWeb text and PIQA signatures are intentionally excluded from Git.
