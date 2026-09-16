# Experimental methodology

This is a **scaled reproduction** of the adaptation mechanism and a local
efficiency study. It is not a matched rerun of the paper's full GLUE protocol.

## Identities and splits

- Model: `FacebookAI/roberta-base`, revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, loaded with a newly
  initialized two-class classifier.
- Data: `nyu-mll/glue`, revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- MRPC distribution: 3,668 train, 408 validation; SST-2: 67,349 train,
  872 validation. The source's unlabeled test splits are never scored here.
- For each seed, `train_test_split(test_size=0.1, seed=seed)` creates a local
  training and tuning partition from the official training split. The
  validation split is used after tuning checkpoint selection. Different seeds
  therefore also vary the local partition, while methods share the same
  partition for a given seed.
- Maximum tokenized sequence length: 128; padding is dynamic per batch.
  The exact truncation rate was not recorded and is a limitation.

Each run records a dataset content-hash manifest and an exact normalized
text-overlap audit. The audit found zero train/validation exact overlaps on
both tasks. SST-2 had two exact train/test overlaps; test labels were not
used. The SST-2 training source contains repeated text, and no group-aware
split was implemented. The local train/tuning partition is deterministic
from the seed but its per-example ID list was not saved separately.

## Training and selection

The 18 core runs compare head-only training, full fine-tuning, and custom
LoRA at seeds 1, 2, and 3 for both tasks. All use three epochs, AdamW,
learning rate 2e-5, weight decay 0, BF16, microbatch 8, accumulation 2,
and the same pinned model and tokenizer. The selected custom LoRA setting
uses rank 8, alpha 16, dropout 0.1, and query/value projections, with the
classifier trained. MRPC selects the best epoch by local tuning positive-class
F1; SST-2 selects by local tuning accuracy. Validation results do not select
the checkpoint.

The reference PEFT comparison is one MRPC run, not a three-seed baseline.
The eight additional MRPC runs are exploratory seed-1 ablations. Rank tests
hold alpha/r = 2; scaling tests vary alpha at rank 4; placement tests compare
query-only and value-only against the query/value core configuration. Integer
rank and actual parameter counts are reported; an equal-parameter placement
study was **not** completed.

## Measurement and aggregation

`training_seconds` measures per-run wall time from the runner start through
final evaluation and artifact writing. Peak allocated and peak reserved GPU
memory come from PyTorch counters reset at run start. They are not total GPU
memory. Parameter counts include the task head; task artifact size includes
all trained values, while resumable checkpoints include the base model and
optimizer state. The core result table reports mean and sample standard
deviation across three seed runs. Three seeds do not resolve small quality
differences.

An old summary counted custom LoRA parameters after merging adapters for
inference. Report generation recomputes custom LoRA training counts from the
saved task artifact, and the runner now captures counts before merge. See
[discrepancies](discrepancies.md) for the audit trail.

Dataset source licensing is separate from this repository's MIT code license.
The [GLUE distribution card](https://huggingface.co/datasets/nyu-mll/glue)
lists `other`; raw dataset rows and model weights are excluded from this Git
repository.
