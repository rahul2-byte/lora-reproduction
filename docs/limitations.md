# Limitations

This is a study of two classification tasks, one RoBERTa-base checkpoint, and
three training seeds per core method on one 6 GB laptop GPU. It does not
establish universal LoRA behavior or reproduce the paper's largest models.

The methods shared one learning rate (2e-5) and three epochs. This makes the
controlled comparison inspectable but may under-tune LoRA, especially on
MRPC. No method-specific tuning-budget comparison was completed. The local
train/tuning partition changes with the seed. The official GLUE validation
split was held out from local checkpoint selection, but there is no local
labeled test set. SST-2 related phrases and repeated source text may make
example independence weaker than a document-level split; group-aware
partitioning was not available.

Rank, alpha, and target-module ablations used one seed. Most MRPC ablations
collapsed to positive-only predictions, so the current data do not support
a stable rank or placement trend. The reference implementation received one
matched training run; strict forward parity was checked, while gradient and
optimizer-step parity were not completed. Merged/unmerged output differences
were checked on a batch, not exhaustively on every prediction.

Memory and time figures are PyTorch allocated-memory and runner wall-time
measurements on one machine. Power limits, temperature, and fresh-process
memory baselines were not systematically controlled. The published task
artifact is not a standalone model: inference also requires the pinned
RoBERTa weights and tokenizer. Generated raw checkpoints and datasets are
kept local and excluded from Git.
