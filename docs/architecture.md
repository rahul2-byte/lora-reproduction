# Architecture

`pipeline.py` defines the 27-job matrix, checks status, prepares MRPC/SST-2
data with at most two CPU threads, and serializes GPU runs. `config.py`
validates resolved settings. `data.py` loads pinned datasets and atomically
caches tokenization. `models.py` loads RoBERTa and its new classifier.
`lora.py` implements the factorized linear update; `injection.py` checks exact
target paths before replacement. `train.py` owns optimizer updates and
accumulation. `evaluate.py` produces metrics and ID-linked predictions.
`checkpoints.py` writes inference artifacts and resumable training state.
`runner.py` connects these pieces for a single job and holds its run lock.
`report.py` and `profile.py` aggregate saved evidence.

The data flow is: pinned model and dataset → tokenized local cache →
seed-specific training/tuning partition → checkpoint-selected training →
official validation predictions → task artifact and summary → generated
tables and figure. GPU training uses one process and one device. No server or
experiment database is required.
