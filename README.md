# LoRA under a 6 GB GPU budget

An independent implementation and measured study of [LoRA: Low-Rank Adaptation
of Large Language Models](https://arxiv.org/abs/2106.09685v2) (Hu et al., ICLR
2022). I implemented the adapter, safe RoBERTa projection injection, checkpoint
and resume path, and an experiment runner. [Hugging Face
PEFT](https://github.com/huggingface/peft) is used only as a reference method.

The study uses a pinned [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base)
checkpoint and the MRPC and SST-2 tasks from [GLUE](https://huggingface.co/datasets/nyu-mll/glue).
It ran 27 jobs on one RTX 4050 laptop GPU: 18 core runs (three seeds per method
and task), one PEFT comparison, and eight MRPC rank, scaling, and target
experiments. The completed suite took about 1 hour 42 minutes on that machine.

## What the experiments found

The table reports the official GLUE **validation** split after selecting a
checkpoint on a separate 10% portion of each task's training split. Values
are mean ± sample standard deviation across seeds 1, 2, and 3. These are
local scores, not GLUE test submissions or directly comparable paper scores.

| Task and metric | Head only | Full fine-tuning | Custom LoRA |
| --- | ---: | ---: | ---: |
| MRPC accuracy | 0.6838 ± 0.0000 | 0.8717 ± 0.0051 | 0.6928 ± 0.0086 |
| MRPC positive-class F1 | 0.8122 ± 0.0000 | 0.9059 ± 0.0018 | 0.8157 ± 0.0034 |
| SST-2 accuracy | 0.7867 ± 0.0284 | 0.9304 ± 0.0024 | 0.9281 ± 0.0046 |

On MRPC, the head-only model predicted the positive class for all 408
validation examples in all three seeds. The custom LoRA model predicted it
for 408, 395, and 400 examples, respectively. Its F1 therefore looks
respectable despite weak discrimination. The chosen rank-8, alpha-16,
query/value configuration **did not approach full fine-tuning on MRPC**. On
SST-2, it reached similar accuracy to full fine-tuning within this small
three-seed study. One shared learning rate (2e-5) was used to control the
comparison; LoRA-specific tuning could change the MRPC result.

| Resource (core configuration) | Full fine-tuning | Custom LoRA |
| --- | ---: | ---: |
| Trainable parameters | 124,647,170 | 887,042 |
| MRPC peak allocated GPU memory, mean | 2.41 GiB | 0.92 GiB |
| SST-2 peak allocated GPU memory, mean | 2.35 GiB | 0.83 GiB |
| Task artifact | 498.7 MB | 3.57 MB |

The LoRA task artifact contains both adapters and the trained classifier.
It still requires the pinned base model for inference. Resumable checkpoints
also contain the base model and optimizer state and are much larger. Parameter
counts come from saved training artifacts: an earlier summary writer counted
merged inference weights as trainable, so publication tables correct that
accounting error without changing run evidence. See [results and
limitations](docs/results.md) before using these numbers in a résumé.

## Mechanism and checks

For a frozen linear weight \(W_0\), the adapter computes
\(y=W_0x+b+(\alpha/r)BAx\), with dropout on the adapter input during
training. `A` starts random and `B` starts at zero; the adapter initially
leaves the pretrained function unchanged. The implementation checks
orientation, first-step gradient behavior, target selection, serialization,
and merged inference. A real RoBERTa forward comparison with PEFT produced
maximum absolute error 0.0 in the tested configuration. The largest
recorded merged versus unmerged logit difference across the 14 custom LoRA
runs was below 2.56e-5. See [paper notes](docs/paper_notes.md) and [reference
parity](docs/reference_parity.md) for the precise boundaries.

## Reproduce

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), enough local disk
space, and a CUDA GPU for the BF16 matrix. The model and dataset are fetched
from their pinned upstream revisions. Dataset files, model weights, full
checkpoints, and logs are excluded from Git.

```bash
UV_CACHE_DIR=.uv-cache uv sync --locked --python 3.12 --group dev
UV_CACHE_DIR=.uv-cache uv run --locked pytest tests -q
UV_CACHE_DIR=.uv-cache uv run --locked pipeline status
UV_CACHE_DIR=.uv-cache uv run --locked pipeline run --resume --max-hours 6
```

`pipeline run` validates completed artifacts and resumes a partial run from
its current optimizer-boundary checkpoint. A SIGINT requests a checkpoint
after the next optimizer update. `--max-hours` stops **starting** new jobs
when the budget expires; it never truncates an active job. GPU jobs are
serial, while the two independent dataset preparations may overlap.

To discard only the generated output of this matrix, first review the exact
cleanup inventory:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked pipeline clean --dry-run
UV_CACHE_DIR=.uv-cache uv run --locked pipeline clean
```

Regenerate the published tables and figure from completed run artifacts:

```bash
UV_CACHE_DIR=.uv-cache uv run --locked python scripts/generate_results.py \
  runs/reliable --completed-only --output results/reliable
UV_CACHE_DIR=.uv-cache uv run --locked python scripts/profile_runs.py \
  runs/reliable --completed-only --output results/reliable/profile.csv
UV_CACHE_DIR=.uv-cache uv run --locked python scripts/audit_project.py \
  --runs runs/reliable --completed-only --fail-on-duplicates
```

The publishable [per-run metrics](results/reliable/metrics.csv), [resource
profile](results/reliable/profile.csv), and [efficiency
figure](results/reliable/figures/efficiency.png) are committed. Raw predictions,
checkpoints, and manifests are generated locally under `runs/reliable/` and
are not redistributed. See [methodology](docs/experimental_methodology.md),
[scope](docs/reproduction_scope.md), [discrepancies](docs/discrepancies.md),
and the [claim ledger](docs/claim_ledger.md) for interpretation.

## Attribution and license

This repository's original code is MIT licensed. The LoRA method and official
[author repository](https://github.com/microsoft/LoRA) belong to Hu et al. and
Microsoft; this project is an independent implementation, not a fork. The
RoBERTa model card lists MIT. The GLUE distribution lists its license as
“other”; the individual source datasets have their own terms. No model
weights or dataset rows are included in this repository.
