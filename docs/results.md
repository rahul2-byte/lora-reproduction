# Results

The [per-run table](../results/reliable/metrics.csv), [resource
profile](../results/reliable/profile.csv), and [figure](../results/reliable/figures/efficiency.png)
were regenerated from 27 completed runs. The Git repository contains compact
outputs; `runs/reliable/` holds the local raw evidence. The run audit found
27 distinct completed summaries and no failed or partial jobs.

## Core comparison

Values below are mean ± sample standard deviation across three training seeds.
The evaluation set is the official GLUE validation split, held out from the
local tuning partition. Each method used the same pinned base model and data,
three epochs, BF16, microbatch 8, accumulation 2, and AdamW at 2e-5.

| Task | Metric | Head only | Full fine-tuning | Custom LoRA |
| --- | --- | ---: | ---: | ---: |
| MRPC | Accuracy | 0.6838 ± 0.0000 | 0.8717 ± 0.0051 | 0.6928 ± 0.0086 |
| MRPC | Positive-class F1 | 0.8122 ± 0.0000 | 0.9059 ± 0.0018 | 0.8157 ± 0.0034 |
| SST-2 | Accuracy | 0.7867 ± 0.0284 | 0.9304 ± 0.0024 | 0.9281 ± 0.0046 |

MRPC's 408 validation examples contain 279 positives. Head only predicted
positive for every example in all three runs; custom LoRA predicted positive
for 408, 395, and 400 examples. This explains why F1 can be high even when
accuracy remains near the 279/408 = 0.6838 majority-class diagnostic. Full
fine-tuning did substantially better on MRPC. The single PEFT LoRA MRPC run
had accuracy 0.6961 and F1 0.8176, consistent with the weak custom LoRA result
under these settings. This is a local observation, not a universal verdict on
LoRA.

On SST-2, custom LoRA mean accuracy was 0.9281 versus 0.9304 for full
fine-tuning. With only three seeds, this small difference cannot establish
statistical equivalence. The task comparison shows that transfer quality
depends on the task and the chosen optimization settings.

## Cost and artifact accounting

The rank-8 query/value configuration trained 887,042 parameters: 294,912
adapter factors plus 592,130 classifier parameters. Full fine-tuning trained
124,647,170 parameters. The complete LoRA task artifact was about 3.57 MB;
the full fine-tuning task artifact was about 498.7 MB. LoRA still loads the
base model, and its resumable checkpoint is about 505 MB because it includes
the frozen backbone and optimizer state.

| Measurement, mean of three core seeds | Full fine-tuning | Custom LoRA |
| --- | ---: | ---: |
| MRPC peak allocated GPU memory | 2.41 GiB | 0.92 GiB |
| SST-2 peak allocated GPU memory | 2.35 GiB | 0.83 GiB |
| MRPC per-run elapsed time | 97.5 s | 36.8 s |
| SST-2 per-run elapsed time | 1,204.6 s | 348.0 s |

Elapsed time is the runner's wall time, including model/data loading,
training, evaluation, and checkpoint work. Memory is PyTorch peak allocated
memory, not total system GPU use. These are measurements on one laptop and
are not a hardware-independent speedup claim.

## Bounded MRPC ablations

The eight additional runs use seed 1 only. Rank 1, 2, and 4 with scaling
alpha/r = 2, two rank-4 alpha variations, and query-only/value-only placement
all predicted the positive class on every validation example. Rank 16 with
alpha 32 reached accuracy 0.7108 and F1 0.8223, but one seed cannot support
a stable rank trend. See each run's resolved configuration and predictions
when investigating placement or scaling.

## Artifact correction

An older runner called `count_parameters` after merging custom LoRA into
dense inference layers. Its raw summaries therefore report an inflated
trainable count and omit adapter factors from the model total. The published
CSV/figure and profile recompute custom LoRA counts from the saved `task.pt`
training artifact; raw summaries remain unchanged as an audit trail. The
runner now captures counts before merge for future runs. This correction
changes parameter accounting only, not trained weights, predictions, task
metrics, or timing.
