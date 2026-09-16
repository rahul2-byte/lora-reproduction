# Claim ledger

Source: [Hu et al., LoRA, arXiv v2](https://arxiv.org/abs/2106.09685v2),
abstract and RoBERTa/GLUE experiments. The official
[author repository](https://github.com/microsoft/LoRA) is a reference, not
the implementation used to train these runs.

| Claim or question | Local evidence | Status | Interpretation |
| --- | --- | --- | --- |
| A low-rank update can adapt frozen pretrained weights with far fewer trained parameters | Run `custom_lora-1-ef495409` in the [corrected table](../results/reliable/metrics.csv) | Approximately reproduced | Core LoRA trained 887,042 values versus 124,647,170 for full fine-tuning. This is a parameter count, not a quality result. |
| LoRA quality can approach full fine-tuning on RoBERTa tasks | MRPC three-seed runs in [per-run metrics](../results/reliable/metrics.csv) | Contradicted in our setting | MRPC LoRA 0.6928 ± 0.0086 versus full fine-tuning 0.8717 ± 0.0051 accuracy. The specific shared learning rate may matter. No global paper claim is disproved. |
| LoRA quality can approach full fine-tuning on RoBERTa tasks | SST-2 three-seed runs in [per-run metrics](../results/reliable/metrics.csv) | Approximately reproduced | SST-2 LoRA 0.9281 ± 0.0046 versus full fine-tuning 0.9304 ± 0.0024 accuracy under the local held-out protocol. |
| Fewer trainable parameters reduce GPU memory and task artifact size | [resource profile](../results/reliable/profile.csv) | Approximately reproduced | LoRA used less peak allocated memory and a smaller task artifact; the base model is still required at inference. |
| Merging can preserve inference outputs | `merged_unmerged_max_abs_error` in the [per-run table](../results/reliable/metrics.csv) | Approximately reproduced | Fourteen custom runs recorded a maximum absolute logit difference below 2.56e-5 on the checked batch. |
| Findings at GPT-3 scale and the full original benchmark suite | No local run | Not attempted | Outside the 6 GB GPU scope. |

All local quality scores use the GLUE **validation** split held out from local
checkpoint selection. They are not official GLUE test scores. The paper's
hyperparameter search and reporting protocol were not matched, so no local
row is labeled an exact paper reproduction.
