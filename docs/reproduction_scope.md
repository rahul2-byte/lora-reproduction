# Reproduction scope

## Implemented and tested

- Frozen pretrained linear weights plus trainable low-rank factors, scaled by
  alpha/r; initial equivalence, gradient flow, serialization, and merge checks.
- Exact query/value projection injection in RoBERTa-base, with a trained
  classifier head.
- MRPC and SST-2, three seeds each for head-only, full fine-tuning, and custom
  LoRA, plus a single trusted PEFT training comparison on MRPC.
- Eight one-seed MRPC rank, scaling, and module-placement ablations.
- A real-model PEFT forward-parity check with copied nonzero adapter weights.

These experiments are a **scaled reproduction** of the mechanism and a local
engineering study. The separate local tuning partition, held-out official
development split, software stack, and hyperparameter budget differ from the
paper's published comparison. The SST-2/MRPC observations are local evidence,
not official GLUE test scores.

## Excluded or incomplete

GPT-3-scale results, the complete GLUE suite, a matched paper hyperparameter
search, equal-parameter module-placement controls, decoder transfer,
quantization, multi-GPU training, production serving, and gradient/optimizer
parity against PEFT are outside the completed evidence.
