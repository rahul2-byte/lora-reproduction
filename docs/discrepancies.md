# Discrepancies

## MRPC LoRA versus full fine-tuning

- **Observation:** Across three seeds, custom LoRA accuracy was 0.6928 ±
  0.0086, while full fine-tuning reached 0.8717 ± 0.0051 on the held-out
  GLUE validation split.
- **Evidence:** LoRA predicted the positive class for 408, 395, and 400 of
  408 examples; a positive-only predictor scores 0.6838 accuracy and 0.8122
  positive-class F1 on this split. A one-seed PEFT LoRA run also scored near
  that level. The saved per-example predictions support this diagnosis.
- **Hypothesis:** The shared 2e-5 learning rate and three-epoch budget may be
  inadequate for this LoRA configuration on MRPC. This has not been tested by
  a matched tuning-budget search, so it is not an established cause.
- **Discriminating experiment:** Compare a small, preregistered LoRA learning
  rate range on the existing tuning partition, with the official validation
  split untouched until selection. Repeat a selected setting across seeds.
- **Status:** Not run. The current result is valid for the declared setting;
  the reason for the gap remains unresolved.

## Parameter reporting after merge

- **Observation:** Older custom LoRA `summary.json` files report roughly
  14.8 million trainable parameters for the core configuration.
- **Cause verified in code:** The summary counted after merged inference
  export, where adapted dense weights inherited `requires_grad=True` and
  adapter factors were removed.
- **Correction:** The corresponding training artifact contains exactly
  887,042 trained parameter values. The report and profile scripts derive
  publication counts from that artifact; the runner now captures counts
  before merging. Raw run summaries are retained.

## Source and protocol differences

The paper's reported GLUE figures and this study use different selection,
training, and aggregation protocols. The official GLUE test labels were not
used locally. No direct numerical “gap from paper” is calculated.
