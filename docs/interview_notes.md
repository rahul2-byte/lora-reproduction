# Interview notes

The trained update is \(W_0+(\alpha/r)BA\). Freezing \(W_0\) and optimizing
the two thin factors constrains the update's rank to at most \(r\). With B
initialized at zero, inference initially matches the pretrained layer. B
receives a useful first-step gradient; A can receive zero gradient until B
becomes nonzero. Adapter parameters for one linear layer are
\(r(d_{in}+d_{out})\); the classifier is a separate trained component.

The core query/value rank-8 run trained 887,042 values including its
592,130-parameter task head. Full fine-tuning trained 124,647,170. Peak
allocated memory fell from roughly 2.4 GiB to under 1 GiB, much less than
the parameter-count ratio, because the base model and activations remain.
The task artifact was 3.57 MB versus 498.7 MB, but LoRA inference still
needs the base model.

The main negative finding matters: MRPC custom LoRA mostly predicted the
positive class, so its 0.8157 mean F1 was close to the 0.8122 positive-only
diagnostic. Full fine-tuning reached 0.9059 mean F1. On SST-2, custom LoRA
and full fine-tuning reached 0.9281 and 0.9304 mean accuracy. Three seeds do
not establish equivalence or a universal rank trend. The next discriminating
experiment would tune LoRA's learning rate on the local tuning partition,
then evaluate a selected setting once on held-out validation.

Implementation questions to be ready for: explain exact query/value module
selection, classifier serialization, why checkpoint state includes optimizer
and RNG, why merge is only for evaluation, and why a post-merge parameter
count was wrong in the initial report. Point to the corrected artifact-based
table and the preserved discrepancy record when discussing that bug.
