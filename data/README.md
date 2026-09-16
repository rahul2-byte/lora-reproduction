# Data

The runner downloads `nyu-mll/glue` at revision
`bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` and saves per-run content
hashes, split counts, and exact normalized text-overlap reports in
`runs/reliable/<task>/<method>/<run-id>/data_manifest.json`. Tokenization uses
the pinned RoBERTa tokenizer, truncates at 128 tokens, and caches locally
under `data/cache/`. Official test labels are not used for local scoring.

The [GLUE distribution card](https://huggingface.co/datasets/nyu-mll/glue)
lists its license as `other`; underlying MRPC and SST-2 source terms must be
assessed separately before redistributing any dataset rows. This repository
publishes no raw dataset files or model weights.
