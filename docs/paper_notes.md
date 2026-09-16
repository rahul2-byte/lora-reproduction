# Paper notes

Primary source: [Hu et al., LoRA, arXiv:2106.09685v2](https://arxiv.org/abs/2106.09685v2),
revised 16 October 2021 and published at ICLR 2022. The
[official author repository](https://github.com/microsoft/LoRA) describes the
method and example experiments. This project uses its own adapter code and
uses pinned PEFT only as a validation reference; it does not vendor or execute
the author's `loralib` source.

For a linear layer with frozen \(W_0\in\mathbb{R}^{d_{out}\times d_{in}}\),
the implemented update is \(\Delta W=(\alpha/r)BA\), where
\(A\in\mathbb{R}^{r\times d_{in}}\) and
\(B\in\mathbb{R}^{d_{out}\times r}\). The adapter adds
\(\Delta W x\) to the base output. The code places optional dropout on the
adapter input, initializes A randomly and B to zero, and trains A/B plus the
task classifier. This starts at the pretrained function. B can receive a
gradient on the first backward pass; A may initially have zero gradient until
B changes. Tests check these behaviors and the dense/factorized equivalence.

The pinned base checkpoint is `FacebookAI/roberta-base` at revision
`e2da8e2f811d1448a5b465c236feacd80ffbac7b`. The distribution of GLUE
used here is `nyu-mll/glue` at revision
`bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`. The source model is a
masked-language-model checkpoint, so the classification head is newly
initialized for each seed.

The paper studies a wider set of model sizes, tasks, configurations, and
selection rules. Its RoBERTa GLUE results should not be subtracted from this
study's held-out development scores as though the protocols matched.
