# Reference parity

On 2026-09-15, a real RoBERTa-base comparison used the pinned model revision,
copied base weights, nonzero adapter factors, rank 4, alpha 8, zero dropout,
and query/value targets in both implementations.

Observed result:

```text
max_abs_error: 0.0
allclose: True
```

This verifies the forward path for that configuration. Gradient, optimizer-step,
and matched-training parity remain separate checks. One MRPC PEFT training run
was subsequently completed: validation accuracy 0.6961 and positive-class F1
0.8176. It used the same stated rank/alpha/task setup as the custom core
configuration, but a one-seed quality comparison is not numerical gradient
parity or proof of identical training trajectories.
