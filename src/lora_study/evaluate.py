"""Task metrics and prediction helpers."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def accuracy(predictions: Iterable[int], labels: Iterable[int]) -> float:
    predictions = list(predictions)
    labels = list(labels)
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must have equal nonzero length")
    return sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels)


def binary_f1(predictions: Iterable[int], labels: Iterable[int], positive: int = 1) -> float:
    predictions = list(predictions)
    labels = list(labels)
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must have equal nonzero length")
    true_positive = sum(p == positive and y == positive for p, y in zip(predictions, labels))
    false_positive = sum(p == positive and y != positive for p, y in zip(predictions, labels))
    false_negative = sum(p != positive and y == positive for p, y in zip(predictions, labels))
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


@torch.no_grad()
def predict(model: nn.Module, batches, device: torch.device) -> tuple[list[int], list[int], list[int]]:
    was_training = model.training
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    example_ids: list[int] = []
    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        targets = batch.pop("labels")
        ids = batch.pop("example_id", None)
        logits = model(**batch).logits
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(targets.cpu().tolist())
        if ids is not None:
            example_ids.extend(ids.cpu().tolist())
        else:
            example_ids.extend(range(len(example_ids), len(example_ids) + len(targets)))
    model.train(was_training)
    return predictions, labels, example_ids
