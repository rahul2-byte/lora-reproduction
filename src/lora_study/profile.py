"""Small resource-measurement helpers."""

from __future__ import annotations

import time
from pathlib import Path

import torch


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "trainable_fraction": trainable / total}


def artifact_parameter_counts(summary: dict, task_path: str | Path) -> dict:
    """Recover training counts from the saved task state for older merged summaries."""
    counts = summary["parameters"]
    if summary.get("method") != "custom_lora":
        return counts
    payload = torch.load(task_path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    trained = sum(tensor.numel() for tensor in state.values())
    adapter = sum(tensor.numel() for name, tensor in state.items()
                  if name.endswith((".lora_a", ".lora_b")))
    if not adapter or trained <= adapter:
        raise ValueError(f"invalid LoRA task artifact: {task_path}")
    if counts["trainable"] == trained:
        return counts
    # Older summaries counted after merge, when adapter factors had been
    # replaced by dense inference weights. Restore their training count.
    total = counts["total"] + adapter
    return {"total": total, "trainable": trained,
            "trainable_fraction": trained / total}


def timed(callable_, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = callable_(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def file_size(path: str | Path) -> int | None:
    """Return an artifact size, preserving null for missing files."""
    candidate = Path(path)
    return candidate.stat().st_size if candidate.is_file() else None
