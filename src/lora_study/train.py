"""Small explicit training loop shared by benchmark methods."""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn


@dataclass(frozen=True)
class TrainSummary:
    mean_loss: float
    examples: int
    optimizer_updates: int
    skipped_updates: int = 0


def train_epoch(
    model: nn.Module,
    batches,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    accumulation: int = 1,
    max_grad_norm: float | None = 1.0,
    scheduler: Any = None,
    scaler: Any = None,
    precision: str = "fp32",
    progress: bool = False,
    progress_label: str = "training",
    progress_interval_seconds: float = 1.0,
    non_blocking: bool = False,
    on_update: Callable[[int, int], None] | None = None,
    initial_batches: int = 0,
) -> TrainSummary:
    if accumulation <= 0:
        raise ValueError("accumulation must be positive")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(f"{precision} requires CUDA")
    if progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    examples = 0
    updates = 0
    windows_completed = 0
    consumed_batches = initial_batches
    total_updates = _ceil_div(len(batches), accumulation)
    progress_started = time.perf_counter()
    last_progress = progress_started
    iterator = iter(batches)
    while True:
        window = []
        for _ in range(accumulation):
            try:
                window.append(next(iterator))
            except StopIteration:
                break
        if not window:
            break
        window_examples = sum(batch["labels"].shape[0] for batch in window)
        for raw_batch in window:
            batch = {key: value.to(device, non_blocking=non_blocking) for key, value in raw_batch.items()}
            batch.pop("example_id", None)
            batch_size = batch["labels"].shape[0]
            context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                enabled=precision != "fp32",
            )
            with context:
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            losses.append(float(loss.detach().cpu()))
            normalized_loss = loss * (batch_size / window_examples)
            if scaler is None:
                normalized_loss.backward()
            else:
                scaler.scale(normalized_loss).backward()
        examples += window_examples
        stepped = _optimizer_step(model, optimizer, scheduler, max_grad_norm, scaler)
        updates += int(stepped)
        windows_completed += 1
        consumed_batches += len(window)
        if on_update is not None:
            on_update(updates, consumed_batches)
        now = time.perf_counter()
        if progress and (now - last_progress >= progress_interval_seconds or windows_completed == total_updates):
            _print_progress(progress_label, windows_completed, total_updates, progress_started)
            last_progress = now
    if progress:
        sys.stderr.write("\n")
        sys.stderr.flush()
    return TrainSummary(sum(losses) / len(losses) if losses else math.nan,
                        examples, updates, windows_completed - updates)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _print_progress(label: str, completed: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed else 0.0
    remaining = (total - completed) / rate if rate else 0.0
    width = 24
    filled = int(width * completed / total) if total else width
    bar = "=" * min(width, filled) + ">" + " " * max(0, width - filled - 1)
    eta = _duration(remaining)
    sys.stderr.write(f"\r{label} [{bar}] {completed}/{total} elapsed={_duration(elapsed)} ETA={eta}")
    sys.stderr.flush()


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _optimizer_step(model, optimizer, scheduler, max_grad_norm, scaler) -> bool:
    if scaler is not None:
        scaler.unscale_(optimizer)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_grad_norm,
        )
    if scaler is None:
        optimizer.step()
        stepped = True
    else:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= old_scale
    optimizer.zero_grad(set_to_none=True)
    if scheduler is not None and stepped:
        scheduler.step()
    return stepped
