"""Validated task artifacts and resumable training checkpoints."""

from __future__ import annotations

import os
import pickle
import tempfile
import shutil
import random
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class CheckpointError(ValueError):
    """Raised for incomplete or incompatible checkpoint artifacts."""


TASK_METADATA_KEYS = {
    "base_model_id",
    "base_model_revision",
    "tokenizer_revision",
    "label_mapping",
    "injection_config",
    "artifact_format",
}


def save_task_artifact(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Atomically save trainable parameters and required model metadata."""
    _validate_metadata(metadata)
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise CheckpointError("task artifact has no trainable parameters")
    payload = {"metadata": metadata, "state_dict": state}
    _atomic_torch_save(payload, Path(path))


def load_task_artifact(
    model: nn.Module,
    path: str | Path,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Load a task artifact after validating identity and trainable keys."""
    _validate_metadata(expected_metadata)
    payload = _load_payload(path)
    metadata = payload.get("metadata")
    state = payload.get("state_dict")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise CheckpointError("invalid task artifact structure")
    for key in TASK_METADATA_KEYS | {"config_hash", "dataset_revision"}:
        if key in expected_metadata and metadata.get(key) != expected_metadata[key]:
            raise CheckpointError(f"checkpoint metadata mismatch: {key}")

    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != trainable:
        missing = sorted(trainable - set(state))
        unexpected = sorted(set(state) - trainable)
        raise CheckpointError(
            f"trainable checkpoint keys differ; missing={missing}, unexpected={unexpected}"
        )
    model.load_state_dict(state, strict=False)
    return metadata


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    metadata: dict[str, Any],
    rng_state: dict[str, Any],
    update: int,
    epoch: int = 0,
    next_batch: int = 0,
    best_metric: float | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    """Atomically save enough state to resume at an optimizer boundary."""
    _validate_metadata(metadata)
    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng_state": rng_state,
        "update": update,
        "epoch": epoch,
        "next_batch": next_batch,
        "best_metric": best_metric,
        "history": history or [],
    }
    _atomic_torch_save(payload, Path(path))


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    scaler: Any = None,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Restore a checkpoint after validating its experiment identity."""
    _validate_metadata(expected_metadata)
    payload = _load_payload(path)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise CheckpointError("training checkpoint has no metadata")
    for key in TASK_METADATA_KEYS | {"config_hash", "dataset_revision"}:
        if key in expected_metadata and metadata.get(key) != expected_metadata[key]:
            raise CheckpointError(f"checkpoint metadata mismatch: {key}")
    if not all(key in payload for key in ("model", "optimizer", "rng_state", "update", "epoch", "next_batch")):
        raise CheckpointError("training checkpoint is incomplete")
    if (not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict)
            or not isinstance(payload["rng_state"], dict)
            or not all(key in payload["rng_state"] for key in ("python", "numpy", "torch", "cuda"))
            or not isinstance(payload["update"], int) or not isinstance(payload["epoch"], int)
            or not isinstance(payload["next_batch"], int)):
        raise CheckpointError("training checkpoint has invalid state fields")
    if (scheduler is None) != (payload.get("scheduler") is None):
        raise CheckpointError("scheduler state does not match configuration")
    if (scaler is None) != (payload.get("scaler") is None):
        raise CheckpointError("scaler state does not match configuration")
    try:
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise CheckpointError("invalid training checkpoint state") from error
    return {
        "metadata": metadata,
        "rng_state": payload.get("rng_state"),
        "update": payload.get("update"),
        "epoch": payload.get("epoch", 0),
        "next_batch": payload.get("next_batch", 0),
        "best_metric": payload.get("best_metric"),
        "history": payload.get("history", []),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    if not isinstance(state, dict) or not all(key in state for key in ("python", "numpy", "torch", "cuda")):
        raise CheckpointError("checkpoint lacks complete RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise CheckpointError("CUDA RNG state requires CUDA")
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.flush()
            os.fsync(file.fileno())
        json.loads(Path(temporary).read_text())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def require_disk_space(path: str | Path, bytes_needed: int) -> None:
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(directory).free
    if available < bytes_needed:
        raise CheckpointError(f"insufficient disk space: need {bytes_needed} bytes, have {available}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(TASK_METADATA_KEYS - set(metadata))
    if missing:
        raise CheckpointError(f"checkpoint metadata missing: {missing}")


def _load_payload(path: str | Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as error:
        raise CheckpointError(f"could not read checkpoint: {path}") from error
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a mapping")
    return payload


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Full fine-tuning checkpoints are roughly 1.5 GB. Reserve room for the
    # temporary file and the next artifact before starting a replacement.
    previous = path.stat().st_size if path.exists() else 0
    require_disk_space(path, max(2_000_000_000, previous + 500_000_000))
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        verify = torch.load(temporary_path, map_location="cpu", weights_only=False)
        if not isinstance(verify, dict) or set(payload) != set(verify):
            raise CheckpointError("temporary checkpoint failed validation")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
