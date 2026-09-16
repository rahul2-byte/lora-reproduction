"""Sequential single-GPU runs with verified artifacts and optimizer-boundary resume."""

from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import math
import os
import platform
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from .checkpoints import (
    CheckpointError, atomic_json, capture_rng_state, load_training_checkpoint,
    restore_rng_state, save_task_artifact, save_training_checkpoint,
)
from .config import StudyConfig
from .data import audit_split_overlap, label_mapping, load_glue_task, split_manifest, tokenize_task_cached
from .evaluate import accuracy, binary_f1, predict
from .injection import inject_lora, merge_lora_modules, resolve_target_paths
from .models import load_roberta_classifier
from .profile import count_parameters
from .reference import apply_reference_lora
from .train import train_epoch

LOGGER = logging.getLogger(__name__)


def run_id_for(config: StudyConfig) -> str:
    return f"{config.experiment.method}-{config.experiment.seed}-{config.resolved_hash[:8]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_completed(run_dir: Path, config: StudyConfig) -> dict | None:
    """Reuse only a run whose config and required artifact hashes still agree."""
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
        manifest = json.loads((run_dir / "artifacts.json").read_text())
        if (summary.get("status") != "completed"
                or summary.get("config_hash") != config.resolved_hash
                or summary.get("run_id") != run_id_for(config)
                or json.loads((run_dir / "config.resolved.json").read_text()) != config.to_dict()):
            return None
        required = {"task.pt", "predictions/validation.jsonl", "config.resolved.json"}
        if not isinstance(manifest, dict) or set(manifest) != required:
            return None
        for name, expected in manifest.items():
            path = run_dir / name
            if not path.is_file() or _sha256(path) != expected:
                return None
        return summary
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def run(config: StudyConfig, output_root: str | Path, resume_checkpoint: str | Path | None = None) -> dict:
    run_dir = Path(output_root) / run_id_for(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CheckpointError(f"run is already active: {run_dir}") from error
        return _run_locked(config, run_dir, resume_checkpoint)


def _run_locked(config: StudyConfig, run_dir: Path, resume_checkpoint: str | Path | None) -> dict:
    completed = validate_completed(run_dir, config)
    if completed is not None:
        LOGGER.info("reusing verified completed run %s", run_dir)
        return completed
    if (run_dir / "summary.json").exists():
        try:
            prior = json.loads((run_dir / "summary.json").read_text())
        except (OSError, ValueError) as error:
            raise CheckpointError(f"run summary is unreadable: {run_dir}") from error
        if prior.get("status") == "completed":
            if not (run_dir / "checkpoints" / "latest.pt").is_file():
                raise CheckpointError(f"completed run failed artifact validation and has no recovery checkpoint: {run_dir}")
            LOGGER.warning("completed artifacts invalid; regenerating from checkpoint: %s", run_dir)
    if (run_dir / "config.resolved.json").exists():
        if json.loads((run_dir / "config.resolved.json").read_text()) != config.to_dict():
            raise CheckpointError(f"existing run has a different config: {run_dir}")
    atomic_json(run_dir / "config.resolved.json", config.to_dict())
    try:
        summary = _run(config, run_dir, resume_checkpoint)
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(run_dir / "artifacts.json", {
            name: _sha256(run_dir / name)
            for name in ("task.pt", "predictions/validation.jsonl", "config.resolved.json")
        })
        return summary
    except BaseException as error:
        atomic_json(run_dir / "summary.json", {
            "run_id": run_id_for(config),
            "status": "cancelled" if isinstance(error, KeyboardInterrupt) else "failed",
            "method": config.experiment.method, "seed": config.experiment.seed,
            "config_hash": config.resolved_hash,
            "failure_reason": f"{type(error).__name__}: {error}",
        })
        raise


def _loader(dataset, collator, config: StudyConfig, salt: int, batch_sampler=None):
    options = {
        "collate_fn": collator, "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "generator": torch.Generator().manual_seed(config.experiment.seed + salt),
    }
    if batch_sampler is None:
        return DataLoader(dataset, batch_size=config.training.microbatch, shuffle=False, **options)
    return DataLoader(dataset, batch_sampler=batch_sampler, **options)


def _epoch_loader(dataset, collator, config: StudyConfig, epoch: int, offset: int):
    generator = torch.Generator().manual_seed(config.experiment.seed + epoch * 1_000_003)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    size = config.training.microbatch
    batches = [order[index:index + size] for index in range(0, len(order), size)]
    return _loader(dataset, collator, config, epoch + 200, batches[offset:])


def _save_progress(path, model, optimizer, scheduler, scaler, config, epoch, next_batch, update, best_metric, history):
    started = time.perf_counter()
    save_training_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        metadata=metadata_for(config), rng_state=capture_rng_state(),
        update=update, epoch=epoch, next_batch=next_batch,
        best_metric=best_metric, history=history,
    )
    LOGGER.info("checkpoint=%s epoch=%d next_batch=%d update=%d bytes=%d seconds=%.2f",
                path, epoch, next_batch, update, path.stat().st_size, time.perf_counter() - started)


def _run(config: StudyConfig, run_dir: Path, resume_checkpoint: str | Path | None) -> dict:
    _seed_everything(config.experiment.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.training.precision != "fp32" and device.type != "cuda":
        raise ValueError(f"{config.training.precision} requires CUDA")
    started = time.perf_counter()
    LOGGER.info("run=%s task=%s method=%s device=%s", run_id_for(config), config.dataset.task, config.experiment.method, device)
    atomic_json(run_dir / "environment.json", {
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_roberta_classifier(config.model.model_id, config.model.revision, 2, str(device))
    _configure_trainable_parameters(model, config)
    if config.experiment.method == "custom_lora":
        inject_lora(model, resolve_target_paths(model, config.lora.target_modules),
                    rank=config.lora.rank, alpha=config.lora.alpha, dropout=config.lora.dropout,
                    initialization=config.lora.initialization, trainable_prefixes=("classifier",))
    elif config.experiment.method == "reference_lora":
        model = apply_reference_lora(model, rank=config.lora.rank, alpha=config.lora.alpha,
                                     dropout=config.lora.dropout)
    training_parameters = count_parameters(model)

    dataset = load_glue_task(config.dataset.task, config.dataset.revision, config.dataset.dataset_id)
    atomic_json(run_dir / "data_manifest.json", {
        "dataset_id": config.dataset.dataset_id, "revision": config.dataset.revision,
        "task": config.dataset.task, "split_manifest": split_manifest(dataset),
        "overlap_audit": audit_split_overlap(dataset, config.dataset.task),
    })
    tokenized = tokenize_task_cached(dataset, tokenizer, config.dataset.task,
                                     config.dataset.max_length, config.dataset.token_cache_dir,
                                     config.dataset.revision)
    parts = tokenized["train"].train_test_split(test_size=0.1, seed=config.experiment.seed)
    if config.training.max_train_examples is not None:
        parts["train"] = parts["train"].select(range(min(len(parts["train"]), config.training.max_train_examples)))
    collator = DataCollatorWithPadding(tokenizer)
    tuning = _loader(parts["test"], collator, config, 100)
    validation = _loader(tokenized["validation"], collator, config, 101)
    batches_per_epoch = math.ceil(len(parts["train"]) / config.training.microbatch)
    updates_per_epoch = math.ceil(batches_per_epoch / config.training.gradient_accumulation)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no trainable parameters; frozen classifier is diagnostic only")
    optimizer = torch.optim.AdamW(trainable, lr=config.training.learning_rate,
                                  weight_decay=config.training.weight_decay)
    scheduler = None
    if config.training.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0,
            total_iters=max(1, config.training.epochs * updates_per_epoch))
    scaler = torch.amp.GradScaler("cuda") if config.training.precision == "fp16" else None
    latest = run_dir / "checkpoints" / "latest.pt"
    best = run_dir / "checkpoints" / "best.pt"
    source = Path(resume_checkpoint) if resume_checkpoint else latest
    if resume_checkpoint is None and not latest.is_file() and best.is_file():
        source = best
    epoch, next_batch, updates, best_metric = 0, 0, 0, float("-inf")
    history: list[dict] = []
    if source.is_file():
        try:
            restored = load_training_checkpoint(source, model=model, optimizer=optimizer,
                                                scheduler=scheduler, scaler=scaler,
                                                expected_metadata=metadata_for(config))
        except CheckpointError:
            if resume_checkpoint is not None or source == best or not best.is_file():
                raise
            LOGGER.warning("latest checkpoint invalid; trying verified best checkpoint %s", best)
            source = best
            restored = load_training_checkpoint(source, model=model, optimizer=optimizer,
                                                scheduler=scheduler, scaler=scaler,
                                                expected_metadata=metadata_for(config))
        epoch = int(restored["epoch"])
        next_batch = int(restored["next_batch"])
        updates = int(restored["update"])
        best_metric = restored["best_metric"] if restored["best_metric"] is not None else float("-inf")
        history = list(restored["history"])
        if not 0 <= epoch <= config.training.epochs or not 0 <= next_batch <= batches_per_epoch:
            raise CheckpointError("checkpoint data position is invalid")
        restore_rng_state(restored["rng_state"])
        LOGGER.info("resumed epoch=%d next_batch=%d update=%d from %s", epoch, next_batch, updates, source)
    elif resume_checkpoint is not None:
        raise CheckpointError(f"resume checkpoint does not exist: {source}")
    elif (run_dir / "metrics.jsonl").exists():
        raise CheckpointError(f"partial run has no latest checkpoint: {run_dir}")

    interrupted = False
    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        LOGGER.warning("interruption requested; saving at next optimizer boundary")
    old_int, old_term = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        LOGGER.info("training-ready run=%s epoch=%d next_batch=%d", run_id_for(config), epoch, next_batch)
        last_checkpoint = time.monotonic()
        for current_epoch in range(epoch, config.training.epochs):
            offset = next_batch if current_epoch == epoch else 0
            loader = _epoch_loader(parts["train"], collator, config, current_epoch, offset)
            epoch_start_updates = updates
            train_started = time.perf_counter()
            def on_update(local_update: int, consumed: int) -> None:
                nonlocal last_checkpoint
                absolute = epoch_start_updates + local_update
                if interrupted or time.monotonic() - last_checkpoint >= 300:
                    _save_progress(latest, model, optimizer, scheduler, scaler, config,
                                   current_epoch, consumed, absolute, best_metric, history)
                    last_checkpoint = time.monotonic()
                    if interrupted:
                        raise KeyboardInterrupt("checkpoint saved after interruption")
            result = train_epoch(
                model, loader, optimizer, device,
                accumulation=config.training.gradient_accumulation,
                max_grad_norm=config.training.max_grad_norm, scheduler=scheduler,
                scaler=scaler, precision=config.training.precision, progress=True,
                progress_label=f"{config.dataset.task}/{config.experiment.method}/seed={config.experiment.seed} epoch={current_epoch + 1}",
                progress_interval_seconds=config.training.progress_interval_seconds,
                non_blocking=config.training.pin_memory,
                on_update=on_update, initial_batches=offset,
            )
            train_seconds = time.perf_counter() - train_started
            updates += result.optimizer_updates
            pred, labels, _ = predict(model, tuning, device)
            record = {
                "epoch": current_epoch + 1, "loss": result.mean_loss,
                "training_seconds": train_seconds, "examples": result.examples,
                "examples_per_second": result.examples / train_seconds,
                "skipped_updates": result.skipped_updates,
                "tuning_accuracy": accuracy(pred, labels),
                "tuning_f1": binary_f1(pred, labels) if config.dataset.task == "mrpc" else None,
            }
            metric = record["tuning_f1"] if config.dataset.task == "mrpc" else record["tuning_accuracy"]
            if metric > best_metric:
                best_metric = metric
                _save_progress(best, model, optimizer, scheduler, scaler, config,
                               current_epoch + 1, 0, updates, best_metric, history + [record])
            history.append(record)
            _save_progress(latest, model, optimizer, scheduler, scaler, config,
                           current_epoch + 1, 0, updates, best_metric, history)
            last_checkpoint = time.monotonic()
            LOGGER.info("epoch=%d update=%d tuning=%s", current_epoch + 1, updates, record)
            next_batch = 0
        if not best.is_file():
            raise CheckpointError(f"best checkpoint missing: {best}")
        load_training_checkpoint(best, model=model, optimizer=optimizer, scheduler=scheduler,
                                 scaler=scaler, expected_metadata=metadata_for(config))
        pred, labels, ids = predict(model, validation, device)
        final = {
            "split": "validation", "accuracy": accuracy(pred, labels),
            "f1": binary_f1(pred, labels) if config.dataset.task == "mrpc" else None,
        }
        save_task_artifact(model, run_dir / "task.pt", metadata_for(config))
        prediction_path = run_dir / "predictions" / "validation.jsonl"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_path.with_suffix(".jsonl.tmp")
        with temporary.open("w") as file:
            for example_id, prediction, label in zip(ids, pred, labels):
                file.write(json.dumps({"example_id": example_id, "prediction": prediction, "label": label}) + "\n")
        if sum(1 for _ in temporary.open()) != len(labels):
            raise CheckpointError("prediction file is incomplete")
        os.replace(temporary, prediction_path)
        merged_error = _check_merged_equivalence(model, validation, device) if config.experiment.method == "custom_lora" else None
        return {
            "run_id": run_id_for(config), "status": "completed", "method": config.experiment.method,
            "seed": config.experiment.seed, "track": config.experiment.track,
            "task": config.dataset.task, "device": str(device), "parameters": training_parameters,
            "history": history + [final], "optimizer_updates": updates,
            "training_seconds": time.perf_counter() - started,
            "peak_allocated_gpu_memory": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_reserved_gpu_memory": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
            "config_hash": config.resolved_hash, "dataset_id": config.dataset.dataset_id,
            "dataset_revision": config.dataset.revision, "model_id": config.model.model_id,
            "model_revision": config.model.revision, "evaluation_protocol": config.dataset.evaluation_protocol,
            "label_mapping": label_mapping(config.dataset.task),
            "merged_unmerged_max_abs_error": merged_error,
        }
    finally:
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def metadata_for(config: StudyConfig) -> dict:
    return {
        "base_model_id": config.model.model_id, "base_model_revision": config.model.revision,
        "tokenizer_revision": config.model.revision, "label_mapping": label_mapping(config.dataset.task),
        "injection_config": {
            "method": config.experiment.method, "rank": config.lora.rank,
            "alpha": config.lora.alpha, "scaling": config.lora.scaling,
            "dropout": config.lora.dropout, "target_modules": list(config.lora.target_modules),
            "initialization": config.lora.initialization,
        },
        "artifact_format": 2, "config_hash": config.resolved_hash,
        "dataset_revision": config.dataset.revision,
    }


def _configure_trainable_parameters(model, config: StudyConfig) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(config.experiment.method == "full_finetune")
    if config.experiment.method == "head_only":
        for name, parameter in model.named_parameters():
            if name.startswith("classifier"):
                parameter.requires_grad_(True)


def _check_merged_equivalence(model, batches, device: torch.device) -> float:
    model.eval()
    batch = {key: value.to(device) for key, value in next(iter(batches)).items()}
    batch.pop("labels", None)
    batch.pop("example_id", None)
    with torch.no_grad():
        unmerged = model(**batch).logits
        merge_lora_modules(model)
        merged = model(**batch).logits
    return float((unmerged - merged).abs().max().cpu())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
