"""Pinned GLUE loading and tokenization helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any


TASK_COLUMNS = {
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
}

LABEL_MAPPINGS = {
    "mrpc": {"not_equivalent": 0, "equivalent": 1},
    "sst2": {"negative": 0, "positive": 1},
}
LOGGER = logging.getLogger(__name__)


def label_mapping(task: str) -> dict[str, int]:
    try:
        return dict(LABEL_MAPPINGS[task])
    except KeyError as error:
        raise ValueError(f"unsupported task: {task}") from error


def load_glue_task(
    task: str,
    revision: str | None = None,
    dataset_id: str = "nyu-mll/glue",
):
    if task not in TASK_COLUMNS:
        raise ValueError(f"unsupported task: {task}")
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("Datasets is required for benchmark loading") from error
    kwargs = {} if revision is None else {"revision": revision}
    for attempt in range(3):
        try:
            return load_dataset(dataset_id, task, **kwargs)
        except (OSError, ConnectionError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def tokenize_task(dataset, tokenizer, task: str, max_length: int):
    columns = TASK_COLUMNS.get(task)
    if columns is None:
        raise ValueError(f"unsupported task: {task}")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    tokenized = dataset.map(
        lambda batch: tokenizer(
            *[batch[column] for column in columns],
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[
            column for column in dataset["train"].column_names if column not in {"label", "idx"}
        ],
    )
    tokenized = tokenized.rename_column("label", "labels")
    if "idx" in dataset["train"].column_names:
        tokenized = tokenized.rename_column("idx", "example_id")
    return tokenized


def tokenize_task_cached(
    dataset,
    tokenizer,
    task: str,
    max_length: int,
    cache_dir: str | Path,
    dataset_revision: str = "unknown",
):
    """Cache tokenized data by task, tokenizer identity, and preprocessing version."""
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as error:
        raise RuntimeError("Datasets is required for tokenization caching") from error
    tokenizer_id = getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__)
    key_payload = json.dumps(
        {
            "task": task,
            "max_length": max_length,
            "tokenizer": tokenizer_id,
            "dataset_revision": dataset_revision,
            "version": 2,
        },
        sort_keys=True,
    ).encode()
    key = hashlib.sha256(key_payload).hexdigest()[:16]
    path = Path(cache_dir) / f"{task}-{key}"
    def valid_cache(candidate: Path):
        cached = load_from_disk(str(candidate))
        if (not isinstance(cached, DatasetDict)
                or set(cached) != set(dataset)
                or any(len(cached[name]) != len(dataset[name]) for name in dataset)
                or any(not {"input_ids", "labels", "example_id"} <= set(cached[name].column_names)
                       for name in cached)):
            raise ValueError(f"tokenization cache is incomplete: {candidate}")
        return cached
    if path.is_dir():
        if path.is_symlink():
            raise ValueError(f"refusing symlink tokenization cache: {path}")
        try:
            return valid_cache(path)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            quarantine = path.with_name(f"{path.name}.invalid-{time.time_ns()}")
            os.replace(path, quarantine)
            LOGGER.warning("invalid tokenization cache moved to %s: %s", quarantine, error)
    tokenized = tokenize_task(dataset, tokenizer, task, max_length)
    if not isinstance(tokenized, DatasetDict):
        raise TypeError("tokenize_task must return a DatasetDict for caching")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        tokenized.save_to_disk(str(temporary))
        valid_cache(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return tokenized


def split_manifest(dataset) -> dict[str, Any]:
    splits = {}
    for name, split in dataset.items():
        records = [
            {key: row[key] for key in row if key in {"idx", "label", "sentence", "sentence1", "sentence2"}}
            for row in split
        ]
        encoded = json.dumps(records, sort_keys=True, ensure_ascii=False).encode()
        splits[name] = {"count": len(records), "sha256": hashlib.sha256(encoded).hexdigest()}
    return {"splits": splits}


def audit_split_overlap(dataset, task: str) -> dict[str, Any]:
    """Report exact normalized text overlap between dataset splits."""
    columns = TASK_COLUMNS.get(task)
    if columns is None:
        raise ValueError(f"unsupported task: {task}")
    split_keys: dict[str, set[str]] = {}
    for name, split in dataset.items():
        keys = set()
        for row in split:
            values = [str(row[column]).strip().casefold() for column in columns if column in row]
            if not values and "sentence" in row:
                values = [str(row["sentence"]).strip().casefold()]
            keys.add("\x1f".join(values))
        split_keys[name] = keys
    overlaps = {}
    names = sorted(split_keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}:{right}"] = len(split_keys[left] & split_keys[right])
    return {"split_sizes": {name: len(keys) for name, keys in split_keys.items()}, "overlap_counts": overlaps}
