"""Strict configuration loading for experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tomllib


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


@dataclass(frozen=True)
class ExperimentConfig:
    method: str = "head_only"
    seed: int = 1
    track: str = "strict"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "FacebookAI/roberta-base"
    revision: str = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


@dataclass(frozen=True)
class DatasetConfig:
    task: str = "mrpc"
    dataset_id: str = "nyu-mll/glue"
    revision: str = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
    max_length: int = 128
    evaluation_protocol: str = "heldout_official_validation"
    token_cache_dir: str = "data/cache"


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("query", "value")
    initialization: str = "paper"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


@dataclass(frozen=True)
class TrainingConfig:
    microbatch: int = 2
    gradient_accumulation: int = 1
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "fp32"
    scheduler: str = "none"
    num_workers: int = 0
    pin_memory: bool = False
    progress_interval_seconds: float = 1.0
    max_train_examples: int | None = None


@dataclass(frozen=True)
class StudyConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    dataset: DatasetConfig
    lora: LoRAConfig
    training: TrainingConfig

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        if self.training.max_train_examples is None:
            values["training"].pop("max_train_examples")
        values["lora"]["target_modules"] = list(self.lora.target_modules)
        values["lora"]["scaling"] = self.lora.scaling
        return values

    @property
    def resolved_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> StudyConfig:
    """Load and validate one TOML experiment configuration."""
    with Path(path).open("rb") as file:
        raw = tomllib.load(file)

    _reject_unknown(raw, {"experiment", "model", "dataset", "lora", "training"}, "root")
    experiment_raw = raw.get("experiment", {})
    model_raw = raw.get("model", {})
    dataset_raw = raw.get("dataset", {})
    lora_raw = raw.get("lora", {})
    training_raw = raw.get("training", {})
    _reject_unknown(experiment_raw, {"method", "seed", "track"}, "experiment")
    _reject_unknown(model_raw, {"model_id", "revision"}, "model")
    _reject_unknown(
        dataset_raw,
        {"task", "dataset_id", "revision", "max_length", "evaluation_protocol", "token_cache_dir"},
        "dataset",
    )
    _reject_unknown(
        lora_raw,
        {"rank", "alpha", "dropout", "target_modules", "initialization"},
        "lora",
    )
    _reject_unknown(
        training_raw,
        {
            "microbatch",
            "gradient_accumulation",
            "epochs",
            "learning_rate",
            "weight_decay",
            "max_grad_norm",
            "precision",
            "scheduler",
            "num_workers",
            "pin_memory",
            "progress_interval_seconds",
            "max_train_examples",
        },
        "training",
    )

    experiment = ExperimentConfig(**experiment_raw)
    model = ModelConfig(**model_raw)
    dataset = DatasetConfig(**dataset_raw)
    if "target_modules" in lora_raw:
        lora_raw = {**lora_raw, "target_modules": tuple(lora_raw["target_modules"])}
    lora = LoRAConfig(**lora_raw)
    training = TrainingConfig(**training_raw)
    _validate(
        experiment,
        model,
        dataset,
        lora,
        training,
        has_lora_section="lora" in raw,
    )
    return StudyConfig(experiment, model, dataset, lora, training)


def _reject_unknown(values: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {section}: {', '.join(unknown)}")


def _validate(
    experiment: ExperimentConfig,
    model: ModelConfig,
    dataset: DatasetConfig,
    lora: LoRAConfig,
    training: TrainingConfig,
    *,
    has_lora_section: bool,
) -> None:
    methods = {"frozen", "head_only", "full_finetune", "custom_lora", "reference_lora"}
    if experiment.method not in methods:
        raise ConfigError(f"invalid method: {experiment.method}")
    if experiment.track not in {"strict", "practical"}:
        raise ConfigError("track must be strict or practical")
    if not isinstance(experiment.seed, int) or isinstance(experiment.seed, bool):
        raise ConfigError("seed must be an integer")
    if not model.model_id or not model.revision:
        raise ConfigError("model_id and revision must be non-empty")
    if dataset.task not in {"mrpc", "sst2"}:
        raise ConfigError("dataset task must be mrpc or sst2")
    if not dataset.dataset_id or not dataset.revision:
        raise ConfigError("dataset_id and dataset revision must be non-empty")
    if not dataset.token_cache_dir:
        raise ConfigError("token_cache_dir must be non-empty")
    if dataset.max_length <= 0:
        raise ConfigError("max_length must be positive")
    if experiment.method not in {"custom_lora", "reference_lora"} and has_lora_section:
        raise ConfigError("LoRA settings require a LoRA method")
    if not isinstance(lora.rank, int) or isinstance(lora.rank, bool) or lora.rank <= 0:
        raise ConfigError("rank must be a positive integer")
    if not isinstance(lora.alpha, (int, float)) or isinstance(lora.alpha, bool):
        raise ConfigError("alpha must be finite")
    if not math.isfinite(lora.alpha):
        raise ConfigError("alpha must be finite")
    if not isinstance(lora.dropout, (int, float)) or isinstance(lora.dropout, bool):
        raise ConfigError("dropout must be in [0, 1)")
    if not 0 <= lora.dropout < 1:
        raise ConfigError("dropout must be in [0, 1)")
    if not lora.target_modules or any(not module for module in lora.target_modules):
        raise ConfigError("target_modules must contain non-empty selectors")
    if lora.initialization not in {"paper", "zero_both"}:
        raise ConfigError("initialization must be paper or zero_both")
    if training.microbatch <= 0 or training.gradient_accumulation <= 0:
        raise ConfigError("microbatch and gradient_accumulation must be positive")
    if training.epochs <= 0 or training.learning_rate <= 0 or training.weight_decay < 0:
        raise ConfigError("epochs and learning rate must be positive; weight decay cannot be negative")
    if training.max_grad_norm <= 0:
        raise ConfigError("max_grad_norm must be positive")
    if training.precision not in {"fp32", "bf16", "fp16"}:
        raise ConfigError("precision must be fp32, bf16, or fp16")
    if training.scheduler not in {"none", "linear"}:
        raise ConfigError("scheduler must be none or linear")
    if training.num_workers < 0:
        raise ConfigError("num_workers cannot be negative")
    if not isinstance(training.pin_memory, bool):
        raise ConfigError("pin_memory must be boolean")
    if training.progress_interval_seconds <= 0:
        raise ConfigError("progress_interval_seconds must be positive")
    if training.max_train_examples is not None and training.max_train_examples <= 0:
        raise ConfigError("max_train_examples must be positive")
