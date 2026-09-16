"""Artifact-derived result aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .profile import artifact_parameter_counts


REQUIRED_SUMMARY_KEYS = {"run_id", "status", "method", "seed", "parameters"}
VALID_STATUSES = {"planned", "running", "completed", "failed", "cancelled"}


def collect_summaries(runs_root: str | Path, statuses: set[str] | None = None) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for path in sorted(Path(runs_root).glob("**/summary.json")):
        with path.open() as file:
            row = json.load(file)
        if statuses is not None and row.get("status") not in statuses:
            continue
        missing = sorted(REQUIRED_SUMMARY_KEYS - set(row))
        if missing:
            raise ValueError(f"summary missing required keys: {missing}")
        if row["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid run status: {row['status']}")
        row["run_id"] = row.get("run_id", path.parent.name)
        if row["run_id"] in seen:
            raise ValueError(f"duplicate run inclusion: {row['run_id']}")
        if row["status"] == "completed" and row.get("method") == "custom_lora":
            row["parameters"] = artifact_parameter_counts(row, path.parent / "task.pt")
        row["study_group"] = "ablation" if path.parent.parent.name == "ablation" else (
            "reference" if row["method"] == "reference_lora" else "core"
        )
        seen.add(row["run_id"])
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: str | Path) -> None:
    if not rows:
        raise ValueError("no run summaries to aggregate")
    keys = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_efficiency_figure(rows: Iterable[dict], path: str | Path) -> None:
    """Plot the core methods by task, keeping seed observations visible."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required for figures") from error
    completed = [row for row in rows if row["status"] == "completed"
                 and row.get("study_group", "core") == "core"]
    if not completed:
        raise ValueError("no completed runs to plot")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharex=True)
    methods = {
        "head_only": ("Head only", "#6b7280"),
        "custom_lora": ("Custom LoRA", "#0f766e"),
        "full_finetune": ("Full fine-tuning", "#c2410c"),
    }
    for axis, task in zip(axes, ("mrpc", "sst2")):
        for method, (label, color) in methods.items():
            subset = [row for row in completed if row.get("task") == task
                      and row["method"] == method]
            axis.scatter([row["parameters"]["trainable"] for row in subset],
                         [row["training_seconds"] for row in subset],
                         label=label, color=color, alpha=0.7, s=55)
        axis.set_xscale("log")
        axis.set_title(task.upper())
        axis.set_xlabel("Trainable parameters (log scale)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Per-run wall time (seconds)")
    axes[1].legend(loc="upper left", frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
