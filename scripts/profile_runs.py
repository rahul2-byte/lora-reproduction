#!/usr/bin/env python
"""Create a resource table from completed run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lora_study.profile import artifact_parameter_counts, file_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", help="directories containing run artifacts")
    parser.add_argument("--output", default="results/profile.csv")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()

    rows = []
    seen = set()
    for root in args.run_roots:
        for summary_path in sorted(Path(root).glob("**/summary.json")):
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "completed":
                if args.completed_only:
                    continue
                print(f"skipping incomplete run {summary.get('run_id', summary_path.parent.name)}")
                continue
            run_id = summary["run_id"]
            if run_id in seen:
                raise ValueError(f"duplicate run inclusion: {run_id}")
            seen.add(run_id)
            run_dir = summary_path.parent
            parameters = artifact_parameter_counts(summary, run_dir / "task.pt")
            rows.append({
                "run_id": run_id,
                "status": summary["status"],
                "method": summary["method"],
                "seed": summary["seed"],
                "trainable_parameters": parameters["trainable"],
                "trainable_fraction": parameters["trainable_fraction"],
                "training_seconds": summary.get("training_seconds"),
                "peak_allocated_gpu_memory": summary.get("peak_allocated_gpu_memory"),
                "peak_reserved_gpu_memory": summary.get("peak_reserved_gpu_memory"),
                "task_artifact_bytes": file_size(run_dir / "task.pt"),
                "resume_checkpoint_bytes": file_size(run_dir / "checkpoints" / "best.pt"),
                "prediction_file_bytes": file_size(run_dir / "predictions" / "validation.jsonl"),
                "merged_unmerged_max_abs_error": summary.get("merged_unmerged_max_abs_error"),
            })
    if not rows:
        raise ValueError("no summaries found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} profiles to {output}")


if __name__ == "__main__":
    main()
