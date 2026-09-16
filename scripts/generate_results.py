#!/usr/bin/env python
"""Generate compact result tables and figures from selected run roots."""

from __future__ import annotations

import argparse
from pathlib import Path

from lora_study.report import collect_summaries, write_csv, write_efficiency_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", help="run directories to include")
    parser.add_argument("--output", default="results")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()

    rows = []
    seen = set()
    for root in args.run_roots:
        statuses = {"completed"} if args.completed_only else None
        for row in collect_summaries(root, statuses=statuses):
            if row["run_id"] in seen:
                raise ValueError(f"duplicate run inclusion: {row['run_id']}")
            seen.add(row["run_id"])
            rows.append(row)
    output = Path(args.output)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "metrics.csv")
    write_efficiency_figure(rows, output / "figures" / "efficiency.png")
    print(f"wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
