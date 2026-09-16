#!/usr/bin/env python
"""Fail if required project evidence or generated artifacts are missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_DOCS = (
    "README.md", "docs/paper_notes.md", "docs/architecture.md",
    "docs/reproduction_scope.md", "docs/experimental_methodology.md",
    "docs/results.md", "docs/discrepancies.md", "docs/limitations.md",
    "docs/interview_notes.md", "docs/claim_ledger.md",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--fail-on-duplicates", action="store_true")
    parser.add_argument("--completed-only", action="store_true")
    args = parser.parse_args()
    missing = [path for path in REQUIRED_DOCS if not Path(path).is_file()]
    if missing:
        raise SystemExit(f"missing required documentation: {missing}")
    rows = []
    duplicates = []
    seen = set()
    invalid = []
    for path in sorted(Path(args.runs).glob("**/summary.json")):
        row = json.loads(path.read_text())
        if args.completed_only and row.get("status") != "completed":
            continue
        run_id = row.get("run_id", path.parent.name)
        if run_id in seen:
            duplicates.append(run_id)
        seen.add(run_id)
        if row.get("status") not in {"planned", "running", "completed", "failed", "cancelled"}:
            invalid.append(run_id)
        rows.append(row)
    failed = [row.get("run_id") for row in rows if row.get("status") == "failed"]
    print(
        f"audited docs={len(REQUIRED_DOCS)} summaries={len(rows)} "
        f"unique_runs={len(seen)} failed_runs={len(failed)} "
        f"duplicates={len(duplicates)} invalid={len(invalid)}"
    )
    if duplicates:
        print("duplicate_run_ids:", ", ".join(sorted(set(duplicates))))
    if args.fail_on_duplicates and duplicates:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
