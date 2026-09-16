"""Bounded, restartable single-GPU experiment matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .checkpoints import atomic_json, require_disk_space
from .config import StudyConfig, load_config
from .data import load_glue_task, tokenize_task_cached
from .runner import run, run_id_for, validate_completed

LOGGER = logging.getLogger(__name__)
ROOT = Path("runs/reliable")
GENERATED_NAMES = {
    "artifacts.json", "best.pt", "latest.pt", "config.resolved.json",
    "data_manifest.json", "environment.json", "validation.jsonl",
    "summary.json", "task.pt",
    ".run.lock",
}
METHODS = {
    "head_only": "configs/mrpc_head_only.toml",
    "full_finetune": "configs/mrpc_full_ft.toml",
    "custom_lora": "configs/mrpc_custom_lora.toml",
}


def jobs(smoke: bool = False) -> list[tuple[StudyConfig, Path]]:
    tasks = ("mrpc",) if smoke else ("mrpc", "sst2")
    methods = ("custom_lora",) if smoke else tuple(METHODS)
    seeds = (1,) if smoke else (1, 2, 3)
    result = []
    for task in tasks:
        for method in methods:
            for seed in seeds:
                base = load_config(METHODS[method])
                training = replace(
                    base.training, precision="fp32" if smoke else "bf16", microbatch=8,
                    gradient_accumulation=2, num_workers=0 if smoke else 2,
                    pin_memory=not smoke,
                    max_train_examples=128 if smoke else None,
                )
                config = replace(
                    base, model=replace(base.model, model_id="tiny-roberta-smoke", revision="synthetic-v1") if smoke else base.model,
                    dataset=replace(base.dataset, task=task),
                    experiment=replace(base.experiment, seed=seed, track="practical"),
                    training=training,
                )
                result.append((config, Path(task) / method / run_id_for(config)))
    if not smoke:
        core = next(config for config, _ in result
                    if config.dataset.task == "mrpc" and config.experiment.method == "custom_lora"
                    and config.experiment.seed == 1)
        reference = replace(core, experiment=replace(core.experiment, method="reference_lora"))
        result.append((reference, Path("mrpc") / "reference_lora" / run_id_for(reference)))
        variants = [replace(core, lora=replace(core.lora, rank=rank, alpha=2.0 * rank))
                    for rank in (1, 2, 4, 8, 16)]
        variants += [replace(core, lora=replace(core.lora, rank=4, alpha=alpha))
                     for alpha in (2.0, 8.0, 16.0)]
        variants += [replace(core, lora=replace(core.lora, target_modules=targets))
                     for targets in (("query",), ("value",), ("query", "value"))]
        seen = {config.resolved_hash for config, _ in result}
        for config in variants:
            if config.resolved_hash not in seen:
                result.append((config, Path("mrpc") / "ablation" / run_id_for(config)))
                seen.add(config.resolved_hash)
    return result


def _inventory(root: Path) -> list[dict]:
    if root.is_symlink():
        raise ValueError(f"refusing symlink output root: {root}")
    if not root.exists():
        return []
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refusing symlink inside output root: {path}")
        if path.is_file():
            if path.name not in GENERATED_NAMES and not (path.name.startswith(".") and path.name.endswith(".tmp")):
                raise ValueError(f"refusing unknown file in cleanup scope: {path}")
            if path.name == ".run.lock":
                with path.open("r") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise ValueError(f"refusing cleanup while run is active: {path}") from error
            relative = path.relative_to(root).as_posix()
            records.append({"path": relative, "size": path.stat().st_size,
                            "mtime_ns": path.stat().st_mtime_ns})
    return records


def clean(root: Path, dry_run: bool) -> None:
    if root.resolve() != ROOT.resolve() and root.resolve() != Path("runs/reliable-smoke").resolve():
        raise ValueError("clean is restricted to runs/reliable or runs/reliable-smoke")
    inventory = _inventory(root)
    preview = Path("runs/.reliable-clean-preview.json" if root == ROOT else "runs/.reliable-smoke-clean-preview.json")
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    if dry_run:
        atomic_json(preview, {"root": str(root.resolve()), "digest": digest, "files": inventory})
        print(f"dry run: {len(inventory)} generated files under {root}; preview saved to {preview}")
        for item in inventory:
            print(f"  {item['path']} ({item['size']} bytes)")
        return
    if not preview.is_file():
        raise ValueError("run 'pipeline clean --dry-run' first")
    expected = json.loads(preview.read_text())
    if expected.get("root") != str(root.resolve()) or expected.get("digest") != digest:
        raise ValueError("cleanup preview is stale; run --dry-run again")
    if root.exists():
        shutil.rmtree(root)
    preview.unlink()
    print(f"removed only generated output root {root}")


def _prepare(config: StudyConfig) -> None:
    from transformers import AutoTokenizer

    if config.model.model_id == "tiny-roberta-smoke":
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, revision=config.model.revision)
    dataset = load_glue_task(config.dataset.task, config.dataset.revision, config.dataset.dataset_id)
    tokenize_task_cached(dataset, tokenizer, config.dataset.task, config.dataset.max_length,
                         config.dataset.token_cache_dir, config.dataset.revision)
    LOGGER.info("prepared task=%s", config.dataset.task)


def status(root: Path, smoke: bool) -> None:
    counts = {"completed": 0, "partial": 0, "failed": 0, "pending": 0, "invalid": 0}
    for config, relative in jobs(smoke):
        path = root / relative
        if validate_completed(path, config) is not None:
            state = "completed"
        elif (path / "checkpoints" / "latest.pt").is_file():
            state = "partial"
        elif (path / "summary.json").is_file():
            try:
                recorded = json.loads((path / "summary.json").read_text()).get("status")
                state = "failed" if recorded in {"failed", "cancelled"} else "invalid"
            except (OSError, ValueError):
                state = "invalid"
        elif path.exists():
            state = "invalid"
        else:
            state = "pending"
        counts[state] += 1
        print(f"{state:9} {relative}")
    print("counts:", counts)


def execute(root: Path, smoke: bool, max_hours: float) -> None:
    if max_hours <= 0:
        raise ValueError("max-hours must be positive")
    matrix = jobs(smoke)
    pending = [(config, relative) for config, relative in matrix
               if validate_completed(root / relative, config) is None]
    if not pending:
        LOGGER.info("all %d runs have verified artifacts", len(matrix))
        return
    # Only the two independent CPU preparations run concurrently. GPU jobs remain serial.
    with ThreadPoolExecutor(max_workers=min(2, len({c.dataset.task for c, _ in pending}))) as pool:
        futures = [pool.submit(_prepare, next(c for c, _ in pending if c.dataset.task == task))
                   for task in dict.fromkeys(c.dataset.task for c, _ in pending)]
        for future in futures:
            future.result()
    started = time.monotonic()
    durations: list[float] = []
    for index, (config, relative) in enumerate(matrix, 1):
        path = root / relative
        if validate_completed(path, config) is not None:
            LOGGER.info("[%d/%d] reused %s", index, len(matrix), relative)
            continue
        elapsed = time.monotonic() - started
        if elapsed >= max_hours * 3600:
            LOGGER.warning("time budget reached; restart later with pipeline run --resume")
            break
        # Leave capacity for the next complete checkpoint and final task artifact.
        require_disk_space(path / "checkpoints" / "latest.pt",
                           5_000_000_000 if config.experiment.method == "full_finetune" else 2_000_000_000)
        eta = (sum(durations) / len(durations) * (len(matrix) - index + 1)) if durations else None
        LOGGER.info("[%d/%d] starting %s elapsed=%.1fm ETA=%s", index, len(matrix), relative,
                    elapsed / 60, f"{eta / 60:.1f}m" if eta is not None else "unknown")
        run_started = time.monotonic()
        run(config, root / relative.parent)
        durations.append(time.monotonic() - run_started)
        LOGGER.info("[%d/%d] completed %s duration=%.1fm", index, len(matrix), relative, durations[-1] / 60)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    clean_parser = sub.add_parser("clean")
    clean_parser.add_argument("--dry-run", action="store_true")
    clean_parser.add_argument("--smoke", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--resume", action="store_true", help="resume is automatic; kept for clarity")
    run_parser.add_argument("--smoke", action="store_true")
    run_parser.add_argument("--max-hours", type=float, default=6.0)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = Path("runs/reliable-smoke") if getattr(args, "smoke", False) else ROOT
    if args.command == "status":
        status(root, args.smoke)
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        with (root.parent / f".{root.name}.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another pipeline process is active for {root}") from error
            if args.command == "clean":
                clean(root, args.dry_run)
            else:
                execute(root, args.smoke, args.max_hours)


if __name__ == "__main__":
    main()
