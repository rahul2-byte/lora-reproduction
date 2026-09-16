#!/usr/bin/env python
"""Probe safe microbatches with a short, isolated CUDA workload."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from lora_study.config import load_config
from lora_study.data import load_glue_task, tokenize_task_cached
from lora_study.injection import inject_lora, resolve_target_paths
from lora_study.models import load_roberta_classifier
from lora_study.runner import _configure_trainable_parameters
from lora_study.train import train_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mrpc", "sst2"], required=True)
    parser.add_argument("--methods", nargs="+", choices=["head_only", "full_finetune", "custom_lora"], required=True)
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--output", default="results/batch_probe.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for batch-size probing")
    candidates = (2, 4, 8, 16, 32)
    results = []
    base_configs = {
        "head_only": "configs/mrpc_head_only.toml",
        "full_finetune": "configs/mrpc_full_ft.toml",
        "custom_lora": "configs/mrpc_custom_lora.toml",
    }
    dataset = load_glue_task(args.task, revision=load_config(base_configs[args.methods[0]]).dataset.revision)
    for method in args.methods:
        base = load_config(base_configs[method])
        base = replace(base, dataset=replace(base.dataset, task=args.task))
        tokenizer, _ = load_roberta_classifier(base.model.model_id, base.model.revision, num_labels=2, device="cpu")
        tokenized = tokenize_task_cached(
            dataset, tokenizer, args.task, base.dataset.max_length,
            base.dataset.token_cache_dir, base.dataset.revision,
        )
        collator = DataCollatorWithPadding(tokenizer)
        train_split = tokenized["train"].select(range(min(len(tokenized["train"]), 256)))
        for microbatch in candidates:
            model = None
            try:
                _, model = load_roberta_classifier(base.model.model_id, base.model.revision, num_labels=2, device="cuda")
                _configure_trainable_parameters(model, base)
                if method == "custom_lora":
                    inject_lora(model, resolve_target_paths(model, base.lora.target_modules), rank=base.lora.rank, alpha=base.lora.alpha, dropout=base.lora.dropout, trainable_prefixes=("classifier",))
                loader = DataLoader(train_split, batch_size=microbatch, shuffle=False, collate_fn=collator, pin_memory=True)
                batches = list(islice(loader, args.updates))
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base.training.learning_rate)
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                train_epoch(model, batches, optimizer, torch.device("cuda"), precision=args.precision, progress=False, non_blocking=True)
                torch.cuda.synchronize()
                results.append({"task": args.task, "method": method, "microbatch": microbatch, "status": "completed", "seconds": time.perf_counter() - started, "peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved()})
            except RuntimeError as error:
                if "out of memory" not in str(error).lower():
                    raise
                results.append({"task": args.task, "method": method, "microbatch": microbatch, "status": "oom", "error": str(error)})
            finally:
                del model
                torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"wrote {len(results)} batch probes to {output}")


if __name__ == "__main__":
    main()
