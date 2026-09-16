"""Optional trusted-framework integration kept outside the custom layer."""

from __future__ import annotations


def require_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("PEFT is required for reference comparisons") from error
    return LoraConfig, TaskType, get_peft_model


def apply_reference_lora(model, *, rank: int, alpha: float, dropout: float):
    """Apply PEFT LoRA to the verified RoBERTa query/value modules."""
    LoraConfig, TaskType, get_peft_model = require_peft()
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )
    return get_peft_model(model, config)
