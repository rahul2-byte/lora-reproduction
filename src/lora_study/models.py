"""Model loading and inspection boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class LinearModuleInfo:
    path: str
    input_features: int
    output_features: int
    module_type: str


def inspect_linear_modules(model: nn.Module) -> tuple[LinearModuleInfo, ...]:
    return tuple(
        LinearModuleInfo(name, module.in_features, module.out_features, type(module).__name__)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )


def load_roberta_classifier(model_id: str, revision: str, num_labels: int, device: str):
    try:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for pretrained model loading") from error
    if model_id == "tiny-roberta-smoke":
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b")
        config = AutoConfig.for_model("roberta", vocab_size=tokenizer.vocab_size,
                                      hidden_size=64, intermediate_size=128,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_config(config).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, num_labels=num_labels
        ).to(device)
    return tokenizer, model
