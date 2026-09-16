import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lora_study.injection import InjectionError, inject_lora, merge_lora_modules, resolve_target_paths
from lora_study.lora import LoRALinear


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.ModuleDict(
            {
                "query": nn.Linear(5, 5),
                "value": nn.Linear(5, 5),
                "key": nn.LayerNorm(5),
            }
        )
        self.classifier = nn.Linear(5, 2)


def test_injects_exact_targets_and_enables_only_adapters_and_head():
    model = TinyModel()
    query_weight = model.attention["query"].weight.detach().clone()
    manifest = inject_lora(
        model,
        ["attention.query", "attention.value"],
        rank=2,
        alpha=4.0,
        dropout=0.0,
        trainable_prefixes=("classifier",),
    )

    assert isinstance(model.attention["query"], LoRALinear)
    assert isinstance(model.attention["value"], LoRALinear)
    assert manifest.paths == ("attention.query", "attention.value")
    assert torch.equal(model.attention["query"].base.weight, query_weight)
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(("attention.query.lora_", "attention.value.lora_", "classifier"))
    )
    assert not model.attention["query"].base.weight.requires_grad
    assert not model.attention["value"].base.weight.requires_grad


def test_missing_target_fails_before_any_mutation():
    model = TinyModel()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    with pytest.raises(InjectionError, match="matched no modules"):
        inject_lora(model, ["attention.missing"], rank=2, alpha=4.0)

    assert isinstance(model.attention["query"], nn.Linear)
    assert all(torch.equal(parameter, before[name]) for name, parameter in model.named_parameters())


def test_unsupported_type_fails_before_any_mutation():
    model = TinyModel()

    with pytest.raises(InjectionError, match="unsupported module type"):
        inject_lora(model, ["attention.key"], rank=2, alpha=4.0)

    assert isinstance(model.attention["query"], nn.Linear)


def test_duplicate_and_repeated_injection_are_rejected():
    model = TinyModel()

    with pytest.raises(InjectionError, match="duplicate"):
        inject_lora(model, ["attention.query", "attention.query"], rank=2, alpha=4.0)

    inject_lora(model, ["attention.query"], rank=2, alpha=4.0)
    with pytest.raises(InjectionError, match="already adapted"):
        inject_lora(model, ["attention.query"], rank=2, alpha=4.0)


def test_leaf_selectors_resolve_only_linear_modules():
    model = TinyModel()

    assert resolve_target_paths(model, ("query", "value")) == (
        "attention.query",
        "attention.value",
    )
    with pytest.raises(InjectionError, match="matched no modules"):
        resolve_target_paths(model, ("key",))


def test_merge_requires_eval_and_replaces_adapter():
    model = TinyModel()
    inject_lora(model, ["attention.query"], rank=2, alpha=4.0)
    with pytest.raises(InjectionError, match="evaluation"):
        merge_lora_modules(model)
    model.eval()
    assert merge_lora_modules(model) == 1
    assert isinstance(model.attention.query, nn.Linear)
