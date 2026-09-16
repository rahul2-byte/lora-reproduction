"""Exact, fail-closed LoRA injection for linear modules."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from .lora import LoRALinear


class InjectionError(ValueError):
    """Raised when an injection plan cannot be applied safely."""


@dataclass(frozen=True)
class InjectionManifest:
    paths: tuple[str, ...]
    rank: int
    alpha: float
    scaling: float
    dropout: float
    adapter_parameter_count: int


def resolve_target_paths(model: nn.Module, selectors: tuple[str, ...]) -> tuple[str, ...]:
    """Resolve exact paths or unique leaf-name selectors."""
    modules = dict(model.named_modules())
    resolved: list[str] = []
    for selector in selectors:
        exact = selector in modules
        matches = [
            name for name, module in modules.items()
            if isinstance(module, nn.Linear)
            and (exact and name == selector or not exact and name.rsplit(".", 1)[-1] == selector)
        ]
        if not matches:
            raise InjectionError(f"selector {selector!r} matched no modules")
        resolved.extend(matches)
    if len(resolved) != len(set(resolved)):
        raise InjectionError("selectors resolve to duplicate targets")
    return tuple(resolved)


def inject_lora(
    model: nn.Module,
    target_paths: list[str] | tuple[str, ...],
    *,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    initialization: str = "paper",
    trainable_prefixes: tuple[str, ...] = (),
) -> InjectionManifest:
    """Replace exact linear module paths after validating the whole plan."""
    paths = tuple(target_paths)
    if len(paths) != len(set(paths)):
        raise InjectionError("duplicate target paths")
    if not paths:
        raise InjectionError("injection requires at least one target path")

    modules: list[tuple[str, nn.Linear]] = []
    for path in paths:
        try:
            module = model.get_submodule(path)
        except AttributeError as error:
            raise InjectionError(f"target {path!r} matched no modules") from error
        if isinstance(module, LoRALinear):
            raise InjectionError(f"target {path!r} is already adapted")
        if not isinstance(module, nn.Linear):
            raise InjectionError(f"target {path!r} has unsupported module type")
        modules.append((path, module))

    parameter_names = tuple(name for name, _ in model.named_parameters())
    for prefix in trainable_prefixes:
        if not any(name == prefix or name.startswith(prefix + ".") for name in parameter_names):
            raise InjectionError(f"trainable prefix {prefix!r} matched no parameters")

    replacements = [
        (path, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, initialization=initialization))
        for path, module in modules
    ]
    for path, replacement in replacements:
        parent_path, _, leaf = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        parent._modules[leaf] = replacement

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if ".lora_a" in name or ".lora_b" in name:
            parameter.requires_grad_(True)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in trainable_prefixes):
            parameter.requires_grad_(True)

    adapter_count = sum(
        replacement.adapter_parameter_count for _, replacement in replacements
    )
    first = replacements[0][1]
    return InjectionManifest(
        paths=paths,
        rank=first.rank,
        alpha=first.alpha,
        scaling=first.scaling,
        dropout=first.dropout,
        adapter_parameter_count=adapter_count,
    )


def merge_lora_modules(model: nn.Module) -> int:
    """Replace every LoRALinear with its merged standard Linear layer."""
    replacements: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                if child.training:
                    raise InjectionError("model must be in evaluation mode before merging")
                replacements.append((parent, name, child.export_merged()))
    for parent, name, replacement in replacements:
        parent._modules[name] = replacement
    return len(replacements)
