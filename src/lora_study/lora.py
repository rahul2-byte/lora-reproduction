"""Small, explicit LoRA implementation for linear layers."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        initialization: str = "paper",
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be torch.nn.Linear")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if rank > min(base.in_features, base.out_features):
            raise ValueError("rank must not exceed the linear layer dimensions")
        if isinstance(alpha, bool) or not isinstance(alpha, Real) or not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, Real)
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0, 1)")
        if initialization not in {"paper", "zero_both"}:
            raise ValueError("initialization must be paper or zero_both")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        if initialization == "paper":
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b)
        else:
            nn.init.zeros_(self.lora_a)
            nn.init.zeros_(self.lora_b)

    @property
    def adapter_parameter_count(self) -> int:
        return self.lora_a.numel() + self.lora_b.numel()

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        adapter_input = F.dropout(inputs, p=self.dropout, training=self.training)
        update = F.linear(F.linear(adapter_input, self.lora_a), self.lora_b)
        return base_output + self.scaling * update

    def export_merged(self) -> nn.Linear:
        """Return a standard linear layer containing the adapter update."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.scaling * self.lora_b @ self.lora_a)
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged
