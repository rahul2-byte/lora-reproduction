import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lora_study.train import train_epoch


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input_ids, labels):
        prediction = self.linear(input_ids)
        return type("Output", (), {"loss": ((prediction - labels) ** 2).mean()})


def test_accumulation_steps_on_final_partial_window():
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [
        {"input_ids": torch.tensor([[1.0]]), "labels": torch.tensor([[2.0]])},
        {"input_ids": torch.tensor([[2.0]]), "labels": torch.tensor([[4.0]])},
        {"input_ids": torch.tensor([[3.0]]), "labels": torch.tensor([[6.0]])},
    ]

    summary = train_epoch(model, batches, optimizer, torch.device("cpu"), accumulation=2)

    assert summary.examples == 3
    assert summary.optimizer_updates == 2
    assert torch.isfinite(model.linear.weight).item()


def test_nonfinite_loss_fails_closed():
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = {"input_ids": torch.tensor([[1.0]]), "labels": torch.tensor([[float("nan")]])}

    with pytest.raises(FloatingPointError, match="nonfinite"):
        train_epoch(model, [batch], optimizer, torch.device("cpu"))


def test_mixed_precision_requires_cuda():
    model = RegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="requires CUDA"):
        train_epoch(model, [], optimizer, torch.device("cpu"), precision="bf16")
