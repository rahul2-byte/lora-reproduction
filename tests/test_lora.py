import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lora_study.lora import LoRALinear


def make_base() -> nn.Linear:
    torch.manual_seed(3)
    return nn.Linear(5, 7, bias=True)


def test_shapes_counts_and_base_are_frozen():
    layer = LoRALinear(make_base(), rank=2, alpha=4.0, dropout=0.0)

    assert layer.lora_a.shape == (2, 5)
    assert layer.lora_b.shape == (7, 2)
    assert layer.adapter_parameter_count == 2 * (5 + 7)
    assert not layer.base.weight.requires_grad
    assert not layer.base.bias.requires_grad
    assert layer.lora_a.requires_grad
    assert layer.lora_b.requires_grad


def test_zero_initialized_update_matches_base_and_only_b_gets_first_gradient():
    base = make_base()
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(3, 5)

    assert torch.allclose(layer(x), base(x))
    assert torch.count_nonzero(layer.lora_a).item() > 0
    assert torch.count_nonzero(layer.lora_b).item() == 0

    layer(x).sum().backward()

    assert torch.count_nonzero(layer.lora_b.grad).item() > 0
    assert torch.count_nonzero(layer.lora_a.grad).item() == 0


def test_both_factors_receive_gradients_after_b_has_updated():
    layer = LoRALinear(make_base(), rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(3, 5)
    layer(x).sum().backward()
    with torch.no_grad():
        layer.lora_b.add_(0.1)
    layer.zero_grad()
    layer(x).sum().backward()

    assert torch.count_nonzero(layer.lora_a.grad).item() > 0
    assert torch.count_nonzero(layer.lora_b.grad).item() > 0


def test_factorized_forward_matches_manual_dense_update():
    layer = LoRALinear(make_base(), rank=2, alpha=3.0, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.normal_()
        layer.lora_b.normal_()
    x = torch.randn(4, 6, 5)

    expected_weight = layer.base.weight + layer.scaling * layer.lora_b @ layer.lora_a
    expected = nn.functional.linear(x, expected_weight, layer.base.bias)

    assert torch.allclose(layer(x), expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rank": 0, "alpha": 1.0, "dropout": 0.0}, "rank"),
        ({"rank": 2, "alpha": float("nan"), "dropout": 0.0}, "alpha"),
        ({"rank": 2, "alpha": 1.0, "dropout": 1.0}, "dropout"),
    ],
)
def test_rejects_invalid_adapter_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LoRALinear(make_base(), **kwargs)


def test_state_dict_round_trip_preserves_outputs():
    source = LoRALinear(make_base(), rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        source.lora_a.normal_()
        source.lora_b.normal_()
    target = LoRALinear(make_base(), rank=2, alpha=4.0, dropout=0.0)
    target.load_state_dict(source.state_dict())
    x = torch.randn(3, 5)

    assert torch.equal(source(x), target(x))


def test_exported_merged_linear_matches_unmerged_output():
    layer = LoRALinear(make_base(), rank=2, alpha=4.0, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.normal_()
        layer.lora_b.normal_()
    merged = layer.export_merged()
    x = torch.randn(3, 5)

    assert torch.allclose(layer(x), merged(x), atol=1e-6, rtol=1e-6)


def test_zero_both_initialization_is_explicit_diagnostic():
    layer = LoRALinear(make_base(), rank=2, alpha=4.0, initialization="zero_both")
    assert torch.count_nonzero(layer.lora_a) == 0
    assert torch.count_nonzero(layer.lora_b) == 0
