import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lora_study.checkpoints import (
    capture_rng_state,
    CheckpointError,
    load_task_artifact,
    load_training_checkpoint,
    save_task_artifact,
    save_training_checkpoint,
)
from lora_study.injection import inject_lora


METADATA = {
    "base_model_id": "tiny",
    "base_model_revision": "abc",
    "tokenizer_revision": "def",
    "label_mapping": {"negative": 0, "positive": 1},
    "injection_config": {"targets": ["query"], "rank": 2},
    "artifact_format": 1,
}


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs):
        return self.classifier(self.query(inputs))


def make_model():
    model = TinyModel()
    inject_lora(model, ["query"], rank=2, alpha=4.0, trainable_prefixes=("classifier",))
    return model


def test_task_artifact_round_trip_preserves_outputs(tmp_path):
    source = make_model()
    with torch.no_grad():
        source.query.lora_a.normal_()
        source.query.lora_b.normal_()
        source.classifier.weight.normal_()
    inputs = torch.randn(3, 4)
    expected = source(inputs).detach()
    path = tmp_path / "task.pt"
    save_task_artifact(source, path, METADATA)

    target = make_model()
    target.query.base.load_state_dict(source.query.base.state_dict())
    load_task_artifact(target, path, METADATA)

    assert torch.equal(target(inputs), expected)


def test_task_artifact_rejects_metadata_mismatch(tmp_path):
    model = make_model()
    path = tmp_path / "task.pt"
    save_task_artifact(model, path, METADATA)
    wrong = {**METADATA, "base_model_revision": "wrong"}

    with pytest.raises(CheckpointError, match="metadata mismatch"):
        load_task_artifact(make_model(), path, wrong)


def test_task_artifact_rejects_missing_trainable_parameter(tmp_path):
    model = make_model()
    path = tmp_path / "task.pt"
    save_task_artifact(model, path, METADATA)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["state_dict"].pop("classifier.bias")
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="trainable checkpoint keys differ"):
        load_task_artifact(make_model(), path, METADATA)


def test_training_checkpoint_restores_model_and_optimizer(tmp_path):
    model = make_model()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
    )
    inputs = torch.randn(2, 4)
    model(inputs).sum().backward()
    optimizer.step()
    path = tmp_path / "resume.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        metadata=METADATA,
        rng_state=capture_rng_state(),
        update=3,
    )

    restored = make_model()
    restored_optimizer = torch.optim.AdamW(
        [parameter for parameter in restored.parameters() if parameter.requires_grad], lr=0.01
    )
    result = load_training_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        expected_metadata=METADATA,
    )

    assert result["update"] == 3
    assert all(
        torch.equal(model.state_dict()[name], restored.state_dict()[name])
        for name in model.state_dict()
    )
