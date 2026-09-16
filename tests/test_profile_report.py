import json

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from lora_study.profile import artifact_parameter_counts, count_parameters
from lora_study.report import collect_summaries, write_csv


def test_parameter_count_reports_trainable_fraction():
    model = nn.Linear(3, 2)
    model.bias.requires_grad_(False)
    result = count_parameters(model)

    assert result["total"] == 8
    assert result["trainable"] == 6
    assert result["trainable_fraction"] == pytest.approx(0.75)


def test_report_reads_summaries_and_writes_csv(tmp_path):
    run = tmp_path / "run-1"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "run_id": "run-1",
                "method": "head_only",
                "seed": 1,
                "parameters": {"trainable": 2},
                "metric": 0.5,
            }
        )
    )

    rows = collect_summaries(tmp_path)
    output = tmp_path / "metrics.csv"
    write_csv(rows, output)

    assert rows == [
        {
            "status": "completed",
            "method": "head_only",
            "seed": 1,
            "parameters": {"trainable": 2},
            "metric": 0.5,
            "run_id": "run-1",
            "study_group": "core",
        }
    ]
    assert "run_id" in output.read_text()


def test_custom_lora_counts_come_from_training_artifact(tmp_path):
    path = tmp_path / "task.pt"
    torch.save({"state_dict": {
        "layer.lora_a": torch.zeros(2, 3),
        "layer.lora_b": torch.zeros(4, 2),
        "classifier.weight": torch.zeros(2, 4),
    }}, path)
    old = {"method": "custom_lora", "parameters": {"total": 100, "trainable": 20}}
    corrected = artifact_parameter_counts(old, path)
    assert corrected == {"total": 114, "trainable": 22,
                         "trainable_fraction": pytest.approx(22 / 114)}
    current = {"method": "custom_lora", "parameters": corrected}
    assert artifact_parameter_counts(current, path) == corrected
