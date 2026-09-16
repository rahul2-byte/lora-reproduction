import pytest

torch = pytest.importorskip("torch")

from lora_study.data import TASK_COLUMNS, audit_split_overlap, label_mapping, tokenize_task
from lora_study.evaluate import accuracy, binary_f1


def test_metrics_use_declared_binary_convention():
    predictions = [1, 1, 0, 0]
    labels = [1, 0, 1, 0]

    assert accuracy(predictions, labels) == 0.5
    assert binary_f1(predictions, labels) == pytest.approx(0.5)


def test_metrics_reject_mismatched_or_empty_inputs():
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        binary_f1([1], [1, 0])


def test_tokenization_uses_expected_columns_and_max_length():
    calls = {}

    class FakeTokenizer:
        def __call__(self, *values, **kwargs):
            calls["values"] = values
            calls["kwargs"] = kwargs
            return {"input_ids": [[1, 2]]}

    class FakeDataset:
        def map(self, function, batched):
            result = function({"sentence1": ["a"], "sentence2": ["b"]})
            assert batched is True
            assert result == {"input_ids": [[1, 2]]}
            return "mapped"

    assert TASK_COLUMNS["mrpc"] == ("sentence1", "sentence2")
    class FakeDatasetDict:
        def __getitem__(self, key):
            return type("Split", (), {"column_names": ["sentence1", "sentence2", "label", "idx"]})()

        def map(self, function, batched, remove_columns):
            assert batched is True
            assert remove_columns == ["sentence1", "sentence2"]
            function({"sentence1": ["a"], "sentence2": ["b"]})
            return self

        def rename_column(self, old, new):
            assert (old, new) in {("label", "labels"), ("idx", "example_id")}
            return self

    assert isinstance(tokenize_task(FakeDatasetDict(), FakeTokenizer(), "mrpc", 32), FakeDatasetDict)
    assert calls["values"] == (["a"], ["b"])
    assert calls["kwargs"] == {"truncation": True, "max_length": 32}


def test_split_overlap_audit_counts_exact_text_overlap():
    dataset = {
        "train": [{"sentence": "Same text"}, {"sentence": "Train only"}],
        "validation": [{"sentence": "same text"}],
    }

    result = audit_split_overlap(dataset, "sst2")

    assert result["overlap_counts"]["train:validation"] == 1


def test_task_label_mappings_are_declared():
    assert label_mapping("mrpc") == {"not_equivalent": 0, "equivalent": 1}
    assert label_mapping("sst2") == {"negative": 0, "positive": 1}
