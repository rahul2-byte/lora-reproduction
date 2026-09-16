import hashlib
import json

import pytest

from lora_study.config import ConfigError, load_config


def test_loads_valid_smoke_config_and_hashes_resolved_values(tmp_path):
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(
        """
        [experiment]
        method = "custom_lora"
        seed = 7

        [model]
        model_id = "FacebookAI/roberta-base"
        revision = "model-sha"

        [dataset]
        task = "mrpc"
        dataset_id = "nyu-mll/glue"
        revision = "data-sha"
        max_length = 128
        evaluation_protocol = "heldout_official_validation"

        [lora]
        rank = 4
        alpha = 8.0
        dropout = 0.1

        [training]
        microbatch = 2
        gradient_accumulation = 4
        """.strip()
    )

    config = load_config(config_path)

    assert config.experiment.method == "custom_lora"
    assert config.lora.scaling == 2.0
    resolved = config.to_dict()
    expected = hashlib.sha256(
        json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert config.resolved_hash == expected


def test_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
        [experiment]
        method = "head_only"
        seed = 1
        unexpected = true
        """.strip()
    )

    with pytest.raises(ConfigError, match="unknown key.*unexpected"):
        load_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", "0", "rank"),
        ("dropout", "1.0", "dropout"),
        ("alpha", '"nan"', "alpha"),
    ],
)
def test_rejects_invalid_lora_values(tmp_path, field, value, message):
    path = tmp_path / "bad.toml"
    path.write_text(
        f"""
        [experiment]
        method = "custom_lora"
        seed = 1

        [dataset]
        revision = "data-sha"

        [lora]
        rank = {value if field == "rank" else "4"}
        alpha = {value if field == "alpha" else "8.0"}
        dropout = {value if field == "dropout" else "0.1"}
        """.strip()
    )

    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_rejects_lora_settings_for_non_lora_method(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        """
        [experiment]
        method = "head_only"
        seed = 1

        [dataset]
        revision = "data-sha"

        [lora]
        rank = 4
        alpha = 8.0
        dropout = 0.1
        """.strip()
    )

    with pytest.raises(ConfigError, match="LoRA settings require a LoRA method"):
        load_config(path)


def test_hash_is_stable_across_loads(tmp_path):
    path = tmp_path / "smoke.toml"
    path.write_text(
        """
        [experiment]
        method = "head_only"
        seed = 1
        """.strip()
    )

    assert load_config(path).resolved_hash == load_config(path).resolved_hash


def test_loads_configured_target_modules(tmp_path):
    path = tmp_path / "targets.toml"
    path.write_text(
        """
        [experiment]
        method = "custom_lora"
        seed = 1

        [lora]
        target_modules = ["query"]
        """.strip()
    )

    assert load_config(path).lora.target_modules == ("query",)
