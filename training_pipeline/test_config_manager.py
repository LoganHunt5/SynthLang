import pytest

from training_pipeline.config_manager import ConfigManager as config_manager


@pytest.fixture
def valid_config():
    return {
        "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 10},
        "dataset": {"path": "data/train.csv", "max_samples": 50},
        "model": {"sample_rate": 16000},
    }


def test_load_config_returns_yaml_dict(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 10
dataset:
  path: data/train.csv
  max_samples: 50
model:
  sample_rate: 16000
""".strip()
    )

    result = config_manager().load_config(str(config_file))

    assert result["training"]["batch_size"] == 32
    assert result["model"]["sample_rate"] == 16000


def test_load_config_raises_runtime_error_for_bad_yaml(tmp_path):
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("training: [bad")

    with pytest.raises(RuntimeError, match="Failed to load config"):
        config_manager.load_config(str(config_file))


def test_config_validator_accepts_valid_config(valid_config):
    errors = config_manager.config_validator(valid_config)

    assert errors == []


def test_config_validator_rejects_non_dict_input():
    assert config_manager.config_validator(None) == ["config must be a dictionary"]
    assert config_manager.config_validator("not-a-dict") == ["config must be a dictionary"]


def test_config_validator_reports_invalid_values():
    config = {
        "training": {"batch_size": 0, "learning_rate": -0.1, "epochs": 0},
        "dataset": {"path": "", "max_samples": 150},
        "model": {"sample_rate": 0},
    }

    errors = config_manager().config_validator(config)

    assert "batch_size must be > 0" in errors
    assert "learning_rate must be > 0" in errors
    assert "epochs must be >= 1" in errors
    assert "Demo only supports datasets <= 100 samples" in errors
    assert "dataset path must be provided" in errors
    assert "sample_rate must be > 0" in errors


def test_config_validator_allows_boundary_values():
    config = {
        "training": {"batch_size": 1, "learning_rate": 1e-6, "epochs": 1},
        "dataset": {"path": "data/sample.csv", "max_samples": 100},
        "model": {"sample_rate": 1},
    }

    assert config_manager.config_validator(config) == []


def test_config_validator_handles_missing_sections():
    config = {}

    errors = config_manager.config_validator(config)

    assert "dataset path must be provided" in errors
    assert "sample_rate must be > 0" in errors
    assert "batch_size must be > 0" not in errors
