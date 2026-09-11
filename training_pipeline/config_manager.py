import yaml


class ConfigManager:
    @staticmethod
    def load_config(path: str) -> dict:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    @staticmethod
    def config_validator(config: dict) -> list:
        errors = []

        if not isinstance(config, dict):
            return ["config must be a dictionary"]

        # --- Training validation ---
        training = config.get("training", {})
        if not isinstance(training, dict):
            training = {}

        if training.get("batch_size", 1) <= 0:
            errors.append("batch_size must be > 0")

        if training.get("learning_rate", 1e-3) <= 0:
            errors.append("learning_rate must be > 0")

        if training.get("epochs", 1) < 1:
            errors.append("epochs must be >= 1")

        # --- Dataset validation ---
        dataset = config.get("dataset", {})
        if not isinstance(dataset, dict):
            dataset = {}

        if dataset.get("max_samples", 0) > 100:
            errors.append("Demo only supports datasets <= 100 samples")

        if not dataset.get("path"):
            errors.append("dataset path must be provided")

        # --- Model validation ---
        model = config.get("model", {})
        if not isinstance(model, dict):
            model = {}

        if model.get("sample_rate", 0) <= 0:
            errors.append("sample_rate must be > 0")

        return errors