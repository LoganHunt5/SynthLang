import yaml

class ModelConfig:
    def __init__(self, config: dict, type: str = "", sample_rate: int = 0):
        self.config = config
        self.type = type
        self.sample_rate = sample_rate

    def get(self, key: str, default=None):
        return self.config.get(key, default)

class TrainingConfig:
    def __init__(self, epochs: int = 1, batch_size: int = 1,
                 eval_batch_size: int = 1, learning_rate: float = 1e-3,
                 output_path: str = "",
                 eval_split_size: float = 0.0, eval_split_max_size: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.output_path = output_path
        self.eval_split_size = eval_split_size
        self.eval_split_max_size = eval_split_max_size

    def get(self, key: str, default=None):
        return getattr(self, key, default)

class DatasetConfig:
    def __init__(self, path:str, max_samples:int, formatter:str, meta_file_train:str):
        self.path = path
        self.max_samples = max_samples
        self.formatter = formatter
        self.meta_file_train = meta_file_train

    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
class ConfigManager:
    def __init__(self):
        self.model_config = None
        self.training_config = None
        self.dataset_config = None
        
    def load_config(self, path: str) -> dict:
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            errors = self.config_validator(config)
            if errors:
                raise ValueError("; ".join(errors))

            model = config["model"]
            training = config["training"]
            dataset = config["dataset"]
            self.model_config = ModelConfig(
                config=model,
                type=model.get("type", ""),
                sample_rate=model["sample_rate"],
            )
            self.training_config = TrainingConfig(
                epochs=training["epochs"],
                batch_size=training["batch_size"],
                eval_batch_size=training.get("eval_batch_size", 1),
                learning_rate=training["learning_rate"],
                output_path=training.get("output_path", ""),
                eval_split_size=training.get("eval_split_size", 0.0),
                eval_split_max_size=training.get("eval_split_max_size", 0),
            )
            self.dataset_config = DatasetConfig(
                path=dataset["path"],
                max_samples=dataset["max_samples"],
                formatter=dataset["formatter"],
                meta_file_train=dataset["meta_file_train"],
            )
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")

    def config_validator(self, config: dict) -> list:
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

        max_samples = dataset.get("max_samples", 0)
        if not isinstance(max_samples, int) or isinstance(max_samples, bool):
            errors.append("max_samples must be an integer")
        elif max_samples <= 0:
            errors.append("max_samples must be > 0")
        elif max_samples > 100:
            errors.append("Demo only supports datasets <= 100 samples")

        if not dataset.get("path"):
            errors.append("dataset path must be provided")

        if not dataset.get("formatter"):
            errors.append("dataset formatter must be provided")

        if not dataset.get("meta_file_train"):
            errors.append("dataset training metadata file must be provided")

        # --- Model validation ---
        model = config.get("model", {})
        if not isinstance(model, dict):
            model = {}

        if model.get("sample_rate", 0) <= 0:
            errors.append("sample_rate must be > 0")

        return errors