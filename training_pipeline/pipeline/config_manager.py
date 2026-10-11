import yaml

DEFAULT_BACKEND = "coqui"
DEFAULT_MODEL_TYPE = "tacotron2"

class ModelConfig:
    def __init__(self, raw_config_dict: dict, type: str = "", sample_rate: int = 0,
                 params: dict | None = None):
        self.raw_config_dict = raw_config_dict
        self.type = type
        self.sample_rate = sample_rate
        # Backend/architecture specific hyperparameters, passed through as-is
        # and validated by the backend that consumes them.
        self.params = params or {}

    def get(self, key: str, default=None):
        return self.raw_config_dict.get(key, default)

class TrainingConfig:
    def __init__(self, epochs: int = 1, batch_size: int = 1,
                 eval_batch_size: int = 1, learning_rate: float = 1e-3,
                 output_path: str = "",
                 eval_split_size: float = 0.0, eval_split_max_size: int = 0,
                 warm_start_enabled: bool = False,
                 warm_start_checkpoint_path: str = ""):
        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.output_path = output_path
        self.eval_split_size = eval_split_size
        self.eval_split_max_size = eval_split_max_size
        self.warm_start_enabled = warm_start_enabled
        self.warm_start_checkpoint_path = warm_start_checkpoint_path

    def get(self, key: str, default=None):
        return getattr(self, key, default)

class DatasetConfig:
    def __init__(self, path:str, formatter:str, meta_file_train:str):
        self.path = path
        #self.max_samples = max_samples
        self.formatter = formatter
        self.meta_file_train = meta_file_train

    def get(self, key: str, default=None):
        return getattr(self, key, default)
    
class ConfigManager:
    def __init__(self):
        self.backend = None
        self.model_config = None
        self.training_config = None
        self.dataset_config = None
        
    def load_config(self, path: str) -> dict:
        try:
            # the whole contents
            config_file = self._read_yaml(path)

            # validation
            errors = self._validate_config_dict(config_file)
            if errors:
                raise ValueError("; ".join(errors))

            if not isinstance(config_file, dict):
                raise ValueError("config must be a dictionary")

            model = config_file["model"]
            training = config_file["training"]
            dataset = config_file["dataset"]
            self.backend = config_file.get("backend", DEFAULT_BACKEND)
            self.model_config = ModelConfig(
                raw_config_dict=model,
                type=model.get("type", DEFAULT_MODEL_TYPE),
                sample_rate=model["sample_rate"],
                params=model.get("params", {}),
            )
            warm_start = training.get("warm_start", {})
            self.training_config = TrainingConfig(
                epochs=training["epochs"],
                batch_size=training["batch_size"],
                eval_batch_size=training.get("eval_batch_size", 1),
                learning_rate=training["learning_rate"],
                output_path=training.get("output_path", ""),
                eval_split_size=training.get("eval_split_size", 0.0),
                eval_split_max_size=training.get("eval_split_max_size", 0),
                warm_start_enabled=warm_start.get("enabled", False),
                warm_start_checkpoint_path=warm_start.get("checkpoint_path", ""),
            )
            self.dataset_config = DatasetConfig(
                path=dataset["path"],
                #max_samples=dataset["max_samples"],
                formatter=dataset["formatter"],
                meta_file_train=dataset["meta_file_train"],
            )
            return config_file
        except Exception as e:
            raise RuntimeError(f"Failed to load config: {e}")
    
    # These functions are to be only called by the load_config method
    # Tyler, if I should make a more stricter access to these methods,
    # let me know.
    def _read_yaml(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _validate_config_dict(self, config_dict) -> list:
        errors = []
        
        if not isinstance(config_dict, dict):
            return ["config must be a dictionary"]

        backend = config_dict.get("backend", DEFAULT_BACKEND)
        if not isinstance(backend, str) or not backend:
            errors.append("backend must be a non-empty string")

        # --- Training validation ---
        training = config_dict.get("training", {})
        if not isinstance(training, dict):
            errors.append("training must be a dictionary")
            training = {}

        batch_size = training.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            errors.append("batch_size must be an integer")
        elif batch_size <= 0:
            errors.append("batch_size must be > 0")

        learning_rate = training.get("learning_rate")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            errors.append("learning_rate must be a number")
        elif learning_rate <= 0:
            errors.append("learning_rate must be > 0")

        epochs = training.get("epochs")
        if not isinstance(epochs, int) or isinstance(epochs, bool):
            errors.append("epochs must be an integer")
        elif epochs < 1:
            errors.append("epochs must be >= 1")

        warm_start = training.get("warm_start", {})
        if not isinstance(warm_start, dict):
            errors.append("warm_start must be a dictionary")
            warm_start = {}

        warm_start_enabled = warm_start.get("enabled", False)
        if not isinstance(warm_start_enabled, bool):
            errors.append("warm_start.enabled must be a boolean")

        checkpoint_path = warm_start.get("checkpoint_path", "")
        if warm_start_enabled and (
            not isinstance(checkpoint_path, str) or not checkpoint_path.strip()
        ):
            errors.append("warm_start.checkpoint_path must be a non-empty string")

        # --- Dataset validation ---
        dataset = config_dict.get("dataset", {})
        if not isinstance(dataset, dict):
            errors.append("dataset must be a dictionary")
            dataset = {}

        #max_samples = dataset.get("max_samples", 0)
        #if not isinstance(max_samples, int) or isinstance(max_samples, bool):
        #    errors.append("max_samples must be an integer")
        #elif max_samples <= 0:
        #    errors.append("max_samples must be > 0")
        #elif max_samples > 100:
        #    errors.append("Demo only supports datasets <= 100 samples")

        if not isinstance(dataset.get("path"), str) or not dataset.get("path"):
            errors.append("dataset path must be provided")

        if not isinstance(dataset.get("formatter"), str) or not dataset.get("formatter"):
            errors.append("dataset formatter must be provided")

        if not isinstance(dataset.get("meta_file_train"), str) or not dataset.get("meta_file_train"):
            errors.append("dataset training metadata file must be provided")

        # --- Model validation ---
        model = config_dict.get("model", {})
        if not isinstance(model, dict):
            errors.append("model must be a dictionary")
            model = {}

        sample_rate = model.get("sample_rate")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
            errors.append("sample_rate must be an integer")
        elif sample_rate <= 0:
            errors.append("sample_rate must be > 0")

        model_type = model.get("type", DEFAULT_MODEL_TYPE)
        if not isinstance(model_type, str) or not model_type:
            errors.append("model type must be a non-empty string")

        if not isinstance(model.get("params", {}), dict):
            errors.append("model params must be a dictionary")

        return errors