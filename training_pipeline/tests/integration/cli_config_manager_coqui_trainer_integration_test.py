import logging
import tempfile
import unittest
from pathlib import Path

import yaml

from pipeline.cli import run
from pipeline.config_manager import ConfigManager
from pipeline.coqui_trainer import CoquiTrainer


class RecordingCoquiTrainer(CoquiTrainer):
    def __init__(self, logger):
        super().__init__(logger)
        self.received_config = None
        self.received_warm_start_checkpoint = None
        self.completed = False

    def train(self, config, *, warm_start_checkpoint=None):
        self.received_config = config
        self.received_warm_start_checkpoint = warm_start_checkpoint
        self.logger.info("Trainer received config for %s", config.training_config.output_path)
        self.completed = True


class TestCliConfigManagerCoquiTrainerIntegration(unittest.TestCase):
    def test_cli_loads_yaml_and_passes_config_and_warm_start_to_trainer(self):
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 2,
                "output_path": "runs/demo",
                "warm_start": {
                    "enabled": True,
                    "checkpoint_path": "checkpoints/model.pth",
                },
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(yaml.safe_dump(config))

            logger = logging.getLogger("cli_config_manager_coqui_trainer_test")
            trainer = RecordingCoquiTrainer(logger)

            result = run(
                config_path,
                trainer_factory=lambda current_logger: trainer,
                logger=logger,
            )

            self.assertEqual(result, 0)
            self.assertTrue(trainer.completed)
            self.assertIsInstance(trainer.received_config, ConfigManager)
            self.assertTrue(trainer.received_config.training_config.warm_start_enabled)
            self.assertEqual(
                trainer.received_config.training_config.warm_start_checkpoint_path,
                "checkpoints/model.pth",
            )
            self.assertEqual(
                trainer.received_warm_start_checkpoint,
                "checkpoints/model.pth",
            )
            self.assertEqual(
                trainer.received_config.model_config.sample_rate,
                16000,
            )
            self.assertEqual(
                trainer.received_config.dataset_config.path,
                "data/train.csv",
            )


if __name__ == "__main__":
    unittest.main()
