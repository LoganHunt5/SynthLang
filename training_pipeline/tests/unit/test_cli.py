import logging
import tempfile
import unittest
from pathlib import Path

import yaml

from unittest.mock import patch

from pipeline.cli import main, run
from pipeline.trainer import Trainer


class RecordingTrainer(Trainer):
    def __init__(self):
        self.calls = []

    def train(self, config, *, warm_start_checkpoint=None, resume_path=None):
        self.calls.append((config, warm_start_checkpoint, resume_path))


class TestCli(unittest.TestCase):
    def _write_config(self, directory, warm_start=None):
        config = {
            "training": {
                "batch_size": 8,
                "learning_rate": 0.001,
                "epochs": 2,
            },
            "dataset": {
                "path": "data/train.csv",
                "formatter": "default",
                "meta_file_train": "metadata.csv",
            },
            "model": {"sample_rate": 16000},
        }
        if warm_start is not None:
            config["training"]["warm_start"] = warm_start

        config_path = Path(directory) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    def test_run_starts_fresh_when_warm_start_is_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            trainer = RecordingTrainer()

            result = run(
                config_path,
                trainer_factory=lambda logger: trainer,
                logger=logging.getLogger("test_cli_fresh"),
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(trainer.calls), 1)
            self.assertIsNone(trainer.calls[0][1])

    def test_run_passes_warm_start_checkpoint_to_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                {
                    "enabled": True,
                    "checkpoint_path": "checkpoints/model.pth",
                },
            )
            trainer = RecordingTrainer()

            result = run(
                config_path,
                trainer_factory=lambda logger: trainer,
                logger=logging.getLogger("test_cli_warm"),
            )

            self.assertEqual(result, 0)
            self.assertEqual(
                trainer.calls[0][1],
                "checkpoints/model.pth",
            )

    def test_run_rejects_warm_start_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                {"enabled": True, "checkpoint_path": ""},
            )

            with self.assertRaisesRegex(RuntimeError, "checkpoint_path"):
                run(
                    config_path,
                    trainer_factory=lambda logger: RecordingTrainer(),
                    logger=logging.getLogger("test_cli_invalid"),
                )

    def test_run_resumes_from_run_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(
                directory,
                {"enabled": True, "checkpoint_path": "checkpoints/model.pth"},
            )
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}")
            (run_dir / "checkpoint_2000.pth").write_bytes(b"")
            trainer = RecordingTrainer()

            result = run(
                config_path,
                resume_path=run_dir,
                trainer_factory=lambda logger: trainer,
                logger=logging.getLogger("test_cli_resume"),
            )

            self.assertEqual(result, 0)
            self.assertEqual(trainer.calls[0][1:], (None, str(run_dir)))

    def test_run_rejects_resume_folder_without_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)

            with self.assertRaisesRegex(ValueError, "containing config.json"):
                run(
                    config_path,
                    resume_path=Path(directory) / "missing",
                    trainer_factory=lambda logger: RecordingTrainer(),
                    logger=logging.getLogger("test_cli_resume_missing"),
                )

    def test_run_rejects_resume_folder_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            (run_dir / "config.json").write_text("{}")

            with self.assertRaisesRegex(ValueError, "no checkpoints found"):
                run(
                    config_path,
                    resume_path=run_dir,
                    trainer_factory=lambda logger: RecordingTrainer(),
                    logger=logging.getLogger("test_cli_resume_empty"),
                )

    def test_main_parses_resume_flag(self):
        with patch("pipeline.cli.run", return_value=0) as fake_run:
            self.assertEqual(main(["config.yaml", "--resume", "runs/sss"]), 0)

        fake_run.assert_called_once_with(Path("config.yaml"), resume_path=Path("runs/sss"))

    def test_run_rejects_unknown_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            config = yaml.safe_load(config_path.read_text())
            config["backend"] = "not_a_backend"
            config_path.write_text(yaml.safe_dump(config))

            with self.assertRaisesRegex(ValueError, "Unknown backend 'not_a_backend'"):
                run(config_path, logger=logging.getLogger("test_cli_backend"))

    def test_registry_resolves_coqui_backend(self):
        from pipeline.coqui_trainer import CoquiTrainer
        from pipeline.registry import get_trainer_factory

        logger = logging.getLogger("test_cli_registry")
        trainer = get_trainer_factory("coqui")(logger)

        self.assertIsInstance(trainer, CoquiTrainer)
        self.assertIs(trainer.logger, logger)


if __name__ == "__main__":
    unittest.main()
