import contextlib
import os
import tempfile
import unittest
from pathlib import Path

from training_pipeline.pipeline import dataset_manager
from training_pipeline.pipeline.config_manager import ConfigManager


class TestConfigDatasetIntegration(unittest.TestCase):
    def test_config_dataset_handoff_loads_samples(self):
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"

        with tempfile.TemporaryDirectory() as tmp_dir:
            working_dir = Path(tmp_dir)
            dataset_dir = working_dir / "tmp" / "LJSpeech-1.1"
            audio_dir = dataset_dir / "audio"
            audio_dir.mkdir(parents=True)

            (audio_dir / "hello.wav").touch()
            (dataset_dir / "metadata.csv").write_text(
                "audio_file|text\nhello.wav|Hello there!\n"
            )

            with contextlib.chdir(working_dir):
                config = ConfigManager()
                config.load_config(str(config_path))

                samples = dataset_manager.load_dataset(config.dataset_config)

            self.assertEqual(
                samples,
                [{
                    "audio": os.path.join(
                        config.dataset_config.path, "audio", "hello.wav"
                    ),
                    "text": "Hello there!",
                }],
            )


if __name__ == "__main__":
    unittest.main()