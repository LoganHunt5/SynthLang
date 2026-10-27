import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, sentinel

from pipeline.coqui_trainer import CoquiTrainer
from training_pipeline.pipeline.logger import configure_logger


class TestLoggerCoquiTrainerIntegration(unittest.TestCase):
    def test_coqui_trainer_messages_are_written_to_configured_log(self):
        logger = logging.getLogger("training_pipeline")
        original_handlers = logger.handlers[:]
        original_level = logger.level
        original_propagate = logger.propagate

        for handler in original_handlers:
            logger.removeHandler(handler)

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                log_path = Path(tmp_dir) / "logs" / "training.log"
                configured_logger = configure_logger(str(log_path))
                config = SimpleNamespace(
                    dataset_config=SimpleNamespace(
                        formatter="ljspeech",
                        meta_file_train="metadata.csv",
                        path=str(Path(tmp_dir) / "dataset"),
                    ),
                    model_config=SimpleNamespace(sample_rate=22050),
                    training_config=SimpleNamespace(
                        batch_size=2,
                        eval_batch_size=2,
                        epochs=1,
                        output_path=str(Path(tmp_dir) / "output"),
                        eval_split_size=0.1,
                        eval_split_max_size=10,
                    ),
                )

                with (
                    patch("pipeline.coqui_trainer.BaseAudioConfig"),
                    patch("pipeline.coqui_trainer.BaseDatasetConfig"),
                    patch("pipeline.coqui_trainer.Tacotron2Config"),
                    patch("pipeline.coqui_trainer.AudioProcessor.init_from_config"),
                    patch(
                        "pipeline.coqui_trainer.TTSTokenizer.init_from_config",
                        return_value=(sentinel.tokenizer, sentinel.coqui_config),
                    ),
                    patch(
                        "pipeline.coqui_trainer.load_tts_samples",
                        return_value=([sentinel.train_sample], [sentinel.eval_sample]),
                    ),
                    patch("pipeline.coqui_trainer.Tacotron2"),
                    patch("pipeline.coqui_trainer.TrainerArgs"),
                    patch("pipeline.coqui_trainer.CoquiTrainerClass") as trainer_class,
                ):
                    CoquiTrainer(configured_logger).train(config)

                log_contents = log_path.read_text()

            self.assertIn("Preparing Coqui training", log_contents)
            self.assertIn(
                "Loaded 1 training samples and 1 evaluation samples", log_contents
            )
            self.assertIn("Starting Coqui trainer", log_contents)
            self.assertIn("Coqui training completed", log_contents)
            trainer_class.return_value.fit.assert_called_once_with()
        finally:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
                handler.close()

            for handler in original_handlers:
                logger.addHandler(handler)
            logger.setLevel(original_level)
            logger.propagate = original_propagate


if __name__ == "__main__":
    unittest.main()