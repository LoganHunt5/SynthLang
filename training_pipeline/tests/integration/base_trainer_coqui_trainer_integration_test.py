import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch, sentinel

from pipeline.coqui_trainer import CoquiTrainer
from pipeline.trainer import Trainer


class TestBaseTrainerCoquiTrainerIntegration(unittest.TestCase):
    def test_coqui_trainer_implements_base_training_contract(self):
        config = SimpleNamespace(
            dataset_config=SimpleNamespace(
                formatter="ljspeech",
                meta_file_train="metadata.csv",
                path="/tmp/dataset",
            ),
            model_config=SimpleNamespace(sample_rate=22050),
            training_config=SimpleNamespace(
                batch_size=2,
                eval_batch_size=2,
                epochs=1,
                output_path="/tmp/output",
                eval_split_size=0.1,
                eval_split_max_size=10,
            ),
        )
        logger = Mock()
        trainer: Trainer = CoquiTrainer(logger)

        self.assertIsInstance(trainer, Trainer)

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
            patch("pipeline.coqui_trainer.TrainerArgs") as trainer_args,
            patch("pipeline.coqui_trainer.CoquiTrainerClass") as coqui_trainer_class,
        ):
            trainer.train(config, warm_start_checkpoint="checkpoint.pth")

        trainer_args.assert_called_once_with(restore_path="checkpoint.pth")
        coqui_trainer_class.return_value.fit.assert_called_once_with()
        logger.info.assert_any_call("Starting Coqui trainer")
        logger.info.assert_any_call("Coqui training completed")


if __name__ == "__main__":
    unittest.main()