from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

from pipeline.coqui_trainer import CoquiTrainer


class TestCoquiTrainer(unittest.TestCase):
    def test_train_builds_model_and_starts_fit(self):
        dataset_cfg = SimpleNamespace(
            sample_rate=22050,
            formatter="ljspeech",
            meta_file_train="metadata.csv",
            path="/tmp/dataset",
        )
        training_cfg = SimpleNamespace(
            batch_size=8,
            eval_batch_size=4,
            epochs=2,
            output_path="/tmp/run",
            eval_split_size=0.1,
            eval_split_max_size=10,
        )
        config = SimpleNamespace(
            dataset_config=dataset_cfg,
            model_config=SimpleNamespace(),
            training_config=training_cfg,
        )
        logger = Mock()

        with (
            patch("pipeline.coqui_trainer.BaseAudioConfig") as audio_config,
            patch("pipeline.coqui_trainer.BaseDatasetConfig") as dataset_config,
            patch("pipeline.coqui_trainer.Tacotron2Config") as model_config,
            patch("pipeline.coqui_trainer.AudioProcessor.init_from_config") as init_audio,
            patch("pipeline.coqui_trainer.TTSTokenizer.init_from_config") as init_tokenizer,
            patch("pipeline.coqui_trainer.load_tts_samples") as load_samples,
            patch("pipeline.coqui_trainer.Tacotron2") as make_model,
            patch("pipeline.coqui_trainer.TrainerArgs") as trainer_args,
            patch("pipeline.coqui_trainer.CoquiTrainerClass") as make_trainer,
        ):
            audio_processor = sentinel.audio_processor
            tokenizer = sentinel.tokenizer
            coqui_config = sentinel.coqui_config
            train_samples = [sentinel.train_sample]
            eval_samples = [sentinel.eval_sample]
            model = sentinel.model

            init_audio.return_value = audio_processor
            init_tokenizer.return_value = (tokenizer, coqui_config)
            load_samples.return_value = (train_samples, eval_samples)
            make_model.return_value = model

            CoquiTrainer(logger).train(config)

        audio_config.assert_called_once()
        self.assertEqual(
            audio_config.call_args.kwargs["sample_rate"],
            dataset_cfg.sample_rate,
        )

        dataset_config.assert_called_once_with(
            format=dataset_cfg.formatter,
            meta_file_train=dataset_cfg.meta_file_train,
            path=dataset_cfg.path,
        )

        model_config.assert_called_once()
        self.assertEqual(model_config.call_args.kwargs["batch_size"], 8)
        self.assertEqual(model_config.call_args.kwargs["epochs"], 2)

        init_audio.assert_called_once_with(model_config.return_value)
        init_tokenizer.assert_called_once_with(model_config.return_value)
        load_samples.assert_called_once()
        make_model.assert_called_once_with(
            coqui_config,
            ap=audio_processor,
            tokenizer=tokenizer,
        )

        make_trainer.assert_called_once()
        make_trainer.return_value.fit.assert_called_once_with()
        logger.info.assert_any_call("Preparing Coqui training")
        logger.info.assert_any_call(
            "Loaded %d training samples and %d evaluation samples",
            len(train_samples),
            len(eval_samples),
        )
        logger.info.assert_any_call("Starting Coqui trainer")
        logger.info.assert_any_call("Coqui training completed")


if __name__ == "__main__":
    unittest.main()