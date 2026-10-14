from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

import numpy as np
import soundfile
import TTS.tts.datasets.dataset as coqui_dataset

from pipeline.coqui_trainer import (
    COQUI_MODELS,
    CoquiModelSpec,
    CoquiTrainer,
    soundfile_audio_size,
)


class TestCoquiTrainer(unittest.TestCase):
    def _config(self, model_type="tacotron2", params=None):
        dataset_cfg = SimpleNamespace(
            formatter="ljspeech",
            meta_file_train="metadata.csv",
            path="/tmp/dataset",
        )
        training_cfg = SimpleNamespace(
            batch_size=8,
            eval_batch_size=4,
            epochs=2,
            learning_rate=0.005,
            output_path="/tmp/run",
            eval_split_size=0.1,
            eval_split_max_size=10,
        )
        model_cfg = SimpleNamespace(
            type=model_type,
            sample_rate=22050,
            params=params or {},
        )
        return SimpleNamespace(
            dataset_config=dataset_cfg,
            model_config=model_cfg,
            training_config=training_cfg,
        )

    def _train(self, config, *, defaults=None, warm_start_checkpoint=None,
               resume_path=None):
        """Run CoquiTrainer.train with Coqui mocked out; return the mocks."""
        spec = CoquiModelSpec(
            config_cls=Mock(name="config_cls"),
            model_cls=Mock(name="model_cls"),
            defaults=defaults or {},
        )
        mocks = SimpleNamespace(spec=spec, logger=Mock())

        with (
            patch.dict(COQUI_MODELS, {"tacotron2": spec}, clear=True),
            patch("pipeline.coqui_trainer.BaseAudioConfig") as mocks.audio_config,
            patch("pipeline.coqui_trainer.BaseDatasetConfig") as mocks.dataset_config,
            patch("pipeline.coqui_trainer.AudioProcessor.init_from_config") as mocks.init_audio,
            patch("pipeline.coqui_trainer.TTSTokenizer.init_from_config") as mocks.init_tokenizer,
            patch("pipeline.coqui_trainer.load_tts_samples") as mocks.load_samples,
            patch("pipeline.coqui_trainer.TrainerArgs") as mocks.trainer_args,
            patch("pipeline.coqui_trainer.CoquiTrainerClass") as mocks.make_trainer,
        ):
            mocks.init_audio.return_value = sentinel.audio_processor
            mocks.init_tokenizer.return_value = (sentinel.tokenizer, sentinel.coqui_config)
            mocks.load_samples.return_value = (
                [sentinel.train_sample],
                [sentinel.eval_sample],
            )
            CoquiTrainer(mocks.logger).train(
                config,
                warm_start_checkpoint=warm_start_checkpoint,
                resume_path=resume_path,
            )
        return mocks

    def test_train_builds_model_and_starts_fit(self):
        config = self._config()
        mocks = self._train(config, defaults={"r": 2})

        self.assertEqual(
            mocks.audio_config.call_args.kwargs["sample_rate"],
            config.model_config.sample_rate,
        )
        mocks.dataset_config.assert_called_once_with(
            formatter="ljspeech",
            meta_file_train="metadata.csv",
            path="/tmp/dataset",
        )

        model_kwargs = mocks.spec.config_cls.call_args.kwargs
        self.assertEqual(model_kwargs["batch_size"], 8)
        self.assertEqual(model_kwargs["epochs"], 2)
        self.assertEqual(model_kwargs["r"], 2)

        coqui_model_cfg = mocks.spec.config_cls.return_value
        mocks.init_audio.assert_called_once_with(coqui_model_cfg)
        mocks.init_tokenizer.assert_called_once_with(coqui_model_cfg)
        mocks.spec.model_cls.assert_called_once_with(
            sentinel.coqui_config,
            ap=sentinel.audio_processor,
            tokenizer=sentinel.tokenizer,
        )
        mocks.trainer_args.assert_called_once_with(restore_path="")
        mocks.make_trainer.return_value.fit.assert_called_once_with()
        mocks.logger.info.assert_any_call("Preparing Coqui %s training", "tacotron2")
        mocks.logger.info.assert_any_call(
            "Loaded %d training samples and %d evaluation samples", 1, 1
        )
        mocks.logger.info.assert_any_call("Coqui training completed")

    def test_train_passes_learning_rate_as_coqui_lr(self):
        mocks = self._train(self._config())

        self.assertEqual(mocks.spec.config_cls.call_args.kwargs["lr"], 0.005)

    def test_model_params_override_defaults_and_generic_settings(self):
        config = self._config(params={"r": 7, "lr": 0.0001, "phoneme_language": "es"})
        mocks = self._train(config, defaults={"r": 2})

        model_kwargs = mocks.spec.config_cls.call_args.kwargs
        self.assertEqual(model_kwargs["r"], 7)
        self.assertEqual(model_kwargs["lr"], 0.0001)
        self.assertEqual(model_kwargs["phoneme_language"], "es")

    def test_nested_audio_and_characters_params_build_coqui_configs(self):
        config = self._config(params={
            "audio": {"preemphasis": 0.99, "signal_norm": True},
            "characters": {"pad": "_", "characters": "abc"},
        })

        with patch("pipeline.coqui_trainer.CharactersConfig") as characters_config:
            mocks = self._train(config)

        audio_kwargs = mocks.audio_config.call_args.kwargs
        self.assertEqual(audio_kwargs["preemphasis"], 0.99)
        self.assertTrue(audio_kwargs["signal_norm"])
        self.assertEqual(audio_kwargs["sample_rate"], 22050)
        characters_config.assert_called_once_with(pad="_", characters="abc")
        model_kwargs = mocks.spec.config_cls.call_args.kwargs
        self.assertIs(model_kwargs["characters"], characters_config.return_value)
        self.assertIs(model_kwargs["audio"], mocks.audio_config.return_value)
        # the user's config object is not mutated
        self.assertIn("audio", config.model_config.params)

    def test_unknown_audio_param_is_rejected(self):
        config = self._config(params={"audio": {"not_a_real_option": 1}})

        with (
            patch("pipeline.coqui_trainer.BaseDatasetConfig"),
            self.assertRaisesRegex(ValueError, "Invalid model.params.audio.*not_a_real_option"),
        ):
            CoquiTrainer(Mock()).train(config)

    def test_train_passes_warm_start_checkpoint_to_coqui(self):
        mocks = self._train(
            self._config(),
            warm_start_checkpoint="checkpoints/model.pth",
        )

        mocks.trainer_args.assert_called_once_with(restore_path="checkpoints/model.pth")
        mocks.make_trainer.assert_called_once()

    def test_resume_continues_the_run_folder(self):
        mocks = self._train(
            self._config(),
            warm_start_checkpoint="checkpoints/model.pth",
            resume_path="runs/sss-run",
        )

        # resuming takes precedence; warm start would reset the optimizer/step
        mocks.trainer_args.assert_called_once_with(continue_path="runs/sss-run")
        mocks.make_trainer.return_value.fit.assert_called_once_with()
        mocks.logger.info.assert_any_call("Resuming Coqui run in %s", "runs/sss-run")

    def test_train_measures_audio_length_without_ffmpeg(self):
        original = coqui_dataset.get_audio_size
        self.addCleanup(setattr, coqui_dataset, "get_audio_size", original)

        self._train(self._config())

        self.assertIs(coqui_dataset.get_audio_size, soundfile_audio_size)

    def test_soundfile_audio_size_reads_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "clip.wav"
            soundfile.write(wav, np.zeros(22050 + 7), 22050)

            self.assertEqual(soundfile_audio_size(wav), 22057)

    def test_soundfile_audio_size_raises_runtime_error_for_unreadable_file(self):
        # Coqui skips a clip only when its length check raises RuntimeError
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "clip.wav"
            bad.write_text("not audio")

            with self.assertRaisesRegex(RuntimeError, "Failed to read audio header"):
                soundfile_audio_size(bad)
            with self.assertRaises(RuntimeError):
                soundfile_audio_size(Path(directory) / "missing.wav")

    def test_unknown_model_type_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "fastspeech9.*Available types: tacotron2"):
            self._train(self._config(model_type="fastspeech9"))

    def test_unknown_model_param_is_rejected(self):
        spec = COQUI_MODELS["tacotron2"]
        config = self._config(params={"not_a_real_option": 1})

        # Use the real Coqui config class so its own validation is exercised.
        with (
            patch.dict(COQUI_MODELS, {"tacotron2": spec}),
            patch("pipeline.coqui_trainer.BaseDatasetConfig"),
            patch("pipeline.coqui_trainer.BaseAudioConfig"),
            self.assertRaisesRegex(ValueError, "Invalid model.params.*not_a_real_option"),
        ):
            CoquiTrainer(Mock()).train(config)


if __name__ == "__main__":
    unittest.main()
