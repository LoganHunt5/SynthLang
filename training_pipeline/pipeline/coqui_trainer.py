import logging
from pipeline.trainer import Trainer as BaseTrainer
from pipeline.config_manager import ConfigManager
import os
from trainer import Trainer as CoquiTrainerClass, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from dataclasses import dataclass, field
from typing import Any, cast


@dataclass(frozen=True)
class CoquiModelSpec:
    """How to build one Coqui architecture selected by ``model.type``."""

    config_cls: type
    model_cls: type
    # Architecture-specific defaults; any key can be overridden by model.params.
    defaults: dict[str, Any] = field(default_factory=dict)


COQUI_MODELS: dict[str, CoquiModelSpec] = {
    "tacotron2": CoquiModelSpec(
        config_cls=Tacotron2Config,
        model_cls=Tacotron2,
        defaults={
            "ga_alpha": 0.0,
            "decoder_loss_alpha": 0.25,
            "postnet_loss_alpha": 0.25,
            "postnet_diff_spec_alpha": 0.0,
            "decoder_diff_spec_alpha": 0.0,
            "decoder_ssim_alpha": 0.0,
            "postnet_ssim_alpha": 0.0,
            "r": 2,
            "attention_type": "dynamic_convolution",
            "double_decoder_consistency": False,
            "max_decoder_steps": 500, # CHANGE TO WHAT RECIPE RECOMMENDS WHEN NOT DEMO
        },
    ),
}


def get_model_spec(model_type: str) -> CoquiModelSpec:
    try:
        return COQUI_MODELS[model_type]
    except KeyError:
        available = ", ".join(sorted(COQUI_MODELS))
        raise ValueError(
            f"Model type '{model_type}' is not supported by the coqui backend. "
            f"Available types: {available}"
        ) from None


class CoquiTrainer(BaseTrainer):
    """
    Trainer class for Coqui models. Inherits from the base Trainer class.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @staticmethod
    def _build_nested(config_cls, kwargs, section: str | None, model_type: str):
        """Build a Coqui config class, reporting bad keys as a config error."""
        location = f"model.params.{section}" if section else "model.params"
        if not isinstance(kwargs, dict):
            raise ValueError(f"{location} must be a dictionary")
        try:
            return config_cls(**kwargs)
        except (TypeError, AssertionError) as error:
            raise ValueError(
                f"Invalid {location} for coqui {model_type}: {error}"
            ) from error

    def train(self, config, *, warm_start_checkpoint: str | None = None):
        """Train the Coqui model, optionally restoring a model checkpoint."""
        dataset_cfg = config.dataset_config
        model_cfg = config.model_config
        training_cfg = config.training_config
        spec = get_model_spec(model_cfg.type)
        self.logger.info("Preparing Coqui %s training", model_cfg.type)

        # nested Coqui configs are given as mappings in model.params
        params = dict(model_cfg.params)
        audio_params = params.pop("audio", {})
        if "characters" in params:
            params["characters"] = self._build_nested(
                CharactersConfig, params["characters"], "characters", model_cfg.type
            )

        # build coqui audio config
        coqui_audio_cfg = self._build_nested(
            BaseAudioConfig,
            {
                "sample_rate": model_cfg.sample_rate,
                "do_trim_silence": True,
                "trim_db": 60,
                "signal_norm": False,
                "mel_fmin": 0.0,
                "mel_fmax": 8000.0,
                "spec_gain": 1.0,
                "log_func": "np.log",
                "ref_level_db": 20,
                "preemphasis": 0.0,
                **audio_params,
            },
            "audio",
            model_cfg.type,
        )

        # build coqui dataset config
        coqui_dataset_cfg = BaseDatasetConfig(
            formatter=dataset_cfg.formatter,
            meta_file_train=dataset_cfg.meta_file_train,
            path=dataset_cfg.path,
        )

        # build coqui model config: architecture defaults, then the generic
        # settings, then model.params so the user always has the final say
        coqui_model_kwargs = dict(spec.defaults)
        coqui_model_kwargs.update(
            audio=coqui_audio_cfg,
            batch_size=training_cfg.batch_size,
            eval_batch_size=training_cfg.eval_batch_size,
            num_loader_workers=0,
            num_eval_loader_workers=0,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=training_cfg.epochs,
            lr=training_cfg.learning_rate,
            text_cleaner="phoneme_cleaners",
            use_phonemes=True,
            phoneme_language="en-us",
            phoneme_cache_path=os.path.join(training_cfg.output_path, "phoneme_cache"),
            print_step=25,
            print_eval=False,
            mixed_precision=False,
            output_path=training_cfg.output_path,
            datasets=[coqui_dataset_cfg],
            eval_split_size=training_cfg.eval_split_size,
            eval_split_max_size=training_cfg.eval_split_max_size,
            run_name="demo_training",
            batch_group_size=5,
        )
        coqui_model_kwargs.update(params)
        coqui_model_cfg = self._build_nested(
            spec.config_cls, coqui_model_kwargs, None, model_cfg.type
        )

        ap = AudioProcessor.init_from_config(coqui_model_cfg)

        tokenizer, config = TTSTokenizer.init_from_config(coqui_model_cfg)

        train_samples, eval_samples = load_tts_samples(
            cast(list[dict], [coqui_dataset_cfg]),
            eval_split=True,
            eval_split_max_size=training_cfg.eval_split_max_size,
            eval_split_size=training_cfg.eval_split_size,
        )
        self.logger.info(
            "Loaded %d training samples and %d evaluation samples",
            len(train_samples),
            len(eval_samples),
        )

        model = spec.model_cls(config, ap=ap, tokenizer=tokenizer)

        trainer = CoquiTrainerClass(
            TrainerArgs(restore_path=warm_start_checkpoint or ""),
            config,
            training_cfg.output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

        self.logger.info("Starting Coqui trainer")
        trainer.fit()
        self.logger.info("Coqui training completed")