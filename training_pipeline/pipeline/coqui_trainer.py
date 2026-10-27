import logging
from pipeline.trainer import Trainer as BaseTrainer
from pipeline.config_manager import ConfigManager
import os
from trainer import Trainer as CoquiTrainerClass, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from typing import cast


class CoquiTrainer(BaseTrainer):
    """
    Trainer class for Coqui models. Inherits from the base Trainer class.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def train(self, config, *, warm_start_checkpoint: str | None = None):
        """Train the Coqui model, optionally restoring a model checkpoint."""
        dataset_cfg = config.dataset_config
        model_cfg = config.model_config
        training_cfg = config.training_config
        self.logger.info("Preparing Coqui training")

        # build coqui audio config
        coqui_audio_cfg =  BaseAudioConfig(
            sample_rate=model_cfg.sample_rate,
            do_trim_silence=True,
            trim_db=60,
            signal_norm=False,
            mel_fmin=0.0,
            mel_fmax=8000.0,
            spec_gain=1.0,
            log_func="np.log",
            ref_level_db=20,
            preemphasis=0.0,
        )

        # build coqui dataset config
        coqui_dataset_cfg = BaseDatasetConfig(
            formatter=dataset_cfg.formatter,
            meta_file_train=dataset_cfg.meta_file_train,
            path=dataset_cfg.path,
        )

        # build coqui model config
        coqui_model_cfg = Tacotron2Config(
            audio=coqui_audio_cfg,
            batch_size=training_cfg.batch_size,
            eval_batch_size=training_cfg.eval_batch_size,
            num_loader_workers=0,
            num_eval_loader_workers=0,
            run_eval=True,
            test_delay_epochs=-1,
            ga_alpha=0.0,
            decoder_loss_alpha=0.25,
            postnet_loss_alpha=0.25,
            postnet_diff_spec_alpha=0.0,
            decoder_diff_spec_alpha=0.0,
            decoder_ssim_alpha=0.0,
            postnet_ssim_alpha=0.0,
            r = 2,
            attention_type="dynamic_convolution",
            double_decoder_consistency=False,
            epochs=training_cfg.epochs,
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
            max_decoder_steps=500, # CHANGE TO WHAT RECIPE RECOMMENDS WHEN NOT DEMO
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

        model = Tacotron2(config, ap=ap, tokenizer=tokenizer)

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