"""Command-line entry point for the TTS training pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pipeline.config_manager import ConfigManager
from pipeline.logger import configure_logger
from pipeline.registry import TrainerFactory, get_trainer_factory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TTS model from a YAML configuration file."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="RUN_DIR",
        help="Continue an interrupted run from its output folder (the folder "
        "with config.json and checkpoints). The run's saved settings are used "
        "and warm_start is ignored.",
    )
    return parser


def run(
    config_path: str | Path,
    *,
    resume_path: str | Path | None = None,
    trainer_factory: TrainerFactory | None = None,
    logger: logging.Logger | None = None,
) -> int:
    """Load a training configuration and run the selected trainer."""
    config_file = Path(config_path)
    config_manager = ConfigManager()
    config_manager.load_config(str(config_file))

    if logger is None:
        logger = configure_logger(
            Path(__file__).resolve().parent.parent / "logs" / "training_pipeline.log"
        )

    trainer_factory = trainer_factory or get_trainer_factory(config_manager.backend)
    trainer = trainer_factory(logger)
    warm_start = config_manager.training_config.warm_start_enabled
    checkpoint = config_manager.training_config.warm_start_checkpoint_path
    if warm_start and not checkpoint:
        raise ValueError(
            "warm_start.checkpoint_path must be provided when warm_start.enabled is true"
        )

    if resume_path is not None:
        resume_dir = Path(resume_path)
        if not (resume_dir / "config.json").is_file():
            raise ValueError(
                f"Cannot resume from {resume_dir}: expected a training run folder "
                "containing config.json"
            )
        if not any(resume_dir.glob("*.pth")):
            raise ValueError(f"Cannot resume from {resume_dir}: no checkpoints found")
        logger.info("Resuming training run %s (config %s)", resume_dir, config_file)
        trainer.train(config_manager, resume_path=str(resume_dir))
        logger.info("Training pipeline completed")
        return 0

    logger.info(
        "Starting %s training from %s",
        "warm-start" if warm_start else "fresh",
        config_file,
    )
    trainer.train(config_manager, warm_start_checkpoint=checkpoint or None)
    logger.info("Training pipeline completed")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return run(args.config, resume_path=args.resume)
    except (OSError, RuntimeError, ValueError) as error:
        # include the traceback: Coqui's cleanup after a failure can raise its
        # own error (e.g. WinError 32 on Windows), and only the chained
        # traceback shows the original cause
        logging.getLogger("training_pipeline").error(
            "Training failed: %s", error, exc_info=True
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
