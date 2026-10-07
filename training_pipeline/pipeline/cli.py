"""Command-line entry point for the TTS training pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from pipeline.coqui_trainer import CoquiTrainer
from pipeline.config_manager import ConfigManager
from pipeline.logger import configure_logger
from pipeline.trainer import Trainer


TrainerFactory = Callable[[logging.Logger], Trainer]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a TTS model from a YAML configuration file."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the training configuration YAML file.",
    )
    return parser


def run(
    config_path: str | Path,
    *,
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

    trainer_factory = trainer_factory or (lambda current_logger: CoquiTrainer(current_logger))
    trainer = trainer_factory(logger)
    warm_start = config_manager.training_config.warm_start_enabled
    checkpoint = config_manager.training_config.warm_start_checkpoint_path
    if warm_start and not checkpoint:
        raise ValueError(
            "warm_start.checkpoint_path must be provided when warm_start.enabled is true"
        )

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
        return run(args.config)
    except (OSError, RuntimeError, ValueError) as error:
        logging.getLogger("training_pipeline").error("Training failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
