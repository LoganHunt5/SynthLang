import logging
from pathlib import Path


def configure_logger(file_path: str = "") -> logging.Logger:
    logger = logging.getLogger("training_pipeline")
    logger.setLevel(logging.DEBUG)  # Let handlers apply their own thresholds.
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if file_path:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger