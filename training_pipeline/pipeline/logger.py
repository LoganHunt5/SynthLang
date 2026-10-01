<<<<<<< HEAD
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
=======
from logging import getLogger, StreamHandler, FileHandler, Formatter
class Logger:
    def __init__(self, console_enabled, file_path="", console_level="INFO", file_level="INFO"):
        self.console_enabled = console_enabled #bool
        self.file_path = file_path
        self.console_level = console_level # determines what errors the console recieves
        self.file_level = file_level # determines what errors the log file receives
        '''
        INFO is the default because it sends INFO, WARNING, ERROR and CRITICAL messages
        but doesn't have DEBUG. But this is probably the most we will ever need.
        '''
        
        self.logger = getLogger(__name__)
        if self.console_enabled:
            console_handler = StreamHandler()
            console_handler.setLevel(self.console_level)
            console_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(console_handler)
        if self.file_path:
            file_handler = FileHandler(self.file_path)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(file_handler)

    def log_info(self, message: str):
        print(f"[INFO] {message}")


    def log_success(self, message: str):
        print(f"\033[92m[SUCCESS]\033[0m {message}")


    def log_error(self, message: str):
        print(f"\033[91m[ERROR]\033[0m {message}")
>>>>>>> 5a15c8d (Added a logger file just to get started on that.)
