<<<<<<< HEAD
from pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
=======
from anyio import Path

from training_pipeline.pipeline import config_manager, trainer
from training_pipeline.pipeline.coqui_trainer import CoquiTrainer
from training_pipeline.pipeline.logger import configure_logger
import logging
def main():
    print("Hello from training-pipeline!")
    
    logger = configure_logger(
        Path(__file__).resolve().parent / "logs" / "training_pipeline.log"
    )
    #config_manager = config_manager.ConfigManager()
    #config_manager.load_config(config_path) # define path to config file

    #dataset_manager = dataset_manager.DatasetManager()
    #samples = dataset_manager.load_dataset(config_manager.DatasetConfig)
    #samples = dataset_manager.preprocess_text(samples)

    #trainer = CoquiTrainer(logger)
    #trainer.train(config_manager)
if __name__ == "__main__":
    main()
>>>>>>> f782fc3 (Changed logger to use the built-in python logging functions. Added potential to run config_manager in main and to run logger on trainer in main file.)
