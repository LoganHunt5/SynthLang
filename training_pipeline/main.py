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
    #config_manager.load_config(config_path)

    #dataset_manager = dataset_manager.DatasetManager()
    #samples = dataset_manager.load_dataset(config_manager.DatasetConfig)
    #samples = dataset_manager.preprocess_text(samples)

    #trainer = CoquiTrainer(logger)
    #trainer.train(config_manager)
if __name__ == "__main__":
    main()
