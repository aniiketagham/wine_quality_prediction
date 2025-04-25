from src.wine_quality.config.configuration import ConfigurationManager
from src.wine_quality.components.model_trainer import ModelTrainer
from src.wine_quality import logger

STAGE_NAME= "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_training(self):
        config=ConfigurationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer=ModelTrainer(config=model_trainer_config)
        model_trainer.train()