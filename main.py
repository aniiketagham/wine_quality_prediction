from src.wine_quality import logger
from src.wine_quality.pipelines.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.wine_quality.pipelines.data_validation_pipeline import DataValidationTrainingPipeline
from src.wine_quality.pipelines.data_transformation_pipeline import DataTransformationTrainingPipeline

STAGE_NAME= "Data Ingestion Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME= "Data Validation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_ingestion()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME= "Data Transformation Stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e