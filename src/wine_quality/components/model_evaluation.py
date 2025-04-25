import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from pathlib import Path
from mlflow.models.signature import infer_signature
from typing import Tuple

from src.wine_quality.entity.config_entity import ModelEvaluationConfig
from src.wine_quality.constants import *
from src.wine_quality.utils.common import read_yaml, create_directories, save_json

# Set MLflow tracking credentials
os.environ["MLFLOW_TRACKING_URL"] = "https://dagshub.com/aniketagham1509/wine_quality_prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "aniketagham1509"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "cbd617ae35cd1768e9e7e7657a853d0ffd61f987"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self._setup_mlflow_experiment()
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment and tracking"""
        try:
            # Verify connection
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            
            # Create experiment if it doesn't exist
            experiment_name = "WineQuality"
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                print(f"Creating new experiment: {experiment_name}")
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    tags={"version": "v1", "project": "wine-quality"}
                )
            else:
                experiment_id = experiment.experiment_id
                
            mlflow.set_experiment(experiment_name)
            
            print(f"MLflow experiment setup complete. Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Current experiment: {mlflow.get_experiment(experiment_id)}")
            
        except Exception as e:
            print(f"Failed to setup MLflow experiment: {e}")
            raise
    
    def eval_metrics(self, actual: pd.Series, pred: pd.Series) -> Tuple[float, float, float]:
        """
        Calculate evaluation metrics
        Args:
            actual: Actual target values
            pred: Predicted values
        Returns:
            Tuple of (rmse, mae, r2)
        """
        rmse = np.sqrt(mean_absolute_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):
        """
        Log model, metrics and parameters to MLflow
        """
        try:
            # Load data and model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)
            
            # Prepare test data
            test_x = test_data.drop([self.config.target_column], axis=1)
            test_y = test_data[self.config.target_column]
            
            # Start MLflow run
            with mlflow.start_run():
                print("MLflow run started...")
                
                # Get predictions
                predicted_qualities = model.predict(test_x)
                
                # Calculate metrics
                rmse, mae, r2 = self.eval_metrics(test_y, predicted_qualities)
                print(f"Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
                
                # Save metrics locally
                scores = {"rmse": rmse, "mae": mae, "r2": r2}
                save_json(path=Path(self.config.metric_file_name), data=scores)
                
                # Log parameters and metrics
                mlflow.log_params(self.config.all_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Create model signature and input example
                signature = infer_signature(test_x, predicted_qualities)
                input_example = test_x.iloc[0:1]  # First row as example
                
                # Determine tracking store type
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                
                # Model registry
                if tracking_url_type_store != "file":
                    print("Registering model in MLflow Model Registry...")
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        registered_model_name="ElasticnetModel",
                        signature=signature,
                        input_example=input_example,
                        metadata={
                            "description": "Wine quality prediction model",
                            "dataset": "wine-quality",
                            "model_type": "ElasticNet"
                        }
                    )
                else:
                    print("Logging model locally...")
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path="model",
                        signature=signature,
                        input_example=input_example
                    )
                
                print("Model successfully logged to MLflow")
                
        except Exception as e:
            print(f"Error in MLflow logging: {e}")
            raise