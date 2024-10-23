from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
import sys

class TrainPipeline:
    def __init__(self):
        logging.info("Initializing TrainPipeline")
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer=ModelTrainer()
    def run_pipeline(self):
        try:
            # Step 1: Data Ingestion
            logging.info("Starting data ingestion")
            train_data_path, test_data_path = self.data_ingestion.initate_data_ingestion()
            logging.info("Data ingestion completed")
            
            # Step 2: Data Transformation
            logging.info("Started data transformation")
            train_array, test_array,_= self.data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
            logging.info("Data Transformation completed")

            #step 3:Model Trainer
            logging.info("Started model trainer")
            model_score=self.model_trainer.initiate_model_trainer(train_arr=train_array,test_arr=test_array)
            logging.info(f'Model Trainer Completed with model score{model_score}')
            
        except Exception as e:
            logging.error("Exception occurred during the pipeline execution")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()
