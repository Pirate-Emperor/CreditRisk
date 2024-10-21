import os ,sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.mlproject.utils import load_data_from_cassandra,rename_columns
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        
    def initate_data_ingestion(self):
    
        try:
            
            logging.info("Starting Data Ingestion")
            
            logging.info("started Reading data from database")          
            # Define your query
            query = "SELECT * FROM bank_credit_new"

            # Load the data
            data = load_data_from_cassandra(query)

            # Proceed with your data processing
            logging.info(f"{data.head()}")
            logging.info(f"{data.columns}")
            logging.info("Reading completed from database")
            
            df= data.drop(columns='id')
            
            logging.info("Renaming columns from German to English")
            
            english_column_names = ['age', 'job', 'employment_duration', 'number_credits', 'other_debtors', 
                                    'personal_status_sex', 'foreign_worker', 'amount', 
                                    'credit_risk', 'status', 'duration', 
                                    'credit_history', 'people_liable', 'installment_rate', 'savings', 
                                    'telephone', 'property', 'purpose', 'other_installment_plans', 
                                    'housing', 'present_residence']
            
            df = rename_columns(df, english_column_names)
            
            logging.info("Renaming of cloumns completed")
            
            target_column='credit_risk'
            columns=[col for col in df.columns if col !=target_column] +[target_column] 
            df=df[columns]
            
            logging.info("Making the target column to last of table/data")
            
            
            logging.info(f"Dataset shape: {df.shape}")   
            logging.info(f"{df.head()}")

            logging.info("Creating artifacts folder")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            
            logging.info("creation artifacts folder completed")
            
            logging.info("Saving data to artifacts folder as raw.csv" )
            
            df.to_csv(self.data_ingestion_config.raw_data_path , index=False , header=True)

            logging.info("Saved the raw.csv in artifacts folder successfully")
            
            logging.info("splitting data into train and test split")
            
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            logging.info("Done the Train and Test Split")
            logging.info(f"Training set shape: {train_set.shape}")
            logging.info(f"Test set shape: {test_set.shape}")
            
            logging.info("Saving train_set and test_set in artifacts folder as train.csv and test.csv" )
                
            train_set.to_csv(self.data_ingestion_config.train_data_path , index=False , header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path , index=False , header=True)
            
            logging.info("Saved Succeddfully train_set and test_set in artifacts folder as train.csv and test.csv" )
            
            logging.info("Data Ingesiton is Completed")

            logging.info(f"Saved file path {self.data_ingestion_config.raw_data_path }")
            logging.info(f"Saved file path {self.data_ingestion_config.train_data_path }")
            logging.info(f"Saved file path {self.data_ingestion_config.test_data_path }")
            
            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path
             
        
        except Exception as e:
            raise CustomException(e,sys)