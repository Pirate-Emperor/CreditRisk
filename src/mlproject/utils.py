import os,sys
from dotenv import load_dotenv
import pickle
from src.mlproject.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from src.mlproject.logger import logging
load_dotenv()


def connect_to_cassandra():
    logging.info("Connecting to Cassandra database")
    try:
        username = os.getenv("CASSANDRA_USER")
        password = os.getenv("CASSANDRA_PASSWORD")
        keyspace = os.getenv("CASSANDRA_KEYSPACE")
        secure_bundle_path = os.getenv("CASSANDRA_SECURE_BUNDLE")

        cloud_config = {
            'secure_connect_bundle': secure_bundle_path
        }
        auth_provider = PlainTextAuthProvider(username=username, password=password)
        cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider , protocol_version=4)
        session = cluster.connect(keyspace)
        
        logging.info("Cassandra connection established")
        return session
    
    except Exception as e:
        logging.error(f"Failed to connect to Cassandra: {e}")
        logging.error(f"CASSANDRA_USER: {username}")
        logging.error(f"CASSANDRA_KEYSPACE: {keyspace}")
        logging.error(f"CASSANDRA_SECURE_BUNDLE: {secure_bundle_path}")
        raise CustomException(e, sys)

def load_data_from_cassandra(query: str) -> pd.DataFrame:
    logging.info("Loading data from Cassandra")
    try:
        session = connect_to_cassandra()
        rows = session.execute(query)
        df = pd.DataFrame(list(rows))
        
        logging.info("Data loaded from Cassandra successfully")
        return df

    except Exception as e:
        raise CustomException(e, sys)

def rename_columns(df: pd.DataFrame, new_column_names: list) -> pd.DataFrame:
    df.columns = new_column_names
    return df

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        
    except Exception as e:
        raise CustomException (e,sys)
    

def evalute_models(X_train,y_train,X_test,y_test,models,param):
    try:
        
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param[list(models.keys())[i]]
            
            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            print(f"y_test[:5]: {y_test[:5]}")
            print(f"y_test_pred[:5]: {y_test_pred[:5]}")
  
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException (e,sys)
    