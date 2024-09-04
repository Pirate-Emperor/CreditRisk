from flask import Flask, render_template, request
import os
import pandas as pd
from src.mlproject.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.mlproject.components.model_trainer import ModelTrainer
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.logger import logging

app = Flask(__name__)

HISTORY_FILE_PATH = 'prediction_history.csv'

# Mappings
status_mapping = {1: 'no checking account', 2: 'less than 0 DM', 3: '0 to 200 DM', 4: '200 DM or more'}
credit_history_mapping = {0: 'delay in paying off in the past', 1: 'critical account/other credits elsewhere', 2: 'no credits taken/all credits paid back duly', 3: 'existing credits paid back duly till now', 4: 'all credits at this bank paid back duly'}
purpose_mapping = {0: 'others', 1: 'car (new)', 2: 'car (used)', 3: 'furniture/equipment', 4: 'radio/television', 5: 'domestic appliances', 6: 'repairs', 7: 'education', 8: 'vacation', 9: 'retraining', 10: 'business'}
savings_mapping = {1: 'unknown/no savings account', 2: 'less than 100 DM', 3: '100 to 500 DM', 4: '500 to 1000 DM', 5: '1000 DM or more'}
employment_duration_mapping = {1: 'unemployed', 2: 'less than 1 year', 3: '1 to 4 yrs', 4: '4 to 7 yrs', 5: '7 yrs or more'}
installment_rate_mapping = {1: '35 or more', 2: '25 to 35', 3: '20 to 25', 4: 'less than 20'}
personal_status_sex_mapping = {1: 'male : divorced/separated', 2: 'female : non-single or male : single', 3: 'male : married/widowed', 4: 'female : single'}
other_debtors_mapping = {1: 'none', 2: 'co-applicant', 3: 'guarantor'}
present_residence_mapping = {1: 'less than 1 year', 2: '1 to 4 yrs', 3: '4 to 7 yrs', 4: '7 yrs or more'}
property_mapping = {1: 'unknown/no property', 2 : 'car or other', 3: 'building soc. savings agr./life insurance', 4 : 'real estate'}
other_installment_options_mapping = {1: 'bank', 2: 'stores', 3: 'none'}
housing_mapping = {1: 'for free', 2: 'rent', 3: 'own'}
job_mapping = {1: 'unemployed/unskilled - non-resident', 2: 'unskilled-resident', 3: 'skilled employee/official', 4: 'manager/self-employed/highly qualified employee'}
number_credits_mapping = {1: '1', 2: '2-3', 3: '4-5', 4: '6 or more'}
people_liable_mapping = {1: '3 or more', 2: '0 to 2'}
telephone_mapping = {1: 'no', 2: 'yes (under customer name)'}
foreign_worker_mapping = { 1 : 'yes', 2: 'no'}

# Reverse Mapping for Display Purposes
reverse_mappings = {
    'status': status_mapping,
    'credit_history': credit_history_mapping,
    'purpose': purpose_mapping,
    'savings': savings_mapping,
    'employment_duration': employment_duration_mapping,
    'installment_rate': installment_rate_mapping,
    'personal_status_sex': personal_status_sex_mapping,
    'other_debtors': other_debtors_mapping,
    'present_residence': present_residence_mapping,
    'property': property_mapping,
    'other_installment_plans': other_installment_options_mapping,
    'housing': housing_mapping,
    'job': job_mapping,
    'number_credits': number_credits_mapping,
    'people_liable': people_liable_mapping,
    'telephone': telephone_mapping,
    'foreign_worker': foreign_worker_mapping
}

@app.route('/')
def index():
    logging.info("Index Page Accessed")
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            logging.info("Training Process Started")
            # Initialize DataTransformation
            data_transformation = DataTransformation()
            
            # Paths to the CSV files
            train_csv_path = os.path.join('artifacts', 'train.csv')
            test_csv_path = os.path.join('artifacts', 'test.csv')
            
            # Transform the data
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path=train_csv_path,
                test_path=test_csv_path
            )
            
            app.logger.info(f"shape of array: {train_arr.shape}")
            app.logger.info(f"Shape of array: {test_arr.shape}")
            
            # Initialize ModelTrainer and train the model
            model_trainer = ModelTrainer()
            accuracy, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
            
            formatted_accuracy = f"{accuracy * 100:.2f}%"
            status = "Training Completed Successfully"        

            logging.info(f"Best Model Trained {best_model}, Accuracy : {formatted_accuracy}")
            
            return render_template('train.html', model=best_model, score=formatted_accuracy, status=status)
    
        except Exception as e:
            logging.info(f"Error during Training {e}")
            return render_template('train.html', score="N/A", model="N/A", status=str(e))       

    else:
        return render_template('train.html', score="N/A", model="N/A", status="")


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('test.html')
    else:
        try:
            logging.info("Prediction Request Received")
            # Capture form data
            data = CustomData(
                status=request.form.get('status'),
                duration=int(request.form.get('duration')),
                credit_history=request.form.get('credit_history'),
                purpose=request.form.get('purpose'),
                amount=int(request.form.get('amount')),
                savings=request.form.get('savings'),
                employment_duration=request.form.get('employment_duration'),
                installment_rate=int(request.form.get('installment_rate')),
                personal_status_sex=request.form.get('personal_status_sex'),
                other_debtors=request.form.get('other_debtors'),
                present_residence=int(request.form.get('present_residence')),
                property=int(request.form.get('property')),
                age=int(request.form.get('age')),
                other_installment_plans=request.form.get('other_installment_plans'),
                housing=request.form.get('housing'),
                number_credits=int(request.form.get('number_credits')),
                job=request.form.get('job'),
                people_liable=int(request.form.get('people_liable')),
                telephone=request.form.get('telephone'),
                foreign_worker=int(request.form.get('foreign_worker'))
            )
            
            # Convert data to DataFrame
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Recevied data for prediction {pred_df}")
            
            # Instantiate the prediction pipeline
            predict_pipeline = PredictPipeline()

            # Perform prediction
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction Result: {results}")

            if results[0] == 0:
                result = "It is not safe to give credit."
            else:
                result = "It is safe to give credit."
              
                    
            # Save prediction to history
            pred_df['prediction'] = result
            save_prediction_to_history(pred_df)

            # Return result to the template
            return render_template("test.html", prediction=result)

        except Exception as e:
            error_message = str(e)
            logging.info(f"Error during prediction : {error_message}")
            return render_template('test.html', prediction="Error occurred: " + error_message)


def save_prediction_to_history(prediction_df):
    
    # Save the prediction along with input data to the history CSV file.
    
    if not os.path.exists(HISTORY_FILE_PATH):
        prediction_df.to_csv(HISTORY_FILE_PATH, index=False)
    else:
        prediction_df.to_csv(HISTORY_FILE_PATH, mode='a', header=False, index=False)


@app.route('/history')
def history_page():
    try:
        logging.info("Accessing prediction history")
        if os.path.exists(HISTORY_FILE_PATH):
            # Load history data
            history_df = pd.read_csv(HISTORY_FILE_PATH)

            # Expected column order
            column_order = [
                'status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
                'employment_duration', 'installment_rate', 'personal_status_sex', 'other_debtors',
                'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
                'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'prediction'
            ]

            # Ensure all expected columns are present in the DataFrame
            for col in column_order:
                if col not in history_df.columns:
                    history_df[col] = "N/A"  # Default value for missing columns

            # Reorder columns according to the expected order
            history_df = history_df[column_order]

            # Apply reverse mappings for categorical columns
            for column, mapping in reverse_mappings.items():
                if column in history_df.columns:
                    history_df[column] = history_df[column].map(mapping)
        
            columns = history_df.columns.tolist()
            rows = history_df.to_dict(orient='records')
        else:
            columns = []
            rows = []

        return render_template('history.html', columns=columns, rows=rows)
    except Exception as e:
        logging.info(f"Error Accessing prediction history : {str(e)}")
        return render_template('history.html', columns=[], rows=[])
        
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

