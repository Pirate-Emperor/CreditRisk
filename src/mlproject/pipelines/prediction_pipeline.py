import pandas as pd
from src.mlproject.exception import CustomException
import sys,os
from src.mlproject.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print('Loading model and preprocessor...')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # Align the features to match the order expected by the preprocessor
            expected_features = preprocessor.get_feature_names_out()
            features = features[expected_features]
            
            print('Transforming data...')
            data_scaled = preprocessor.transform(features)
            
            print('Predicting...')
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(
        self,
        status: int,
        duration: int,
        credit_history: int,
        purpose: int,
        amount: int,
        savings: int,
        employment_duration: int,
        installment_rate: int,
        personal_status_sex: int,
        other_debtors: int,
        present_residence: int,
        property: int,
        age: int,
        other_installment_plans: int,
        housing: int,
        number_credits: int,
        job: int,
        people_liable: int,
        telephone: int,
        foreign_worker: int,
    ):

        try:
            self.status = status
            self.duration = duration
            self.credit_history = credit_history
            self.purpose =  purpose
            self.amount = amount
            self.savings = savings
            self.employment_duration =  employment_duration
            self.installment_rate =  installment_rate
            self.personal_status_sex = personal_status_sex
            self.other_debtors = other_debtors
            self.present_residence = present_residence
            self.property = property
            self.age = age
            self.other_installment_plans =  other_installment_plans
            self.housing = housing
            self.number_credits = number_credits
            self.job = job
            self.people_liable = people_liable
            self.telephone = telephone
            self.foreign_worker = foreign_worker

        except KeyError as e:
            print(f"Error mapping value: {e}")
            raise CustomException(f"Error mapping value: {e}", sys)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "status": [self.status],
                "duration": [self.duration],
                "credit_history": [self.credit_history],
                "purpose": [self.purpose],
                "amount": [self.amount],
                "savings": [self.savings],
                "employment_duration": [self.employment_duration],
                "installment_rate": [self.installment_rate],
                "personal_status_sex": [self.personal_status_sex],
                "other_debtors": [self.other_debtors],
                "present_residence": [self.present_residence],
                "property": [self.property],
                "age": [self.age],
                "other_installment_plans": [self.other_installment_plans],
                "housing": [self.housing],
                "number_credits": [self.number_credits],
                "job": [self.job],
                "people_liable": [self.people_liable],
                "telephone": [self.telephone],
                "foreign_worker": [self.foreign_worker],
            }

            # Create DataFrame and reorder columns
            df = pd.DataFrame(custom_data_input_dict)
            return df

        except Exception as e:
            raise CustomException(e, sys)
