import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model = load_object(file_path=self.model_path)
            self.preprocessor = load_object(file_path=self.preprocessor_path)
            print("Model and Preprocessor loaded successfully")
        except Exception as e:
            print(f"Error loading model or preprocessor: {e}")
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        try:
            print("Features before scaling:", features)
            data_scaled = self.preprocessor.transform(features)
            print("Features after scaling:", data_scaled)
            preds = self.model.predict(data_scaled)
            print("Predictions:", preds)
            return preds
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            df = pd.DataFrame(custom_data_input_dict)
            print("DataFrame created from custom data:", df)
            return df
        except Exception as e:
            print(f"Error creating DataFrame from custom data: {e}")
            raise CustomException(e, sys)
