import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_feature = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps={
                    ("imputer", SimpleImputer(strategy="mode")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())
                }
            )

            logging.info(f"Numerical Columns: {numerical_features}")
            logging.info(f"Categorical Columns: {categorical_feature}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_feature)
                ]
            )

            return preprocessor
        

        except Exception as e:
            CustomException(e, sys)

    def initate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(r"/home/ragnar-lothbrok/Documents/Ml_project/artifacts/train.csv")
            test_df = pd.read_csv(r"/home/ragnar-lothbrok/Documents/Ml_project/artifacts/test.csv")

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformer_object()
            
            target_column_name = "math_score"
            numerical_features = ['writing_score', 'reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            

        except Exception as e:
            CustomException(e,sys)