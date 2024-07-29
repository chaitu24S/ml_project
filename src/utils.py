import os
import sys

import numpy as np
import pandas as pd
import dill  # or pickle if dill is not necessary
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging  # Ensure you have a logging setup in src.logger

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = {
                "train_score": train_model_score,
                "test_score": test_model_score,
                "best_params": gs.best_params_
            }

            logging.info(f"Completed evaluation for model: {model_name}")

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
