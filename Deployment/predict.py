import os
import pickle

import pandas as pd
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE_DIR, 'Models')

transformer_path = os.path.join(MODELS_PATH, 'preprocessor.bin')
if not os.path.exists(transformer_path):
    raise FileNotFoundError(f"Missing transformer at: {transformer_path}")

with open(transformer_path, 'rb') as f:
    transformer = pickle.load(f)


def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
    return transformer.transform(features)


def predict(X_test: pd.DataFrame) -> list:
    model_path = os.path.join(MODELS_PATH, 'model.xgb')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at: {model_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    features = preprocess_features(X_test)
    preds = model.predict(features)
    return preds.tolist()
