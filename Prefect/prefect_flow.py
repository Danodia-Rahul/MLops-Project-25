import os
import pickle

import pandas as pd
import xgboost as xgb
from prefect import flow, task
from sklearn.metrics import root_mean_squared_error


@task
def load_preprocessor(path='Models/preprocessor.bin'):
    with open(path, 'rb') as f:
        return pickle.load(f)


@task
def load_model(model_path='Models/model.xgb'):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


@task
def load_data(path='Data/train.csv'):

    df = pd.read_csv(path)
    features = [
        'Age',
        'Annual Income',
        'Number of Dependents',
        'Occupation',
        'Credit Score',
        'Property Type',
    ]

    target = 'Premium Amount'

    return df[features], df[target]


@task
def evaluate(preds, y_test):
    rmse = root_mean_squared_error(y_test, preds)
    print(f"RMSE: {rmse}")
    return rmse


@flow(log_prints=True)
def run():
    transformer = load_preprocessor()
    xgb_model = load_model()
    X_test, y_test = load_data()

    x = transformer.transform(X_test)
    preds = xgb_model.predict(x)

    rmse = evaluate(preds, y_test)
    print(f'RMSE: {rmse}')


if __name__ == "__main__":
    run()
