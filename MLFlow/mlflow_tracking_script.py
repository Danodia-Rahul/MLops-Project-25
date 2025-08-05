import pickle

import boto3
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow import MlflowClient
from mlflow.entities import ViewType
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

MLFLOW_TRACKING_URI = (
    'http://ec2-56-228-36-113.eu-north-1.compute.amazonaws.com:5000'
)

"""### Preprocess data"""


def dump_pickle(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out)


def prepare_data(path='/content/train.csv'):
    target = 'Premium Amount'
    features = [
        'Age',
        'Annual Income',
        'Number of Dependents',
        'Occupation',
        'Credit Score',
        'Property Type',
    ]

    df = pd.read_csv(path)
    df = df[features + [target]].copy()

    categorical = df.select_dtypes(include=['object']).columns.tolist()

    cat_col_transformer = Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(strategy='constant', fill_value='missing'),
            ),
            (
                'encoder',
                OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[('cat', cat_col_transformer, categorical)],
        remainder='passthrough',
    )

    transformed = transformer.fit_transform(df[features])

    with open('preprocessor.bin', 'wb') as f_out:
        pickle.dump(transformer, f_out)

    X_train, X_temp, y_train, y_temp = train_test_split(
        transformed, df[target], test_size=0.3
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5
    )

    dump_pickle((X_train, y_train), 'train.pkl')
    dump_pickle((X_valid, y_valid), 'valid.pkl')
    dump_pickle((X_test, y_test), 'test.pkl')


prepare_data()


def load_file(file_path):
    with open(file_path, 'rb') as f_in:
        return pickle.load(f_in)


"""### Hyper parameter tuning"""


def hyper_parameter_tuning(num_trials=10):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment('XGBoost Tuning')

    X_train, y_train = load_file('train.pkl')
    X_valid, y_valid = load_file('valid.pkl')
    X_test, y_test = load_file('test.pkl')

    def objective(params):

        with mlflow.start_run():

            mlflow.log_params(params)

            model = xgb.XGBRegressor(**params, device='cuda')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            rmse = root_mean_squared_error(y_valid, y_pred)

            mlflow.log_metric('rmse', rmse)

            return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 100, 2000, 50)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -3, 1),
        'reg_lambda': hp.loguniform('reg_lambda', -3, 1),
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
    )


hyper_parameter_tuning()

"""### Model Registry"""


def register_model():

    HPO_EXPERIMENT_NAME = 'XGBoost Tuning'
    EXPERIMENT_NAME = 'XGB MODELS'

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    def train_and_log_model(params):

        X_train, y_train = load_file('train.pkl')
        X_valid, y_valid = load_file('valid.pkl')
        X_test, y_test = load_file('test.pkl')

        with mlflow.start_run():
            parsed_params = {
                k: int(v) if v.isdigit() else float(v)
                for k, v in params.items()
            }

            model = xgb.XGBRegressor(**parsed_params, device='cuda')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            test_rmse = root_mean_squared_error(y_test, y_pred)
            mlflow.log_metric("test_rmse", test_rmse)

            mlflow.log_artifact('preprocessor.bin')
            mlflow.xgboost.log_model(model, artifact_path='model')

    def run_register_model(top_n=3):

        client = MlflowClient()

        experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.rmse ASC"],
        )
        for run in runs:
            train_and_log_model(run.data.params)

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        best_run = client.search_runs(
            experiment_ids=experiment.experiment_id,
            order_by=['metrics.test_rmse ASC'],
        )[0]

        mlflow.register_model(
            model_uri=f'runs:/{best_run.info.run_id}/model',
            name='BEST XGBOOST MODEL',
        )

    run_register_model()


register_model()
