import os
import sys
import time
import pandas as pd
import numpy as np
import json
import warnings
import importlib.util
from shutil import copyfile
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function to load Python configuration files
def load_config(config_dir):
    spec = importlib.util.spec_from_file_location("config_module", config_dir + "/config.py")
    if spec is None:
        raise FileNotFoundError(f"Config file not found at {config_dir}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

# Log-transform helper
log_transform = FunctionTransformer(func=np.log, validate=True)

# Outlier capper class
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, X, y=None):
        self.lower_bound = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound = np.quantile(X, self.upper_quantile, axis=0)
        return self
    
    def transform(self, X):
        X_capped = np.clip(X, self.lower_bound, self.upper_bound)
        return X_capped

# Function to create the pipeline
def create_pipeline(model, numerical_features, categorical_features, encoding):
    if encoding == 'onehot':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('onehot', OneHotEncoder(use_cat_names=True))
        ])
    elif encoding == 'target':
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('target_enc', TargetEncoder())
        ])
    else:
        raise ValueError("Unsupported encoding type. Choose 'onehot' or 'target'.")
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('outlier', OutlierCapper())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return pipeline

# Main function to run the model
def model(config, config_dir, data_dir):
    np.random.seed(42)
    encoding = config.get('encoding', 'onehot')

    # Feature engineering parameters
    manual_drop = config['feature_engineering']['drop_columns']
    highly_correlated = config['feature_engineering']['highly_correlated']
    potential_non_linear = config['feature_engineering']['potential_non_linear']
    sparse_categories = config['feature_engineering']['sparse_categories']
    numeric_as_categorical = config['feature_engineering'].get('numeric_as_categorical', [])

    DROP_COLS = ['PID'] + highly_correlated + potential_non_linear + sparse_categories + manual_drop

    # Load training data
    train_path = os.path.join(data_dir, 'train.csv')
    train = pd.read_csv(train_path).drop(columns=DROP_COLS)

    # Separate response variable and features
    y = train['Sale_Price']
    X = train.drop(['Sale_Price'], axis=1)

    # Set difference for numerical columns
    numerical_cols = list(set(X.select_dtypes(include=['int64', 'float64']).columns.tolist()) - set(numeric_as_categorical))

    # Combine categorical columns with numeric columns to be treated as categorical
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist() + numeric_as_categorical

    # Create pipelines for models defined in config
    models = {}
    for model_name, model_info in config['models'].items():
        model_type = model_info['type']
        model_params = model_info['params']

        if model_type == 'ElasticNetCV':
            regressor = ElasticNetCV(**model_params)
        elif model_type == 'ElasticNet':
            regressor = ElasticNet(**model_params)
        elif model_type == 'XGBRegressor':
            regressor = XGBRegressor(**model_params)
        elif model_type == 'RandomForestRegressor':
            regressor = RandomForestRegressor(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model_instance = TransformedTargetRegressor(
            regressor=regressor,
            transformer=log_transform
        )

        pipeline = create_pipeline(
            model=model_instance,
            numerical_features=numerical_cols,
            categorical_features=categorical_cols,
            encoding=encoding
        )
        models[model_name] = pipeline

    # Logging setup
    experiment_log = {
        'encoding': encoding,
        'feature_engineering': config['feature_engineering'],
        'models': config['models'],
        'metrics': {}
    }

    # Train models and log results
    for name, pipeline in models.items():
        start_time = time.perf_counter()
        pipeline.fit(X, y)
        train_time = time.perf_counter() - start_time
        print(f"{name} model trained in {train_time:.2f} seconds.")
        experiment_log['metrics'][name] = {'train_time': train_time}

    test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'test.csv')).drop(columns=DROP_COLS, errors='ignore')

    for name, pipeline in models.items():
        preds = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(np.log(test_y['Sale_Price']), preds))
        experiment_log['metrics'][name]['RMSE'] = round(rmse, 5)

    # Save experiment log
    log_file = os.path.join(config_dir, 'experiment_log.json')
    with open(log_file, 'w') as f:
        json.dump(experiment_log, f, indent=4)

# Function to aggregate results across experiments
def aggregate_results(results_root_dir):
    experiment_logs = []
    for exp_dir in os.listdir(results_root_dir):
        exp_path = os.path.join(results_root_dir, exp_dir)
        log_file = os.path.join(exp_path, 'experiment_log.json')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log = json.load(f)
                experiment_logs.append(log)

    # Flatten the logs into a DataFrame
    records = []
    for log in experiment_logs:
        exp_name = log['experiment_name']
        for model_name, metrics in log['metrics'].items():
            record = {
                'experiment_name': exp_name,
                'model_name': model_name,
                'RMSE': metrics.get('RMSE'),
                'train_time': metrics.get('train_time'),
                'encoding': log['encoding'],
                'data_dir': log['data_dir'],
            }
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(results_root_dir, 'aggregated_results.csv'), index=False)
    print(df)

# Main entry point
if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "aggregate":
        aggregate_results("experiments")
    elif len(sys.argv) == 2:
        config_path = sys.argv[1]
        config = load_config(config_path)

        # Get all subdirectories in the data folder (e.g., fold1, fold2, ..., fold10)
        data_root = 'data'
        data_folders = [f.path for f in os.scandir(data_root) if f.is_dir()]

        all_rmses = []

        # Loop through each fold directory (e.g., data/fold1)
        for data_dir in data_folders:
            print(f"Running experiment on {data_dir}...")
            model(config, config_path, data_dir)

            # After running the model, load the RMSE from the log for that fold
            log_file = os.path.join(config_path, 'experiment_log.json')

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    experiment_log = json.load(f)
                    for model_name, metrics in experiment_log['metrics'].items():
                        rmse = metrics.get('RMSE')
                        train_time = metrics.get('train_time')
                        if rmse is not None and train_time is not None:
                            all_rmses.append({
                                'fold': data_dir, 
                                'model': model_name, 
                                'RMSE': rmse,
                                'Train Time (seconds)': train_time
                            })

        # Print out all RMSEs for each fold
        if all_rmses:
            
            final_results = pd.DataFrame(all_rmses).sort_values(by='RMSE')
            final_results['fold_num'] = final_results['fold'].str.extract(r'fold(\d+)').astype(int)
            final_results = final_results.sort_values(by='fold_num').drop(columns='fold')

            # Save as a markdown table
            with open(config_path + "/" + "results.md", "w") as f:
                f.write(final_results.to_markdown(index=False))
        else:
            print("No RMSE results found.")
    else:
         print("Usage: python script.py <config_dir> or python script.py aggregate")
