import os
import sys
import pandas as pd
import numpy as np
import warnings
import importlib.util
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings("ignore")

# Custom Transformers for Feature Engineering

class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Date"] = pd.to_datetime(X["Date"])
        if "year" in self.features:
            X["Year"] = X["Date"].dt.year
        if "month" in self.features:
            X["Month"] = X["Date"].dt.month
        if "weekofyear" in self.features:
            X["WeekOfYear"] = X["Date"].dt.isocalendar().week.astype(int)
        if "dayofweek" in self.features:
            X["DayOfWeek"] = X["Date"].dt.dayofweek
        return X

class HolidayProximityAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.holiday_dates_ = X[X["IsHoliday"]]["Date"].unique()
        return self

    def transform(self, X):
        X = X.copy()
        X["DaysUntilHoliday"] = X["Date"].apply(
            lambda x: min([(h - x).days for h in self.holiday_dates_ if h >= x], default=0)
        )
        X["DaysSinceHoliday"] = X["Date"].apply(
            lambda x: min([(x - h).days for h in self.holiday_dates_ if h <= x], default=0)
        )
        return X

class InteractionFeaturesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["Store_Dept"] = X["Store"].astype(str) + "_" + X["Dept"].astype(str)
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from the DataFrame.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

# Function to load configuration

def load_config(config_dir):
    spec = importlib.util.spec_from_file_location(
        "config_module", os.path.join(config_dir, "config.py")
    )
    if spec is None:
        raise FileNotFoundError(f"Config file not found at {config_dir}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

# Function to create the pipeline dynamically based on config

def create_pipeline(model, config):
    feature_steps = []
    
    # Date Features
    if config["feature_engineering"]["date_features"]["include"]:
        feature_steps.append(
            ("date_features", DateFeaturesAdder(
                features=config["feature_engineering"]["date_features"]["features"]
            ))
        )
    
    # Holiday Proximity
    if config["feature_engineering"]["holiday_proximity"]["include"]:
        feature_steps.append(
            ("holiday_proximity", HolidayProximityAdder())
        )
    
    # Interaction Features
    if config["feature_engineering"]["interaction_features"]["include"]:
        feature_steps.append(
            ("interaction_features", InteractionFeaturesAdder())
        )
    
    # Combine feature engineering steps into a pipeline
    feature_pipeline = Pipeline(feature_steps)
    
    # Preprocessing for numerical and categorical features
    numerical_features = config["preprocessing"]["numerical_features"]
    categorical_features = config["preprocessing"]["categorical_features"]
    
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Combine feature engineering and preprocessing
    full_preprocessor = Pipeline(
        steps=[
            ("feature_engineering", feature_pipeline),
            ("preprocessing", preprocessor)
        ]
    )

    # Create the full pipeline with preprocessing and the model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", full_preprocessor),
            ("model", model)
        ]
    )

    return pipeline

# Custom Weighted MAE Metric

def weighted_mae(y_true, y_pred, holiday_weights):
    weights = np.where(holiday_weights, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

# Function to process each fold

def process_fold(fold_path, test_labels, config):
    print(f"Processing {fold_path}")

    # Load training data
    train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))

    # Define target and features
    X_train = train_df
    y_train = train_df["Weekly_Sales"]

    model_config = config["model"]

    if model_config["type"] == "ElasticNet":
        model = ElasticNet(**model_config["params"])
    elif model_config["type"] == "ElasticNetCV":
        model = ElasticNetCV(**model_config["params"])
    elif model_config["type"] == "RandomForestRegressor":
        model = RandomForestRegressor(**model_config["params"])
    elif model_config["type"] == "XGBRegressor":
        model = XGBRegressor(**model_config["params"])
    else:
        raise ValueError(f"Unsupported model type: {model_config['type']}")

    # Create and train pipeline
    pipeline = create_pipeline(model, config)
    pipeline.fit(X_train, y_train)

    # Load test data
    test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))

    X_test = test_df

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Prepare submission dataframe
    submission_df = test_df[["Store", "Dept", "Date", "IsHoliday"]].copy()
    submission_df["Weekly_Pred"] = predictions

    # Calculate weighted MAE using the test labels
    merged_df = pd.merge(
        submission_df, test_labels, on=["Store", "Dept", "Date", "IsHoliday"]
    )

    if len(merged_df) != len(submission_df):
        print(
            f"Warning: Mismatch in number of samples. Predictions: {len(submission_df)}, Labels: {len(merged_df)}"
        )

    wmae = weighted_mae(
        merged_df["Weekly_Sales"], merged_df["Weekly_Pred"], merged_df["IsHoliday"]
    )

    return {
        "fold": os.path.basename(fold_path),
        "num_train_samples": len(train_df),
        "num_test_samples": len(test_df),
        "weighted_mae": wmae,
    }


# Main Function

def main(config_path):
    config = load_config(config_path)
    data_dir = "data"

    results = []

    # Load the test labels once
    test_labels_path = os.path.join(data_dir, "test_with_label.csv")
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"test_with_label.csv not found at {test_labels_path}")

    test_labels = pd.read_csv(test_labels_path)

    for fold in sorted(os.listdir(data_dir)):
        fold_path = os.path.join(data_dir, fold)
        if os.path.isdir(fold_path) and fold.startswith("fold_"):
            result = process_fold(fold_path, test_labels, config)
            results.append(result)

    # Aggregate results into a DataFrame
    results_df = pd.DataFrame(results)
    print("\nAggregated Results:")
    print(results_df)

    # Save aggregated results
    aggregated_results_path = os.path.join(config_path, "aggregated_results.csv")
    results_df.to_csv(aggregated_results_path, index=False)
    print(f"Aggregated results saved to {aggregated_results_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_path>")
    else:
        config_path = sys.argv[1]
        main(config_path)
