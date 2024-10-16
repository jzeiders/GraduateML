import os
import sys
import time
import pandas as pd
import numpy as np
import json
import warnings
import importlib.util
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, ElasticNet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")


def load_config(config_dir):
    spec = importlib.util.spec_from_file_location(
        "config_module", config_dir + "/config.py"
    )
    if spec is None:
        raise FileNotFoundError(f"Config file not found at {config_dir}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def create_pipeline(model, numerical_features, categorical_features):
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

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline


def weighted_mae(y_true, y_pred, holiday_weights):
    weights = np.where(holiday_weights, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)


def process_fold(fold_path, test_labels, config):
    print(f"Processing {fold_path}")

    try:
        # Load and preprocess training data
        train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
        train_df["Date"] = pd.to_datetime(train_df["Date"])

        # Prepare features and target for training
        X_train = train_df.drop(["Weekly_Sales", "Date"], axis=1)
        y_train = train_df["Weekly_Sales"]

        numerical_features = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X_train.select_dtypes(
            include=["object"]
        ).columns.tolist()
        

        # Create model based on config
        model_config = config["model"]
        
        print(config)

        if model_config["type"] == "ElasticNet":
            model = ElasticNet(**model_config["params"])
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")

        # Create and train pipeline
        pipeline = create_pipeline(model, numerical_features, categorical_features)
        pipeline.fit(X_train, y_train)

        # Load and preprocess test data
        test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        # Prepare features for prediction
        X_test = test_df.drop(["Date"], axis=1)

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

    except Exception as e:
        print(f"Error processing {fold_path}: {str(e)}")
        return {"fold": os.path.basename(fold_path), "error": str(e)}


def main(config_path):
    config_path = sys.argv[1]
    data_dir = "data"

    config = load_config(config_path)

    results = []

    # Load the test labels once
    test_labels_path = data_dir + "/" + "test_with_label.csv"
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"test_with_label.csv not found at {test_labels_path}")

    test_labels = pd.read_csv(test_labels_path)
    test_labels["Date"] = pd.to_datetime(test_labels["Date"])

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
    results_df.to_csv(os.path.join(config_path, "aggregated_results.csv"), index=False)
    print(
        f"Aggregated results saved to {os.path.join(config_path, 'aggregated_results.csv')}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_path>")
    else:
        config_path = sys.argv[1]
        main(config_path)
