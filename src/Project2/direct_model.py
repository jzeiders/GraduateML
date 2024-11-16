import os
import sys
from typing import Dict, Set, Tuple
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
warnings.filterwarnings("ignore")

class GlobalSalesModel:
    def __init__(self):
            # Define categorical and numeric features
            self.categorical_features = ['WeekOfYear', 'DayOfWeek', 'HolidayCategory']
            self.numeric_features = ['Year']
            self.interaction_features = ['Store', 'Dept']
            
            # Create main preprocessor
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', self.numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), 
                     self.categorical_features),
                    ('store_dept', Pipeline([
                        ('onehot', OneHotEncoder(handle_unknown='ignore')),
                        ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
                    ]), self.interaction_features)
                ])
            
            self.model = LinearRegression(fit_intercept=False)
        
    def transform_input(self, df):
        # Create base features DataFrame
        X = df[["Store", "Dept"]].copy()
        Dates = pd.to_datetime(df["Date"])
        
        # Add temporal features
        X["Year"] = Dates.dt.year.astype(int)
        X["WeekOfYear"] = Dates.dt.isocalendar().week.astype(str)
        X["DayOfWeek"] = Dates.dt.dayofweek.astype(str)
        
        # Define holiday dates
        holidays = {
            "Super Bowl": ["2010-02-07", "2011-02-06", "2012-02-05", "2013-02-03"],
            "Labor Day": ["2010-09-06", "2011-09-05", "2012-09-03", "2013-09-02"],
            "Thanksgiving": ["2010-11-25", "2011-11-24", "2012-11-22", "2013-11-28"],
            "Christmas": ["2010-12-25", "2011-12-25", "2012-12-25", "2013-12-25"]
        }
        
        # Convert holiday dates to datetime
        holidays = {key: pd.to_datetime(value) for key, value in holidays.items()}
        
        # Initialize holiday category column
        X["HolidayCategory"] = "none"
        
        # Add holiday features
        for holiday_name, dates in holidays.items():
            for date in dates:
                diff = (Dates - date).dt.days
                X.loc[diff == 0, "HolidayCategory"] = f"{holiday_name}_is"
                X.loc[diff == -1, "HolidayCategory"] = f"{holiday_name}_is1daybefore"
                X.loc[diff == -2, "HolidayCategory"] = f"{holiday_name}_is2daysbefore"
        
        # Convert Store and Dept to string to ensure proper categorical handling
        X["Store"] = X["Store"].astype(str)
        X["Dept"] = X["Dept"].astype(str)
        
        return X
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Transform the input data
        X_transformed = self.transform_input(X)
        
        # Fit the preprocessor and transform the data
        X_processed = self.preprocessor.fit_transform(X_transformed)
        
        # Fit the model
        self.model.fit(X_processed, y)
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Transform the input data
        X_transformed = self.transform_input(X)
        
        # Transform using the fitted preprocessor
        X_processed = self.preprocessor.transform(X_transformed)
        
        # Make predictions
        return self.model.predict(X_processed)

def weighted_mae(y_true, y_pred, holiday_weights):
    weights = np.where(holiday_weights, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def process_fold(fold_path, test_labels):
    print(f"Processing {fold_path}")

    # Load training data
    train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))

    # Define target and features
    X_train = train_df
    y_train = train_df["Weekly_Sales"]
    
    # Create and fit global model
    model = GlobalSalesModel()
    model.fit(X_train, y_train)

    # Load test data
    test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))
    X_test = test_df

    # Make predictions using global model
    predictions = model.predict(X_test)

    # Prepare submission dataframe
    submission_df = test_df[["Store", "Dept", "Date", "IsHoliday"]].copy()
    submission_df["Weekly_Pred"] = predictions

    # Calculate weighted MAE using the test labels
    merged_df = pd.merge(
        submission_df, test_labels, on=["Store", "Dept", "Date", "IsHoliday"]
    )
    merged_df["error"] = np.abs(merged_df["Weekly_Sales"] - merged_df["Weekly_Pred"])

    if len(merged_df) != len(submission_df):
        print(
            f"Warning: Mismatch in number of samples. Predictions: {len(submission_df)}, Labels: {len(merged_df)}"
        )

    wmae = weighted_mae(
        merged_df["Weekly_Sales"], merged_df["Weekly_Pred"], merged_df["IsHoliday"]
    )

    # Add additional metrics for store-dept performance
    store_dept_metrics = merged_df.groupby(['Store', 'Dept']).agg({
        'error': ['mean', 'std'],
        'Weekly_Sales': 'count'
    }).reset_index()
    
    store_dept_metrics.columns = ['Store', 'Dept', 'Mean_Error', 'Std_Error', 'Sample_Count']
    
    metrics_path = os.path.join(save_path, f"store_dept_metrics_{os.path.basename(fold_path)}.csv")
    store_dept_metrics.to_csv(metrics_path, index=False)

    return [
        {
            'fold': os.path.basename(fold_path),
            'num_train_samples': len(train_df),
            'num_test_samples': len(test_df),
            'weighted_mae': wmae
        },
        merged_df
    ]

save_path = "/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project2/data/global_model_results"
os.makedirs(save_path, exist_ok=True)

def main(fold_count=10):
    data_dir = "/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project2/data"

    results = []
    dfs = []

    # Load the test labels once
    test_labels_path = os.path.join(data_dir, "test_with_label.csv")
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"test_with_label.csv not found at {test_labels_path}")

    test_labels = pd.read_csv(test_labels_path)

    folds = sorted(os.listdir(data_dir))[: fold_count + 1]

    for fold in folds:
        fold_path = os.path.join(data_dir, fold)
        if os.path.isdir(fold_path) and fold.startswith("fold_"):
            result, df = process_fold(fold_path, test_labels)
            results.append(result)
            dfs.append(df)

    # Aggregate results into a DataFrame
    results_df = pd.DataFrame(results)
    print("\nAggregated Results:")
    print(results_df)

    # Save aggregated results
    aggregated_results_path = os.path.join(save_path, "aggregated_results.csv")
    results_df.to_csv(aggregated_results_path, index=False)

    # Save submission files
    submission_dir = os.path.join(save_path, "submissions")
    os.makedirs(submission_dir, exist_ok=True)
    for i, df in enumerate(dfs):
        df.to_csv(os.path.join(submission_dir, f"submission_{i}.csv"), index=False)
    print(f"Aggregated results saved to {aggregated_results_path}")

if __name__ == "__main__":
    fold_count = 2
    if len(sys.argv) == 2:
        fold_count = int(sys.argv[1])
    main(fold_count)