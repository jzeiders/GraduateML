from datetime import datetime
import logging
import os
import sys
from typing import Dict, Set, Tuple
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
warnings.filterwarnings("ignore")

class GlobalSalesModel:
    def __init__(self):
        # Set up logging
        self.setup_logging()
        
        # Define categorical and numeric features
        self.categorical_features = ['HolidayCategory']
        self.numeric_features = ['Year']
        self.interaction_features = ['Store', 'Dept', 'WeekOfYear', 'DayOfWeek', 'HolidayCategory']
        
        
        # Create main preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), 
                 self.categorical_features),
                ('interactions', Pipeline([
                    ('onehot', OneHotEncoder(handle_unknown='ignore')),
                    ('interactions', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
                ]), self.interaction_features)
            ])
        
        self.model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
            n_alphas=100,
            cv=5,
            random_state=42,
            selection='random',
            max_iter=1000
        )
    
    def setup_logging(self):
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(save_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging configuration
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'model_interpretation_{timestamp}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging
        
    def get_feature_names(self):
        """Get all feature names after transformation"""
        # Get numeric feature names
        feature_names = self.numeric_features.copy()
        
        # Get categorical feature names
        cat_features = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_features)
        
        # Get store-dept interaction feature names
        store_dept_encoder = self.preprocessor.named_transformers_['interactions'].named_steps['onehot']
        store_dept_features = store_dept_encoder.get_feature_names_out(self.interaction_features)
        
        # Get interaction terms
        poly = self.preprocessor.named_transformers_['interactions'].named_steps['interactions']
        store_dept_interactions = poly.get_feature_names_out(store_dept_features)
        
        feature_names.extend(store_dept_interactions)
        
        return np.array(feature_names)
    
    def analyze_coefficients(self, feature_importance_threshold=0.01):
        """Analyze and log model coefficients and their importance"""
        feature_names = self.get_feature_names()
        coefficients = self.model.coef_
        
        # Create DataFrame with feature names and coefficients
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })
        
        # Calculate relative importance (absolute value of coefficients)
        coef_df['importance'] = np.abs(coef_df['coefficient'])
        coef_df['relative_importance'] = coef_df['importance'] / coef_df['importance'].sum()
        
        # Sort by importance
        coef_df = coef_df.sort_values('relative_importance', ascending=False)
        
        # Log most important features
        self.logger.info("\n=== Model Interpretation ===")
        self.logger.info(f"\nTop 20 Most Important Features:")
        for _, row in coef_df.head(20).iterrows():
            self.logger.info(f"{row['feature']}: {row['coefficient']:.4f} (Relative Importance: {row['relative_importance']:.4%})")
        
        # Analyze Store-Dept interactions
        interaction_features = coef_df[coef_df['feature'].str.contains('x')]
        self.logger.info("\nTop 10 Store-Department Interactions:")
        for _, row in interaction_features.head(10).iterrows():
            self.logger.info(f"{row['feature']}: {row['coefficient']:.4f} (Relative Importance: {row['relative_importance']:.4%})")
        
        # Analyze temporal features
        temporal_features = coef_df[coef_df['feature'].str.contains('Week|Day|Year')]
        self.logger.info("\nTop 10 Temporal Features:")
        for _, row in temporal_features.head(10).iterrows():
            self.logger.info(f"{row['feature']}: {row['coefficient']:.4f} (Relative Importance: {row['relative_importance']:.4%})")
        
        # Analyze holiday effects
        holiday_features = coef_df[coef_df['feature'].str.contains('Holiday')]
        self.logger.info("\nHoliday Effects:")
        for _, row in holiday_features.iterrows():
            self.logger.info(f"{row['feature']}: {row['coefficient']:.4f} (Relative Importance: {row['relative_importance']:.4%})")
        
        # Save detailed coefficient analysis to CSV
        coef_file = os.path.join(save_path, 'logs', 'coefficient_analysis.csv')
        coef_df.to_csv(coef_file, index=False)
        self.logger.info(f"\nDetailed coefficient analysis saved to: {coef_file}")
        
        return coef_df
    
    def analyze_predictions(self, X: pd.DataFrame, y_pred: np.ndarray):
        """Analyze prediction patterns"""
        analysis_df = X.copy()
        analysis_df['predicted_sales'] = y_pred
        
        # Analyze predictions by store and department
        store_dept_analysis = analysis_df.groupby(['Store', 'Dept'])['predicted_sales'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        self.logger.info("\n=== Prediction Analysis ===")
        self.logger.info("\nTop 10 Store-Department Combinations by Average Predicted Sales:")
        self.logger.info(store_dept_analysis.nlargest(10, 'mean').to_string())
        
        # Save detailed prediction analysis
        analysis_file = os.path.join(save_path, 'logs', 'prediction_analysis.csv')
        store_dept_analysis.to_csv(analysis_file, index=False)
        self.logger.info(f"\nDetailed prediction analysis saved to: {analysis_file}")
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.logger.info("\n=== Training Model ===")
        self.logger.info(f"Training data shape: {X.shape}")
        
        # Transform the input data
        X_transformed = self.transform_input(X)
        
        # Fit the preprocessor and transform the data
        X_processed = self.preprocessor.fit_transform(X_transformed)
        
        # Fit the model
        self.model.fit(X_processed, y)
        
        # Analyze coefficients and log interpretation
        self.analyze_coefficients()
        
        # Analyze predictions
        y_pred = self.predict(X)
        self.analyze_predictions(X, y_pred)
        
        # Log model performance metrics
        r2_score = self.model.score(X_processed, y)
        self.logger.info(f"\nModel R² Score: {r2_score:.4f}")
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_transformed = self.transform_input(X)
        X_processed = self.preprocessor.transform(X_transformed)
        return self.model.predict(X_processed)
    
    def transform_input(self, df):
        # [Previous transform_input implementation remains the same]
        X = df[["Store", "Dept"]].copy()
        Dates = pd.to_datetime(df["Date"])
        
        X["Year"] = Dates.dt.year.astype(int)
        X["WeekOfYear"] = Dates.dt.isocalendar().week.astype(str)
        X["DayOfWeek"] = Dates.dt.dayofweek.astype(str)
        
        holidays = {
            "Super Bowl": ["2010-02-07", "2011-02-06", "2012-02-05", "2013-02-03"],
            "Labor Day": ["2010-09-06", "2011-09-05", "2012-09-03", "2013-09-02"],
            "Thanksgiving": ["2010-11-25", "2011-11-24", "2012-11-22", "2013-11-28"],
            "Christmas": ["2010-12-25", "2011-12-25", "2012-12-25", "2013-12-25"]
        }
        
        holidays = {key: pd.to_datetime(value) for key, value in holidays.items()}
        
        X["HolidayCategory"] = "none"
        
        for holiday_name, dates in holidays.items():
            for date in dates:
                diff = (Dates - date).dt.days
                X.loc[diff == 0, "HolidayCategory"] = f"{holiday_name}_is"
                X.loc[diff == -1, "HolidayCategory"] = f"{holiday_name}_is1daybefore"
                X.loc[diff == -2, "HolidayCategory"] = f"{holiday_name}_is2daysbefore"
        
        X["Store"] = X["Store"].astype(str)
        X["Dept"] = X["Dept"].astype(str)
        
        return X

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