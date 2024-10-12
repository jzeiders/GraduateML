import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import StandardScaler
import os 
import sys
from sklearn.metrics import mean_squared_error
import time
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor



# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


log_transform = FunctionTransformer(func=np.log,inverse_func=np.exp, validate=True)

def create_pipeline(model, numerical_features, categorical_features, encoding):
    """
    Creates a preprocessing and modeling pipeline.

    Parameters:
    - model: The machine learning model to integrate into the pipeline.
    - numerical_features: List of numerical feature names.
    - categorical_features: List of categorical feature names.
    - encoding: Type of encoding for categorical variables ('onehot' or 'target').

    Returns:
    - A scikit-learn Pipeline object.
    """
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
        ('scaler', StandardScaler())
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

def model(data_dir, results_dir, encoding='onehot'):
    # Construct paths for train.csv and test.csv based on the given directory
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')


    manual_drop = ['Latitude', 'Longitude']
    highly_correlated = ['TotRms_AbvGrd', 'Garage_Yr_Blt', 'Garage_Area', 'Latitude']

    # Columns identified as potentially non-linear
    potential_non_linear = ['BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 'Low_Qual_Fin_SF', 
                        'Bsmt_Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr', 'Enclosed_Porch', 
                        'Three_season_porch', 'Screen_Porch', 'Pool_Area', 'Misc_Val', 
                        'Mo_Sold', 'Year_Sold']

    sparse_categories = ['Street', 'Utilities', 'Land_Slope', 'Condition_2', 'Roof_Matl', 'Heating', 'Central_Air', 'Electrical', 'Functional', 'Garage_Qual', 'Garage_Cond', 'Paved_Drive', 'Pool_QC', 'Fence', 'Misc_Feature', 'Sale_Type']


    # Define columns to drop, most are due to many NA values
    DROP_COLS = ['PID'] + highly_correlated + potential_non_linear + sparse_categories + manual_drop

    # Load training data
    train = pd.read_csv(train_path).drop(columns=DROP_COLS)
    
    # Separate response variable and features
    y = train['Sale_Price']
    X = train.drop(['Sale_Price'], axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    
    elastic_cv_pipeline = create_pipeline(
        model= TransformedTargetRegressor(regressor=ElasticNetCV(cv=5, random_state=42), transformer=log_transform),
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        encoding=encoding
    )
    
    # Create XGBoost pipeline
    xgb_pipeline_from_note = create_pipeline(
        model = TransformedTargetRegressor(regressor=XGBRegressor(
            n_estimators=5000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        ), transformer=log_transform),
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        encoding='onehot'  # XGBoost often works well with target encoding
    )
    
        # Train models and measure time
    models = {
        'ElasticNetCV': elastic_cv_pipeline,
        # 'XGBoost_From_Note' : xgb_pipeline_from_note
    }

    for name, pipeline in models.items():
        start_time = time.perf_counter()
        pipeline.fit(X, y)
        train_time = time.perf_counter() - start_time
        print(f"{name} model trained in {train_time:.2f} seconds.")


    # Step 2: Preprocess test data and make predictions
    # Load test data
    test = pd.read_csv(test_path)
    test_pid = test['PID']
    X_test = test.drop(DROP_COLS, axis=1)

    # Make sure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Make predictions and create submissions
    for name, pipeline in models.items():
        preds = pipeline.predict(X_test)
        submission = pd.DataFrame({
            'PID': test_pid,
            'Sale_Price': preds,
        })
        submission_path = os.path.join(results_dir, f'submission_{name.lower().replace(" ", "_")}.txt')
        submission.to_csv(submission_path, sep=',', index=False, header=True)

    

    if is_test_dev:
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'))

        results = []
        for name, pipeline in models.items():
            preds = pipeline.predict(X_test)
            rmse = np.sqrt(mean_squared_error(test_y['Sale_Price'], preds))
            results.append({
                'Model': name,
                'RMSE': round(rmse, 5),
                'Time': train_time  # Note: This will only have the time for the last trained model
            })

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(results)
    
        # Write the results to a CSV file
        results_file = os.path.join(results_dir, 'rmse_results.csv')
        results_df.to_csv(results_file, index=False)
        
def evaluate():
    # Set the RMSE performance targets
    initial_target_rmse = 0.125
    subsequent_target_rmse = 0.135

    # Track evaluation results
    all_results = []

    # Go through all the data folders in /data
    data_folders = os.listdir("data")
    for folder in data_folders:
        folder_path = os.path.join("data", folder)
        if os.path.isdir(folder_path):
            data_dir = folder_path
            results_dir = os.path.join("results", folder)

            # Ensure the results directory exists
            os.makedirs(results_dir, exist_ok=True)

            # Run the model function
            model(data_dir, results_dir)

            # Check if the rmse_results.csv file exists
            rmse_file = os.path.join(results_dir, "rmse_results.csv")
            if os.path.exists(rmse_file):
                try:
                    # Load RMSE results and evaluate performance
                    results_df = pd.read_csv(rmse_file)

                    # Determine if this is part of the initial or subsequent split
                    target_rmse = initial_target_rmse if len(all_results) < 5 else subsequent_target_rmse

                    # Extract RMSE values for each model
                    model_rmses = {}
                    for _, row in results_df.iterrows():
                        model_name = row['Model']
                        rmse = row['RMSE']
                        model_rmses[f'{model_name} RMSE'] = rmse

                    # Check if all models meet the target
                    meets_target = all(rmse < target_rmse for rmse in model_rmses.values())

                    # Prepare result dictionary
                    result = {'Folder': folder, 'Meets Target': meets_target}
                    result.update(model_rmses)

                    all_results.append(result)

                except Exception as e:
                    print(f"Error processing {rmse_file}: {e}")
            else:
                print(f"Warning: RMSE results file not found for folder {folder}")

    if not all_results:
        print("No results to evaluate.")
        return

    # Convert the list of results to a DataFrame
    all_results_df = pd.DataFrame(all_results).sort_values(by='Folder')

    # Save the summary to a CSV file
    summary_file = os.path.join("results", "evaluation_summary.csv")
    try:
        all_results_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
        
        print(all_results_df.to_markdown())
    except Exception as e:
        print(f"Error saving summary to {summary_file}: {e}")

    # Print a message if any folders failed to meet the target
    if not all_results_df['Meets Target'].all():
        print("Warning: Some models did not meet the RMSE targets.")
    else:
        print("All models met the RMSE targets.")
    

def identify_problematic_features(df, target_column, correlation_threshold=0.8, missing_threshold=0.1, unique_threshold=0.95):
    problematic_features = {}
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # Check for highly correlated features
    corr_matrix = numeric_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]
    if highly_correlated:
        problematic_features['highly_correlated'] = highly_correlated

    # Check for features with high percentage of missing values
    missing_percentage = numeric_df.isnull().mean()
    high_missing = missing_percentage[missing_percentage > missing_threshold].index.tolist()
    if high_missing:
        problematic_features['high_missing'] = high_missing

    # Check for features with low variance (potentially irrelevant)
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
    low_variance = scaled_df.columns[scaled_df.var() < 0.1].tolist()
    if low_variance:
        problematic_features['low_variance'] = low_variance

    # Check for categorical variables with many levels (this check remains on all columns)
    high_cardinality = [col for col in df.select_dtypes(include=['object', 'category']).columns 
                        if df[col].nunique() / len(df) > unique_threshold]
    if high_cardinality:
        problematic_features['high_cardinality'] = high_cardinality

    # Sparse Categories
    sparse_categories = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        category_counts = df[col].value_counts(normalize=True)
        if category_counts.max() > 0.95:
            sparse_categories.append(col)
    problematic_features['sparse_categories'] = sparse_categories

    # Check for potential non-linear relationships with target
    if target_column in numeric_df.columns:
        X = numeric_df.drop(columns=[target_column])
        y = numeric_df[target_column]
        non_linear = []
        for col in X.columns:
            correlation = np.corrcoef(X[col], y)[0, 1]
            if abs(correlation) < 0.2:  # Weak linear correlation might indicate non-linear relationship
                non_linear.append(col)
        if non_linear:
            problematic_features['potential_non_linear'] = non_linear

    return problematic_features

def feature_eng():
    print("Feature Engineering")
    # Load the training data
    train = pd.read_csv("data/fold1/train.csv")
    problem = identify_problematic_features(train, 'Sale_Price')
    print(problem)


if __name__ == "__main__":
    is_test_dev = len(sys.argv) > 1

    if len(sys.argv) == 2 and sys.argv[1] == "evaluate":
        evaluate()
    elif len(sys.argv) == 2 and sys.argv[1] == "feature_eng":
        feature_eng()
    else: 
        data_dir = sys.argv[1] if is_test_dev else "."
        results_dir = os.path.join("results", sys.argv[1].split("/")[1]) if is_test_dev else "."
        model(data_dir, results_dir)
        
