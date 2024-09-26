import pandas as pd
import numpy as np
import warnings
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder, TargetEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os 
import sys
from sklearn.metrics import mean_squared_error
import time


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

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

    # Define columns to drop, most are due to many NA values
    DROP_COLS = ['PID', 'Mas_Vnr_Type',"Garage_Yr_Blt","Misc_Feature"]

    # Load training data
    train = pd.read_csv(train_path).drop(columns=DROP_COLS)
    
    # Separate response variable and features
    y = np.log1p(train['Sale_Price'])  # Using log1p to handle possible zero or skewed values
    X = train.drop(['Sale_Price'], axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    

        # Create pipelines for different models
    # Lasso Pipeline (with onehot encoding)
    lasso_pipeline = create_pipeline(
        model=Lasso(alpha=0.001, max_iter=1000000, random_state=42),
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        encoding=encoding  # You can choose 'onehot' or 'target'
    )
    
    # Random Forest Pipeline (with target encoding)
    rf_pipeline = create_pipeline(
        model=RandomForestRegressor(
            n_estimators=500, 
            max_depth=None, 
            min_samples_split=2, 
            min_samples_leaf=1, 
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        numerical_features=numerical_cols,
        categorical_features=categorical_cols,
        encoding='target'  # Target encoding often works better with tree-based models
    )

    start_time = time.perf_counter()
    lasso_pipeline.fit(X, y)
    lasso_time = time.perf_counter() - start_time
    print(f"Lasso model trained in {lasso_time:.2f} seconds.")
    
    start_time = time.perf_counter()
    rf_pipeline.fit(X, y)
    rf_time = time.perf_counter() - start_time
    print(f"Random Forest model trained in {rf_time:.2f} seconds.")

    # Step 2: Preprocess test data and make predictions
    # Load test data
    test = pd.read_csv(test_path)
    test_pid = test['PID']
    X_test = test.drop(DROP_COLS, axis=1)

    # Make predictions
    preds_lasso = lasso_pipeline.predict(X_test)
    preds_rf = rf_pipeline.predict(X_test)

    # Prepare submission files
    submission_lasso = pd.DataFrame({
        'PID': test_pid,
        'Sale_Price': np.expm1(preds_lasso)
    })

    submission_rf = pd.DataFrame({
        'PID': test_pid,
        'Sale_Price': np.expm1(preds_rf)
    })

    
    # Make sure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    
    submission_1_path = os.path.join(results_dir, 'mysubmission1.txt')
    submission_2_path = os.path.join(results_dir, 'mysubmission2.txt')

    # Save to text files without index and header
    submission_lasso.to_csv(submission_1_path, sep=',', index=False, header=True)
    submission_rf.to_csv(submission_2_path, sep=',', index=False, header=True)
    
    if is_test_dev:
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'))
        actual_prices = np.log1p(test_y['Sale_Price']) 

        preds_lasso_test = lasso_pipeline.predict(X_test)
        preds_rf_test = rf_pipeline.predict(X_test)

        rmse_lasso = np.sqrt(mean_squared_error(actual_prices, preds_lasso_test))
        rmse_rf = np.sqrt(mean_squared_error(actual_prices, preds_rf_test))

        # Write thes results to a file using a pandas dataframe
        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Model': ['Lasso', 'Random Forest'],
            'RMSE': [round(rmse_lasso,5), round(rmse_rf,5)],
            'Time': [lasso_time / 1_000_000, rf_time / 1_000_000]
        })
    
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
    # Ensure you're iterating over the desired range; adjust indices if needed
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
                    if len(all_results) < 5:
                        target_rmse = initial_target_rmse
                    else:
                        target_rmse = subsequent_target_rmse

                    # Extract RMSE values for each model
                    lasso_rmse = results_df.loc[results_df['Model'] == 'Lasso', 'RMSE'].values
                    rf_rmse = results_df.loc[results_df['Model'] == 'Random Forest', 'RMSE'].values

                    # Handle cases where RMSE values might be missing
                    if len(lasso_rmse) == 0 or len(rf_rmse) == 0:
                        print(f"Warning: Missing RMSE values in {rmse_file}")
                        continue

                    lasso_rmse = lasso_rmse[0]
                    rf_rmse = rf_rmse[0]

                    meets_target = lasso_rmse < target_rmse and rf_rmse < target_rmse
                    all_results.append({
                        'Folder': folder,
                        'Lasso RMSE': lasso_rmse,
                        'Random Forest RMSE': rf_rmse,
                        'Meets Target': meets_target
                    })
                except Exception as e:
                    print(f"Error processing {rmse_file}: {e}")
            else:
                print(f"Warning: RMSE results file not found for folder {folder}")

    if not all_results:
        print("No results to evaluate.")
        return

    # Convert the list of results to a DataFrame
    all_results_df = pd.DataFrame(all_results)

    # Save the summary to a CSV file
    summary_file = os.path.join("results", "evaluation_summary.csv")
    try:
        all_results_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Error saving summary to {summary_file}: {e}")

    # Print a message if any folders failed to meet the target
    if not all_results_df['Meets Target'].all():
        print("Warning: Some models did not meet the RMSE targets.")
    else:
        print("All models met the RMSE targets.")
    

if __name__ == "__main__":
    is_test_dev = len(sys.argv) > 1

    if len(sys.argv) == 2 and sys.argv[1] == "evaluate":
        evaluate()
    else: 
        data_dir = sys.argv[1] if is_test_dev else "."
        results_dir = os.path.join("results", sys.argv[1].split("/")[1]) if is_test_dev else "."
        model(data_dir, results_dir)
        
