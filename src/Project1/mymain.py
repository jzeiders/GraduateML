import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import os 
import sys
from sklearn.metrics import mean_squared_error


# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def model(data_dir, results_dir):
    # Construct paths for train.csv and test.csv based on the given directory
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')

    # Step 1: Preprocess the training data and fit the models
    # Load training data

    train = pd.read_csv(train_path).drop(columns=['PID'])
    
    # Separate PID and response variable
    y = np.log1p(train['Sale_Price'])  # Using log1p to handle possible zero values

    X = train.drop(['Sale_Price'], axis=1)

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Handle missing values
    X_num = X[numerical_cols].fillna(X[numerical_cols].median())
    X_cat = X[categorical_cols].fillna('Missing')

    # Encode categorical variables
    encoder = OneHotEncoder(cols=categorical_cols, use_cat_names=True)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Combine numerical and encoded categorical features
    X_processed = pd.concat([X_num, X_cat_encoded], axis=1)

    # Optional: Feature Scaling for Lasso
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_processed)

    # Fit Lasso Regression
    lasso = Lasso(alpha=0.001, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    # Fit Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_processed, y)

    # Step 2: Preprocess test data and make predictions
    # Load test data
    test = pd.read_csv(test_path)
    test_pid = test['PID']
    X_test = test.drop(['PID'], axis=1)

    # Handle missing values in test data
    X_test_num = X_test[numerical_cols].fillna(X[numerical_cols].median())
    X_test_cat = X_test[categorical_cols].fillna('Missing')

    # Encode categorical variables using the same encoder
    X_test_cat_encoded = encoder.transform(X_test_cat)

    # Combine numerical and encoded categorical features
    X_test_processed = pd.concat([X_test_num, X_test_cat_encoded], axis=1)

    # Align test data with training data
    X_test_processed = X_test_processed.reindex(columns=X_processed.columns, fill_value=0)

    # Optional: Feature Scaling for Lasso
    X_test_scaled = scaler.transform(X_test_processed)

    # Make predictions
    preds_lasso = lasso.predict(X_scaled)  # Not needed
    preds_lasso_test = lasso.predict(X_test_scaled)
    preds_rf_test = rf.predict(X_test_processed)

    # Prepare submission files
    submission_lasso = pd.DataFrame({
        'PID': test_pid,
        'Sale_Price': preds_lasso_test
    })

    submission_rf = pd.DataFrame({
        'PID': test_pid,
        'Sale_Price': preds_rf_test
    })

    
    # Make sure the directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    submission_1_path = os.path.join(results_dir, 'mysubmission1.txt')
    submission_2_path = os.path.join(results_dir, 'mysubmission2.txt')

    # Save to text files without index and header
    submission_lasso.to_csv(submission_1_path, sep=',', index=False, header=True)
    submission_rf.to_csv(submission_2_path, sep=',', index=False, header=True)
    
    if is_test_dev:
        test_y = pd.read_csv(os.path.join(data_dir, 'test_y.csv'))
        actual_prices = np.log1p(test_y['Sale_Price']) 
        rmse_lasso = np.sqrt(mean_squared_error(actual_prices, preds_lasso_test))
        rmse_rf = np.sqrt(mean_squared_error(actual_prices, preds_rf_test))

        # Write thes results to a file using a pandas dataframe
        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Model': ['Lasso', 'Random Forest'],
            'RMSE': [rmse_lasso, rmse_rf]
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
    for folder in data_folders[1:3]:
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
        
