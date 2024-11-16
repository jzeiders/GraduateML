import os
import sys
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from functools import partial
from scipy.linalg import svd

def svd_dept(train_df, n_comp=8):
    """
    Perform SVD-based dimensionality reduction on department-wise sales data.
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Input DataFrame with columns: Store, Dept, Date, Weekly_Sales
    n_comp : int, default=8
        Number of SVD components to retain
        
    Returns:
    --------
    pandas.DataFrame
        Transformed DataFrame with reconstructed sales data
    """
    # Pivot the data to get stores as columns and (dept, date) as index
    pivot_df = train_df.pivot_table(
        index=['Dept', 'Date'],
        columns='Store',
        values='Weekly_Sales',
        fill_value=0
    )
    
    new_data = []
    
    # Process each department separately
    for dept in pivot_df.index.get_level_values('Dept').unique():
        # Filter data for current department
        dept_data = pivot_df.loc[dept]
        
        # Get raw sales matrix X
        X = dept_data.values
        
        # Calculate store means for centering
        store_means = np.mean(X, axis=0)
        
        # Center the data
        X_centered = X - store_means
        
        # Apply SVD if we have enough rows
        if X_centered.shape[0] > n_comp:
            # Perform SVD
            U, s, Vt = svd(X_centered, full_matrices=False)
            
            # Reconstruct using n_comp components
            X_reconstructed = U[:, :n_comp] @ np.diag(s[:n_comp]) @ Vt[:n_comp, :]
            
            # Add back the means
            X_reconstructed += store_means
        else:
            # If not enough rows, use original data
            X_reconstructed = X
            
        # Create DataFrame with reconstructed data
        dept_dates = dept_data.index
        stores = dept_data.columns
        
        # Reshape to long format
        for i, date in enumerate(dept_dates):
            for j, store in enumerate(stores):
                new_data.append({
                    'Store': int(store),
                    'Dept': dept,
                    'Date': date,
                    'Weekly_Sales': X_reconstructed[i, j]
                })
    
    # Convert to DataFrame and sort
    result_df = pd.DataFrame(new_data)
    
    return result_df



def AddDateFeatures(df):
    Date = pd.to_datetime(df['Date'])
    df["Year"] = Date.dt.year
    df["Year"] = df["Year"].astype(int) 
    df["Week"] = Date.dt.isocalendar().week.astype(pd.CategoricalDtype(categories=range(1,53)))

    # Conver to dummies on the week
    df = pd.get_dummies(df, columns=["Week"])

    return df

def run_pipeline(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    is_fold_5 = test_df['Date'].between("2011-11-04", "2011-12-30").any()
    
    # train_df_svd = train_df.drop(columns=["IsHoliday"]); # svd_dept(train_df)
    train_df_svd = svd_dept(train_df)

    train_df_pairs = train_df[['Store', 'Dept']].drop_duplicates()
    test_df_pairs = test_df[['Store', 'Dept']].drop_duplicates()
    
    train_df_svd = AddDateFeatures(train_df_svd)
    test_df = AddDateFeatures(test_df)

    common_pairs = pd.merge(train_df_pairs, test_df_pairs, on=['Store', 'Dept'], how='inner').drop_duplicates()
    models: Dict[Tuple[int, int], LinearRegression] = {}
    preds = pd.DataFrame()
    for row in common_pairs.iterrows():
        dept = row[1]['Dept']
        store = row[1]['Store']
        
        data = train_df_svd[(train_df_svd['Dept'] == dept) & (train_df_svd['Store'] == store)]
        model = Ridge(alpha=1.0)
        model.fit(data.drop(['Date', 'Weekly_Sales', "Store","Dept"], axis=1), data['Weekly_Sales'])
        models[(store, dept)] = model
        
        
        test_data = test_df[(test_df['Dept'] == dept) & (test_df['Store'] == store)].copy()
        pred = model.predict(test_data.drop(['Date', 'IsHoliday', 'Store','Dept'], axis=1))
        test_data['Weekly_Pred'] = pred
        preds = pd.concat([preds, test_data])
        
    predictions = test_df.merge(preds, on=['Store', 'Dept', 'Date', 'IsHoliday'], how='left')['Weekly_Pred']
    
    
    # TODO: Add in the re-balancing of the predictions
    

    return predictions, models


def weighted_mae(y_true, y_pred, holiday_weights):
    weights = np.where(holiday_weights, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def process_fold(fold_path, test_labels):
    print(f"Processing {fold_path}")

    # Load training data
    train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))


    predictions, models= run_pipeline(os.path.join(fold_path, "train.csv"),os.path.join(fold_path, "test.csv"))
        # Write the model coefficients to a file
    model_coefs = pd.DataFrame()
    
    # Extract and save model coefficients
    coef_records = []
    for (store, dept), model in models.items():
        features = model.feature_names_in_
        
        # Create a dictionary with store and dept
        coef_dict = {
            'Store': store,
            'Dept': dept,
            'Intercept': model.intercept_
        }
        
        # Add coefficients with their feature names
        for feature_name, coef in zip(features, model.coef_):
            coef_dict[feature_name] = coef
            
        coef_records.append(coef_dict)
    
    # Convert to DataFrame and save
    model_coefs = pd.DataFrame(coef_records)
    model_coefs.to_csv(os.path.join(save_path, f"model_coefs_{os.path.basename(fold_path)}.csv"), index=False)

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