import os
import time
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np



def svd_dept(train_df, n_comp=8):
    # Pivot the data to get stores as columns and (dept, date) as index
    pivot_df = train_df.pivot_table(
        index=["Dept", "Date"], columns="Store", values="Weekly_Sales", fill_value=0
    )

    new_data = []

    # Process each department separately
    for dept in pivot_df.index.get_level_values("Dept").unique():
        # Filter data for current department
        dept_data = pivot_df.loc[dept]

        # Get raw sales matrix X
        X = dept_data.values

        # Skip if no data
        if X.size == 0:
            continue

        # Calculate store means for centering
        store_means = np.mean(X, axis=0)

        pca = PCA(n_components=n_comp)
        try:
            X_reconstructed = (
                pca.inverse_transform(pca.fit_transform(X - store_means)) + store_means
            )
        except Exception as e:
            print(
                f"Warning: PCA failed for department {dept}. Using original data. Error: {e}"
            )
            X_reconstructed = X

  
        dept_dates = dept_data.index
        stores = dept_data.columns

        # Reshape to long format
        data_array = (
            pd.DataFrame(X_reconstructed, index=dept_dates, columns=stores)
            .stack()
            .reset_index()
        )
        data_array.columns = ["Date", "Store", "Weekly_Sales"]
        data_array.insert(0, "Dept", dept)

        new_data.append(data_array)

    # Concatenate all reconstructed data
    result_df = pd.concat(new_data, ignore_index=True)

    return result_df


def AddDateFeatures(df):
    Date = pd.to_datetime(df["Date"])
    df["Week"] = Date.dt.isocalendar().week.astype(pd.CategoricalDtype(range(1, 53)))
    df["Year_2010"] = (Date.dt.year == 2010).astype(int)
    df["Year_2011"] = (Date.dt.year == 2011).astype(int)
    df["Year_2012"] = (Date.dt.year == 2012).astype(int)

    # Conver to dummies on the week
    df = pd.get_dummies(df, columns=["Week"])

    return df


def run_pipeline(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df_svd = svd_dept(train_df)

    train_df_pairs = train_df[["Store", "Dept"]].drop_duplicates()
    test_df_pairs = test_df[["Store", "Dept"]].drop_duplicates()

    train_df_svd = AddDateFeatures(train_df_svd)
    test_df = AddDateFeatures(test_df)

    common_pairs = pd.merge(
        train_df_pairs, test_df_pairs, on=["Store", "Dept"], how="inner"
    ).drop_duplicates()
    preds = pd.DataFrame()
    for row in common_pairs.iterrows():
        dept = row[1]["Dept"]
        store = row[1]["Store"]

        data = train_df_svd[
            (train_df_svd["Dept"] == dept) & (train_df_svd["Store"] == store)
        ]
        model = Ridge(alpha=0.01, random_state=42)
        model.fit(
            data.drop(["Date", "Weekly_Sales", "Store", "Dept"], axis=1),
            data["Weekly_Sales"],
        )

        test_data = test_df[
            (test_df["Dept"] == dept) & (test_df["Store"] == store)
        ].copy()
        pred = model.predict(
            test_data.drop(["Date", "IsHoliday", "Store", "Dept"], axis=1)
        )
        if (pred > 10e8).any():
            print(
                f"Warning: Predicted value is too high for Store {store}, Dept {dept}"
            )
        pred[pred < 0] = 0

        test_data["Weekly_Pred"] = pred
        preds = pd.concat([preds, test_data])

    predictions = test_df.merge(
        preds, on=["Store", "Dept", "Date", "IsHoliday"], how="left"
    )[["Store", "Dept", "Date", 'IsHoliday', "Weekly_Pred"]]

    return predictions 



def weighted_mae(y_true, y_pred, holiday_weights):
    weights = np.where(holiday_weights, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)

def process_fold(fold_path, test_labels):
    print(f"Processing {fold_path}")

    # Load training data
    train_df = pd.read_csv(os.path.join(fold_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(fold_path, "test.csv"))


    submission_df = run_pipeline(os.path.join(fold_path, "train.csv"),os.path.join(fold_path, "test.csv"))
        # Write the model coefficients to a file
    
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

    # Load the test labels once
    test_labels_path = os.path.join(data_dir, "test_with_label.csv")
    if not os.path.exists(test_labels_path):
        raise FileNotFoundError(f"test_with_label.csv not found at {test_labels_path}")

    test_labels = pd.read_csv(test_labels_path)

    folds = sorted(os.listdir(data_dir))[: fold_count + 1]
    
    # Save submission files
    submission_dir = os.path.join(save_path, "submissions")
    os.makedirs(submission_dir, exist_ok=True)

    for fold in folds:
        fold_path = os.path.join(data_dir, fold)
        if os.path.isdir(fold_path) and fold.startswith("fold_"):
            start_time = time.perf_counter()
            result, df = process_fold(fold_path, test_labels)
            train_time = time.perf_counter() - start_time
            print(f"{fold} model trained in {train_time:.2f} seconds.")
            results.append(result)
            df.to_csv(os.path.join(submission_dir, f"submission_{fold}.csv"), index=False)

    # Aggregate results into a DataFrame
    results_df = pd.DataFrame(results)
    print("\nAggregated Results:")
    print(results_df)
    print(results_df["weighted_mae"].describe())

    # Save aggregated results
    aggregated_results_path = os.path.join(save_path, "aggregated_results.csv")
    results_df.to_csv(aggregated_results_path, index=False)
       
    print(f"Aggregated results saved to {aggregated_results_path}")

if __name__ == "__main__":
    fold_count = 100
    main(fold_count)