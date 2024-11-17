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


def main():
    train_path = 'train.csv'
    test_path = 'test.csv'

    predictions = run_pipeline(train_path, test_path)
    predictions.to_csv('mypred.csv', index=False)

if __name__ == '__main__':
    main()