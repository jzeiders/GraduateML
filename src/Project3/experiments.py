# Filename: mymain.py

import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import joblib
import os
import sys
import time
import argparse

def load_data(train_path, test_path, test_y_path):
    """
    Load training and testing data using Polars.
    """
    print(f"Loading training data from {train_path}...")
    train = pl.read_csv(train_path)
    
    print(f"Loading test data from {test_path}...")
    test = pl.read_csv(test_path)
    
    print(f"Loading test labels from {test_y_path}...")
    test_y = pl.read_csv(test_y_path)
    
    return train, test, test_y

def preprocess_data(train, test, test_y):
    """
    Preprocess the data by selecting relevant features and separating labels.
    """
    # Drop 'id' and 'review' columns
    X_train = train.drop(['id', 'sentiment', 'review'])
    y_train = train['sentiment']
    
    X_test = test.drop(['id', 'review'])
    y_test = test_y['sentiment']
    
    # Convert Polars DataFrames to NumPy arrays for XGBoost
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()
    
    return X_train_np, y_train_np, X_test_np, y_test_np

def train_xgboost_model(X_train, y_train):
    """
    Initialize and train the XGBoost classifier.
    """
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Predict probabilities on the test set and calculate AUC.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc, y_pred_prob

def save_model(model, split_number, save_dir='models/'):
    """
    Save the trained model using joblib.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, f'xgb_model_split{split_number}.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def generate_submission(model, test_data_path, output_path='mysubmission.csv'):
    """
    Generate submission file based on the trained model and test data.

    Parameters:
    - model: Trained XGBoost model.
    - test_data_path (str): Path to the test.csv file.
    - output_path (str): Path to save the submission CSV.
    """
    # Load test data
    print(f"Loading test data for submission from {test_data_path}...")
    test = pl.read_csv(test_data_path)
    
    # Preprocess test data
    X_test = test.drop(['id', 'review'])
    X_test_np = X_test.to_numpy()
    
    # Predict probabilities
    print("Predicting probabilities for submission...")
    y_pred_prob = model.predict_proba(X_test_np)[:, 1]
    
    # Create submission DataFrame
    submission = pl.DataFrame({
        'id': test['id'],
        'prob': y_pred_prob
    })
    
    # Save to CSV
    submission.write_csv(output_path)
    print(f"Submission file saved to {output_path}")

def main():
    """
    Main function to train the model and generate submission.
    """
    parser = argparse.ArgumentParser(description='Train XGBoost model and generate submission.')
    parser.add_argument('--data-dir', type=str, help='Path to the data directory', default='/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/data', required=False)
    args = parser.parse_args()
    
    for split_folder in os.listdir(args.data_dir):
        if split_folder.startswith('split') == False:
            continue
        train_path = os.path.join(args.data_dir, split_folder, 'train.csv')
    
    
        train_path = os.path.join(args.data_dir,split_folder, 'train.csv')
        test_path = os.path.join(args.data_dir, split_folder,'test.csv')
        test_y_path = os.path.join(args.data_dir, split_folder,'test_y.csv') 
        
        train, test, test_y = load_data(train_path, test_path, test_y_path)
        
        # Preprocess data
        X_train, y_train, X_test, y_test = preprocess_data(train, test, test_y)
        
        # Train model
        print("Training XGBoost model...")
        model = train_xgboost_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        auc, y_pred_prob = evaluate_model(model, X_test, y_test)
        print(f"AUC: {auc:.6f}")
        
        # Save the model
        save_model(model, split_folder, os.path.join(args.data_dir, 'models'))
        
        # Check if AUC meets the threshold
        if auc < 0.986:
            print(f"⚠️ Warning: AUC {auc:.6f} is below the threshold of 0.986.")
        else:
            print(f"✅ AUC meets the threshold.")


if __name__ == "__main__":
    main()
