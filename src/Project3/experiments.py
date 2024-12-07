import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import joblib
import os
import argparse
import pandas as pd
import time


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
    

    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """
    Train a logistic regression model with optimized parameters.
    Returns the model and training time.
    """
    start_time = time.time()
    model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        C=10,  # Based on previous grid search results
        random_state=42
    )
    
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    return model, training_time

def evaluate_model(model, X_test, y_test):
    """
    Predict probabilities on the test set and calculate AUC.
    Also measures inference time.
    """
    start_time = time.time()
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc, y_pred_prob, inference_time

def perform_grid_search(X_train, y_train):
    start_time = time.time()
    model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting GridSearch for Logistic Regression...")
    grid_search.fit(X_train, y_train)
    grid_search_time = time.time() - start_time
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best CV score: {grid_search.best_score_:.6f}")
    print(f"Grid search time: {grid_search_time:.2f} seconds")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_, grid_search_time

def save_grid_search_results(cv_results, split_number, save_dir='grid_search_results/'):
    """
    Save grid search results to a CSV file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    results_df = pd.DataFrame(cv_results)
    results_path = os.path.join(save_dir, f'grid_search_results_split{split_number}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Grid search results saved to {results_path}")

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
    """
    print(f"Loading test data for submission from {test_data_path}...")
    test = pl.read_csv(test_data_path)
    
    X_test = test.drop(['id', 'review'])
    X_test_np = X_test.to_numpy()
    
    print("Predicting probabilities for submission...")
    y_pred_prob = model.predict_proba(X_test_np)[:, 1]
    
    submission = pl.DataFrame({
        'id': test['id'],
        'prob': y_pred_prob
    })
    
    submission.write_csv(output_path)
    print(f"Submission file saved to {output_path}")

def main():
    """
    Main function to perform grid search, train the model and generate submission.
    """
    parser = argparse.ArgumentParser(description='Train XGBoost model with GridSearch and generate submission.')
    
    data_dir = '/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/data'
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search for hyperparameter tuning')
    args = parser.parse_args()
    
    best_params_all_splits = {}
    total_training_time = 0
    total_inference_time = 0
    num_splits = 0
    
    for split_folder in os.listdir(data_dir):
        if not split_folder.startswith('split'):
            continue
            
        print(f"\nProcessing {split_folder}...")
        num_splits += 1
        
        train_path = os.path.join(data_dir, split_folder, 'train.csv')
        test_path = os.path.join(data_dir, split_folder, 'test.csv')
        test_y_path = os.path.join(data_dir, split_folder, 'test_y.csv')
        
        train, test, test_y = load_data(train_path, test_path, test_y_path)
        X_train, y_train, X_test, y_test = preprocess_data(train, test, test_y)
        
        if args.grid_search:
            best_model, best_params, cv_results, grid_time = perform_grid_search(X_train, y_train)
            best_params_all_splits[split_folder] = best_params
            print(f"Grid search time for {split_folder}: {grid_time:.2f} seconds")
            save_grid_search_results(cv_results, split_folder)
        else:
            # Use default parameters if not performing grid search
            best_model, training_time = train_model(X_train, y_train)
            total_training_time += training_time
            print(f"Training time for {split_folder}: {training_time:.2f} seconds")
        
        # Evaluate model
        print("\nEvaluating model...")
        auc, y_pred_prob, inference_time = evaluate_model(best_model, X_test, y_test)
        total_inference_time += inference_time
        print(f"AUC for {split_folder}: {auc:.6f}")
        print(f"Inference time for {split_folder}: {inference_time:.2f} seconds")
        
        # Save the model
        save_model(best_model, split_folder, os.path.join(data_dir, 'models'))
        
        # Check if AUC meets the threshold
        if auc < 0.986:
            print(f"⚠️ Warning: AUC {auc:.6f} is below the threshold of 0.986.")
        else:
            print(f"✅ AUC meets the threshold.")
    
    # Print average timing statistics
    if num_splits > 0:
        print("\nTiming Statistics:")
        print(f"Average training time per split: {total_training_time/num_splits:.2f} seconds")
        print(f"Average inference time per split: {total_inference_time/num_splits:.2f} seconds")
        print(f"Total training time: {total_training_time:.2f} seconds")
        print(f"Total inference time: {total_inference_time:.2f} seconds")

if __name__ == "__main__":
    main()