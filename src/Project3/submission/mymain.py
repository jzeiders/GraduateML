import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import os

def load_data(train_path, test_path, test_y_path=None):
    """
    Load training and testing data using Polars.
    Returns test_y as None if test_y_path is not provided.
    """
    train = pl.read_csv(train_path)
    test = pl.read_csv(test_path)
    test_y = pl.read_csv(test_y_path) if test_y_path else None
    
    return train, test, test_y

def preprocess_data(train, test, test_y=None):
    """
    Preprocess the data by selecting relevant features and separating labels.
    """
    X_train = train.drop(['id', 'sentiment', 'review'])
    y_train = train['sentiment']
    
    X_test = test.drop(['id', 'review'])
    y_test = test_y['sentiment'] if test_y is not None else None
    
    return X_train, y_train, X_test, y_test

def train_model(X_train, y_train):
    """
    Train a logistic regression model with optimized parameters.
    """
    model = LogisticRegression(
        solver='liblinear',
        max_iter=1000,
        C=10,  # Based on previous grid search results
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using AUC score.
    """
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc

def generate_submission(model, test, output_path='mysubmission.csv'):
    """
    Generate submission file with predicted probabilities.
    """
    X_test = test.drop(['id', 'review'])
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    submission = pl.DataFrame({
        'id': test['id'],
        'prob': y_pred_prob
    })
    
    submission.write_csv(output_path)

def main():
    """
    Main function to train model and generate predictions.
    """
    # Define data paths
    data_dir = ''  # Assuming data directory is in the same level as the script
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    test_y_path = None
    
    # Load and preprocess data
    train, test, test_y = load_data(train_path, test_path, test_y_path)
    X_train, y_train, X_test, y_test = preprocess_data(train, test, test_y)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model if test labels are available
    if y_test is not None:
        auc = evaluate_model(model, X_test, y_test)
        print(f"Model AUC: {auc:.6f}")
    
    # Generate submission file
    generate_submission(model, test)

if __name__ == "__main__":
    main()
