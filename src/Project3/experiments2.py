import os
import json
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from transformers import BertTokenizer, BertModel
import torch
import requests
import tempfile
from lime.lime_text import LimeTextExplainer
import hashlib
from tqdm import tqdm


# -----------------------------
# Section 1: Setting Up
# -----------------------------

# Constants
BERT_DIM = 768
OPENAI_DIM = 1536
TRANSFORMATION_MATRIX_FILE = 'bert_to_openai_W.pkl'
BINARY_CLASSIFIER_FILE = 'https://raw.githubusercontent.com/jzeiders/GraduateML/main/src/Project3/submission/xgb_model_splitsplit_1.joblib'

torch.manual_seed(42)

# -----------------------------
# Section 2: Loading and Preparing Data
# -----------------------------

def load_data(file_path):
    """
    Loads paired embeddings data from a CSV file.

    The CSV file is expected to have the following columns:
    - 'text': The original text data.
    - 'openai_embedding': Stringified list of OpenAI embeddings (length 1536).

    BERT embeddings will be computed on the fly.

    Parameters:
    - file_path: Path to the CSV file.

    Returns:
    - DataFrame with 'text' and 'openai_embedding' columns.
    """
    df = pd.read_csv(file_path)
    # Convert the stringified embeddings to actual numpy arrays
    df['openai_embedding'] = df['openai_embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))
    return df

# -----------------------------
# Section 3: Computing BERT Embeddings
# -----------------------------
class CacheManager:
    def __init__(self, cache_dir: str):
        """
        Manages the file-backed cache for embeddings.
        
        Parameters:
        - cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.embeddings_dir = self.cache_dir / "embeddings"
        
        # Create cache directories if they don't exist
        self.cache_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from disk or initialize if not exists."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _get_hash(self, text: str) -> str:
        """Generate a hash for the input text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache if it exists.
        
        Returns None if not found.
        """
        text_hash = self._get_hash(text)
        if text_hash not in self.metadata:
            return None
            
        embedding_path = self.embeddings_dir / f"{text_hash}.npy"
        if not embedding_path.exists():
            return None
            
        return np.load(embedding_path)
    
    def put(self, text: str, embedding: np.ndarray):
        """Store embedding in cache."""
        text_hash = self._get_hash(text)
        
        # Save embedding
        embedding_path = self.embeddings_dir / f"{text_hash}.npy"
        np.save(embedding_path, embedding)
        
        # Update metadata
        self.metadata[text_hash] = {
            'text': text,
            'shape': embedding.shape
        }
        self._save_metadata()
    
    def clear(self):
        """Clear all cache contents."""
        # Remove all embedding files
        for f in self.embeddings_dir.glob("*.npy"):
            f.unlink()
        
        # Clear metadata
        self.metadata = {}
        self._save_metadata()

class BERTEmbedder:
    def __init__(self, model_name='bert-base-uncased', cache_dir='/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/bert_cache'):
        """
        BERT embedder with file-backed caching.
        
        Parameters:
        - model_name: Name of the BERT model to use
        - cache_dir: Directory to store embedding cache
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        self.cache = CacheManager(cache_dir)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for text, using cache if available.
        
        Parameters:
        - text: Input text string
        
        Returns:
        - Numpy array of shape (768,)
        """
        # Check cache first
        cached_embedding = self.cache.get(text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Compute embedding if not in cache
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                  padding=True, max_length=512)
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        assert cls_embedding.shape == (BERT_DIM,), \
            f"Expected shape: {(BERT_DIM,)}, Got: {cls_embedding.shape}"
        
        # Store in cache
        self.cache.put(text, cls_embedding)
        
        return cls_embedding
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()

def compute_bert_embeddings(df: pl.DataFrame, embedder: BERTEmbedder, batch_size=32):
    """
    Computes BERT embeddings for all texts in the DataFrame.
    """
    bert_embeddings = []
    for text in tqdm(df['review'], total=len(df), desc="Computing BERT embeddings"):
        embedding = embedder.get_embedding(text)
        bert_embeddings.append(embedding)
            
    embeddings = np.vstack(bert_embeddings)
    assert embeddings.shape == (len(df), BERT_DIM)
    return embeddings

# -----------------------------
# Section 4: Fitting the Linear Regression Model
# -----------------------------

def fit_linear_regression(X_train, Y_train):
    """
    Fits a linear regression model to map BERT embeddings to OpenAI embeddings.

    Parameters:
    - X_train: Training BERT embeddings (n_samples, 768)
    - Y_train: Training OpenAI embeddings (n_samples, 1536)

    Returns:
    - Trained LinearRegression model
    """
    linear_reg = LinearRegression()
    print("Fitting the linear regression model...")
    linear_reg.fit(X_train, Y_train)
    print("Model fitting complete.")
    return linear_reg

# -----------------------------
# Section 5: Evaluating the Transformation
# -----------------------------

def evaluate_transformation(model, X_test, Y_test):
    """
    Evaluates the linear regression model using Mean Squared Error.

    Parameters:
    - model: Trained LinearRegression model
    - X_test: Testing BERT embeddings (n_samples, 768)
    - Y_test: Testing OpenAI embeddings (n_samples, 1536)

    Returns:
    - Mean Squared Error
    """
    Y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    return mse

# -----------------------------
# Section 6: Saving and Loading Models
# -----------------------------

def save_transformation_matrix(model, file_path):
    """
    Saves the transformation matrix W to a file.

    Parameters:
    - model: Trained LinearRegression model
    - file_path: Path to save the transformation matrix
    """
    W = model.coef_.T  # Shape: (768, 1536)
    joblib.dump(W, file_path)
    print(f"Transformation matrix W saved as '{file_path}'.")

def load_transformation_matrix(file_path):
    """
    Loads the transformation matrix W from a file.

    Parameters:
    - file_path: Path to the transformation matrix file.

    Returns:
    - Numpy array of shape (768, 1536)
    """
    W = joblib.load(file_path)
    print(f"Transformation matrix W loaded from '{file_path}'.")
    return W

def load_binary_classifier(file_path: str) -> LogisticRegression:
    """
    Loads the pre-trained binary classification model.

    Parameters:
    - file_path: Path to the binary classifier file.

    Returns:
    - Loaded binary classification model
    """
import requests
import joblib
import tempfile
from typing import Union
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse

def load_binary_classifier(file_path: Union[str, bytes]) -> LogisticRegression:
    """
    Loads the pre-trained binary classification model from either a local path or remote URL.

    Parameters:
    - file_path: Path to the binary classifier file. Can be either:
                - Local file path (str)
                - Remote URL (str)
                - Bytes object containing the model data

    Returns:
    - Loaded binary classification model

    Raises:
    - ValueError: If the URL is invalid or file cannot be downloaded
    - Exception: If model loading fails
    """
    # Check if input is already bytes
    if isinstance(file_path, bytes):
        with tempfile.NamedTemporaryFile() as tmp_file:
            tmp_file.write(file_path)
            tmp_file.flush()
            return joblib.load(tmp_file.name)
    
    # Check if the path is a URL
    parsed = urlparse(file_path)
    if parsed.scheme in ('http', 'https'):
        try:
            # Download the file
            response = requests.get(file_path)
            response.raise_for_status()
            
            # Save to temporary file and load
            with tempfile.NamedTemporaryFile() as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file.flush()
                return joblib.load(tmp_file.name)
        except Exception as e:
            raise ValueError(f"Failed to download or load binary classifier from URL: {e}")
    else:
        return joblib.load(file_path)

# -----------------------------
# Section 7: Classifying New Text Inputs
# -----------------------------

class SentimentClassifier:
    def __init__(self, bert_embedder, binary_classifier, convert_model: LinearRegression):
        self.embedder = bert_embedder
        self.convert_model = convert_model
        self.classifier = binary_classifier

    def classify_sentiment(self, text: str) -> float:
        """Classifies the sentiment of a given text."""
        bert_emb = self.embedder.get_embedding(text)
        openai_emb = self.convert_model.predict(bert_emb.reshape(1, -1))
        
        # Create DataFrame with numbered embedding columns
        embedding_df = pd.DataFrame(
            openai_emb, 
            columns=[f"embedding_{i+1}" for i in range(openai_emb.shape[1])]
        )
        
        return self.classifier.predict_proba(embedding_df)[0][1]

def analyze_multiple_reviews(test_data, sentiment_classifier: SentimentClassifier):
    """
    Analyzes 5 positive and 5 negative reviews and combines their visualizations.
    """
    # Create results directory
    results_dir = os.path.join('results', 'combined_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    positive_indices = []
    negative_indices = []
    
    # Process reviews with progress bar
    pbar = tqdm(test_data['review'], desc="Finding reviews")
    for idx, review in enumerate(pbar):
        if len(positive_indices) >= 5 and len(negative_indices) >= 5:
            break
            
        sentiment = sentiment_classifier.classify_sentiment(review)
        
        if len(positive_indices) < 5 and sentiment > 0.8:
            positive_indices.append(idx)
            pbar.set_postfix({'positive': len(positive_indices), 'negative': len(negative_indices)})
        elif len(negative_indices) < 5 and sentiment < 0.2:
            negative_indices.append(idx)
            pbar.set_postfix({'positive': len(positive_indices), 'negative': len(negative_indices)})
    
    if len(positive_indices) < 5 or len(negative_indices) < 5:
        print("Warning: Could not find enough reviews of each type.")
        print(f"Found {len(positive_indices)} positive and {len(negative_indices)} negative reviews.")
    
    # Create figure with subplots for all reviews
    fig = plt.figure(figsize=(20, 25))
    
    # Process each review with progress bar
    for idx, review_idx in enumerate(tqdm(negative_indices + positive_indices, desc="Generating LIME explanations")):
        review = test_data['review'][review_idx]
        sentiment = sentiment_classifier.classify_sentiment(review)
        
        # Create LIME explanation
        def predictor(texts):
            probs = []
            for t in texts:
                p = sentiment_classifier.classify_sentiment(t)
                probs.append(np.array([1-p, p]))
            return np.array(probs)
        
        explainer = LimeTextExplainer(class_names=['negative', 'positive'])
        print(f"Generating LIME explanation for review {idx + 1}")
        exp = explainer.explain_instance(
            review, 
            predictor,
            num_features=30,
            num_samples=10
        )
        
        # Get feature weights
        features_with_weights = exp.as_list()
        word_weights = {feature[0]: feature[1] for feature in features_with_weights}
        
        # Create subplot
        ax = fig.add_subplot(5, 2, idx + 1)
        
        # Split review into words and create colored text
        words = review.split()
        y_position = 1.0
        x_position = 0.0
        line_height = 0.05
        current_line = []
        current_line_width = 0
        max_line_width = 0.9  # Maximum width of a line
        
        # Normalize weights for coloring
        weights = np.array(list(word_weights.values()))
        max_abs_weight = max(abs(weights.min()), abs(weights.max())) if len(weights) > 0 else 1
        
        for word in words:
            clean_word = word.lower().strip('.,!?()[]{}":;')
            weight = word_weights.get(clean_word, 0)
            
            # Calculate word width (approximate)
            word_width = len(word) * 0.01
            
            # Check if we need to start a new line
            if current_line_width + word_width > max_line_width:
                y_position -= line_height
                x_position = 0.0
                current_line_width = 0
                current_line = []
            
            # Add word to visualization
            if clean_word in word_weights:
                color = 'green' if weight > 0 else 'red'
                alpha = min(abs(weight) / max_abs_weight, 1)
            else:
                color = 'gray'
                alpha = 0.2
            
            ax.text(x_position, y_position, word + ' ', 
                   color=color, alpha=alpha, transform=ax.transAxes)
            
            x_position += word_width
            current_line_width += word_width
            current_line.append(word)
        
        # Set title and remove axes
        sentiment_label = "Positive" if sentiment > 0.5 else "Negative"
        ax.set_title(f"Review {idx + 1} ({sentiment_label}, prob: {sentiment:.3f})")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the combined visualization
    output_path = os.path.join(results_dir, 'combined_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined analysis saved to: {output_path}")

# -----------------------------
# Section 8: Main Execution Flow
# -----------------------------

def main():
    # Load data
    data_dir = "/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/data/split_1"
    print("Loading data...")
    train_data = pl.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pl.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Use subset for training
    train_subset = train_data[:1000]

    # Initialize BERT embedder and compute embeddings
    bert_embedder = BERTEmbedder()
    bert_embeddings = compute_bert_embeddings(train_subset, bert_embedder)

    # Prepare features
    X = bert_embeddings
    Y = np.vstack(train_subset.drop(['id','sentiment', 'review']).to_numpy())

    # Train model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    print("Training linear regression model...")
    linear_reg_model = fit_linear_regression(X_train, Y_train)
    
    # Evaluate and save model
    mse = evaluate_transformation(linear_reg_model, X_test, Y_test)
    save_transformation_matrix(linear_reg_model, TRANSFORMATION_MATRIX_FILE)

    # Load binary classifier and create sentiment classifier
    binary_classifier = load_binary_classifier(BINARY_CLASSIFIER_FILE)
    sentiment_classifier = SentimentClassifier(bert_embedder, binary_classifier, linear_reg_model)
    
    # Analyze reviews
    analyze_multiple_reviews(test_data, sentiment_classifier)

if __name__ == "__main__":
    main()
