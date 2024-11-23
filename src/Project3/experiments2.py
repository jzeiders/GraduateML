import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib  # For saving/loading models
from transformers import BertTokenizer, BertModel
import torch
import polars as pl
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
import re
import html


# -----------------------------
# Section 1: Setting Up
# -----------------------------

# Define constants
BERT_DIM = 768          # Dimension of BERT [CLS] embeddings
OPENAI_DIM = 1536       # Dimension of OpenAI embeddings
TRANSFORMATION_MATRIX_FILE = 'bert_to_openai_W.pkl'
BINARY_CLASSIFIER_FILE = '/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/data/models/xgb_model_splitsplit_1.joblib'  # Replace with your actual file
DATA_FILE = 'paired_embeddings.csv'  # Replace with your actual data file
torch.manual_seed(42)  # For reproducibility

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

    Parameters:
    - df: DataFrame containing 'text' column.
    - embedder: Instance of BERTEmbedder.
    - batch_size: Number of samples to process at once.

    Returns:
    - Numpy array of shape (n_samples, 768)
    """
    bert_embeddings = []
    for idx, text in enumerate(df['review']):
        embedding = embedder.get_embedding(text)
        bert_embeddings.append(embedding)
        if (idx + 1) % 1000 == 0:
            print(f"Computed BERT embeddings for {idx + 1} samples.")
    out = np.vstack(bert_embeddings)
    assert out.shape == (len(df), BERT_DIM), f"Expected shape: {(len(df), BERT_DIM)}, Got: {out.shape}"
    return out

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
    joblib_model = joblib.load(file_path)
    return joblib_model

# -----------------------------
# Section 7: Classifying New Text Inputs
# -----------------------------

class SentimentClassifier:
    def __init__(self, bert_embedder, W, binary_classifier, convert_model: LinearRegression):
        """
        Initializes the SentimentClassifier.

        Parameters:
        - bert_embedder: Instance of BERTEmbedder
        - W: Transformation matrix (768, 1536)
        - binary_classifier: Pre-trained binary classification model
        """
        self.embedder = bert_embedder
        self.W = W
        self.convert_model = convert_model
        self.classifier = binary_classifier

    def transform_bert_to_openai(self, bert_emb):
        """
        Transforms a BERT embedding to the OpenAI embedding space.

        Parameters:
        - bert_emb: Numpy array of shape (768,)

        Returns:
        - Numpy array of shape (1536,)
        """
        return self.convert_model.predict(bert_emb.reshape(1, -1))

    def classify_sentiment(self, text):
        """
        Classifies the sentiment of a given text.

        Parameters:
        - text: Input text string.

        Returns:
        - Probability of positive sentiment
        """
        bert_emb = self.embedder.get_embedding(text)
        openai_emb = self.transform_bert_to_openai(bert_emb)
        num_features = openai_emb.shape[1]
        column_names = [f"embedding_{i+1}" for i in range(num_features)]

        # Convert embedding to DataFrame with column names
        embedding_df = pd.DataFrame(openai_emb, columns=column_names)

        sentiment_prob = self.classifier.predict_proba(embedding_df)[0][1]
        return sentiment_prob


# -----------------------------
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')


def interpreability_analysis(text, sentiment_classifier: SentimentClassifier, path):
    """
    Performs composite interpretability analysis by assessing the importance of each word in the text
    based on its impact at sentence, trigram, and word levels.

    Parameters:
    - text: The original text (review) as a string.
    - sentiment_classifier: An instance of SentimentClassifier to compute sentiments.
    - max_sentence_length: Maximum number of characters to display for each sentence in logs.

    Outputs:
    - Saves a CSV table and an HTML file with highlighted text in the 'results' directory.
    """
    # Ensure results directory exists
    results_dir = os.path.join('results',path)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize data structures
    word_impacts = defaultdict(list)  # To accumulate impacts per word

    # Step 1: Compute original sentiment
    original_sentiment = sentiment_classifier.classify_sentiment(text)
    print(f"Original Sentiment: {original_sentiment}")

    # Step 2: Tokenize text into sentences
    sentences = sent_tokenize(text)
    unique_sentences = list(set(sentences))  # Ensure uniqueness
    print(f"Total Sentences: {len(unique_sentences)}")

    # Step 3: Sentence-Level Analysis
    print("\nStarting Sentence-Level Analysis...")
    for sentence in unique_sentences:
        # Remove the sentence from the text
        modified_text = text.replace(sentence, '')
        modified_text = ' '.join(modified_text.split())  # Clean up any extra spaces

        # Compute sentiment of modified text
        modified_sentiment = sentiment_classifier.classify_sentiment(modified_text)

        # Calculate impact as the difference
        impact = original_sentiment - modified_sentiment

        # Tokenize the sentence into words
        words_in_sentence = word_tokenize(sentence)
        words_in_sentence = [word.lower() for word in words_in_sentence if word.isalpha()]

        if not words_in_sentence:
            continue  # Skip if no valid words

        # Distribute impact equally among words in the sentence
        impact_per_word = impact / len(words_in_sentence)
        for word in words_in_sentence:
            word_impacts[word].append(impact_per_word)
        
    # Step 4: Trigram-Level Analysis
    print("\nStarting Trigram-Level Analysis...")
    # Generate trigrams
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    trigrams = [' '.join(trigram) for trigram in zip(tokens, tokens[1:], tokens[2:])]
    unique_trigrams = list(set(trigrams))
    print(f"Total Unique Trigrams: {len(unique_trigrams)}")

    for trigram in unique_trigrams:
        # Remove the trigram from the text
        pattern = re.escape(trigram)
        modified_text = re.sub(r'\b' + pattern + r'\b', '', text, flags=re.IGNORECASE)
        modified_text = ' '.join(modified_text.split())  # Clean up any extra spaces

        # Compute sentiment of modified text
        modified_sentiment = sentiment_classifier.classify_sentiment(modified_text)

        # Calculate impact as the difference
        impact = original_sentiment - modified_sentiment

        # Split trigram into words
        words_in_trigram = trigram.split()
        
        # Distribute impact equally among words in the trigram
        impact_per_word = impact / len(words_in_trigram)
        for word in words_in_trigram:
            word_impacts[word].append(impact_per_word)
        
    # Step 5: Word-Level Analysis
    print("\nStarting Word-Level Analysis...")
    unique_words = list(set(tokens))
    print(f"Total Unique Words: {len(unique_words)}")

    for word in unique_words:
        # Remove the word from the text (all occurrences)
        modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '', text, flags=re.IGNORECASE)
        modified_text = ' '.join(modified_text.split())  # Clean up any extra spaces

        # Compute sentiment of modified text
        modified_sentiment = sentiment_classifier.classify_sentiment(modified_text)

        # Calculate impact as the difference
        impact = original_sentiment - modified_sentiment

        # Assign impact directly to the word
        word_impacts[word].append(impact)

    # Step 6: Aggregate Impacts
    print("\nAggregating Impacts...")
    aggregated_impacts = {}
    for word, impacts in word_impacts.items():
        average_impact = sum(impacts) / len(impacts)
        aggregated_impacts[word] = average_impact

    # Step 7: Create a DataFrame for results
    results_df = pd.DataFrame({
        'Word': list(aggregated_impacts.keys()),
        'Average_Impact': list(aggregated_impacts.values())
    })

    # Step 8: Save the table as CSV
    csv_path = os.path.join(results_dir, 'word_importance_composite.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults table saved to '{csv_path}'.")

    # Step 9: Visualization - Highlighted Text
    print("\nGenerating Highlighted Text Visualization...")
    # Normalize impact scores for coloring
    impacts = results_df['Average_Impact']
    norm = Normalize(vmin=impacts.min(), vmax=impacts.max())
    cmap = plt.get_cmap('RdYlGn')  # Red for negative, Green for positive

    # Create a mapping from word to color
    word_color_map = {row['Word']: cm.colors.to_hex(cmap(norm(row['Average_Impact']))) for idx, row in results_df.iterrows()}

    # Function to color words
    def color_word(word):
        clean_word = re.sub(r'\W+', '', word.lower())
        color = word_color_map.get(clean_word, '#FFFFFF')  # Default to white if word not found
        return f'<span style="background-color: {color}">{html.escape(word)}</span>'

    # Split the original text into words while keeping punctuation
    def split_text_preserve_punctuation(text):
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

    tokens_with_punctuation = split_text_preserve_punctuation(text)
    highlighted_text = ' '.join([color_word(word) if word.isalpha() else html.escape(word) for word in tokens_with_punctuation])

    # Generate HTML content
    html_content = f"""
    <html>
    <head>
        <title>Interpretability Analysis - Highlighted Text</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                padding: 20px;
            }}
            .highlighted-text {{
                white-space: pre-wrap;
            }}
            span {{
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <h2>Interpretability Analysis - Highlighted Text</h2>
        <div class="highlighted-text">
            {highlighted_text}
        </div>
        <h3>Legend</h3>
        <p><span style="background-color: #d73027;">&nbsp;&nbsp;&nbsp;</span> Negative Impact</p>
        <p><span style="background-color: #ffffbf;">&nbsp;&nbsp;&nbsp;</span> Neutral Impact</p>
        <p><span style="background-color: #1a9850;">&nbsp;&nbsp;&nbsp;</span> Positive Impact</p>
    </body>
    </html>
    """

    # Save HTML file
    html_path = os.path.join(results_dir, 'highlighted_text.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Highlighted text visualization saved to '{html_path}'.")

    # Optional: Generate a color-coded bar chart of word impacts
    print("\nGenerating Color-Coded Bar Chart...")
    fig, ax = plt.subplots(figsize=(12, max(6, len(results_df) * 0.3)))  # Adjust height based on number of words
    results_df_sorted = results_df.sort_values(by='Average_Impact', ascending=True)
    bars = ax.barh(results_df_sorted['Word'], results_df_sorted['Average_Impact'], color='grey', edgecolor='black')

    # Apply color based on impact
    for bar, imp in zip(bars, results_df_sorted['Average_Impact']):
        bar.set_color(cmap(norm(imp)))

    ax.set_xlabel('Average Impact (Original - Modified Sentiment)')
    ax.set_title('Word Importance Based on Composite Sentiment Impact')
    fig.tight_layout()

    # Create a colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Average Impact')

    # Save the image
    image_path = os.path.join(results_dir, 'word_importance_composite.png')
    plt.savefig(image_path)
    plt.close()
    print(f"Color-coded bar chart saved to '{image_path}'.")
# -----------------------------
# Section 8: Main Execution Flow
# -----------------------------

def main():
    # Step 1: Load Data
    # Replace 'paired_embeddings.csv' with your actual data file path
    data_dir = "/Users/jzeiders/Documents/Code/Learnings/GraduateML/src/Project3/data/split_1"
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    test_labels_path = os.path.join(data_dir, 'test_y.csv')
    
    train_data = pl.read_csv(train_path)
    test_data = pl.read_csv(test_path)
    
    
    df = train_data[:10000]

    # Step 2: Initialize BERT Embedder
    bert_embedder = BERTEmbedder()

    # Step 3: Compute BERT Embeddings
    print("Computing BERT embeddings...")
    bert_embeddings = compute_bert_embeddings(df, bert_embedder)
    print("BERT embeddings computed.")

    # Step 4: Prepare Feature Matrices
    X = bert_embeddings  # Shape: (n_samples, 768)
    Y = np.vstack(df.drop(['id','sentiment', 'review']).to_numpy())  # Shape: (n_samples, 1536)

    # Step 5: Split Data into Training and Testing Sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.5, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Step 6: Fit Linear Regression Model
    linear_reg_model = fit_linear_regression(X_train, Y_train)
    

    # Step 7: Evaluate the Transformation
    evaluate_transformation(linear_reg_model, X_test, Y_test)
    


    # Step 8: Save the Transformation Matrix
    save_transformation_matrix(linear_reg_model, TRANSFORMATION_MATRIX_FILE)

    # Step 9: Load the Transformation Matrix and Binary Classifier
    W = load_transformation_matrix(TRANSFORMATION_MATRIX_FILE)
    binary_classifier = load_binary_classifier(BINARY_CLASSIFIER_FILE)

    # Step 10: Initialize Sentiment Classifier
    sentiment_classifier = SentimentClassifier(bert_embedder, W, binary_classifier, linear_reg_model)
    
    # New Predictions | Confirm the translation matrix is decent
    predictions = binary_classifier.predict(df[:,3:])
    
    errors = 0
    for i in range(1000):
        baseline_pred = predictions[i]
        bert_pred = round(sentiment_classifier.classify_sentiment(df['review'][i]))
        
        if(baseline_pred != bert_pred):
            errors +=1
            
    print(f"Error Rate on Classification: {errors / 1000}")
 
    for i in range(0,5):
        interpreability_analysis(test_data['review'][i], sentiment_classifier, str(i))

if __name__ == "__main__":
    main()
