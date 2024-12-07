import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from collections import defaultdict
import json
import logging
import time
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self):
        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('recommender.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.ratings_matrix = None
        self.movie_data = {}  # Initialize as empty dictionary
        self.similarity_matrix = None
        self.popularity_ranks = None
        self.movie_ids = None
        
    def load_data(self, ratings_file: str, movies_file: str) -> None:
        """Load and process the ratings and movies data."""
        start_time = time.time()
        self.logger.info("Starting data loading process...")
        
        # Read ratings matrix
        self.logger.info(f"Loading ratings from {ratings_file}")
        self.ratings_matrix = pd.read_csv(ratings_file, index_col=0).T
        self.logger.info(f"Loaded ratings matrix with shape: {self.ratings_matrix.shape}")
        
        # Convert column names
        if not all(col.startswith('u') for col in self.ratings_matrix.columns):
            self.logger.debug("Converting column names to user ID format")
            self.ratings_matrix.columns = [f'u{col}' for col in self.ratings_matrix.columns]
        
        # Load movie data
        self.logger.info(f"Loading movies from {movies_file}")
        movies_loaded = 0
        try:
            with open(movies_file, 'r', encoding='latin-1') as f:
                for line in f:
                    try:
                        movie_id, title, genres = line.strip().split('::')
                        self.movie_data[f"m{movie_id}"] = {
                            'title': title,
                            'genres': genres.split('|')
                        }
                        movies_loaded += 1
                    except ValueError:
                        self.logger.warning(f"Skipped malformed line in movies file: {line.strip()}")
                        continue
        except Exception as e:
            self.logger.error(f"Error loading movies file: {str(e)}")
            
        self.logger.info(f"Successfully loaded {movies_loaded} movies")
        self.movie_ids = self.ratings_matrix.index.tolist()
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Data loading completed in {elapsed_time:.2f} seconds")

    def compute_popularity(self) -> List[Tuple[str, float]]:
        """Compute movie popularity based on number of ratings and average rating."""
        # Calculate metrics for each movie
        popularity_scores = []
        
        for movie_id in self.ratings_matrix.index:
            ratings = self.ratings_matrix.loc[movie_id]
            num_ratings = ratings.notna().sum()
            avg_rating = ratings.mean()
            
            # Only consider movies with at least 10 ratings and average rating > 3.5
            if num_ratings >= 10 and avg_rating > 3.5:
                # Popularity score combines both metrics
                popularity_score = (avg_rating * num_ratings) / self.ratings_matrix.shape[1]
                popularity_scores.append((movie_id, popularity_score))
        
        # Sort by popularity score
        popularity_scores.sort(key=lambda x: x[1], reverse=True)
        self.popularity_ranks = popularity_scores
        return popularity_scores

    def compute_similarity_matrix(self) -> None:
        """Compute movie similarity matrix using sklearn's cosine_similarity."""
        start_time = time.time()
        self.logger.info("Starting similarity matrix computation...")
        
        # Center the ratings matrix
        self.logger.info("Centering ratings matrix...")
        means = self.ratings_matrix.mean(axis=1)
        centered_matrix = self.ratings_matrix.sub(means, axis=0)
        
        # Fill NaN values with 0 for cosine similarity computation
        self.logger.info("Converting to dense matrix for similarity computation...")
        dense_matrix = centered_matrix.fillna(0).values
        
        # Compute similarities using sklearn
        self.logger.info("Computing cosine similarities...")
        similarities = cosine_similarity(dense_matrix)
        
        # Convert to DataFrame with proper indices
        self.logger.info("Converting similarities to DataFrame...")
        similarity_matrix = pd.DataFrame(
            similarities,
            index=self.movie_ids,
            columns=self.movie_ids
        )
        
        # Transform similarities to [0,1] range
        similarity_matrix = (similarity_matrix + 1) / 2
        
        # Keep only top 30 similarities per movie
        self.logger.info("Filtering top 30 similarities per movie...")
        for movie in self.movie_ids:
            # Get similarities for current movie
            movie_similarities = similarity_matrix.loc[movie].copy()
            # Set self-similarity to NaN
            movie_similarities[movie] = np.nan
            # Sort similarities
            sorted_indices = movie_similarities.sort_values(ascending=False).index
            # Keep only top 30
            keep_indices = sorted_indices[:30]
            # Set all other similarities to NaN
            similarity_matrix.loc[movie, ~similarity_matrix.columns.isin(keep_indices)] = np.nan
        
        self.similarity_matrix = similarity_matrix
        
        # Save matrix
        self.logger.info("Saving similarity matrix to file...")
        self.similarity_matrix.to_csv('similarity_matrix.csv')
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Similarity matrix computation completed in {elapsed_time:.2f} seconds")

    def myIBCF(self, newuser: pd.Series) -> List[str]:
        """Implement IBCF recommendation for a new user."""
        start_time = time.time()
        self.logger.info("Starting IBCF recommendation process...")
        
        if self.similarity_matrix is None:
            self.logger.info("Loading similarity matrix from file...")
            self.similarity_matrix = pd.read_csv('similarity_matrix.csv', index_col=0)
        
        self.logger.info(f"User has rated {newuser.notna().sum()} movies")
        predictions = []
        
        for movie_i in self.movie_ids:
            if pd.isna(newuser[movie_i]):
                similar_movies = self.similarity_matrix.loc[movie_i].dropna()
                rated_similar = similar_movies[newuser[similar_movies.index].notna()]
                
                if len(rated_similar) > 0:
                    weights = rated_similar.values
                    ratings = newuser[rated_similar.index].values
                    prediction = np.sum(weights * ratings) / np.sum(weights)
                    predictions.append((movie_i, prediction))
        
        self.logger.info(f"Generated {len(predictions)} predictions")
        
        # Sort and get recommendations
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended = [p[0] for p in predictions[:10]]
        
        # Fill with popular movies if needed
        if len(recommended) < 10:
            self.logger.info(f"Only {len(recommended)} predictions generated, filling with popular movies")
            popular_movies = [m[0] for m in self.popularity_ranks 
                            if m[0] not in recommended and pd.isna(newuser[m[0]])]
            recommended.extend(popular_movies[:10-len(recommended)])
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"IBCF recommendations completed in {elapsed_time:.2f} seconds")
        return recommended[:10]

    def generate_recommendations(self) -> None:
        """Print recommendations in table format."""
        # Get popular movies
        popular_movies = self.compute_popularity()[:10]
        
        # Create example new user with some ratings
        newuser = pd.Series(np.nan, index=self.movie_ids)
        # Add some example ratings
        example_ratings = {
            'm1': 5, 'm10': 4, 'm100': 3, 'm1510': 5,
            'm260': 4, 'm3212': 5
        }
        for movie, rating in example_ratings.items():
            if movie in newuser.index:
                newuser[movie] = rating
        
        ibcf_recommendations = self.myIBCF(newuser)
        
        # Print Popularity-Based Recommendations
        print("\nSystem I: Popularity-Based Recommendations")
        print("-" * 80)
        print(f"{'Movie ID':<10} {'Title':<50} {'Score':<10}")
        print("-" * 80)
        for movie_id, score in popular_movies:
            title = self.movie_data.get(movie_id, {}).get('title', 'Unknown')
            print(f"{movie_id:<10} {title[:50]:<50} {score:.3f}")
        
        # Print IBCF Recommendations
        print("\nSystem II: Item-Based Collaborative Filtering Recommendations")
        print("-" * 70)
        print(f"{'Movie ID':<10} {'Title':<60}")
        print("-" * 70)
        for movie_id in ibcf_recommendations:
            title = self.movie_data.get(movie_id, {}).get('title', 'Unknown')
            print(f"{movie_id:<10} {title[:60]:<60}")

def main():
    recommender = MovieRecommender()
    recommender.load_data('rating_matrix.csv', 'movies.dat')
    recommender.compute_similarity_matrix()
    recommender.generate_recommendations()

def test_recommendations():
    """Test the IBCF recommendations for two users."""
    recommender = MovieRecommender()
    recommender.load_data('rating_matrix.csv', 'movies.dat')
    recommender.compute_similarity_matrix()
    
    # Test for existing user u1181
    print("\nRecommendations for user u1181:")
    print("-" * 70)
    user_ratings = recommender.ratings_matrix.loc['u1181']
    recommendations = recommender.myIBCF(user_ratings)
    for movie_id in recommendations:
        title = recommender.movie_data.get(movie_id, {}).get('title', 'Unknown')
        print(f"{movie_id}: {title}")
    
    # Test for hypothetical user
    print("\nRecommendations for hypothetical user (m1613:5, m1755:4):")
    print("-" * 70)
    hypo_user = pd.Series(np.nan, index=recommender.movie_ids)
    hypo_user['m1613'] = 5
    hypo_user['m1755'] = 4
    recommendations = recommender.myIBCF(hypo_user)
    for movie_id in recommendations:
        title = recommender.movie_data.get(movie_id, {}).get('title', 'Unknown')
        print(f"{movie_id}: {title}")


if __name__ == "__main__":
    main()
    test_recommendations()