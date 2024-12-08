import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import os
from collections import defaultdict
import json
import logging
import time
from multiprocessing import Pool, cpu_count
from itertools import combinations


def compute_similarity_chunk(args):
    chunk, R, rated_mask = args
    results = []
    processed_count = 0
    for i, j in chunk:
        processed_count += 1
        # Find users who rated both movies
        common = rated_mask[:, i] & rated_mask[:, j]
        n_common = np.sum(common)
        
        if n_common >= 3:
            # Extract ratings from common users
            Ri = R[common, i]
            Rj = R[common, j]
            
            numerator = np.sum(Ri * Rj)
            denom = np.sqrt(np.sum(Ri**2)) * np.sqrt(np.sum(Rj**2))
            
            if denom != 0:
                cos_sim = numerator / denom
                # Transform similarity to be in [0, 1]
                sim = 0.5 + 0.5 * cos_sim
                results.append((i, j, sim))
    print(f"Processed {processed_count} pairs for chunk {chunk[0][0]}")
    return results


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
        
        # Read ratings matrix - no transpose needed now
        self.logger.info(f"Loading ratings from {ratings_file}")
        self.ratings_matrix = pd.read_csv(ratings_file, index_col=0)
        self.logger.info(f"Loaded ratings matrix with shape: {self.ratings_matrix.shape}")
        
        # Convert row indices if needed
        if not all(idx.startswith('u') for idx in self.ratings_matrix.index):
            self.logger.debug("Converting row indices to user ID format")
            self.ratings_matrix.index = [f'u{idx}' for idx in self.ratings_matrix.index]
        
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
        self.movie_ids = self.ratings_matrix.columns.tolist()
        
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


    def compute_transformed_cosine_similarity(self, Rdf):
        """Compute cosine similarities between movies in parallel."""
        # Check if cached matrix exists
        cache_file = 'part_2_matrix.csv'
        if os.path.exists(cache_file):
            self.logger.info("Loading cached similarity matrix...")
            try:
                similarities = pd.read_csv(cache_file, index_col=0)
                # Ensure the cached matrix matches our current data
                if set(similarities.columns) == set(Rdf.columns):
                    self.logger.info("Successfully loaded cached similarity matrix")
                    return similarities
                else:
                    self.logger.warning("Cached matrix columns don't match current data. Recomputing...")
            except Exception as e:
                self.logger.warning(f"Error loading cached matrix: {e}. Recomputing...")
        
        self.logger.info("Computing similarity matrix...")
        R = np.asarray(Rdf)
        num_users, num_movies = R.shape
        
        # Boolean mask of rated entries
        rated_mask = ~np.isnan(R)
        
        # Initialize similarity matrix
        similarities = pd.DataFrame(
            np.full((num_movies, num_movies), np.nan, dtype=np.float64), 
            index=Rdf.columns, 
            columns=Rdf.columns
        )
        
        # Generate all pairs of movie indices
        all_pairs = list(combinations(range(num_movies), 2))
        
        # Split pairs into chunks for parallel processing
        n_cores = cpu_count()
        chunk_size = len(all_pairs) // (n_cores * 4)  # Divide work into smaller chunks
        chunks = [all_pairs[i:i + chunk_size] for i in range(0, len(all_pairs), chunk_size)]
        
        # Prepare arguments for parallel processing
        chunk_args = [(chunk, R, rated_mask) for chunk in chunks]
        
        self.logger.info(f"Starting parallel similarity computation using {n_cores} cores...")
        
        # Process chunks in parallel
        with Pool(processes=n_cores) as pool:
            results = pool.map(compute_similarity_chunk, chunk_args)
        
        self.logger.info("Recombining results...")
        # Combine results
        for chunk_results in results:
            for i, j, sim in chunk_results:
                similarities.iloc[i, j] = sim
                similarities.iloc[j, i] = sim  # Maintain symmetry
        
        self.logger.info("Parallel similarity computation completed")
        
        # Cache the results
        self.logger.info("Saving similarity matrix to cache...")
        similarities.to_csv(cache_file)
        
        return similarities

    def compute_similarity_matrix(self) -> None:
        """Compute movie similarity matrix using sklearn's cosine_similarity."""
        start_time = time.time()
        self.logger.info("Starting similarity matrix computation...")
        
        # Center the ratings matrix by user (row) means
        self.logger.info("Centering ratings matrix...")
        self.user_means = self.ratings_matrix.mean(axis=1)  # Mean rating for each user
        centered_matrix = self.ratings_matrix.sub(self.user_means, axis=0)  # Subtract along rows
        print(centered_matrix)
        
        # Fill NaN values with 0 for cosine similarity computation
        self.logger.info("Converting to dense matrix for similarity computation...")
        
        # Compute similarities using sklearn on the transposed matrix for movie-movie similarities
        self.logger.info("Computing cosine similarities...")
        print(centered_matrix.shape)
        similarity_matrix = self.compute_transformed_cosine_similarity(centered_matrix)
        print(similarity_matrix)
        
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

    def validate_recommendations(self, user_id: str, recommendations: List[str]) -> None:
        """Validate recommendations against known correct movies."""
        if user_id == 'u1181':
            required_top_3 = ['m3732', 'm749', 'm3899']
            expected_label = "Required top 3"
            top_k = 3
        else:  # hypothetical user
            required_top_10 = ['m1017', 'm2805', 'm3269', 'm691', 'm74', 
                              'm765', 'm1100', 'm1468', 'm1541', 'm158']
            required_top_3 = required_top_10  # Use all 10 for validation
            expected_label = "Expected top 10"
            top_k = 10
            
        # Check top k
        top_k_found = [movie for movie in required_top_3 if movie in recommendations[:top_k]]
        print(f"\nValidation Results for {'u1181' if user_id == 'u1181' else 'hypothetical user'}:")
        print("-" * 50)
        print(f"{expected_label}: {', '.join(required_top_3)}")
        print(f"Found in top {top_k}: {', '.join(top_k_found)}")
        print(f"Top {top_k} accuracy: {len(top_k_found)}/{len(required_top_3)}")
        
        # Check if required movies are in rest of recommendations
        remaining_required = [m for m in required_top_3 if m not in recommendations[:top_k]]
        found_in_rest = [m for m in remaining_required if m in recommendations[top_k:]]
        if remaining_required:
            print(f"\nExpected movies found later in list: {', '.join(found_in_rest)}")
        
        # Print full recommendation list with positions
        print("\nFull recommendation list with positions:")
        for i, movie_id in enumerate(recommendations, 1):
            title = self.movie_data.get(movie_id, {}).get('title', 'Unknown')
            is_required = '(EXPECTED)' if movie_id in required_top_3 else ''
            print(f"{i}. {movie_id}: {title} {is_required}")

def myIBCF(logger: logging.Logger, similarity_matrix: pd.DataFrame, newuser: pd.Series) -> List[str]:
    """Implement IBCF recommendation for a new user."""
    start_time = time.time()
    logger.info("Starting IBCF recommendation process...")
    
    if similarity_matrix is None:
        logger.info("Loading similarity matrix from file...")
        similarity_matrix = pd.read_csv('similarity_matrix.csv', index_col=0)
    
    # Calculate mean rating for new user
    user_mean = newuser[newuser.notna()].mean()
    
    logger.info(f"User has rated {newuser.notna().sum()} movies")
    predictions = []

    for movie_i in similarity_matrix.index:
        if pd.isna(newuser[movie_i]):
            similar_movies = similarity_matrix.loc[movie_i].dropna()
            rated_similar = similar_movies[newuser[similar_movies.index].notna()]
            
            if len(rated_similar) > 0:
                weights = rated_similar.values
                ratings = newuser[rated_similar.index].values
                # Center the ratings using user's mean
                centered_ratings = ratings - user_mean
                # Compute prediction with baseline adjustment
                weighted_sum = np.sum(weights * centered_ratings)
                weight_sum = np.sum(np.abs(weights))  # Use absolute weights for normalization
                if weight_sum > 0:  # Avoid division by zero
                    prediction = user_mean + (weighted_sum / weight_sum)
                    # Clip predictions to valid rating range (1-5)
                    prediction = np.clip(prediction, 1, 5)
                    predictions.append((movie_i, prediction))
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Sort and get recommendations
    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended = [p[0] for p in predictions[:10]]
    
    elapsed_time = time.time() - start_time
    logger.info(f"IBCF recommendations completed in {elapsed_time:.2f} seconds")
    return recommended[:10]

def main():
    recommender = MovieRecommender()
    recommender.load_data('rating_matrix.csv', 'movies.dat')
    recommender.compute_similarity_matrix()
    recommender.generate_recommendations()

def test_recommendations():
    """Test the IBCF recommendations for two users."""
    
    recommender = MovieRecommender()
    recommender.load_data('rating_matrix.csv', 'movies.dat')
    user_ratings = recommender.ratings_matrix.loc['u1181']
    recommender.compute_similarity_matrix()
    
    # Test for existing user u1181
    print("\nRecommendations for user u1181:")
    print("-" * 70)
    recommendations = myIBCF(recommender.logger, recommender.similarity_matrix, user_ratings)
    recommender.validate_recommendations('u1181', recommendations)
    
    # Test for hypothetical user
    print("\nRecommendations for hypothetical user (m1613:5, m1755:4):")
    print("-" * 70)
    hypo_user = pd.Series(np.nan, index=recommender.movie_ids)
    hypo_user['m1613'] = 5
    hypo_user['m1755'] = 4
    recommendations = myIBCF(recommender.logger, recommender.similarity_matrix, hypo_user)
    recommender.validate_recommendations('hypothetical', recommendations)


if __name__ == "__main__":
    # main()
    test_recommendations()