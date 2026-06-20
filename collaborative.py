import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies shape:", movies.shape)
print("Ratings shape:", ratings.shape)

# Merge
movie_ratings = ratings.merge(movies, on='movieId')

# User-Movie Matrix
user_movie_matrix = movie_ratings.pivot_table(
    index='userId',
    columns='title',
    values='rating'
)

# Normalize (mean centering)
user_movie_matrix_norm = user_movie_matrix.subtract(
    user_movie_matrix.mean(axis=1),
    axis=0
).fillna(0)

print("\nUser-Movie Matrix shape:", user_movie_matrix.shape)

# Movie similarity (based on normalized data)
movie_similarity = cosine_similarity(user_movie_matrix_norm.T)

print("Movie similarity matrix shape:", movie_similarity.shape)

# Correct index mapping (IMPORTANT: must match similarity matrix)
movie_indices = pd.Series(
    range(len(user_movie_matrix.columns)),
    index=user_movie_matrix.columns
)

def recommend(movie_name, top_n=5):
    if movie_name not in movie_indices:
        print("Movie not found!")
        return

    idx = movie_indices[movie_name]

    similarity_scores = list(enumerate(movie_similarity[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:top_n+1]

    print(f"\nTop {top_n} collaborative recommendations for {movie_name}:\n")

    movie_titles = user_movie_matrix.columns

    for movie_idx, score in similarity_scores:
        print(f"{movie_titles[movie_idx]} | Similarity: {round(score, 3)}")


# Test
recommend("Toy Story (1995)")