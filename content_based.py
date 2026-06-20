import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Show sample before cleaning
print("Before cleaning:")
print(movies[['title', 'genres']].head())

# Replace | with spaces
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)
movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi', regex=False)
movies['genres'] = movies['genres'].str.replace('Film-Noir', 'FilmNoir', regex=False)
movies['genres'] = movies['genres'].str.replace('(no genres listed)', '', regex=False)

print("\nAfter cleaning:")
print(movies[['title', 'genres']].head())

cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

print("\nGenre matrix shape:")
print(genre_matrix.shape)

print("\nGenres found:")
print(cv.get_feature_names_out())

similarity = cosine_similarity(genre_matrix)

print("\nSimilarity matrix shape:")
print(similarity.shape)

movie_indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

def recommend(movie_name, top_n=5):
    if movie_name not in movie_indices:
        print("Movie not found!")
        return
    
    idx = movie_indices[movie_name]

    similarity_scores = list(enumerate(similarity[idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similarity_scores = similarity_scores[1:top_n+1]

    print(f"\nTop {top_n} recommendations for {movie_name}:\n")

    for movie_idx, score in similarity_scores:
        print(movies.iloc[movie_idx]['title'], " | Similarity:", round(score, 3))

recommend("Toy Story (1995)")