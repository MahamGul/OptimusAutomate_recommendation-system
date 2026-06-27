import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings


def build_content_model(movies):
    df = movies.copy()
    df['genres'] = df['genres'].fillna("").str.replace('|', ' ', regex=False)

    # Extract year from title for richer features
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').fillna("").astype(str)
    df['features'] = df['genres'] + " " + df['year']

    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(df['features'])
    sim = cosine_similarity(matrix)

    # Handle duplicate titles by keeping first occurrence
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    return sim, indices, df


def build_collab_model(ratings, movies):
    data = ratings.merge(movies, on='movieId')

    matrix = data.pivot_table(
        index='userId',
        columns='title',
        values='rating'
    ).fillna(0)

    sim = cosine_similarity(matrix.T)
    idx = pd.Series(range(len(matrix.columns)), index=matrix.columns)

    return sim, idx, matrix


def recommend_content(movie, sim, indices, movies_df, top_n=5):
    if movie not in indices:
        return []

    i = indices[movie]
    scores = list(enumerate(sim[i]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    movie_clean = movie.strip().lower()

    for j, s in scores:
        row = movies_df.iloc[j]

        # Skip same movie
        if row['title'].strip().lower() == movie_clean:
            continue

        genre_list = row['genres'].replace(' ', ', ')
        results.append({
            "title": row['title'],
            "score": float(s),
            "genres": genre_list
        })

        if len(results) == top_n:
            break

    return results


def recommend_collab(movie, sim, idx, matrix, top_n=5):
    if movie not in idx:
        return []

    i = idx[movie]
    scores = list(enumerate(sim[i]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    cols = matrix.columns
    results = []
    movie_clean = movie.strip().lower()

    for j, s in scores:
        if cols[j].strip().lower() == movie_clean:
            continue

        results.append({
            "title": cols[j],
            "score": float(s),
            "genres": ""
        })

        if len(results) == top_n:
            break

    return results


def hybrid_recommend(movie, content_sim, content_idx, movies_df,
                     collab_sim, collab_idx, matrix, top_n=5):
    """Blend content + collaborative scores."""

    c_results = {
        r["title"]: r["score"] * 0.5
        for r in recommend_content(
            movie, content_sim, content_idx, movies_df, top_n * 2
        )
    }

    cf_results = {
        r["title"]: r["score"] * 0.5
        for r in recommend_collab(
            movie, collab_sim, collab_idx, matrix, top_n * 2
        )
    }

    all_titles = set(c_results) | set(cf_results)
    blended = {}
    movie_clean = movie.strip().lower()

    for t in all_titles:
        if t.strip().lower() == movie_clean:
            continue
        blended[t] = c_results.get(t, 0) + cf_results.get(t, 0)

    top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]

    results = []
    for title, score in top:
        genres = ""
        if title in content_idx:
            genres = movies_df.iloc[content_idx[title]]['genres'].replace(' ', ', ')

        results.append({
            "title": title,
            "score": float(score),
            "genres": genres
        })

    return results