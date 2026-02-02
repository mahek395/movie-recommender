import pandas as pd
import joblib
import ast

# -----------------------------
# Load Model & Data
# -----------------------------
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
nn_model = joblib.load("model/nn_model.joblib")
movies = pd.read_csv("model/movie_metadata.csv")

# -----------------------------
# Helper functions
# -----------------------------
def parse_list(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []

movies["genres_parsed"] = movies["genres"].apply(parse_list)

def create_content(row):
    parts = []

    # title
    if pd.notna(row["title"]):
        parts.append(str(row["title"]))

    # genres
    for g in row["genres_parsed"]:
        parts.append(str(g))

    # release year
    if pd.notna(row["release_year"]):
        parts.append(str(int(row["release_year"])))

    return " ".join(parts)

# Create content safely
movies["content"] = movies.apply(create_content, axis=1)

# Vectorize
X = vectorizer.transform(movies["content"])

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_movie(title, top_n=10):
    matches = movies[movies["title"].str.lower().str.contains(title.lower(), na=False)]

    if matches.empty:
        print("❌ Movie not found!")
        return

    idx = matches.index[0]
    distances, indices = nn_model.kneighbors(X[idx], n_neighbors=top_n + 1)

    print(f"\n🎬 Recommendations for '{movies.loc[idx, 'title']}':\n")

    for i, movie_idx in enumerate(indices[0][1:]):
        print(
            f"{i+1}. {movies.loc[movie_idx, 'title']} "
            f"({movies.loc[movie_idx, 'release_year']})"
        )

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    while True:
        query = input("\nEnter movie name (or 'exit'): ")
        if query.lower() == "exit":
            break
        recommend_movie(query)
