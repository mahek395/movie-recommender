import pandas as pd
import numpy as np
import ast
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/raw_titles.csv")

print("Dataset loaded:", df.shape)

# -----------------------------
# 2. Parse list-like columns
# -----------------------------
def parse_list(x):
    if pd.isna(x):
        return []
    try:
        return [i.lower().strip() for i in ast.literal_eval(x)]
    except:
        return []

df["genres_parsed"] = df["genres"].apply(parse_list)
df["countries_parsed"] = df["production_countries"].apply(parse_list)

# -----------------------------
# 3. Create content feature
# -----------------------------
def create_content(row):
    parts = []
    if pd.notna(row["title"]):
        parts.append(row["title"])
    parts.extend(row["genres_parsed"])
    parts.extend(row["countries_parsed"])
    if pd.notna(row["release_year"]):
        parts.append(str(row["release_year"]))
    if pd.notna(row["runtime"]):
        parts.append(str(int(row["runtime"])) + "min")
    return " ".join(parts)

df["content"] = df.apply(create_content, axis=1)

print("Content feature created")

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(df["content"])

print("TF-IDF matrix shape:", X.shape)

# -----------------------------
# 5. Nearest Neighbors Model
# -----------------------------
nn_model = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute"
)

nn_model.fit(X)

print("Nearest Neighbors model trained")

# -----------------------------
# 6. Save Model & Vectorizer
# -----------------------------
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
joblib.dump(nn_model, "model/nn_model.joblib")

# Save minimal metadata
df[["title", "release_year", "genres", "imdb_score"]].to_csv(
    "model/movie_metadata.csv", index=False
)

print("Model saved successfully")
