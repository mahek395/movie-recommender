import pandas as pd
import numpy as np
import ast
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("data/raw_titles.csv")
print("Loaded:", df.shape)

# -----------------------------
# Parse list-like columns
# -----------------------------
def parse_list(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except:
        return []

df["genres_parsed"] = df["genres"].apply(parse_list)
df["countries_parsed"] = df["production_countries"].apply(parse_list)

# -----------------------------
# Create semantic content
# -----------------------------
def create_content(row):
    parts = []

    if pd.notna(row["title"]):
        parts.append(str(row["title"]))

    parts.extend(row["genres_parsed"])
    parts.extend(row["countries_parsed"])

    if pd.notna(row["release_year"]):
        parts.append(f"released in {int(row['release_year'])}")

    if pd.notna(row["runtime"]):
        parts.append(f"runtime {int(row['runtime'])} minutes")

    return " ".join(parts)

df["content"] = df.apply(create_content, axis=1)

# -----------------------------
# Sentence-BERT Model
# -----------------------------
print("Loading Sentence-BERT model...")
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")


print("Encoding movie content...")
embeddings = embedder.encode(
    df["content"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)

# -----------------------------
# Nearest Neighbors
# -----------------------------
nn_model = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute"
)

nn_model.fit(embeddings)

# -----------------------------
# Save everything
# -----------------------------
joblib.dump(embeddings, "model/movie_embeddings.joblib")
joblib.dump(nn_model, "model/semantic_nn_model.joblib")

df[["title", "release_year", "genres", "imdb_score"]].to_csv(
    "model/movie_metadata.csv", index=False
)

print("✅ Semantic model training complete")
