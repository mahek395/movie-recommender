import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load models and data
# -----------------------------
embeddings = joblib.load("model/movie_embeddings.joblib")
nn_model = joblib.load("model/semantic_nn_model.joblib")
movies = pd.read_csv("model/movie_metadata.csv")

# Load the SAME embedding model used during training
embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# -----------------------------
# Semantic Recommendation Function
# -----------------------------
def recommend_movie(query, top_n=10):
    """
    Takes a free-text query (keyword, phrase, concept)
    and returns semantically similar movies.
    """

    # Encode user query
    query_embedding = embedder.encode(
        [query],
        convert_to_numpy=True
    )

    # Find nearest movies
    distances, indices = nn_model.kneighbors(
        query_embedding,
        n_neighbors=top_n
    )

    print(f"\n🎬 Semantic Recommendations for '{query}':\n")

    for i, movie_idx in enumerate(indices[0]):
        title = movies.loc[movie_idx, "title"]
        year = movies.loc[movie_idx, "release_year"]

        print(f"{i+1}. {title} ({year})")

# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":
    print("🎥 Semantic Movie Recommendation System")
    print("Type anything: movie name, genre, concept, mood")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() == "exit":
            print("👋 Exiting recommender")
            break

        if query == "":
            print("⚠ Please enter a valid query")
            continue

        recommend_movie(query)
