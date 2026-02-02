from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)
CORS(app)

# -----------------------------
# Load ML models once
# -----------------------------
print("🔄 Loading models...")

embeddings = joblib.load("model/movie_embeddings.joblib")
nn_model = joblib.load("model/semantic_nn_model.joblib")
movies = pd.read_csv("model/movie_metadata.csv")

embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

print("✅ Models loaded")

# -----------------------------
# Health check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Movie Recommendation API is live"
    })

# -----------------------------
# Recommendation endpoint
# -----------------------------
@app.route("/recommend", methods=["GET"])
def recommend():
    query = request.args.get("query")
    top_n = int(request.args.get("top_n", 10))

    if not query:
        return jsonify({"error": "query parameter is required"}), 400

    query_embedding = embedder.encode([query], convert_to_numpy=True)

    distances, indices = nn_model.kneighbors(
        query_embedding,
        n_neighbors=top_n
    )

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "title": str(movies.loc[idx, "title"]),
            "release_year": int(movies.loc[idx, "release_year"])
                if pd.notna(movies.loc[idx, "release_year"]) else None,
            "genres": str(movies.loc[idx, "genres"]),
            "imdb_score": float(movies.loc[idx, "imdb_score"])
                if pd.notna(movies.loc[idx, "imdb_score"]) else None,
            "similarity": float(1 - dist)
        })

    return jsonify({
        "query": query,
        "results": results
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=False)
