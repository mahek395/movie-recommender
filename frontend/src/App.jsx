import { useState } from "react";
import { FiSearch } from "react-icons/fi";
import { HiSparkles } from "react-icons/hi2";
import { FaFilm } from "react-icons/fa";

function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchRecommendations = async (q = query) => {
    if (!q.trim()) return;

    setLoading(true);
    setError("");
    setResults([]);

    try {
      const res = await fetch(
        `http://127.0.0.1:5000/recommend?query=${encodeURIComponent(q)}`
      );
      if (!res.ok) throw new Error("Failed");

      const data = await res.json();
      setResults(data.results);
    } catch {
      setError("Backend not reachable");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      {/* HERO */}
      <section className="hero">
        <div className="logo">
          <FaFilm />
        </div>

        <h1 className="title">
          Movie<span>Finder</span>
        </h1>

        <p className="subtitle">
          Discover movies tailored to your taste using{" "}
          <span>AI-powered</span> recommendations
        </p>

        <div className="search-box">
          <FiSearch className="search-icon" />
          <input
            type="text"
            placeholder="Search movies, genres, or moods..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && fetchRecommendations()}
          />
          <button onClick={() => fetchRecommendations()}>
            <HiSparkles /> Search
          </button>
        </div>

        <p className="hint">
          Try: <span>“space adventure”</span>,{" "}
          <span>“romantic comedy”</span>,{" "}
          <span>“dark thriller”</span>
        </p>
      </section>

      {/* RESULTS */}
      <section className="results">
        {loading && <p className="status">Finding the best matches…</p>}
        {error && <p className="error">{error}</p>}

        <div className="grid">
          {results.map((movie, i) => (
            <div className="card" key={i}>
              <div className="card-icon">
                <FaFilm />
              </div>

              <h3>{movie.title}</h3>
              <p>📅 {movie.release_year ?? "N/A"}</p>
              <p>⭐ IMDb: {movie.imdb_score ?? "N/A"}</p>

              {i < 3 && <div className="badge">Recommended</div>}
            </div>
          ))}
        </div>
      </section>

      {/* DISCOVER SECTION (BOTTOM) */}
      {results.length === 0 && !loading && (
        <section className="discover">
          <div className="discover-card">
            <FaFilm className="discover-icon" />
          </div>

          <h2 className="discover-title">
            Discover Your Next Favorite Movie
          </h2>

          <p className="discover-subtitle">
            Start typing in the search bar above to get{" "}
            <span>AI-powered</span> movie recommendations based on your mood.
          </p>

          <div className="chips">
            {["Action", "Sci-Fi", "Comedy", "Thriller", "Romance"].map(
              (genre) => (
                <button
                  key={genre}
                  className="chip"
                  onClick={() => {
                    setQuery(genre);
                    fetchRecommendations(genre);
                  }}
                >
                  {genre}
                </button>
              )
            )}
          </div>
        </section>
      )}
    </div>
  );
}

export default App;
