import logging
import math
import os
import pickle
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from flask import Flask, jsonify, render_template, request


# Configure app-wide logging so hosting platforms can surface startup/model issues.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class KDramaRecommender:
    """Load model artifacts and generate K-Drama recommendations."""

    # Store the artifact filenames in one place so future notebook exports are easy to wire in.
    DRAMAS_FILE = "dramas.pkl"
    METADATA_FILE = "metadata.pkl"
    NEIGHBORS_FILE = "nn.pkl"
    VECTORS_FILE = "vectors.pkl"

    def __init__(self, data_dir: Path):
        # Keep model state on the instance so routes can reuse loaded artifacts across requests.
        self.data_dir = data_dir
        self.dramas_df: pd.DataFrame = pd.DataFrame()
        self.metadata_df: pd.DataFrame = pd.DataFrame()
        self.neighbor_model: Any = None
        self.vectors: Any = None

        # Precompute lookup data used by the form, suggestions, and recommendation search.
        self.drama_names: List[str] = []
        self.title_lookup: Dict[str, str] = {}
        self.indices: pd.Series = pd.Series(dtype=int)
        self.genres: List[str] = []
        self.load_error = ""

        self._load_data()

    def _load_pickle(self, filename: str) -> Any:
        # Load a serialized artifact from the project root using an explicit binary read.
        with open(self.data_dir / filename, "rb") as file:
            return pickle.load(file)

    def _load_data(self) -> None:
        # Load the dataframe and nearest-neighbor artifacts exported from the notebook.
        try:
            self.dramas_df = self._load_pickle(self.DRAMAS_FILE)
            self.metadata_df = self._load_pickle(self.METADATA_FILE)
            self.neighbor_model = self._load_pickle(self.NEIGHBORS_FILE)
            self.vectors = self._load_pickle(self.VECTORS_FILE)
            self._build_indices()
            logger.info("Loaded K-Drama recommendation artifacts successfully.")
        except FileNotFoundError as error:
            self.load_error = f"Missing model file: {error.filename}"
            logger.exception(self.load_error)
        except ModuleNotFoundError as error:
            self.load_error = f"Missing Python dependency required by a model file: {error.name}"
            logger.exception(self.load_error)
        except Exception as error:
            self.load_error = "Could not load recommendation artifacts."
            logger.exception("Model startup failed: %s", error)

    def _build_indices(self) -> None:
        # Build title and genre indexes once so requests stay fast.
        self.drama_names = sorted(self.dramas_df["name"].dropna().astype(str).tolist())
        self.title_lookup = {name.lower(): name for name in self.drama_names}
        self.indices = pd.Series(
            self.dramas_df.index,
            index=self.dramas_df["name"].str.lower()
        ).drop_duplicates()
        self.genres = sorted({
            genre
            for genres in self.metadata_df["genres"].dropna()
            for genre in self._split_values(genres)
        })

    @staticmethod
    def _split_values(value: Any) -> List[str]:
        # Convert comma-separated metadata fields into clean lists for filtering and display.
        if pd.isna(value):
            return []
        return [item.strip() for item in str(value).split(",") if item.strip()]

    def _normalize_record(self, row: pd.Series) -> Dict[str, Any]:
        # Convert pandas values into template/API-safe primitives and add list-friendly fields.
        record = row.to_dict()
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = ""
            elif isinstance(value, float) and value.is_integer():
                record[key] = int(value)

        record["genres_list"] = self._split_values(record.get("genres"))
        record["cast_list"] = self._split_values(record.get("main_role"))
        return record

    def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        # Return direct substring matches plus fuzzy matches when a user mistypes a title.
        query = (query or "").strip().lower()
        if not query:
            return []

        direct = [name for name in self.drama_names if query in name.lower()][:limit]
        fuzzy = get_close_matches(
            query,
            [name.lower() for name in self.drama_names],
            n=limit,
            cutoff=0.45
        )
        merged = direct + [self.title_lookup.get(name, name) for name in fuzzy]
        return list(dict.fromkeys(merged))[:limit]

    def _vector_at(self, index: int) -> Any:
        # Select one vector while supporting both scipy sparse matrices and numpy-like arrays.
        if hasattr(self.vectors, "getrow"):
            return self.vectors.getrow(index)
        return self.vectors[index:index + 1]

    def _neighbor_scores(self, index: int, pool_size: int) -> List[Tuple[int, float]]:
        # Ask the notebook-trained nearest-neighbor model for candidate dramas and match scores.
        query_vector = self._vector_at(index)
        neighbor_count = min(pool_size + 1, len(self.dramas_df))
        distances, neighbors = self.neighbor_model.kneighbors(
            query_vector,
            n_neighbors=neighbor_count
        )

        scores = []
        for item_idx, distance in zip(neighbors[0], distances[0]):
            if int(item_idx) == int(index):
                continue
            match_score = max(0.0, 1.0 - float(distance))
            scores.append((int(item_idx), match_score))
        return scores

    def recommend(
        self,
        drama_name: str,
        limit: int = 8,
        genre: str = "",
        min_rating: float = 0.0,
        sort_by: str = "match",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        # Validate the requested title and return suggestions instead of failing on typos.
        drama_name = (drama_name or "").strip()
        key = drama_name.lower()
        if self.load_error or key not in self.indices:
            return [], self.get_suggestions(drama_name)

        # Fetch a larger candidate pool so genre and rating filters still have room to work.
        recommendations = []
        query_index = int(self.indices[key])
        for item_idx, score in self._neighbor_scores(query_index, pool_size=60):
            row = self._normalize_record(self.metadata_df.iloc[item_idx])
            row["match_score"] = round(score * 100, 1)

            if genre and genre not in row["genres_list"]:
                continue

            rating = row.get("rating") or 0
            if rating and not math.isnan(float(rating)) and float(rating) < min_rating:
                continue

            recommendations.append(row)

        # Sort the filtered results according to the user's selected ranking mode.
        if sort_by == "rating":
            recommendations.sort(key=lambda row: float(row.get("rating") or 0), reverse=True)
        elif sort_by == "popularity":
            recommendations.sort(key=lambda row: float(row.get("popularity") or 999999))

        return recommendations[:limit], []

    def get_ui_options(self) -> Dict[str, Any]:
        # Provide shared template options used by the homepage and recommendation results page.
        return {
            "dramas": self.drama_names,
            "genres": self.genres,
            "counts": [5, 8, 12],
            "ratings": [0, 7, 8, 8.5, 9],
            "load_error": self.load_error,
        }


# Resolve artifact paths relative to this file so the app works from any launch directory.
BASE_DIR = Path(__file__).resolve().parent


# Create the Flask app and load the recommendation engine once at startup.
app = Flask(__name__)
engine = KDramaRecommender(data_dir=BASE_DIR)


@app.route("/")
def home():
    # Render the search experience with dropdown data from the loaded artifacts.
    return render_template("index.html", **engine.get_ui_options())


@app.route("/recommend", methods=["POST"])
def recommend_route():
    # Read the search form and sanitize numeric filters before generating results.
    drama_name = request.form.get("drama", "")
    genre = request.form.get("genre", "")
    sort_by = request.form.get("sort_by", "match")
    try:
        limit = int(request.form.get("limit", 8))
        min_rating = float(request.form.get("min_rating", 0.0))
    except ValueError:
        limit, min_rating = 8, 0.0

    # Render the same template with recommendation data and the user's selected filters.
    recommendations, suggestions = engine.recommend(drama_name, limit, genre, min_rating, sort_by)
    return render_template(
        "index.html",
        **engine.get_ui_options(),
        recommendations=recommendations,
        suggestions=suggestions,
        selected=drama_name,
        selected_genre=genre,
        selected_limit=limit,
        selected_rating=min_rating,
        selected_sort=sort_by,
    )


@app.route("/api/recommend")
def recommend_api():
    # Expose the same recommendation logic as JSON for integrations or future frontends.
    drama_name = request.args.get("drama", "")
    genre = request.args.get("genre", "")
    sort_by = request.args.get("sort_by", "match")
    try:
        limit = int(request.args.get("limit", 8))
        min_rating = float(request.args.get("min_rating", 0.0))
    except ValueError:
        limit, min_rating = 8, 0.0

    # Return recommendations, suggestions, and model health in a machine-readable response.
    recommendations, suggestions = engine.recommend(drama_name, limit, genre, min_rating, sort_by)
    return jsonify({
        "query": drama_name,
        "count": len(recommendations),
        "suggestions": suggestions,
        "recommendations": recommendations,
        "load_error": engine.load_error,
    })


if __name__ == "__main__":
    # Run a simple local server; production hosting uses gunicorn via the Procfile.
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
