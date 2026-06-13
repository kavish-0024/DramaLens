
# K-Drama Recommender
Live demo: https://dramalens.onrender.com/
Detailed writeup: https://drive.google.com/file/d/1EOm_rtN688hmp0r7FTeMCM6O5vnMgBGb/view
A K-Drama Recommendation System built with Python and Flask. This application suggests similar dramas based on a user's selected title, utilizing a machine learning model to find the closest matches.

## How It Works

The recommendation engine relies on pre-processed data and a machine learning model to serve fast and accurate suggestions. Here is a breakdown of how the application functions:

1. **Model Loading:** On startup, the Flask application loads several pre-trained artifacts (`.pkl` files) using Python's built-in `pickle` module:
   - `dramas.pkl`: The dataset of K-Drama titles.
   - `metadata.pkl`: Detailed metadata for each drama, including genres, cast, and ratings.
   - `vectors.pkl`: Vectorized numerical representations of the dramas.
   - `nn.pkl`: A pre-trained K-Nearest Neighbors model.

2. **Smart Search & Fuzzy Matching:** When a user searches for a drama, the application uses Python's built-in `difflib` library (`get_close_matches`). This allows the app to handle typos and minor misspellings by providing fuzzy-matched suggestions if the exact title isn't found.

3. **Recommendation Engine:** 
   - The application uses **Scikit-Learn**'s K-Nearest Neighbors algorithm to find dramas with the most similar vectors to the selected title.
   - It calculates a "match score" based on the vector distance.
   - **Pandas** is used extensively to manage, filter, and normalize the metadata, allowing users to filter the recommendations by specific genres or a minimum rating threshold.

4. **Web Interface & API:** 
   - **Flask** handles the web routing, rendering the HTML frontend (`index.html`) using Jinja templates.
   - The app also exposes a REST API endpoint (`/api/recommend`) that returns the recommendation results in JSON format using Flask's `jsonify`.

## Technologies Used

This project strictly utilizes the following technologies and libraries:

- **Python:** The core programming language.
- **Flask:** The web framework used to serve the frontend and the API endpoints.
- **Scikit-Learn:** Used for the K-Nearest Neighbors (`KNeighbors`) machine learning model.
- **Pandas:** Used for data manipulation, filtering, and structuring the metadata.
- **NumPy:** Underlying mathematical operations for the vectors.
- **Gunicorn:** A Python WSGI HTTP Server for UNIX, used for production deployment.
- **Built-in Python Libraries:** 
  - `pickle` (for serializing and deserializing the model and data artifacts)
  - `difflib` (for fuzzy string matching and auto-suggestions)
  - `math`, `os`, `logging`, `pathlib`, `typing`

## Running the Application Locally

1. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server:**
   Run the Flask application:
   ```bash
   python app.py
   ```

3. **View the app:**
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Deployment

The project includes a `Procfile` configured for **Gunicorn** (`web: gunicorn app:app`), making it ready for deployment on platforms that support Python WSGI applications.

