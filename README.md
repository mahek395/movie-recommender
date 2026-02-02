# Movie Recommender

A machine learning-based movie recommendation system with a modern web interface.

## Project Structure

- **backend/**: Flask API server with recommendation models
- **frontend/**: React + Vite web interface
- **data/**: Dataset files
- **model/**: Pre-trained machine learning models

## Features

- Semantic and TF-IDF based movie recommendations
- Neural network models for improved accuracy
- REST API backend
- Interactive web UI

## Setup Instructions

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Models

The system uses two recommendation approaches:
- **TF-IDF Model**: `tfidf_vectorizer.joblib` with nearest neighbors
- **Semantic Model**: `semantic_nn_model.joblib` with embeddings

## Dependencies

- Python 3.x
- Flask
- Scikit-learn
- Node.js
- React
- Vite
