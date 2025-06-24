# RAOP Pizza Success Predictor

A machine learning-powered API that predicts the likelihood of a Reddit pizza request receiving a pizza on the r/Random_Acts_Of_Pizza subreddit.

## Project Overview

This project uses natural language processing and machine learning techniques to analyze Reddit pizza requests and predict their success rate. By examining features like request text, account age, and posting patterns, the model predicts whether a request is likely to receive a pizza.

## Technical Stack

- **Backend**: FastAPI, Python 3.11
- **Machine Learning**: scikit-learn, Sentence Transformers, CatBoost
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite (for logging predictions)
- **Containerization**: Docker

## Features

- **Text Analysis**: Processes the text of pizza requests using BERT-based embeddings
- **Feature Engineering**: Creates textual and social features from request data
- **ML Pipeline**: Implements a stacked ensemble model for prediction
- **REST API**: Provides simple endpoints for prediction and health checks
- **Web Interface**: Offers a user-friendly way to test the prediction algorithm
- **Logging**: Records all predictions for future analysis

## Project Structure

```
raop-pizza/
├── all-MiniLM-L6-v2/       # Sentence transformer model
├── data/                   # Data files
├── models/                 # Trained ML models
├── notebooks/              # Jupyter notebooks for analysis
├── pizza_service/          # FastAPI service
│   ├── src/                # Source code
│   ├── static/             # Web interface
│   ├── tests/              # Unit tests
│   └── Dockerfile          # Docker configuration
└── src/                    # Core Python package
```

## Installation & Setup

### Prerequisites

- Python 3.11+
- pip (Python package manager)

### Local Development

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/raop-pizza.git
   cd raop-pizza
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Run the FastAPI server:
   ```
   cd raop-pizza
   python -m uvicorn pizza_service.src.main:app --reload
   ```

4. Access the web interface at `http://127.0.0.1:8000/app` or API docs at `http://127.0.0.1:8000/docs`

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t raop-pizza-predictor -f pizza_service/Dockerfile .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 raop-pizza-predictor
   ```

## API Documentation

### Endpoints

- `GET /healthz`: Health check endpoint
- `GET /app`: Web interface for testing predictions
- `POST /predict`: Prediction endpoint

### Prediction Request Format

```json
{
  "request_title": "Broke student craving pizza",
  "request_text_edit_aware": "Hey kind folks, finals drained my wallet. Could anyone spare a pizza tonight? Thank you!",
  "request_time_utc": "2025-06-24T14:30:00Z",
  "requester_account_age_in_days_at_request": 200
}
```

### Prediction Response Format

```json
{
  "score": 0.65,
  "label": 1
}
```
Where `score` is the probability of success (0-1) and `label` is the binary prediction (1 = success, 0 = failure).
