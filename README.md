# financial-market-sentiment-mlops

End-to-end MLOps pipeline for financial news sentiment analysis and market movement prediction.  
Includes data ingestion, preprocessing, model training, experiment tracking with MLflow, REST API deployment (FastAPI), and containerisation with Docker.

---

## Project Structure

```
financial-market-sentiment-mlops/
│
├── data/                   # Raw and processed CSV datasets (git-tracked placeholders)
│
├── src/
│   ├── ingestion.py        # Fetch financial news from Alpha Vantage / NewsAPI
│   ├── preprocessing.py    # Text cleaning, feature engineering, train/val/test splits
│   ├── train.py            # TF-IDF + Logistic Regression training with MLflow logging
│   └── evaluate.py         # Model evaluation – metrics, reports, confusion matrix
│
├── app/
│   └── main.py             # FastAPI inference service (single & batch prediction)
│
├── models/                 # Serialised model artifacts (git-tracked placeholders)
│
├── Dockerfile              # Production container image
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Quick Start

### 1 – Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 – Configure API keys (optional)

Create a `.env` file in the project root:

```env
ALPHA_VANTAGE_API_KEY=your_key_here
NEWSAPI_KEY=your_key_here
MLFLOW_TRACKING_URI=file:./mlruns   # or a remote MLflow server URI
```

### 3 – Run the pipeline

```bash
# Step 1: Ingest financial news
python -m src.ingestion

# Step 2: Preprocess & split data
python -m src.preprocessing

# Step 3: Train the model (logged to MLflow)
python -m src.train

# Step 4: Evaluate on the test set
python -m src.evaluate
```

### 4 – Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### 5 – Docker

```bash
# Build the image
docker build -t market-sentiment-api .

# Run the container
docker run -p 8000:8000 market-sentiment-api
```

---

## API Endpoints

| Method | Path              | Description                            |
|--------|-------------------|----------------------------------------|
| GET    | `/health`         | Liveness probe                         |
| POST   | `/predict`        | Predict sentiment for a single text    |
| POST   | `/predict/batch`  | Predict sentiment for multiple texts   |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Apple reports record quarterly earnings, beating analyst expectations."}'
```

```json
{
  "label_id": 4,
  "label": "Bullish",
  "confidence": 0.8732
}
```

---

## Sentiment Labels

| ID | Label            |
|----|------------------|
| 0  | Bearish          |
| 1  | Somewhat-Bearish |
| 2  | Neutral          |
| 3  | Somewhat-Bullish |
| 4  | Bullish          |

---

## MLflow Experiment Tracking

All training runs and evaluation metrics are logged under the **`financial-sentiment`** experiment.  
Launch the MLflow UI with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## License

MIT
