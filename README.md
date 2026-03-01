# 🚗 Car Repair Cost Estimator

AI-powered vehicle damage cost estimation using deep learning. Upload a photo of car damage and get an instant repair cost estimate in US Dollars ($).

## Project Structure

```
Car_repair_cost_detection/
├── backend/                          # FastAPI REST API
│   ├── main.py                       # App entry + API routes
│   ├── config.py                     # Centralised settings
│   ├── database.py                   # SQLite helpers
│   ├── ml_model.py                   # Keras model loader + prediction
│   └── requirements.txt              # Python dependencies
├── frontend/                         # Vite + Vanilla JS SPA
│   ├── index.html                    # SPA shell
│   ├── style.css                     # Design system (dark theme)
│   ├── vite.config.js                # Dev-server + API proxy
│   └── src/
│       ├── main.js                   # SPA router
│       ├── toast.js                  # Toast notifications
│       └── pages/
│           ├── home.js               # Landing page
│           ├── predict.js            # Image upload + prediction
│           └── history.js            # Past prediction results
├── models/                           # Pre-trained ML artefacts
│   ├── repair_cost_regression_model.h5
│   ├── repair_cost_scaler.pkl
│   └── repair_cost_labels.csv
├── Vehicle damage severity.ipynb     # Model training notebook
├── app.py                            # Original Streamlit app (legacy)
└── uploads/                          # Uploaded images (auto-created)
```

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- The pre-trained model files (`repair_cost_regression_model.h5` and `repair_cost_scaler.pkl`) placed in `models/`

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` with auto-proxy to the backend.

## Features

- **Drag-and-drop image upload** — click or drag a photo of vehicle damage
- **AI-powered estimation** — Keras deep-learning regression model predicts repair cost
- **Prediction history** — browse all past estimates with image thumbnails
- **Premium dark UI** — glassmorphism cards, gradient accents, micro-animations
- **No login required** — upload and predict instantly

## Tech Stack

| Layer    | Technology                         |
|----------|------------------------------------|
| Backend  | Python, FastAPI, Uvicorn           |
| ML Model | TensorFlow / Keras, scikit-learn   |
| Frontend | Vite, Vanilla JavaScript, CSS3     |
| Database | SQLite                             |

## API Endpoints

| Method | Path                  | Description                    |
|--------|-----------------------|--------------------------------|
| POST   | `/api/predict`        | Upload image → cost estimate   |
| GET    | `/api/history`        | List all past predictions      |
| GET    | `/uploads/{filename}` | Serve an uploaded image        |
| GET    | `/api/health`         | Health check                   |
