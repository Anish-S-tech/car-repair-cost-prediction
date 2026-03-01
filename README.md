# 🚗 Car Repair Cost Estimator

AI-powered vehicle damage cost estimation. Upload a photo of car damage and get an instant repair cost estimate in US Dollars ($).

## Project Structure

```
Car_repair_cost_detection/
├── backend/               # FastAPI REST API
│   ├── main.py            # App entry + routes
│   ├── config.py          # Centralised settings
│   ├── database.py        # SQLite helpers
│   ├── ml_model.py        # Keras model loader
│   ├── auth.py            # JWT + bcrypt auth
│   └── requirements.txt   # Python dependencies
├── frontend/              # Vite + Vanilla JS SPA
│   ├── index.html         # SPA shell
│   ├── style.css          # Design system (dark theme)
│   ├── vite.config.js     # Dev-server + proxy
│   └── src/
│       ├── main.js        # Router
│       ├── api.js         # Fetch wrapper + auth
│       ├── toast.js       # Notifications
│       └── pages/         # Page components
│           ├── home.js
│           ├── login.js
│           ├── signup.js
│           ├── predict.js
│           └── history.js
├── models/                # Pre-trained ML artefacts
│   ├── repair_cost_regression_model.h5
│   ├── repair_cost_scaler.pkl
│   └── repair_cost_labels.csv
└── uploads/               # Uploaded images (auto-created)
```

## Quick Start

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Docs at `/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` with auto-proxy to the backend.

## Features

- **JWT authentication** — signup, login, protected routes
- **Image upload** — drag-and-drop or click to upload
- **AI prediction** — Keras deep-learning model estimates repair cost
- **History** — browse past predictions with thumbnails
- **Premium UI** — dark theme, glassmorphism, gradient accents, micro-animations
