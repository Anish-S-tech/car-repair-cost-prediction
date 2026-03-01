"""
Centralised application configuration.
"""

import os

# Paths
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_ROOT)
MODELS_FOLDER = os.path.join(PROJECT_ROOT, "models")
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
DB_PATH = os.path.join(PROJECT_ROOT, "database.db")

MODEL_PATH = os.path.join(MODELS_FOLDER, "repair_cost_regression_model.h5")
SCALER_PATH = os.path.join(MODELS_FOLDER, "repair_cost_scaler.pkl")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Image processing
IMG_SIZE = (224, 224)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Currency
CURRENCY_PREFIX = "$"

# JWT
JWT_SECRET = os.getenv("JWT_SECRET", "car-repair-super-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24  # 24 hours
