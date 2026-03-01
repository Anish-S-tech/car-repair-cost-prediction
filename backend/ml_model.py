"""
ML model loading & prediction service.
"""

import os
import joblib
import numpy as np
from PIL import Image
from io import BytesIO

from config import MODEL_PATH, SCALER_PATH, IMG_SIZE

# ── Lazy singleton ───────────────────────────────────────────────
_model = None
_scaler = None


def _load():
    """Load model + scaler once."""
    global _model, _scaler

    # Suppress TF info logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        from tensorflow.keras.models import load_model as keras_load
    except ImportError:
        from keras.models import load_model as keras_load

    _model = keras_load(MODEL_PATH)

    if os.path.exists(SCALER_PATH):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _scaler = joblib.load(SCALER_PATH)


def predict(image_bytes: bytes) -> float:
    """
    Run inference on raw image bytes.
    Returns the estimated repair cost (₹).
    """
    global _model, _scaler
    if _model is None:
        _load()

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    y_pred = _model.predict(arr, verbose=0)[0][0]

    if _scaler is not None:
        y_pred = _scaler.inverse_transform(np.array([[y_pred]])).ravel()[0]

    return float(y_pred)
