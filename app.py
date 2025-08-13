import os
import sqlite3
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_from_directory
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
import pickle

# -----------------------------
# Config
# -----------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
MODELS_FOLDER = os.path.join(APP_ROOT, "models")
DB_PATH = os.path.join(APP_ROOT, "database.db")

MODEL_PATH = os.path.join(MODELS_FOLDER, "repair_cost_regression_model.h5")
SCALER_PATH = os.path.join(MODELS_FOLDER, "repair_cost_scaler.pkl")  # optional

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE = (224, 224)  # must match your model input
CURRENCY_PREFIX = "₹"   # change if needed

app = Flask(__name__)
app.secret_key = "CHANGE_THIS_TO_A_RANDOM_SECRET"  # generate with secrets.token_hex(16)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# -----------------------------
# Load model + optional scaler
# -----------------------------
model = load_model(MODEL_PATH)

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("[INFO] Loaded scaler.pkl")
    except Exception as e:
        print(f"[WARN] Could not load scaler: {e}. Proceeding without it.")

# -----------------------------
# DB helpers
# -----------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            prediction REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -----------------------------
# Utils
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_storage) -> np.ndarray:
    """
    Takes a Werkzeug FileStorage, returns np array shaped (1, 224, 224, 3) normalized to [0,1]
    """
    img = Image.open(file_storage).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def format_money(v: float) -> str:
    try:
        return f"{CURRENCY_PREFIX}{float(v):,.2f}"
    except Exception:
        return str(v)

# -----------------------------
# Routes - Auth
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        if not username or not password:
            flash("Username and password are required.", "warning")
            return redirect(url_for("signup"))

        pw_hash = generate_password_hash(password)
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                (username, pw_hash, datetime.utcnow().isoformat())
            )
            conn.commit()
            flash("Signup successful. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Choose another.", "danger")
        finally:
            conn.close()

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = (request.form.get("password") or "").strip()

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()

        if row and check_password_hash(row["password"], password):
            session["user_id"] = row["id"]
            session["username"] = username
            flash("Logged in successfully.", "success")
            return redirect(url_for("predict"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("home"))

# -----------------------------
# Routes - Prediction + History
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user_id" not in session:
        flash("Please log in to access predictions.", "warning")
        return redirect(url_for("login"))

    result = None
    saved_filename = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the form.", "danger")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No file selected.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Unsupported file type. Please upload a JPG/PNG image.", "danger")
            return redirect(request.url)

        try:
            # Preprocess for model
            processed = preprocess_image(file)

            # Predict (scaled or raw)
            y_pred = model.predict(processed)[0][0]

            # If a scaler was used during training for the target, inverse-transform
            if scaler is not None:
                y_pred = scaler.inverse_transform(np.array([[y_pred]])).ravel()[0]

            # Save uploaded image for history (save the original file separately)
            filename = secure_filename(file.filename)
            ts_prefix = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")
            saved_filename = ts_prefix + filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], saved_filename)
            # Rewind file pointer and save original
            file.stream.seek(0)
            file.save(save_path)

            # Store history
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO history (user_id, image_path, prediction, created_at) VALUES (?, ?, ?, ?)",
                (session["user_id"], saved_filename, float(y_pred), datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()

            result = format_money(y_pred)
            flash("Prediction completed.", "success")

        except Exception as e:
            flash(f"Error during prediction: {e}", "danger")

    return render_template("predict.html", result=result, uploaded_image=saved_filename)

@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please log in to view history.", "warning")
        return redirect(url_for("login"))

    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT image_path, prediction, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (session["user_id"],))
    rows = cur.fetchall()
    conn.close()

    # Convert to serializable dicts
    records = [
        {
            "image_path": r["image_path"],
            "prediction": format_money(r["prediction"]),
            "created_at": r["created_at"]
        }
        for r in rows
    ]

    return render_template("history.html", records=records)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Make sure model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    app.run(debug=True, host="0.0.0.0", port=5000)
