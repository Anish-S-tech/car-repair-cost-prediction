import os
import sqlite3
from datetime import datetime

import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model
=======

from keras.models import load_model

import pickle

# -----------------------------
# Config
# -----------------------------
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "uploads")
MODELS_FOLDER = os.path.join(APP_ROOT, "models")
DB_PATH = os.path.join(APP_ROOT, "database.db")

MODEL_PATH = os.path.join(MODELS_FOLDER, "repair_cost_regression_model.h5")
SCALER_PATH = os.path.join(MODELS_FOLDER, "repair_cost_scaler.pkl")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
IMG_SIZE = (224, 224)
CURRENCY_PREFIX = "₹"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# -----------------------------
# Load model and scaler
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    model = load_model(MODEL_PATH)
    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# -----------------------------
# Database
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
# Helpers
# -----------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image) -> np.ndarray:
    img = image.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def format_money(v: float) -> str:
    try:
        return f"{CURRENCY_PREFIX}{float(v):,.2f}"
    except Exception:
        return str(v)

# -----------------------------
# Auth with session_state
# -----------------------------
def login_user(username, password):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row["password"] == password:  # Simplified password check for demo
        st.session_state["user_id"] = row["id"]
        st.session_state["username"] = username
        return True
    return False

def signup_user(username, password):
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username, password, datetime.utcnow().isoformat())
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Car Repair Cost Estimator", layout="wide")

st.title("🚗 Car Repair Cost Estimator")

menu = ["Home", "Signup", "Login", "Predict", "History"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Home":
    st.write("Welcome! Use the sidebar to navigate.")

elif choice == "Signup":
    st.subheader("Create Account")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Signup"):
        if signup_user(uname, pwd):
            st.success("Account created! Please login.")
        else:
            st.error("Username already exists!")

elif choice == "Login":
    st.subheader("Login")
    uname = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(uname, pwd):
            st.success(f"Welcome {uname}!")
        else:
            st.error("Invalid credentials")

elif choice == "Predict":
    if "user_id" not in st.session_state:
        st.warning("Please login first!")
    else:
        st.subheader("Upload Image to Predict Repair Cost")
        file = st.file_uploader("Upload JPG/PNG", type=list(ALLOWED_EXTENSIONS))

        if file is not None:
            img = Image.open(file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict Cost"):
                try:
                    processed = preprocess_image(img)
                    y_pred = model.predict(processed)[0][0]
                    if scaler is not None:
                        y_pred = scaler.inverse_transform(np.array([[y_pred]])).ravel()[0]

                    # Save uploaded file
                    ts_prefix = datetime.utcnow().strftime("%Y%m%d_%H%M%S_")
                    saved_filename = ts_prefix + file.name
                    save_path = os.path.join(UPLOAD_FOLDER, saved_filename)
                    img.save(save_path)

                    # Insert history
                    conn = get_db()
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO history (user_id, image_path, prediction, created_at) VALUES (?, ?, ?, ?)",
                        (st.session_state["user_id"], saved_filename, float(y_pred), datetime.utcnow().isoformat())
                    )
                    conn.commit()
                    conn.close()

                    st.success(f"Estimated Repair Cost: {format_money(y_pred)}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

elif choice == "History":
    if "user_id" not in st.session_state:
        st.warning("Please login first!")
    else:
        st.subheader("Prediction History")
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT image_path, prediction, created_at FROM history WHERE user_id=? ORDER BY created_at DESC",
            (st.session_state["user_id"],)
        )
        rows = cur.fetchall()
        conn.close()

        if rows:
            for r in rows:
                st.image(os.path.join(UPLOAD_FOLDER, r["image_path"]), width=200)
                st.write(f"Prediction: {format_money(r['prediction'])}")
                st.caption(f"Time: {r['created_at']}")
                st.markdown("---")
        else:
            st.info("No history yet.")
