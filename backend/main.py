"""
Car Repair Cost Estimator — FastAPI backend (no auth).
"""

import os
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, CURRENCY_PREFIX
from database import get_db, init_db
import ml_model

# ── Initialise ───────────────────────────────────────────────────
init_db()

app = FastAPI(
    title="Car Repair Cost Estimator API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Prediction endpoint (no auth) ──────────────────────────────
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # Validate extension
    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type .{ext} not allowed. Use: {ALLOWED_EXTENSIONS}")

    image_bytes = await file.read()

    # Run ML inference
    try:
        cost = ml_model.predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Save uploaded image
    ts_prefix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_")
    saved_name = ts_prefix + (file.filename or "upload.jpg")
    save_path = os.path.join(UPLOAD_FOLDER, saved_name)
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    # Persist to history (user_id = 0 since no auth)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO history (user_id, image_path, prediction, created_at) VALUES (?, ?, ?, ?)",
        (0, saved_name, cost, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    formatted = f"{CURRENCY_PREFIX}{cost:,.2f}"
    return {"estimated_cost": cost, "formatted_cost": formatted, "image_path": saved_name}


# ── History endpoint (no auth) ─────────────────────────────────
@app.get("/api/history")
def history():
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, image_path, prediction, created_at FROM history ORDER BY created_at DESC"
    )
    rows = cur.fetchall()
    conn.close()

    items = []
    for r in rows:
        items.append({
            "id": r["id"],
            "image_path": r["image_path"],
            "prediction": r["prediction"],
            "formatted_cost": f"{CURRENCY_PREFIX}{r['prediction']:,.2f}",
            "created_at": r["created_at"],
            "image_url": f"/uploads/{r['image_path']}",
        })
    return items


# ── Serve uploaded images ───────────────────────────────────────
@app.get("/uploads/{filename}")
def serve_upload(filename: str):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


# ── Health check ────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Car Repair Cost Estimator API"}
