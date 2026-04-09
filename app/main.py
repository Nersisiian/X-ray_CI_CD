import os
import io
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

from config import MODEL_PATH, LOG_LEVEL, BASE_DIR, API_PORT

logging.basicConfig(level=getattr(logging, LOG_LEVEL),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("xray-api")

if not Path(MODEL_PATH).exists():
    logger.error(f"Model not found at {MODEL_PATH}. Run train.py first.")
    raise RuntimeError("Model missing")

model = tf.keras.models.load_model(MODEL_PATH)
logger.info(f"Model loaded from {MODEL_PATH}")

def prepare_image(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

app = FastAPI(title="X-ray Pneumonia Detector", version="2.0")

static_dir = BASE_DIR / "app" / "static"
templates_dir = BASE_DIR / "app" / "templates"
static_dir.mkdir(parents=True, exist_ok=True)
templates_dir.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    try:
        contents = await file.read()
        input_tensor = prepare_image(contents)
        preds = model.predict(input_tensor)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])
        label = "PNEUMONIA" if class_idx == 1 else "NORMAL"
        logger.info(f"Prediction: {label} ({confidence:.3f}) from {file.filename}")
        return JSONResponse({
            "filename": file.filename,
            "prediction": label,
            "confidence": confidence,
            "probabilities": {
                "NORMAL": float(preds[0]),
                "PNEUMONIA": float(preds[1])
            }
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Internal error: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=API_PORT, reload=True)
