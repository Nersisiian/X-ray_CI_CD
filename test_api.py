import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path
import tensorflow as tf
from PIL import Image

def create_dummy_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

MODEL_PATH = Path("models/xray_model.h5")
if not MODEL_PATH.exists():
    print("Creating dummy model for testing...")
    os.makedirs("models", exist_ok=True)
    model = create_dummy_model()
    model.save(MODEL_PATH)
    print(f"Dummy model saved to {MODEL_PATH}")

from app.main import app
client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422

def test_predict_with_image(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
