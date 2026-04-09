import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422  # missing file

def test_predict_with_image(tmp_path):
    # создадим фиктивное изображение
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    with open(img_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["NORMAL", "PNEUMONIA"]