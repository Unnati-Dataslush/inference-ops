import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import cv2

from app import app

client = TestClient(app)

def make_test_image():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_predict_returns_200():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")}
    )
    assert response.status_code == 200

def test_predict_response_has_detections():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")}
    )
    body = response.json()
    assert "detections" in body
    assert "inference_time" in body

def test_predict_image_returns_200():
    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")}
    )
    assert response.status_code == 200

def test_predict_image_content_type():
    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", make_test_image(), "image/jpeg")}
    )
    assert "image" in response.headers["content-type"]
