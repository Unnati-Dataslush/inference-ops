import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app import app
from model import model, run_inference

client = TestClient(app)

# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────

def make_blank_image():
    """A plain blue image with no cricket objects in it."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (100, 150, 200)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def load_real_image():
    """Your actual cricket image — model should detect something here."""
    img = cv2.imread("tests/test_image.jpg")
    return img

# ─────────────────────────────────────────
# Block 1 — Model tests
# ─────────────────────────────────────────

# Checks that best.pt loaded successfully and is not empty
def test_model_loaded():
    assert model is not None

# Checks that the model knows exactly 3 classes — not more, not less
def test_model_has_three_classes():
    assert len(model.names) == 3

# Checks that those 3 classes are specifically bat, ball and stump
def test_model_class_names_are_correct():
    class_names = list(model.names.values())
    assert "bat" in class_names
    assert "ball" in class_names
    assert "stump" in class_names

# ─────────────────────────────────────────
# Block 2 — run_inference() tests
# ─────────────────────────────────────────

# Checks that run_inference() returns exactly 2 things — image and list
def test_run_inference_returns_two_things():
    img = load_real_image()
    result = run_inference(img)
    assert len(result) == 2

# Checks that the second thing returned is always a list
def test_run_inference_returns_list_of_detections():
    img = load_real_image()
    _, detections = run_inference(img)
    assert isinstance(detections, list)

# Checks that each detection has all 4 required fields
def test_run_inference_detection_has_correct_fields():
    img = load_real_image()
    _, detections = run_inference(img)
    for detection in detections:
        assert "bbox" in detection
        assert "confidence" in detection
        assert "class_id" in detection
        assert "class_name" in detection

# Checks that bbox always has exactly 4 numbers
def test_run_inference_bbox_has_four_values():
    img = load_real_image()
    _, detections = run_inference(img)
    for detection in detections:
        assert len(detection["bbox"]) == 4

# Checks that confidence is always a number between 0 and 1
def test_run_inference_confidence_is_valid():
    img = load_real_image()
    _, detections = run_inference(img)
    for detection in detections:
        assert 0.0 <= detection["confidence"] <= 1.0

# Checks that class_name is always bat, ball or stump — never anything else
def test_run_inference_class_name_is_valid():
    img = load_real_image()
    _, detections = run_inference(img)
    valid_classes = ["bat", "ball", "stump"]
    for detection in detections:
        assert detection["class_name"] in valid_classes

# ─────────────────────────────────────────
# Block 3 — /health endpoint tests
# ─────────────────────────────────────────

# Checks that the server is alive and responding
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

# Checks that the response says exactly {"status": "running"}
def test_health_returns_correct_status():
    response = client.get("/health")
    assert response.json()["status"] == "running"

# ─────────────────────────────────────────
# Block 4 — /predict endpoint tests
# ─────────────────────────────────────────

# Checks that /predict responds without crashing for a valid image
def test_predict_returns_200():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert response.status_code == 200

# Checks that the response always has a detections field
def test_predict_has_detections_field():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert "detections" in response.json()

# Checks that the response always has an inference_time field
def test_predict_has_inference_time_field():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert "inference_time" in response.json()

# Checks that detections is always a list even when nothing is detected
def test_predict_detections_is_always_list():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert isinstance(response.json()["detections"], list)

# ─────────────────────────────────────────
# Block 5 — /predict-image endpoint tests
# ─────────────────────────────────────────

# Checks that /predict-image responds without crashing for a valid image
def test_predict_image_returns_200():
    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert response.status_code == 200

# Checks that what comes back is actually an image and not JSON or text
def test_predict_image_returns_actual_image():
    response = client.post(
        "/predict-image",
        files={"file": ("test.jpg", make_blank_image(), "image/jpeg")}
    )
    assert "image" in response.headers["content-type"]