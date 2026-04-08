from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import logging
import time

from model import run_inference

app = FastAPI()

# Logging setup
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Health check (important for ops)
@app.get("/health")
def health():
    return {"status": "running"}

# JSON endpoint (engineers use this)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    logging.info("Request received (JSON endpoint)")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    _, detections = run_inference(img)

    inference_time = time.time() - start
    logging.info(f"Detections: {len(detections)}, Time: {inference_time:.3f}s")

    return {
        "detections": detections,
        "inference_time": inference_time
    }

# IMAGE endpoint (for demo / managers)
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    start = time.time()
    logging.info("Request received (IMAGE endpoint)")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    output_img, detections = run_inference(img)

    inference_time = time.time() - start
    logging.info(f"Image response, Time: {inference_time:.3f}s")

    _, buffer = cv2.imencode('.jpg', output_img)

    return StreamingResponse(io.BytesIO(buffer), media_type="image/jpeg")