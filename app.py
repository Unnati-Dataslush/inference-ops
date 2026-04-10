from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
import logging
import time

from model import run_inference
import os
import tempfile
from fastapi.responses import FileResponse

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




@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    start = time.time()
    logging.info("Request received (VIDEO endpoint)")

    # Step 1 — Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        contents = await file.read()
        tmp_input.write(contents)
        tmp_input_path = tmp_input.name

    # Step 2 — Open video with OpenCV
    cap = cv2.VideoCapture(tmp_input_path)

    # Step 3 — Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Step 4 — Create output temp file for annotated video
    tmp_output_path = tmp_input_path.replace(".mp4", "_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height))

    # Step 5 — Process frame by frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, _ = run_inference(frame)
        out.write(annotated_frame)
        frame_count += 1

    # Step 6 — Release everything
    cap.release()
    out.release()

    inference_time = time.time() - start
    logging.info(f"Video processed: {frame_count} frames, Time: {inference_time:.3f}s")

    # Step 7 — Return annotated video
    return FileResponse(
        tmp_output_path,
        media_type="video/mp4",
        filename="annotated_output.mp4"
    )