from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.ml_model import load_trained_model, predict_image
from app.utils import save_prediction_log
import os
import cv2
import numpy as np
import pandas as pd
from collections import deque

app = FastAPI(title="Proyecto Lenguaje de Señas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "sign_model.h5")
MODEL = load_trained_model(MODEL_PATH)
LABELS = [chr(i) for i in range(65, 91)] + ["space", "nothing"]

def get_camera():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("No se pudo acceder a la cámara.")
    return cam

camera = get_camera()
pred_buffer = deque(maxlen=5)
current_word = ""

@app.get("/")
async def get_index():
    return FileResponse(os.path.join("app/frontend", "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global current_word
    try:
        img_bytes = await file.read()
        predicted_label = predict_image(MODEL, img_bytes, LABELS)

        if predicted_label.isalpha():
            current_word += predicted_label
        elif predicted_label == "space":
            current_word += " "
        elif predicted_label == "nothing":
            current_word = ""

        save_prediction_log(predicted_label, source="camera")
        return JSONResponse(content={"prediction": predicted_label, "word": current_word})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history")
async def get_history():
    try:
        df = pd.read_csv("prediction_history.csv")
        return JSONResponse(content=df.tail(10).to_dict(orient="records"))
    except FileNotFoundError:
        return JSONResponse(content=[])

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        x1, y1 = int(w * 0.3), int(h * 0.3)
        x2, y2 = int(w * 0.7), int(h * 0.7)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]

        roi_bytes = cv2.imencode(".jpg", roi)[1].tobytes()
        pred = predict_image(MODEL, roi_bytes, LABELS)

        pred_buffer.append(pred)
        if len(pred_buffer) == pred_buffer.maxlen:
            pred = max(set(pred_buffer), key=pred_buffer.count)

        cv2.putText(frame, f"Pred: {pred}", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3, cv2.LINE_AA)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    if camera.isOpened():
        camera.release()