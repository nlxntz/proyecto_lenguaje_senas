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
import mediapipe as mp
import time

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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


def get_camera():
    for idx in [0, 1, 2, 3]:
        cam = cv2.VideoCapture(idx)
        if cam.isOpened():
            print(f"[INFO] Usando cámara índice {idx}")
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            return cam
        cam.release()
    raise RuntimeError("No se pudo acceder a ninguna cámara (índices 0–3).")


camera = get_camera()
pred_buffer = deque(maxlen=5)
current_word = ""
frame_counter = 0
last_pred = "nothing"


@app.get("/")
async def get_index():
    return FileResponse(os.path.join("app/frontend", "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicción desde una imagen subida (usa predict_image).
    """
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

        save_prediction_log(predicted_label, source="upload")
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
    """
    Lee la cámara, usa MediaPipe para encontrar la mano, dibuja un rectángulo
    que sigue la mano y clasifica el ROI con el modelo cada N frames.
    """
    global frame_counter, last_pred, current_word, pred_buffer

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        pred = last_pred

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            pad = 10
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            roi = frame[y_min:y_max, x_min:x_max]

            frame_counter += 1

            if frame_counter % 5 == 0 and roi.size > 0:
                try:
                    roi_resized = cv2.resize(roi, (64, 64))
                    roi_normalized = roi_resized / 255.0
                    roi_input = np.expand_dims(roi_normalized, axis=0)

                    pred_probs = MODEL.predict(roi_input, verbose=0)
                    pred_single = LABELS[np.argmax(pred_probs)]

                    last_pred = pred_single
                    pred_buffer.append(pred_single)

                    if len(pred_buffer) == pred_buffer.maxlen:
                        pred = max(set(pred_buffer), key=pred_buffer.count)
                        if pred.isalpha():
                            current_word += pred
                        elif pred == "space":
                            current_word += " "
                        elif pred == "nothing":
                            current_word = ""

                except Exception as e:
                    print("Error en clasificación:", e)
        else:
            pred = "nothing"
            last_pred = pred

        cv2.putText(
            frame,
            f"Pred: {pred}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(1 / 60)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.on_event("shutdown")
def shutdown_event():
    if camera.isOpened():
        camera.release()
