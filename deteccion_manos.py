import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from app.utils import detect_hand_and_crop, save_prediction_log
from app.ml_model import load_trained_model
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8x-tuned-hand-gestures", "weights", "best.pt")
SIGN_MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")

yolo_model = YOLO(YOLO_MODEL_PATH)
sign_model = load_trained_model(SIGN_MODEL_PATH)

LABELS = [chr(i) for i in range(65, 91)]  # A-Z

cap = cv2.VideoCapture(0)
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        cv2.imshow("Lenguaje de Señas", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = boxes[0].astype(int)
    hand_crop = frame[y1:y2, x1:x2]

    if hand_crop.size == 0:
        continue
    roi_resized = cv2.resize(hand_crop, (64, 64))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    pred = sign_model.predict(roi_input, verbose=0)
    predicted_label = LABELS[np.argmax(pred)]

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Seña: {predicted_label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Lenguaje de Señas", frame)

    save_prediction_log(predicted_label, source="camera")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()