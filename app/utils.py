import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from datetime import datetime
from ultralytics import YOLO

YOLO_MODEL_PATH = os.path.join("yolov8x-tuned-hand-gestures", "weights", "best.pt")
yolo_model = YOLO(YOLO_MODEL_PATH)

def detect_hand_and_crop(image_bytes, img_size=(64, 64)):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    results = yolo_model.predict(source=img, conf=0.5, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if boxes.shape[0] == 0:
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    hand_crop = img[y1:y2, x1:x2]

    hand_crop = cv2.resize(hand_crop, img_size)
    hand_crop = hand_crop.astype("float32") / 255.0
    hand_crop = np.expand_dims(hand_crop, axis=0)
    return hand_crop

def save_prediction_log(prediction, source="camera"):
    log_path = "prediction_history.csv"
    entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": prediction,
        "source": source
    }])
    try:
        df = pd.read_csv(log_path)
        df = pd.concat([df, entry], ignore_index=True)
    except FileNotFoundError:
        df = entry
    df.to_csv(log_path, index=False)

def load_and_preprocess_image(img_path, img_size=(64, 64)):
    if isinstance(img_path, bytes):
        from io import BytesIO
        img = Image.open(BytesIO(img_path))
    else:
        img = Image.open(img_path)
    img = img.convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    return img_array

def load_dataset(dataset_path, img_size=(64, 64)):
    X = []
    y = []
    labels = []
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        if not file.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        label = file[0].upper()
        if label not in labels:
            labels.append(label)
        try:
            img_array = load_and_preprocess_image(file_path, img_size)
            X.append(img_array)
            y.append(labels.index(label))
        except Exception as e:
            print(f"Error cargando {file_path}: {e}")
    X = np.array(X)
    y = np.array(y)
    return X, y, labels