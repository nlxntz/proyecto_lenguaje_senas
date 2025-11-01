from PIL import Image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

def detect_hand_and_crop(image_bytes, img_size=(64, 64)):
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return None

    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(img_ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 500:
        return None

    x, y, w, h = cv2.boundingRect(largest_contour)

    x_min = max(x - 20, 0)
    y_min = max(y - 20, 0)
    x_max = min(x + w + 20, img.shape[1])
    y_max = min(y + h + 20, img.shape[0])

    cropped = img[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None

    cropped = cv2.resize(cropped, img_size)
    cropped = cropped / 255.0
    return np.expand_dims(cropped, axis=0)

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