import tensorflow as tf
from app.utils import load_dataset
import numpy as np
import mediapipe as mp
import cv2
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")

mp_hands = mp.solutions.hands

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path, img_size=(64,64), epochs=30):
    X, y, labels = load_dataset(dataset_path, img_size)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(labels))

    model = create_model((img_size[0], img_size[1], 3), len(labels))

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "sign_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y_cat,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        callbacks=[checkpoint, early_stop],
        shuffle=True
    )

    print("Modelo entrenado y guardado como sign_model.h5")
    return model, labels, history

def load_trained_model(model_path="sign_model.h5"):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_bytes, labels, img_size=(64, 64)):
    img_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        return "nothing"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:
        results = hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return "nothing"

        h, w, _ = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        for lm in results.multi_hand_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)

        pad = 10
        x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
        x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)

        roi = frame[y_min:y_max, x_min:x_max]

    roi_resized = cv2.resize(roi, img_size)
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)

    pred = model.predict(roi_input, verbose=0)
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

def classify_roi(model, roi, labels, img_size=(64, 64)):
    """Clasifica una imagen ya recortada (ROI de la mano)."""
    roi_resized = cv2.resize(roi, img_size)
    roi_normalized = roi_resized / 255.0
    roi_input = np.expand_dims(roi_normalized, axis=0)
    pred = model.predict(roi_input, verbose=0)
    return labels[np.argmax(pred)]


if __name__ == "__main__":
    dataset_path = "dataset/asl_alphabet_test"
    model, labels, history = train_model(dataset_path)