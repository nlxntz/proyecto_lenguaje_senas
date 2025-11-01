import tensorflow as tf
from app.utils import load_dataset, detect_hand_and_crop
import numpy as np

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(dataset_path, img_size=(64,64), epochs=10):
    X, y, labels = load_dataset(dataset_path, img_size)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=len(labels))
    model = create_model((img_size[0], img_size[1], 3), len(labels))
    model.fit(X, y_cat, epochs=epochs, batch_size=32, validation_split=0.1)
    model.save("sign_model.h5")
    print("Modelo entrenado y guardado como sign_model.h5")
    return model, labels

def load_trained_model(model_path="sign_model.h5"):
    return tf.keras.models.load_model(model_path)

def predict_image(model, image_bytes, labels, img_size=(64,64)):
    cropped = detect_hand_and_crop(image_bytes, img_size)
    if cropped is None:
        return "No se detectó mano"
    pred = model.predict(cropped)
    predicted_label = labels[np.argmax(pred)]
    return predicted_label

if __name__ == "__main__":
    dataset_path = "dataset/asl_alphabet_test"
    train_model(dataset_path)