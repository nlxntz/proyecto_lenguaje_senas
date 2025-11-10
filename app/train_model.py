import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

CLASSES = [chr(i) for i in range(65, 91)] + ["space", "nothing"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "asl_alphabet_test")
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.h5")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {DATASET_PATH}")

files = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(".jpg")]
labels = []

for f in files:
    name = f.split("_")[0].upper()
    if name in [chr(i) for i in range(65, 91)]:
        labels.append(name)
    elif "space" in f.lower():
        labels.append("space")
    elif "nothing" in f.lower():
        labels.append("nothing")
    else:
        labels.append("unknown")

df = pd.DataFrame({
    "filename": [os.path.join(DATASET_PATH, f) for f in files],
    "class": labels
})
df = df[df["class"] != "unknown"]

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

train_generator = datagen.flow_from_dataframe(
    train_df,
    x_col="filename",
    y_col="class",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    classes=CLASSES
)

val_generator = datagen.flow_from_dataframe(
    val_df,
    x_col="filename",
    y_col="class",
    target_size=(64, 64),
    batch_size=32,
    class_mode="categorical",
    classes=CLASSES
)

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = create_model()

    checkpoint = ModelCheckpoint(
        MODEL_PATH, monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=30,
        callbacks=[checkpoint, early_stop]
    )

    print(f"\n Modelo entrenado y guardado en: {MODEL_PATH}")

    plt.figure()
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.show()