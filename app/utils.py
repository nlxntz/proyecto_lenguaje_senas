from PIL import Image
import numpy as np
import os

def load_and_preprocess_image(img_path, img_size=(64, 64)):
    """
    Carga una imagen, la redimensiona y normaliza los valores.
    """
    img = Image.open(img_path).convert('RGB')  # Mantener color
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0  # Normalizar 0-1
    return img_array

def load_dataset(dataset_path, img_size=(64, 64)):
    """
    Carga todas las imágenes del directorio (sin subcarpetas) y asigna etiquetas
    basadas en la primera letra del nombre del archivo (por ejemplo A_test.jpg → A).
    """
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