from PIL import Image
import numpy as np
import os

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