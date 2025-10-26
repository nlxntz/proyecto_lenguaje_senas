Proyecto Lenguaje de Señas

Reconocimiento de lenguaje de señas en tiempo real usando FastAPI, TensorFlow y Tailwind CSS. Este proyecto permite subir imágenes o usar la cámara para identificar letras en lenguaje de señas.

Tecnologías:

- Backend: FastAPI

- Frontend: HTML + Tailwind CSS + JavaScript

- Machine Learning: TensorFlow / Keras

- Bases de datos: No aplica (modelo entrenado en memoria)

- Otros: OpenCV opcional para procesamiento de imagen en tiempo real 

Instalación:

1. Clonar el repositorio

git clone https://github.com/nlxntz/proyecto_lenguaje_senas.git
cd proyecto_lenguaje_senas

2. Crar y activar entorno virtual

python -m venv venv
venv\Scripts\activate

3. instalar dependencias

pip install -r requirements.txt

4. Entrenar el modelo

python -m app.ml_model

5. Ejecutar el servidor

uvicorn app.main:app --reload

Funcionalidades:

- Subir imagen: Sube una imagen de tu mano realizando una seña para predecir la letra.

- Cámara en tiempo real: El sistema detecta tu mano y predice las letras en tiempo real cada 0.8 segundos.

- Interfaz atractiva: Diseño moderno y responsive con Tailwind CSS y efectos de sombra y blur.
