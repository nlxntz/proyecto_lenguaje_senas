# proyecto_lenguaje_senas


Integrantes: Felipe Aguayo, Ignacio Barrientos, Jose Briones, Nicolas Lintz.
Asignatura: Aplicaciones De Inteligencia Artificial.
Docente: Vedran Hrvoj Tomicic Cantizano.



1) Mencionar y justificar el problema (con datos y su fuente respectiva)
R: En Chile, las personas con discapacidad siguen enfrentando barreras importantes para acceder a la educación, al trabajo y a los servicios básicos. La Encuesta Nacional de Discapacidad y Dependencia (ENDIDE 2022) estima que un 17,6 % de la población adulta presenta alguna discapacidad, es decir, alrededor de 2,7 millones de personas, y que en la Región de Los Ríos la prevalencia llega aproximadamente al 21,6 % (61.404 personas). Dentro de estas discapacidades, las de tipo sensorial (visión y audición) generan barreras comunicacionales especialmente críticas en contextos de atención ciudadana, educación y salud. La OMS calcula que más de 430 millones de personas en el mundo tienen una pérdida de audición discapacitante, proyección que podría superar los 700 millones en 2050. En Chile, se estima que más de 800.000 personas presentan algún grado de hipoacusia o sordera, lo que refuerza la necesidad de soluciones que faciliten la comunicación entre personas sordas y oyentes. En la Región de Los Ríos cuyo sector productivo se apoya fuertemente en servicios, comercio, turismo y educación esta brecha comunicacional impacta directamente la inclusión laboral y el acceso a información de calidad (trámites municipales, salud, transporte, etc.). No siempre hay intérpretes de Lengua de Señas Chilena (LSCh) disponibles, y muchas interacciones diarias quedan fuera de la cobertura de los servicios formales.

Links Referencias: 

https://observatorio.ministeriodesarrollosocial.gob.cl/endide-2022
https://observatorio.ministeriodesarrollosocial.gob.cl/encuesta-casen-2022
https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss
https://www.elmostrador.cl/dia/2022/09/23/envejecimiento-y-sordera-una-realidad-invisibilizada/

   
2) Explicar la solución
R: La solución propuesta es un sistema de reconocimiento de lenguaje de señas basado en redes neuronales, que funciona como una aplicación web:

  El usuario se conecta desde un navegador (PC o notebook) y usa la cámara en tiempo real o bien sube una imagen de su mano realizando una seña.

  En el backend se ejecuta una API construida con FastAPI (main.py), que:

    Toma cada frame del video o la imagen subida.
    Usa un modelo YOLOv8 ajustado a gestos de mano para detectar y recortar automáticamente la mano en la imagen (utils.py, carpeta yolov8x-tuned-hand-gestures).
    Envía ese recorte a una red neuronal convolucional (CNN) entrenada para clasificar la seña (ml_model.py y train_model.py).

  El modelo predice una letra del alfabeto (A–Z), o las clases especiales “space” y “nothing”, y esa predicción se muestra en el frontend (index.html) como:

    Señal detectada (última letra)
    Palabra formada (concatenación de letras cuando se reconocen varias señas seguidas).

De esta manera, la aplicación actúa como un traductor básico de señas a texto, permitiendo que una persona sorda pueda deletrear palabras que luego son mostradas en pantalla para un interlocutor oyente. El enfoque inicial es el alfabeto dactilológico, pero la misma infraestructura sirve como base para extenderse a palabras y frases completas en el futuro.

   
3) Ver la competencia y explicar cómo se resuelve el problema actualmente
R: Actualmente, el problema de comunicación entre personas sordas y oyentes se aborda principalmente de tres formas:
1.- Intérpretes humanos de lengua de señas: En Chile, la Lengua de Señas Chilena (LSCh) fue reconocida legalmente, y en algunos contextos formales (televisión, servicios públicos, educación superior) se exigen intérpretes presenciales o en pantalla.
Ventaja: alta calidad de interpretación y contexto cultural.
Desventaja: costo elevado, disponibilidad limitada y difícil cobertura para interacciones cotidianas (comercio, pequeñas empresas, trámites simples, turismo, etc.).

2.- Aplicaciones de apoyo, pero no de reconocimiento automático: Existen apps y cursos para aprender lengua de señas (por ejemplo, cursos online de LSCh y materiales educativos digitales), pero suelen funcionar como guías de aprendizaje, no como traductores en tiempo real.

3.- Sistemas de IA para reconocimiento de lenguaje de señas (principalmente investigación / otros países): Proyectos comerciales e investigativos como SignAll desarrollan cabinas y sistemas de cámaras múltiples para traducir señas de ASL a texto o voz. Varios trabajos académicos usan visión por computador y deep learning (CNN, RNN, Transformers) para reconocimiento de signos o frases, incluidos prototipos para alfabetos de distintas lenguas de señas. En Chile hay tesis y prototipos que abordan el reconocimiento del alfabeto de la Lengua de Señas Chilena usando visión por computador, pero suelen quedar como proyectos académicos aislados y no llegan a producción masiva.

La solución se ubica en este tercer grupo, pero con un enfoque ligero y replicable (solo una cámara web + servidor con GPU/CPU), pensado para ser usado en espacios productivos locales: oficinas públicas, cajas de atención, instituciones educativas o empresas de servicios de la Región de Los Ríos. No depende de Internet por lo que es útil en oficinas públicas o instituciones que no quieren subir video a la nube y está pensada para integrarse con una interfaz simple tipo kiosk / escritorio que incluso un funcionario pueda usar sin previa capacitación.


Link Referencias:

https://signall.ai
https://github.com/ultralytics/ultralytics

   
4) Profundizar el diseño de su solución (Red neuronal)
R: En el proyecto se combinan dos redes neuronales principales:

Una red neuronal convolucional (CNN) es un tipo de red neuronal de aprendizaje profundo diseñada para procesar datos con una estructura de cuadrícula, como las imágenes. Funciona imitando la corteza visual humana para identificar patrones en los datos, como bordes, formas y texturas, a través de capas jerárquicas. A diferencia de las redes neuronales tradicionales, las CNN utilizan la operación matemática de convolución para extraer características de manera jerárquica, lo que les permite reconocer objetos independientemente de su posición o tamaño. 


1.- YOLOv8 especializado en manos (detección de objetos): Arquitectura tipo CNN de detección que recibe la imagen completa y devuelve coordenadas de las manos detectadas. El modelo pre-entrenado está afinado sobre un dataset de gestos de mano, y se carga desde la carpeta yolov8x-tuned-hand-gestures (utils.py y deteccion_manos.py). Su rol es localizar la mano, recortarla y entregar un ROI “limpio” al clasificador.

2.- Clasificador de señas basado en CNN (tu red principal de reconocimiento): Definida en ml_model.py / train_model.py:
  Entrada: imágenes RGB de la mano de tamaño 64×64×3, normalizadas en el rango [0,1].
  Capas convolucionales:
    3 bloques Conv2D → ReLU → BatchNormalization → MaxPooling2D:
    Conv2D(32 filtros, 3×3)
    Conv2D(64 filtros, 3×3)
    Conv2D(128 filtros, 3×3)
  Estas capas extraen patrones espaciales (bordes, formas de dedos, orientación de la mano).

Capa de aplanado y densas:
  Flatten()
  Dense(256, activation='relu') + Dropout(0.5) para reducir sobreajuste.

Capa de salida:
  Dense(num_classes, activation='softmax'), donde num_classes = 28 (A–Z + space + nothing).

Función de pérdida: categorical_crossentropy, coherente con una salida softmax multiclase.
Optimizador: Adam, recomendado en las diapositivas como optimizador estable y rápido para la mayoría de los modelos.

Esta arquitectura corresponde a una Red Neuronal Convolucional (CNN) utilizada para clasificación de imágenes, alineado con capas de entrada (imágenes), capas ocultas convolucionales + ReLU + BatchNorm + MaxPool, y capa de salida con activación y función de pérdida compatibles.

   
5) Aplicar técnicas de entrenamiento y de optimización
R: En tu código se aplican varias técnicas vistas en la clase:
  Normalización y preprocesamiento de imágenes
    Todas las imágenes se reescalan a 64×64 píxeles.
    Se normalizan dividiendo por 255 (rescale=1./255 en ImageDataGenerator).
  Data Augmentation (aumento de datos) – ImageDataGenerator en train_model.py:
    rotation_range=20
    width_shift_range y height_shift_range=0.1
    shear_range y zoom_range=0.1
    horizontal_flip=True
    Esto genera versiones rotadas, desplazadas y espejadas de las manos, aumentando la robustez del modelo frente a variaciones de posición y ángulo.
  Regularización:
    Dropout(0.5) en la capa densa final, para reducir sobreajuste.
    BatchNormalization en cada capa convolucional, estabilizando el entrenamiento y permitiendo tasas de aprendizaje más altas.

  Callbacks de entrenamiento:
    ModelCheckpoint → guarda el modelo con mejor precisión de validación (val_accuracy), asegurando que uses la mejor versión.
    EarlyStopping → detiene el entrenamiento si la val_accuracy no mejora tras 5 épocas, restaurando los mejores pesos. Esto es una implementación práctica de early stopping.
    
  Optimizador Adam:
    Se usa optimizer='adam', que combina ideas de RMSProp y momentum y suele converger más rápido en problemas de visión.
Todo esto corresponde a la aplicacion de técnicas de entrenamiento y optimización estándar para CNN: normalización, data augmentation, batch norm, dropout, early stopping y checkpointing.
    
    
6) Lograr de forma parcial la generalización del modelo
R: La generalización se aborda en tu proyecto de varias maneras:
    Separación entrenamiento/validación
    En train_model.py se crea un DataFrame con todas las imágenes y luego se hace un split 80 % / 20 % (train_df y val_df). Esto permite medir qué tan bien el modelo funciona sobre datos que no ha visto durante el entrenamiento.

    Uso de data augmentation
    El aumento de datos obliga a la red a aprender características más robustas (forma de la mano) en lugar de memorizar una posición exacta de los píxeles.

    Monitoreo de val_accuracy y val_loss
    Los callbacks EarlyStopping y ModelCheckpoint usan la precisión de validación como métrica para evitar sobreajuste.

Sin embargo, se habla de generalización parcial porque:
  El dataset usado es el asl_alphabet_test, con fondo y condiciones relativamente controladas.
  El modelo reconoce principalmente letras aisladas, no frases ni señas dinámicas completas.
  En condiciones reales (iluminación compleja, fondos ruidosos, manos muy cercanas o lejanas) el rendimiento puede bajar.

    
7) Explicar el tratamiento de los datos
R: El flujo de tratamiento de datos en tu solución es:
1.- Captura / entrada: Frames desde la cámara (cv2.VideoCapture(0)) o imágenes subidas vía formulario (/predict en main.py).
2.- Segmentación de la mano: Se ejecuta el modelo YOLO (yolov8x-tuned-hand-gestures) para detectar la mano en la imagen (detect_hand_and_crop en utils.py). Si se detecta mano, se recorta solo la región de interés (ROI). Si no se detecta, se intenta una segunda estrategia con MediaPipe Hands (en ml_model.py) o se clasifica como "nothing".
3.- Preprocesamiento: El recorte se redimensiona a 64×64 píxeles (cv2.resize). Se normaliza a flotantes en [0,1] (/ 255.0). Se añade una dimensión extra para formar un batch de tamaño 1 (np.expand_dims).
4.- Etiquetado: Las clases se definen en CLASSES = [A–Z] + ["space", "nothing"]. Durante el entrenamiento, las etiquetas se representan como one-hot vectors (class_mode="categorical" y to_categorical).
5.- Registro de predicciones: Cada predicción se guarda en prediction_history.csv con timestamp, prediction y source (save_prediction_log en utils.py). Esto permite analizar el rendimiento posterior y usar los datos para futuros reentrenamientos.

En resumen, el tratamiento de datos sigue el pipeline recomendado en la presentación: normalización, estandarización de tamaño, data augmentation, vectorización de clases y logging de resultados.

     
8) Ver mejoras a futuro
1.- Pasar de alfabeto a palabras/frases reales: Incorporar secuencias de video (no solo imágenes estáticas) usando modelos tipo RNN/LSTM/GRU o Transformers temporales, como se describe en las diapositivas para datos de series de tiempo y texto. Detectar no solo letras, sino señas completas frecuentes (ej. “hola”, “gracias”, “hospital”, “ayuda”).

2.- Adaptación a Lengua de Señas Chilena (LSCh): Construir o utilizar un dataset específico de LSCh, con apoyo de instituciones locales de la Región de Los Ríos. Ajustar el modelo para incorporar diferencias culturales y lingüísticas respecto al ASL.

3.- Mejor experiencia de usuario: Agregar salida de voz sintetizada del texto reconocido (Text-to-Speech). Incorporar un historial de frases, opciones para corregir errores y exportar conversaciones.

4.- Optimización y despliegue en dispositivos móviles: Convertir el modelo a TensorFlow Lite para ejecutarlo en celulares Android o en dispositivos edge de bajo costo. Reducir tamaño de modelo (podas, cuantización) para acelerar la inferencia.

5.- Evaluación sistemática: Medir métricas como precision, recall, F1-score y matriz de confusión, además de la accuracy.

    
9) Aproximar cómo se presentaría esta solución en producción.
Una posible arquitectura de producción para tu sistema sería:

  Frontend web ligero: Página en HTML + Tailwind CSS (index.html, styles.css) servida desde un hosting estático (por ejemplo, un bucket en la nube o un servidor web en la institución). Interacción vía HTTPS con la API (/video_feed, /predict, /history).

  Backend de inferencia: Servicio FastAPI (main.py) ejecutándose en un servidor (on-premise en la institución o en la nube). El servicio carga en memoria: El modelo YOLO para detección de manos. El modelo CNN sign_model.h5 para clasificación. Se podrían empaquetar   ambos en un contenedor Docker, facilitando el despliegue y la escalabilidad.

Hardware: Para uso en una oficina de atención al público, bastaría un mini-PC con GPU modesta o CPU potente + cámara HD. El sistema podría instalarse en puntos de atención de municipalidades, servicios de salud o centros educativos de la Región de Los Ríos.

Registro y monitoreo: Las predicciones podrían almacenarse en una base de datos en lugar de un CSV, para: Monitorear uso. Detectar tipos de errores frecuentes. Generar nuevos datasets para reentrenar el modelo.

Integración con otros sistemas: A futuro, la API podría integrarse con: Sistemas de gestión de filas. Plataformas de atención ciudadana. Aplicaciones móviles institucionales.

De esta forma, el prototipo pasaría de ser un proyecto academico a un módulo reutilizable de accesibilidad, que distintas organizaciones del sector servicios en la Región de Los Ríos podrían adoptar para reducir barreras de comunicación con personas sordas.

