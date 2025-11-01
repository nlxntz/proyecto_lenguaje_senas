from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.ml_model import load_trained_model, predict_image
from app.utils import load_dataset, save_prediction_log
import os
import pandas as pd

app = FastAPI(title="Proyecto Lenguaje de Señas")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="app/frontend"), name="frontend")

MODEL_PATH = "sign_model.h5"
DATASET_PATH = "dataset/asl_alphabet_test"

_, _, LABELS = load_dataset(DATASET_PATH)
MODEL = load_trained_model(MODEL_PATH)

current_word = ""

@app.get("/")
async def get_index():
    return FileResponse(os.path.join("app/frontend", "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global current_word
    try:
        img_bytes = await file.read()
        predicted_label = predict_image(MODEL, img_bytes, LABELS)

        if len(predicted_label) == 1 and predicted_label.isalpha():
            current_word += predicted_label
        elif predicted_label == "No se detectó mano":
            current_word = ""

        save_prediction_log(predicted_label, source="camera")
        return JSONResponse(content={"prediction": predicted_label, "word": current_word})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/history")
async def get_history():
    try:
        df = pd.read_csv("prediction_history.csv")
        return JSONResponse(content=df.tail(10).to_dict(orient="records"))
    except FileNotFoundError:
        return JSONResponse(content=[])