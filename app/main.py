from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.ml_model import load_trained_model, predict_image
from app.utils import load_dataset
import os

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

@app.get("/")
async def get_index():
    return FileResponse(os.path.join("app/frontend", "index.html"))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        predicted_label = predict_image(MODEL, img_bytes, LABELS)
        return JSONResponse(content={"prediction": predicted_label})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})