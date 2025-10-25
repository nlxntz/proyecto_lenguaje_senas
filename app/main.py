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

app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

MODEL_PATH = "sign_model.h5"
DATASET_PATH = "dataset/asl_alphabet_test"
_, _, LABELS = load_dataset(DATASET_PATH)
MODEL = load_trained_model(MODEL_PATH)

@app.get("/")
async def get_index():
    return FileResponse(os.path.join("app/frontend", "index.html"))

@app.get("/api")
def read_root():
    return {"message": "API de lenguaje de señas funcionando"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(img_bytes)

        predicted_label = predict_image(MODEL, temp_path, LABELS)

        os.remove(temp_path)

        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})