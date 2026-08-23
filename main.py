import os
import urllib.request
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

# =========================================================================
URL_MODEL_BISINDO = "https://huggingface.co/elfrumoasa/bisindo-yolo/blob/main/best-bisindo.pt"
URL_MODEL_KESEHARIAN = "https://huggingface.co/elfrumoasa/bisindo-yolo/blob/main/best-keseharian.pt"
# =========================================================================

def download_model(url, filename):
    # Hanya download jika file belum ada
    if not os.path.exists(filename):
        print(f"Downloading {filename} from Hugging Face...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"✅ Downloaded {filename} successfully!")
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")

# Proses download dijalankan sebelum API menyala
download_model(URL_MODEL_BISINDO, "best-bisindo.pt")
download_model(URL_MODEL_KESEHARIAN, "best-keseharian.pt")

app = FastAPI()

# Load model hasil fine-tuning
model_bisindo = YOLO("best-bisindo.pt")
model_keseharian = YOLO("best-keseharian.pt")

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "BISINDO YOLO API is running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    results = model_bisindo.predict(
        source=image,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    predictions = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        predictions.append({
            "class": model_bisindo.names[class_id],
            "confidence": round(confidence, 4)
        })

    return {
        "predictions": predictions
    }

@app.post("/predict-keseharian")
async def predict_keseharian(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    results = model_keseharian.predict(
        source=image,
        conf=0.5,
        verbose=False
    )

    result = results[0]

    predictions = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        predictions.append({
            "class": model_keseharian.names[class_id],
            "confidence": round(confidence, 4)
        })

    return {
        "predictions": predictions
    }