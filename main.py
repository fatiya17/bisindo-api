from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load model hasil fine-tuning
model = YOLO("best.pt")


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

    results = model.predict(
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
            "class": model.names[class_id],
            "confidence": round(confidence, 4)
        })

    return {
        "predictions": predictions
    }