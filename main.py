from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from deepface import DeepFace
import tempfile, os

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Python AI service is running"}


@app.post("/deepface-test")
async def deepface_test(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        img_path = tmp.name

    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=True
        )

        if not result or len(result) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        embedding = result[0]["embedding"]

        return {
            "embedding": embedding
        }

    finally:
        os.unlink(img_path)
