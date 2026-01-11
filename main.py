from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
from deepface import DeepFace

app = FastAPI()

face_cascade = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

@app.get("/")
def root():
    return {"message": "Python AI service is running"}

@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    image_bytes = await file.read()

    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No face detected")

    if len(faces) > 1:
        raise HTTPException(status_code=400, detail="Multiple faces detected")

    return {
        "message": "Exactly one face detected",
        "faces_count": len(faces)
    }

@app.post("/face-embedding")
async def face_embedding(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    if len(faces) != 1:
        raise HTTPException(status_code=400, detail="Image must contain exactly one face")

    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (200, 200))

    # Train with single image (temporary identity = 0)
    recognizer.train([face_resized], np.array([0]))

    histogram = recognizer.getHistograms()[0].flatten()

    return {
        "embedding_length": len(histogram),
        "embedding_sample": histogram[:10].tolist()
    }

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


@app.post("/compare-faces")
async def compare_faces(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    import tempfile, os

    def get_embedding(file: UploadFile):
        img_bytes = file.file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(img_bytes)
            path = tmp.name

        try:
            result = DeepFace.represent(
                img_path=path,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False
            )
            return result[0]["embedding"]
        finally:
            os.remove(path)

    emb1 = get_embedding(file1)
    emb2 = get_embedding(file2)

    similarity = cosine_similarity(emb1, emb2)

    THRESHOLD = 0.80  # 🔥 realistic for ArcFace
    return {
        "similarity_score": similarity,
        "match": similarity >= THRESHOLD
    }


from deepface import DeepFace

@app.post("/deepface-test")
async def deepface_test(file: UploadFile = File(...)):
    import tempfile, os

    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(image_bytes)
        img_path = tmp.name

    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=True
        )

        # 🔑 IMPORTANT PART
        if not result or len(result) == 0:
            return {"embedding": None}

        embedding = result[0]["embedding"]

        return {
            "embedding": embedding
        }

    finally:
        os.unlink(img_path)
