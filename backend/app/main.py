import numpy as np
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

import tensorflow as tf

app = FastAPI()

IMG_SIZE = 224
MODEL_PATH = 'backend/models/model_test.keras'

model = tf.keras.models.load_model(MODEL_PATH)
class PredictionValue(BaseModel):
    Label: int
    Probability: list[float]


@app.get("/health", tags=["health"])
async def health():
    return {"message": "API working fine"}


@app.post("/predict", tags=['prediction'],  response_model=PredictionValue)
async def predict(file: UploadFile = File(...)):
    content = await file.read()

    image = tf.keras.utils.load_img(
        BytesIO(content),
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    results = model.predict(image)[0]

    return {
        "Label" : int(np.argmax(results)),
        "Probability": results.tolist(),
    }