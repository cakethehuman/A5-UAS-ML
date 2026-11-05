from fastapi import FastAPI, File, UploadFile

from pydantic import BaseModel

app = FastAPI()

class PredictionValue(BaseModel):
    label: int | None = None
    probabilty: float | None = None
    

@app.get("/health")
async def health():
    return {"message": "API working fine"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {file.filename}

