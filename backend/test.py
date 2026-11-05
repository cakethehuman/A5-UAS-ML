from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_image_upload_with_file():
    with open(r"backend/handwritting.jpg", "rb") as image_file:
        file_payload = {"file": ("sample.jpg", image_file, "image/jpeg")}
        response = client.post("/upload-image/", files=file_payload)
        
    assert response.status_code == 200
    assert response.json()["filename"] == "sample.jpg"
