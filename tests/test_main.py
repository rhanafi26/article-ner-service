from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Article NER Service ready"}

def test_predict_entities_success():
    payload = {
        "text": "Jokowi mengunjungi Jakarta minggu lalu."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert isinstance(data["entities"], list)

def test_predict_entities_empty_text():
    payload = {
        "text": "   "
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Teks artikel tidak boleh kosong."