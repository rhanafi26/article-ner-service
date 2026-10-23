from fastapi import FastAPI, HTTPException
from app.schemas import ArticleRequest, NERResponse
from app.ner_engine import ner_engine

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Article NER Service",
    description="API untuk ekstraksi Named Entity Recognition (NER) dari artikel",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Endpoint health check untuk memastikan API berjalan."""
    return {"message": "Article NER Service ready"}

@app.post("/predict", response_model=NERResponse)
def predict_entities(request: ArticleRequest):
    """Endpoint untuk mengekstraksi entitas dari teks artikel."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Teks artikel tidak boleh kosong.")
    
    try:
        entities = ner_engine.extract_entities(request.text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses teks: {str(e)}")