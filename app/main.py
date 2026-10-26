from fastapi import FastAPI, HTTPException
from app.schemas import ArticleRequest, NERResponse
from src.ner_model import ner_service  

app = FastAPI(
    title="Article NER Service",
    description="API untuk ekstraksi Named Entity Recognition (NER) dari artikel",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Article NER Service ready"}

@app.post("/extract-entities", response_model=NERResponse)
def extract_entities_endpoint(request: ArticleRequest):
    if not request.body.strip():
        raise HTTPException(status_code=400, detail="Isi artikel tidak boleh kosong.")
    
    try:
        result = ner_service.extract_entities(
            title=request.title,
            body=request.body,
            model_type=request.model_type or "adapted"
        )
        return result
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses teks: {str(exc)}")