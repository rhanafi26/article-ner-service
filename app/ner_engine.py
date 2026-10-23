import spacy
from typing import List, Dict, Any

MODEL_PATH = "models/ner_adapted"

class NEREngine:
    def __init__(self):
        # Memuat model spaCy yang sudah dilatih/di-adaptasi
        self.nlp = spacy.load(MODEL_PATH)

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            })
        return entities

# Instansiasi objek agar siap digunakan oleh FastAPI
ner_engine = NEREngine()