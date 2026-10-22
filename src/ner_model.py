import os
import spacy
from typing import Dict, Any, List

LABEL_MAPPING = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "LOC",
    "DATE": "DATE",
    "EVENT": "EVENT",
    "FAC": "LOC",
    "NORP": "ORG",
}

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "ner_adapted")
MAX_TEXT_LENGTH = 50000

class ArticleNERService:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = model_dir
        self._baseline_nlp = None
        self._adapted_nlp = None

    def get_baseline_nlp(self):
        if self._baseline_nlp is None:
            self._baseline_nlp = spacy.load("en_core_web_sm")
        return self._baseline_nlp

    def get_adapted_nlp(self):
        if self._adapted_nlp is None:
            if os.path.exists(self.model_dir):
                self._adapted_nlp = spacy.load(self.model_dir)
            else:
                nlp = spacy.load("en_core_web_sm")
                self._adapted_nlp = nlp
        return self._adapted_nlp

    def validate_input(self, title: Any, body: Any) -> None:
        if not isinstance(title, str) or not isinstance(body, str):
            raise ValueError("Both 'title' and 'body' must be strings.")
        
        cleaned_title = title.strip()
        cleaned_body = body.strip()
        
        if not cleaned_title or not cleaned_body:
            raise ValueError("'title' and 'body' cannot be empty or whitespace only.")
            
        total_len = len(cleaned_title) + len(cleaned_body)
        if total_len > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Combined input length ({total_len} chars) exceeds maximum allowed length of {MAX_TEXT_LENGTH} chars."
            )

    def extract_entities(self, title: str, body: str, model_type: str = "adapted") -> Dict[str, Any]:
        self.validate_input(title, body)

        combined_text = f"{title.strip()}\n\n{body.strip()}"

        if model_type == "baseline":
            nlp = self.get_baseline_nlp()
            model_name = "spacy-en_core_web_sm-baseline"
        else:
            nlp = self.get_adapted_nlp()
            model_name = "spacy-ner-malaysia-banking-adapted"

        doc = nlp(combined_text)

        entities: List[Dict[str, Any]] = []
        seen_spans = set()

        for ent in doc.ents:
            mapped_label = LABEL_MAPPING.get(ent.label_, None)
            if mapped_label is None:
                continue

            span_key = (ent.start_char, ent.end_char, mapped_label)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            entities.append({
                "text": ent.text,
                "label": mapped_label,
                "start": ent.start_char,
                "end": ent.end_char
            })

        entities.sort(key=lambda x: (x["start"], x["end"]))

        return {
            "entities": entities,
            "meta": {
                "entity_count": len(entities),
                "model": model_name
            }
        }

ner_service = ArticleNERService()
