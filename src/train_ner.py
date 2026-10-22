import json
import os
import random
import time
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

RANDOM_SEED = 42
TRAIN_SPLIT_PATH = os.path.join("data", "processed", "train_split.json")
VAL_SPLIT_PATH = os.path.join("data", "processed", "val_split.json")
OUTPUT_MODEL_DIR = os.path.join("models", "ner_adapted")

# Malaysian Banking & Corporate Domain Patterns
DOMAIN_PATTERNS = [
    # Banking & Financial Institutions
    {"label": "ORG", "pattern": "Maybank"},
    {"label": "ORG", "pattern": "Malayan Banking Berhad"},
    {"label": "ORG", "pattern": "Maybank Islamic"},
    {"label": "ORG", "pattern": "Maybank Investment Bank"},
    {"label": "ORG", "pattern": "Maybank Foundation"},
    {"label": "ORG", "pattern": "CIMB"},
    {"label": "ORG", "pattern": "CIMB Bank"},
    {"label": "ORG", "pattern": "CIMB Group"},
    {"label": "ORG", "pattern": "CIMB Investment Bank"},
    {"label": "ORG", "pattern": "CIMB Securities"},
    {"label": "ORG", "pattern": "RHB"},
    {"label": "ORG", "pattern": "RHB Bank"},
    {"label": "ORG", "pattern": "RHB Investment Bank"},
    {"label": "ORG", "pattern": "Public Bank"},
    {"label": "ORG", "pattern": "Public Bank Berhad"},
    {"label": "ORG", "pattern": "Hong Leong Bank"},
    {"label": "ORG", "pattern": "HLB"},
    {"label": "ORG", "pattern": "HLB Islamic"},
    {"label": "ORG", "pattern": "Hong Leong Investment Bank"},
    {"label": "ORG", "pattern": "Bank Simpanan Nasional"},
    {"label": "ORG", "pattern": "BSN"},
    {"label": "ORG", "pattern": "Bank Negara Malaysia"},
    {"label": "ORG", "pattern": "BNM"},
    {"label": "ORG", "pattern": "Bursa Malaysia"},
    {"label": "ORG", "pattern": "Bursa Malaysia Securities"},
    {"label": "ORG", "pattern": "Bursa Securities"},
    {"label": "ORG", "pattern": "Securities Commission Malaysia"},
    {"label": "ORG", "pattern": "Securities Commission"},
    {"label": "ORG", "pattern": "Malaysia Debt Ventures"},
    {"label": "ORG", "pattern": "Malaysia Debt Ventures Bhd"},
    {"label": "ORG", "pattern": "MDV"},
    {"label": "ORG", "pattern": "Khazanah Nasional"},
    {"label": "ORG", "pattern": "Khazanah Nasional Bhd"},
    {"label": "ORG", "pattern": "Khazanah"},
    {"label": "ORG", "pattern": "PETRONAS"},
    {"label": "ORG", "pattern": "Petroliam Nasional Bhd"},
    {"label": "ORG", "pattern": "Tenaga Nasional"},
    {"label": "ORG", "pattern": "Tenaga Nasional Bhd"},
    {"label": "ORG", "pattern": "TNB"},
    {"label": "ORG", "pattern": "Inland Revenue Board"},
    {"label": "ORG", "pattern": "IRB"},
    {"label": "ORG", "pattern": "Audit Oversight Board"},
    {"label": "ORG", "pattern": "AOB"},
    {"label": "ORG", "pattern": "IOI Properties Group Bhd"},
    {"label": "ORG", "pattern": "IOI Properties"},
    {"label": "ORG", "pattern": "IOIPG"},
    {"label": "ORG", "pattern": "IOIPG REIT"},
    {"label": "ORG", "pattern": "Gamuda"},
    {"label": "ORG", "pattern": "Gamuda Bhd"},
    {"label": "ORG", "pattern": "ViTrox"},
    {"label": "ORG", "pattern": "ViTrox Corp Bhd"},
    {"label": "ORG", "pattern": "AirAsia"},
    {"label": "ORG", "pattern": "AirAsia Group Bhd"},
    {"label": "ORG", "pattern": "Batu Kawan"},
    {"label": "ORG", "pattern": "Batu Kawan Bhd"},
    {"label": "ORG", "pattern": "MKH"},
    {"label": "ORG", "pattern": "Gentari"},
    {"label": "ORG", "pattern": "Gentari Sdn Bhd"},
    {"label": "ORG", "pattern": "SD Guthrie"},
    {"label": "ORG", "pattern": "SD Guthrie Bhd"},
    {"label": "ORG", "pattern": "SPRM"},
    {"label": "ORG", "pattern": "Suruhanjaya Pencegahan Rasuah Malaysia"},
    {"label": "ORG", "pattern": "PAC"},
    {"label": "ORG", "pattern": "Jawatankuasa Kira-Kira Wang Negara"},
    {"label": "ORG", "pattern": "Jabatan Audit Negara"},
    {"label": "ORG", "pattern": "UiTM"},
    {"label": "ORG", "pattern": "UiTM Holdings"},
    {"label": "ORG", "pattern": "KMUiTM"},
    {"label": "ORG", "pattern": "Kolektif Mahasiswa UiTM"},
    {"label": "ORG", "pattern": "Aliran"},
    {"label": "ORG", "pattern": "Persatuan Aliran Kesedaran Negara"},
    {"label": "ORG", "pattern": "KPMG"},
    {"label": "ORG", "pattern": "Ernst & Young"},
    {"label": "ORG", "pattern": "EY"},

    # Locations (GPE / LOC)
    {"label": "GPE", "pattern": "Malaysia"},
    {"label": "GPE", "pattern": "Kuala Lumpur"},
    {"label": "GPE", "pattern": "Petaling Jaya"},
    {"label": "GPE", "pattern": "Putrajaya"},
    {"label": "GPE", "pattern": "Johor"},
    {"label": "GPE", "pattern": "Penang"},
    {"label": "GPE", "pattern": "Sabah"},
    {"label": "GPE", "pattern": "Sarawak"},
    {"label": "GPE", "pattern": "Kuching"},
    {"label": "GPE", "pattern": "Langkawi"},
    {"label": "GPE", "pattern": "Seremban"},
    {"label": "GPE", "pattern": "Puncak Alam"},
    {"label": "GPE", "pattern": "Tapah"},
    {"label": "GPE", "pattern": "Kota Samarahan"},
    {"label": "GPE", "pattern": "Pasir Gudang"},
    {"label": "GPE", "pattern": "Jasin"},
    {"label": "GPE", "pattern": "Gambang"},
    {"label": "GPE", "pattern": "Batu Kawan"},
    {"label": "GPE", "pattern": "Sungai Buloh"},
    {"label": "GPE", "pattern": "Singapore"},

    # Prominent Persons
    {"label": "PERSON", "pattern": "Anwar Ibrahim"},
    {"label": "PERSON", "pattern": "Rafizi Ramli"},
    {"label": "PERSON", "pattern": "Steven Sim"},
    {"label": "PERSON", "pattern": "Charles Santiago"},
    {"label": "PERSON", "pattern": "Dafinah Ahmed Hilmi"},
    {"label": "PERSON", "pattern": "Tony Fernandes"},
    {"label": "PERSON", "pattern": "Lee Yeow Seng"},
    {"label": "PERSON", "pattern": "Chu Jenn Weng"},
    {"label": "PERSON", "pattern": "Amir Hamzah Azizan"},
    {"label": "PERSON", "pattern": "Amir Hamzah"},
    {"label": "PERSON", "pattern": "Abdul Rasheed Ghaffour"},
    {"label": "PERSON", "pattern": "Faizoull Ahmad"},
    {"label": "PERSON", "pattern": "Mahathir Mohamad"},
    {"label": "PERSON", "pattern": "Hasmah Mohd Ali"},
    {"label": "PERSON", "pattern": "R Ramanan"},
    {"label": "PERSON", "pattern": "Khairul Firdaus Akbar Khan"},
    {"label": "PERSON", "pattern": "Veerinderjeet Singh"},
    {"label": "PERSON", "pattern": "Dr Veerinderjeet Singh"},
]

def load_data(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_training_examples(nlp, articles, max_samples=250):
    """
    Constructs gold/silver training examples combining domain patterns and existing entity knowledge.
    """
    examples = []
    ruler_nlp = spacy.blank("en")
    ruler = ruler_nlp.add_pipe("entity_ruler")
    ruler.add_patterns(DOMAIN_PATTERNS)

    class_counts = {"ORG": 0, "GPE": 0, "PERSON": 0, "DATE": 0, "EVENT": 0}

    for item in articles[:max_samples]:
        title = item.get("title", "")
        body = item.get("body", "")
        text = f"{title}\n\n{body}"
        if len(text.strip()) == 0:
            continue

        # Extract gold spans using ruler on the text
        doc_ruler = ruler_nlp(text)
        gold_spans = []
        for ent in doc_ruler.ents:
            gold_spans.append((ent.start_char, ent.end_char, ent.label_))
            class_counts[ent.label_] = class_counts.get(ent.label_, 0) + 1

        # Also leverage baseline predictions for standard entities not overlapping
        doc_base = nlp.make_doc(text)
        entities_dict = {"entities": gold_spans}
        try:
            example = Example.from_dict(doc_base, entities_dict)
            examples.append(example)
        except Exception:
            pass

    return examples, class_counts

def train_and_adapt_model():
    random.seed(RANDOM_SEED)
    start_time = time.time()

    print(f"Loading training data from {TRAIN_SPLIT_PATH}...")
    train_data = load_data(TRAIN_SPLIT_PATH)
    print(f"Loading validation data from {VAL_SPLIT_PATH}...")
    val_data = load_data(VAL_SPLIT_PATH)

    print(f"Total train documents: {len(train_data)}")
    print(f"Total val documents: {len(val_data)}")

    print("Initializing base spaCy pipeline (en_core_web_sm)...")
    nlp = spacy.load("en_core_web_sm")

    # Add Entity Ruler before NER to ground domain-specific financial entities
    if "entity_ruler" in nlp.pipe_names:
        nlp.remove_pipe("entity_ruler")
    
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(DOMAIN_PATTERNS)

    print("Preparing training examples...")
    train_examples, class_distribution = create_training_examples(nlp, train_data, max_samples=300)
    print(f"Created {len(train_examples)} training examples.")
    print("Class distribution:", class_distribution)

    # Fine-tune the statistical NER component
    ner = nlp.get_pipe("ner")
    for pattern in DOMAIN_PATTERNS:
        ner.add_label(pattern["label"])

    pipe_exceptions = ["ner"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    print("Starting adaptation and fine-tuning iterations...")
    epochs = 10
    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.resume_training()
        for epoch in range(epochs):
            random.shuffle(train_examples)
            losses = {}
            batches = minibatch(train_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer)
            print(f"Epoch {epoch + 1}/{epochs} - NER Loss: {losses.get('ner', 0.0):.4f}")

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    nlp.to_disk(OUTPUT_MODEL_DIR)
    duration = time.time() - start_time
    print(f"Trained model artifact successfully saved to: {OUTPUT_MODEL_DIR}")
    print(f"Total training/adaptation duration: {duration:.2f} seconds")

if __name__ == "__main__":
    train_and_adapt_model()
