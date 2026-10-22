import json
import os
import sys

# Ensure root dir in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding='utf-8')

from collections import defaultdict
from typing import Dict, List, Set, Tuple
from src.ner_model import ner_service

EVAL_PATH = os.path.join("data", "processed", "evaluation_examples.json")

def calculate_metrics(gold_set: Set[Tuple[int, int, str]], pred_set: Set[Tuple[int, int, str]]):
    tp = len(gold_set.intersection(pred_set))
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return tp, fp, fn, precision, recall, f1

def evaluate_model(model_type: str = "adapted"):
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_examples = json.load(f)

    overall_gold: Set[Tuple[int, int, str]] = set()
    overall_pred: Set[Tuple[int, int, str]] = set()

    per_class_gold = defaultdict(set)
    per_class_pred = defaultdict(set)

    error_cases = []

    for idx, ex in enumerate(eval_examples):
        title = ex.get("title", "")
        body = ex.get("body", "")
        gold_entities = ex.get("gold_entities", [])

        # Build gold set for this doc
        doc_gold: Set[Tuple[int, int, str]] = set()
        for g in gold_entities:
            span = (g["start"], g["end"], g["label"])
            doc_gold.add(span)
            per_class_gold[g["label"]].add((idx, span[0], span[1]))
            overall_gold.add((idx, span[0], span[1], span[2]))

        # Predict
        try:
            pred_res = ner_service.extract_entities(title, body, model_type=model_type)
            doc_pred: Set[Tuple[int, int, str]] = set()
            for p in pred_res.get("entities", []):
                span = (p["start"], p["end"], p["label"])
                doc_pred.add(span)
                per_class_pred[p["label"]].add((idx, span[0], span[1]))
                overall_pred.add((idx, span[0], span[1], span[2]))
        except Exception as e:
            print(f"Error predicting article {idx+1}: {e}")
            doc_pred = set()

        # Check errors
        combined_text = f"{title}\n\n{body}"
        fps = doc_pred - doc_gold
        fns = doc_gold - doc_pred

        for fp in fps:
            ent_text = combined_text[fp[0]:fp[1]]
            error_cases.append({
                "type": "False Positive",
                "article_id": ex.get("original_id"),
                "text": ent_text,
                "label": fp[2],
                "start": fp[0],
                "end": fp[1]
            })

        for fn in fns:
            ent_text = combined_text[fn[0]:fn[1]]
            error_cases.append({
                "type": "False Negative",
                "article_id": ex.get("original_id"),
                "text": ent_text,
                "label": fn[2],
                "start": fn[0],
                "end": fn[1]
            })

    # Overall metrics
    tp, fp, fn, prec, rec, f1 = calculate_metrics(overall_gold, overall_pred)

    print(f"\n================ EVALUATION RESULTS: {model_type.upper()} MODEL ================")
    print(f"Total Gold Spans: {len(overall_gold)} | Total Predicted Spans: {len(overall_pred)}")
    print(f"Overall Precision : {prec:.4f}")
    print(f"Overall Recall    : {rec:.4f}")
    print(f"Overall F1-Score  : {f1:.4f}\n")

    print("--- Per-Class Metrics ---")
    all_classes = sorted(list(set(list(per_class_gold.keys()) + list(per_class_pred.keys()))))
    for cls in all_classes:
        c_gold = per_class_gold[cls]
        c_pred = per_class_pred[cls]
        _, _, _, c_prec, c_rec, c_f1 = calculate_metrics(c_gold, c_pred)
        print(f"Class {cls:8s} -> P: {c_prec:.4f} | R: {c_rec:.4f} | F1: {c_f1:.4f} (Gold: {len(c_gold)}, Pred: {len(c_pred)})")

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "error_cases": error_cases
    }

if __name__ == "__main__":
    print("Evaluating Baseline Model (spaCy en_core_web_sm)...")
    base_res = evaluate_model("baseline")

    print("\nEvaluating Adapted Model...")
    adapt_res = evaluate_model("adapted")
