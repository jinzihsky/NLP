"""Predict sentiment for one Chinese movie review.

Examples:
    python predict.py --text "這部電影節奏很好，演員也很有感染力"
    python predict.py --model-type tfidf --text "劇情很拖，看到一半就想睡"
    python predict.py --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABEL_ORDER = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict positive/negative sentiment for Chinese reviews.")
    parser.add_argument(
        "--model-type",
        choices=["bert", "tfidf"],
        default="bert",
        help="Model type to use.",
    )
    parser.add_argument(
        "--bert-model-dir",
        default="models/bert_chinese_full/best_model",
        help="BERT model directory.",
    )
    parser.add_argument(
        "--tfidf-model",
        default="models/tfidf_logreg_full/tfidf_logreg_model.joblib",
        help="TF-IDF model path.",
    )
    parser.add_argument("--text", help="Review text to classify.")
    parser.add_argument("--interactive", action="store_true", help="Read review text repeatedly from stdin.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum BERT token length.")
    parser.add_argument("--cpu", action="store_true", help="Force BERT inference on CPU.")
    return parser.parse_args()


def predict_with_tfidf(model_path: Path, text: str) -> tuple[str, float | None]:
    if not model_path.exists():
        raise FileNotFoundError(f"TF-IDF model not found: {model_path}")

    model = joblib.load(model_path)
    label = str(model.predict([text])[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probabilities = model.predict_proba([text])[0]
        confidence = float(probabilities[classes.index(label)])

    return label, confidence


def load_bert(model_dir: Path, force_cpu: bool):
    if not model_dir.exists():
        raise FileNotFoundError(f"BERT model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model, device


@torch.no_grad()
def predict_with_bert(tokenizer, model, device: torch.device, text: str, max_length: int) -> tuple[str, float]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    prediction_id = int(torch.argmax(probabilities).item())
    label = model.config.id2label.get(prediction_id, LABEL_ORDER[prediction_id])
    confidence = float(probabilities[prediction_id].item())
    return label, confidence


def print_prediction(text: str, label: str, confidence: float | None) -> None:
    zh_label = "正面" if label == "positive" else "負面"
    print(f"輸入：{text}")
    print(f"預測：{label} ({zh_label})")
    if confidence is not None:
        print(f"信心分數：{confidence:.4f}")


def main() -> None:
    args = parse_args()

    bert_assets = None
    if args.model_type == "bert":
        bert_assets = load_bert(Path(args.bert_model_dir), args.cpu)
        print(f"Loaded BERT model: {args.bert_model_dir}")
        print(f"Device: {bert_assets[2]}")
    else:
        print(f"Loaded TF-IDF model: {args.tfidf_model}")

    def predict_one(text: str) -> None:
        text = text.strip()
        if not text:
            print("請輸入非空白文字。")
            return

        if args.model_type == "bert":
            assert bert_assets is not None
            tokenizer, model, device = bert_assets
            label, confidence = predict_with_bert(tokenizer, model, device, text, args.max_length)
        else:
            label, confidence = predict_with_tfidf(Path(args.tfidf_model), text)

        print_prediction(text, label, confidence)

    if args.interactive:
        print("輸入中文評論後按 Enter；輸入 q 離開。")
        while True:
            text = input("> ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break
            predict_one(text)
            print()
    elif args.text:
        predict_one(args.text)
    else:
        raise SystemExit("請使用 --text 輸入一句評論，或使用 --interactive 進入互動模式。")


if __name__ == "__main__":
    main()
