"""Fine-tune a Chinese BERT sentiment classifier for Douban movie reviews.

Example:
    python train_bert_model.py --max-per-class 100000 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


LABEL_ORDER = ["negative", "positive"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Chinese BERT for sentiment classification.")
    parser.add_argument(
        "--input",
        default="data/processed/douban_movie_reviews_clean.csv",
        help="Cleaned CSV from preprocess_douban.py.",
    )
    parser.add_argument(
        "--model-name",
        default="bert-base-chinese",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/bert_chinese",
        help="Directory for model and evaluation outputs.",
    )
    parser.add_argument("--text-column", default="comment_clean", help="Text feature column.")
    parser.add_argument("--label-column", default="sentiment", help="Target label column.")
    parser.add_argument("--max-per-class", type=int, default=100_000, help="Rows sampled per class. Use 0 for all rows.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test split ratio.")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum token length.")
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Evaluation batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.06, help="Warmup ratio for linear scheduler.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Accumulate gradients across steps.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. 0 is safest on Windows.")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable weighted loss for imbalanced data.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(args: argparse.Namespace) -> pd.DataFrame:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, usecols=[args.text_column, args.label_column])
    df = df.dropna(subset=[args.text_column, args.label_column]).copy()
    df[args.text_column] = df[args.text_column].astype(str).str.strip()
    df = df[df[args.text_column].ne("")]
    df = df[df[args.label_column].isin(LABEL_ORDER)]

    if args.max_per_class and args.max_per_class > 0:
        parts = []
        for label in LABEL_ORDER:
            part = df[df[args.label_column] == label]
            sample_size = min(args.max_per_class, len(part))
            parts.append(part.sample(sample_size, random_state=args.random_state))
        df = pd.concat(parts).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    return df


def make_loaders(args: argparse.Namespace, tokenizer, df: pd.DataFrame):
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.label_column],
    )

    train_dataset = ReviewDataset(
        train_df[args.text_column].tolist(),
        train_df[args.label_column].map(LABEL_TO_ID).tolist(),
        tokenizer,
        args.max_length,
    )
    test_dataset = ReviewDataset(
        test_df[args.text_column].tolist(),
        test_df[args.label_column].map(LABEL_TO_ID).tolist(),
        tokenizer,
        args.max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
    )
    return train_df, test_df, train_loader, test_loader


def compute_class_weights(train_df: pd.DataFrame, label_column: str, device: torch.device) -> torch.Tensor:
    counts = train_df[label_column].value_counts().reindex(LABEL_ORDER).astype(float)
    total = counts.sum()
    weights = total / (len(LABEL_ORDER) * counts)
    return torch.tensor(weights.to_list(), dtype=torch.float32, device=device)


def train_one_epoch(
    model,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    gradient_accumulation_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(train_loader, start=1):
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(**inputs).logits
            loss = criterion(logits, labels) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if step % gradient_accumulation_steps == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * gradient_accumulation_steps

        if step % 200 == 0:
            print(f"  step {step}/{len(train_loader)} - loss {total_loss / step:.4f}")

    return total_loss / max(1, len(train_loader))


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, device: torch.device) -> tuple[list[str], list[str]]:
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []

    for batch in data_loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        predictions = torch.argmax(logits, dim=-1)
        y_true.extend(ID_TO_LABEL[int(label)] for label in labels.cpu())
        y_pred.extend(ID_TO_LABEL[int(prediction)] for prediction in predictions.cpu())

    return y_true, y_pred


def save_confusion_matrix(y_true: list[str], y_pred: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    display.plot(ax=ax, cmap="Purples", values_format="d", colorbar=False)
    ax.set_title("Chinese BERT Sentiment Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading dataset...")
    df = load_dataset(args)
    label_counts = df[args.label_column].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    print(f"Rows used: {len(df)}")
    print(f"Label counts: {label_counts.to_dict()}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_df, test_df, train_loader, test_loader = make_loaders(args, tokenizer, df)
    class_weights = None if args.no_class_weights else compute_class_weights(train_df, args.label_column, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    started_at = time.time()
    best_macro_f1 = -1.0
    best_dir = output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            scaler,
            args.gradient_accumulation_steps,
        )
        y_true, y_pred = evaluate(model, test_loader, device)
        macro_f1 = f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss, "accuracy": accuracy, "macro_f1": macro_f1}
        history.append(epoch_metrics)
        print(f"  accuracy {accuracy:.4f} - macro_f1 {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

    y_true, y_pred = evaluate(model, test_loader, device)
    report = classification_report(y_true, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)

    metrics = {
        "input": args.input,
        "model_name": args.model_name,
        "rows_used": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "label_counts": label_counts.to_dict(),
        "test_size": args.test_size,
        "max_per_class": args.max_per_class,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "seconds": round(time.time() - started_at, 2),
        "history": history,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0),
        "classification_report": report,
    }

    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_model")
    joblib.dump(LABEL_ORDER, output_dir / "labels.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    print(report_text)
    print(f"Saved best model: {best_dir}")
    print(f"Saved final model: {output_dir / 'final_model'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print(f"Saved confusion matrix: {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
