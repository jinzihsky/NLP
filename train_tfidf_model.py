"""Train a TF-IDF sentiment classifier for cleaned Douban movie reviews.

Example:
    conda run -n project python train_tfidf_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline


LABEL_ORDER = ["negative", "positive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TF-IDF sentiment classifier.")
    parser.add_argument(
        "--input",
        default="data/processed/douban_movie_reviews_clean.csv",
        help="Cleaned CSV from preprocess_douban.py.",
    )
    parser.add_argument("--text-column", default="comment_clean", help="Text feature column.")
    parser.add_argument("--label-column", default="sentiment", help="Target label column.")
    parser.add_argument(
        "--output-dir",
        default="models/tfidf_logreg",
        help="Directory for model and evaluation outputs.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=200_000,
        help="Maximum rows sampled per class. Use 0 to train on all rows.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=300_000,
        help="Maximum TF-IDF features.",
    )
    return parser.parse_args()


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
        sampled_parts = []
        for label in LABEL_ORDER:
            part = df[df[args.label_column] == label]
            sample_size = min(args.max_per_class, len(part))
            sampled_parts.append(part.sample(sample_size, random_state=args.random_state))
        df = pd.concat(sampled_parts).sample(frac=1, random_state=args.random_state).reset_index(drop=True)

    return df


def build_pipeline(max_features: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                    min_df=3,
                    max_df=0.95,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="saga",
                    random_state=42,
                    verbose=0,
                ),
            ),
        ]
    )


def save_confusion_matrix(y_true: pd.Series, y_pred: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABEL_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Douban Review Sentiment Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = load_dataset(args)
    label_counts = df[args.label_column].value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)
    print(f"Rows used: {len(df)}")
    print(f"Label counts: {label_counts.to_dict()}")

    x_train, x_test, y_train, y_test = train_test_split(
        df[args.text_column],
        df[args.label_column],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.label_column],
    )

    print("Training TF-IDF + Logistic Regression model...")
    model = build_pipeline(args.max_features)
    model.fit(x_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, labels=LABEL_ORDER, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_pred, labels=LABEL_ORDER, zero_division=0)

    metrics = {
        "input": args.input,
        "model": "TF-IDF char 2-4 grams + LogisticRegression",
        "rows_used": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "label_counts": label_counts.to_dict(),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_per_class": args.max_per_class,
        "max_features": args.max_features,
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_precision": precision_score(y_test, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, y_pred, labels=LABEL_ORDER, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, labels=LABEL_ORDER, average="weighted", zero_division=0),
        "classification_report": report,
    }

    model_path = output_dir / "tfidf_logreg_model.joblib"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "classification_report.txt"
    matrix_path = output_dir / "confusion_matrix.png"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")
    save_confusion_matrix(y_test, y_pred, matrix_path)

    print(report_text)
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")


if __name__ == "__main__":
    main()
