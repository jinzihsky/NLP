"""Preprocess Douban movie review data for Chinese sentiment analysis.

Example:
    conda run -n project python preprocess_douban.py --input DMSC.csv
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.util import hash_pandas_object


HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


OUTPUT_COLUMNS = [
    "original_id",
    "movie_name_cn",
    "movie_name_en",
    "review_date",
    "star",
    "sentiment",
    "like_count",
    "comment",
    "comment_clean",
    "comment_length",
]


def clean_comment(value: object) -> str:
    """Normalize and clean one review comment."""
    if pd.isna(value):
        return ""

    text = str(value)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def build_sentiment(star: int, mode: str) -> str | None:
    if mode == "none":
        return None
    if star <= 2:
        return "negative"
    if star >= 4:
        return "positive"
    if mode == "three-class":
        return "neutral"
    return None


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def process_chunk(
    chunk: pd.DataFrame,
    label_mode: str,
    min_length: int,
    seen_hashes: set[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats = {
        "rows_read": len(chunk),
        "invalid_star_rows": 0,
        "empty_comment_rows": 0,
        "short_comment_rows": 0,
        "neutral_dropped_rows": 0,
        "duplicate_rows": 0,
    }

    ensure_columns(chunk, ["ID", "Movie_Name_EN", "Movie_Name_CN", "Date", "Star", "Comment", "Like"])

    df = chunk.copy()
    df["star"] = pd.to_numeric(df["Star"], errors="coerce")
    valid_star = df["star"].between(1, 5)
    stats["invalid_star_rows"] = int((~valid_star).sum())
    df = df.loc[valid_star].copy()
    df["star"] = df["star"].astype("int8")

    df["comment_clean"] = df["Comment"].map(clean_comment)
    non_empty = df["comment_clean"].ne("")
    stats["empty_comment_rows"] = int((~non_empty).sum())
    df = df.loc[non_empty].copy()

    df["comment_length"] = df["comment_clean"].str.len()
    long_enough = df["comment_length"].ge(min_length)
    stats["short_comment_rows"] = int((~long_enough).sum())
    df = df.loc[long_enough].copy()

    if label_mode != "none":
        df["sentiment"] = df["star"].map(lambda star: build_sentiment(int(star), label_mode))
        keep_label = df["sentiment"].notna()
        stats["neutral_dropped_rows"] = int((~keep_label).sum())
        df = df.loc[keep_label].copy()
    else:
        df["sentiment"] = ""

    df = df.rename(
        columns={
            "ID": "original_id",
            "Movie_Name_EN": "movie_name_en",
            "Movie_Name_CN": "movie_name_cn",
            "Date": "review_date",
            "Like": "like_count",
            "Comment": "comment",
        }
    )

    keys = df[["movie_name_cn", "comment_clean"]].fillna("")
    hashes = hash_pandas_object(keys, index=False).astype("uint64")
    duplicate_mask = hashes.map(lambda value: int(value) in seen_hashes)
    stats["duplicate_rows"] = int(duplicate_mask.sum())

    new_hashes = hashes.loc[~duplicate_mask]
    seen_hashes.update(int(value) for value in new_hashes)
    df = df.loc[~duplicate_mask, OUTPUT_COLUMNS].copy()

    return df, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Douban movie review CSV data.")
    parser.add_argument("--input", default="DMSC.csv", help="Path to raw Douban CSV file.")
    parser.add_argument(
        "--output",
        default="data/processed/douban_movie_reviews_clean.csv",
        help="Path to cleaned output CSV.",
    )
    parser.add_argument(
        "--summary",
        default="data/processed/preprocess_summary.json",
        help="Path to preprocessing summary JSON.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows to process per chunk.")
    parser.add_argument("--min-length", type=int, default=2, help="Minimum cleaned comment length.")
    parser.add_argument(
        "--label-mode",
        choices=["binary", "three-class", "none"],
        default="binary",
        help="binary drops 3-star reviews; three-class keeps them as neutral.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {
        "rows_read": 0,
        "rows_written": 0,
        "invalid_star_rows": 0,
        "empty_comment_rows": 0,
        "short_comment_rows": 0,
        "neutral_dropped_rows": 0,
        "duplicate_rows": 0,
    }
    sentiment_counts: dict[str, int] = {}
    star_counts: dict[str, int] = {}
    seen_hashes: set[int] = set()

    first_chunk = True
    reader = pd.read_csv(input_path, chunksize=args.chunksize)
    for index, chunk in enumerate(reader, start=1):
        cleaned, stats = process_chunk(chunk, args.label_mode, args.min_length, seen_hashes)

        for key, value in stats.items():
            totals[key] += value
        totals["rows_written"] += len(cleaned)

        for label, count in cleaned["sentiment"].value_counts(dropna=False).items():
            sentiment_counts[str(label)] = sentiment_counts.get(str(label), 0) + int(count)
        for star, count in cleaned["star"].value_counts(dropna=False).items():
            star_counts[str(star)] = star_counts.get(str(star), 0) + int(count)

        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",
        )
        first_chunk = False
        print(f"Processed chunk {index}: wrote {len(cleaned)} rows")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "label_mode": args.label_mode,
        "min_length": args.min_length,
        "chunksize": args.chunksize,
        "totals": totals,
        "sentiment_counts": dict(sorted(sentiment_counts.items())),
        "star_counts": dict(sorted(star_counts.items(), key=lambda item: int(item[0]))),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
