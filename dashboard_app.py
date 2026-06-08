"""Streamlit dashboard for the Douban sentiment analysis project.

Run:
    streamlit run dashboard_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "DMSC.csv"
CLEAN_DATA_PATH = ROOT / "data" / "processed" / "douban_movie_reviews_clean.csv"
PREPROCESS_SUMMARY_PATH = ROOT / "data" / "processed" / "preprocess_summary.json"
MOVIE_SUMMARY_PATH = ROOT / "data" / "processed" / "movie_star_summary.csv"

TFIDF_METRICS_PATH = ROOT / "models" / "tfidf_logreg_full" / "metrics.json"
BERT_FULL_METRICS_PATH = ROOT / "models" / "bert_chinese_full" / "metrics.json"
BERT_200K_METRICS_PATH = ROOT / "models" / "bert_chinese_200k" / "metrics.json"
TFIDF_MODEL_PATH = ROOT / "models" / "tfidf_logreg_full" / "tfidf_logreg_model.joblib"
BERT_MODEL_DIR = ROOT / "models" / "bert_chinese_full" / "best_model"

LABEL_ORDER = ["negative", "positive"]


st.set_page_config(
    page_title="Douban Movie Review Sentiment Dashboard",
    layout="wide",
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, dict]:
    return {
        "TF-IDF Full": read_json(TFIDF_METRICS_PATH),
        "BERT 200k": read_json(BERT_200K_METRICS_PATH),
        "BERT Full": read_json(BERT_FULL_METRICS_PATH),
    }


@st.cache_data(show_spinner=False)
def load_preprocess_summary() -> dict:
    return read_json(PREPROCESS_SUMMARY_PATH)


@st.cache_data(show_spinner=False)
def load_movie_summary() -> pd.DataFrame:
    if MOVIE_SUMMARY_PATH.exists():
        return pd.read_csv(MOVIE_SUMMARY_PATH)
    return build_movie_summary()


@st.cache_data(show_spinner=True)
def build_movie_summary() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame()

    chunks = []
    usecols = ["Movie_Name_CN", "Movie_Name_EN", "Star", "Like"]
    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=usecols, chunksize=250_000):
        chunk["Star"] = pd.to_numeric(chunk["Star"], errors="coerce")
        chunk = chunk.dropna(subset=["Movie_Name_CN", "Star"])
        chunk["Star"] = chunk["Star"].astype(int)
        chunk["Like"] = pd.to_numeric(chunk["Like"], errors="coerce").fillna(0)
        chunk["positive_flag"] = chunk["Star"].ge(4).astype(int)
        chunk["negative_flag"] = chunk["Star"].le(2).astype(int)
        chunk["neutral_flag"] = chunk["Star"].eq(3).astype(int)

        star_counts = pd.crosstab(chunk["Movie_Name_CN"], chunk["Star"])
        for star in range(1, 6):
            if star not in star_counts.columns:
                star_counts[star] = 0
        star_counts = star_counts[[1, 2, 3, 4, 5]]
        star_counts.columns = [f"star_{star}" for star in range(1, 6)]

        grouped = chunk.groupby("Movie_Name_CN").agg(
            movie_name_en=("Movie_Name_EN", "first"),
            review_count=("Star", "size"),
            avg_star=("Star", "mean"),
            like_sum=("Like", "sum"),
            positive_count=("positive_flag", "sum"),
            negative_count=("negative_flag", "sum"),
            neutral_count=("neutral_flag", "sum"),
        )
        chunks.append(grouped.join(star_counts))

    if not chunks:
        return pd.DataFrame()

    combined = pd.concat(chunks).reset_index()
    star_columns = [f"star_{star}" for star in range(1, 6)]
    summary = combined.groupby("Movie_Name_CN").agg(
        movie_name_en=("movie_name_en", "first"),
        review_count=("review_count", "sum"),
        avg_star=("avg_star", lambda values: 0),
        like_sum=("like_sum", "sum"),
        positive_count=("positive_count", "sum"),
        negative_count=("negative_count", "sum"),
        neutral_count=("neutral_count", "sum"),
        **{column: (column, "sum") for column in star_columns},
    )
    total_star_points = sum(summary[f"star_{star}"] * star for star in range(1, 6))
    summary["avg_star"] = total_star_points / summary["review_count"]
    summary["positive_rate"] = summary["positive_count"] / summary["review_count"]
    summary["negative_rate"] = summary["negative_count"] / summary["review_count"]
    summary["neutral_rate"] = summary["neutral_count"] / summary["review_count"]

    global_avg = total_star_points.sum() / summary["review_count"].sum()
    confidence_reviews = 300
    summary["bayesian_star"] = (
        summary["review_count"] / (summary["review_count"] + confidence_reviews) * summary["avg_star"]
        + confidence_reviews / (summary["review_count"] + confidence_reviews) * global_avg
    )
    summary["recommendation"] = summary.apply(recommend_movie, axis=1)
    summary = summary.reset_index().rename(columns={"Movie_Name_CN": "movie_name_cn"})
    summary = summary.sort_values(["bayesian_star", "review_count"], ascending=[False, False])

    MOVIE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(MOVIE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return summary


def recommend_movie(row: pd.Series) -> str:
    if row["review_count"] < 100:
        return "資料不足"
    if row["bayesian_star"] >= 4.0 and row["positive_rate"] >= 0.7:
        return "推薦"
    if row["bayesian_star"] >= 3.5 and row["positive_rate"] >= 0.55:
        return "可考慮"
    return "不推薦"


def metric_rows(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, item in metrics.items():
        if not item:
            continue
        rows.append(
            {
                "model": name,
                "rows_used": item.get("rows_used"),
                "test_rows": item.get("test_rows"),
                "accuracy": item.get("accuracy"),
                "macro_f1": item.get("macro_f1"),
                "weighted_f1": item.get("weighted_f1"),
                "macro_precision": item.get("macro_precision"),
                "macro_recall": item.get("macro_recall"),
            }
        )
    return pd.DataFrame(rows)


def class_report_rows(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for model_name, item in metrics.items():
        report = item.get("classification_report", {})
        for label in LABEL_ORDER:
            values = report.get(label, {})
            if values:
                rows.append(
                    {
                        "model": model_name,
                        "label": label,
                        "precision": values.get("precision"),
                        "recall": values.get("recall"),
                        "f1": values.get("f1-score"),
                        "support": values.get("support"),
                    }
                )
    return pd.DataFrame(rows)


def show_overview(metrics: dict[str, dict]) -> None:
    summary = load_preprocess_summary()
    totals = summary.get("totals", {})
    sentiment_counts = summary.get("sentiment_counts", {})
    metric_df = metric_rows(metrics)

    st.subheader("專案總覽")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("原始評論數", f"{totals.get('rows_read', 0):,}")
    c2.metric("清洗後二分類評論", f"{totals.get('rows_written', 0):,}")
    c3.metric("Positive", f"{sentiment_counts.get('positive', 0):,}")
    c4.metric("Negative", f"{sentiment_counts.get('negative', 0):,}")

    st.markdown("#### 模型整體比較")
    st.dataframe(
        metric_df.style.format(
            {
                "accuracy": "{:.4f}",
                "macro_f1": "{:.4f}",
                "weighted_f1": "{:.4f}",
                "macro_precision": "{:.4f}",
                "macro_recall": "{:.4f}",
            }
        ),
        use_container_width=True,
    )

    chart_df = metric_df.melt(
        id_vars="model",
        value_vars=["accuracy", "macro_f1", "weighted_f1"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        chart_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        text_auto=".3f",
        range_y=[0, 1],
        title="模型 Accuracy / Macro F1 / Weighted F1 比較",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_analysis(metrics: dict[str, dict]) -> None:
    st.subheader("模型細部分析")
    report_df = class_report_rows(metrics)

    selected_metric = st.radio("選擇要比較的分類指標", ["precision", "recall", "f1"], horizontal=True)
    fig = px.bar(
        report_df,
        x="model",
        y=selected_metric,
        color="label",
        barmode="group",
        text_auto=".3f",
        range_y=[0, 1],
        title=f"各模型 {selected_metric} by label",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### TF-IDF Full confusion matrix")
        image_path = ROOT / "models" / "tfidf_logreg_full" / "confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
    with c2:
        st.markdown("#### BERT Full confusion matrix")
        image_path = ROOT / "models" / "bert_chinese_full" / "confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)

    st.markdown("#### Classification report")
    st.dataframe(
        report_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "support": "{:.0f}"}),
        use_container_width=True,
    )


def show_movie_recommendations() -> None:
    st.subheader("不同電影星等統計與推薦")
    movie_df = load_movie_summary()
    if movie_df.empty:
        st.warning("找不到電影星等統計資料，請確認 DMSC.csv 是否存在。")
        return

    min_reviews = st.slider("最少評論數", 50, 20_000, 1_000, step=50)
    recommendation_filter = st.multiselect(
        "推薦分類",
        ["推薦", "可考慮", "不推薦", "資料不足"],
        default=["推薦", "可考慮", "不推薦"],
    )

    filtered = movie_df[
        movie_df["review_count"].ge(min_reviews) & movie_df["recommendation"].isin(recommendation_filter)
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("電影數", f"{len(filtered):,}")
    c2.metric("最高 Bayesian 星等", f"{filtered['bayesian_star'].max():.2f}" if len(filtered) else "-")
    c3.metric("平均推薦率", f"{filtered['positive_rate'].mean():.1%}" if len(filtered) else "-")

    st.markdown("#### 推薦排行榜")
    display_columns = [
        "movie_name_cn",
        "movie_name_en",
        "recommendation",
        "review_count",
        "avg_star",
        "bayesian_star",
        "positive_rate",
        "negative_rate",
        "star_1",
        "star_2",
        "star_3",
        "star_4",
        "star_5",
    ]
    st.dataframe(
        filtered[display_columns]
        .sort_values(["recommendation", "bayesian_star"], ascending=[True, False])
        .style.format(
            {
                "avg_star": "{:.2f}",
                "bayesian_star": "{:.2f}",
                "positive_rate": "{:.1%}",
                "negative_rate": "{:.1%}",
            }
        ),
        use_container_width=True,
        height=420,
    )

    top_n = filtered.sort_values("bayesian_star", ascending=False).head(20)
    fig = px.bar(
        top_n.sort_values("bayesian_star"),
        x="bayesian_star",
        y="movie_name_cn",
        color="positive_rate",
        orientation="h",
        text="recommendation",
        title="Top 20 推薦電影 Bayesian 星等",
        labels={"movie_name_cn": "電影", "bayesian_star": "Bayesian 星等", "positive_rate": "推薦率"},
    )
    st.plotly_chart(fig, use_container_width=True)

    selected_movie = st.selectbox("選擇電影查看星等分布", filtered["movie_name_cn"].tolist())
    row = filtered[filtered["movie_name_cn"] == selected_movie].iloc[0]
    star_dist = pd.DataFrame(
        {
            "star": ["1 星", "2 星", "3 星", "4 星", "5 星"],
            "count": [row[f"star_{star}"] for star in range(1, 6)],
        }
    )
    c1, c2 = st.columns([1, 1])
    with c1:
        fig = px.pie(star_dist, names="star", values="count", title=f"{selected_movie} 星等比例")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(star_dist, x="star", y="count", text_auto=True, title=f"{selected_movie} 星等分布")
        st.plotly_chart(fig, use_container_width=True)


@st.cache_resource(show_spinner=False)
def load_tfidf_model():
    return joblib.load(TFIDF_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_bert_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR).to(device)
    model.eval()
    return tokenizer, model, device


def predict_tfidf(text: str) -> tuple[str, float]:
    model = load_tfidf_model()
    label = str(model.predict([text])[0])
    probability = 0.0
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        probability = float(model.predict_proba([text])[0][classes.index(label)])
    return label, probability


@torch.no_grad()
def predict_bert(text: str) -> tuple[str, float]:
    tokenizer, model, device = load_bert_model()
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()
    prediction_id = int(torch.argmax(probabilities).item())
    label = model.config.id2label.get(prediction_id, LABEL_ORDER[prediction_id])
    return label, float(probabilities[prediction_id].item())


def show_live_prediction() -> None:
    st.subheader("單句評論情緒測試")
    model_type = st.radio("模型", ["BERT Full", "TF-IDF Full"], horizontal=True)
    text = st.text_area("輸入中文電影評論", value="這部電影很好看，劇情感人，演員表現也很棒", height=120)

    if st.button("開始預測", type="primary"):
        if not text.strip():
            st.warning("請先輸入評論文字。")
            return
        if model_type == "BERT Full":
            label, confidence = predict_bert(text.strip())
        else:
            label, confidence = predict_tfidf(text.strip())

        zh_label = "正面" if label == "positive" else "負面"
        st.metric("預測結果", f"{label} ({zh_label})", f"信心分數 {confidence:.4f}")


def main() -> None:
    st.title("豆瓣電影評論情緒分析儀表板")
    st.caption("中文評論情緒分類：TF-IDF baseline vs Chinese BERT，並依電影星等統計判斷推薦程度。")

    metrics = load_metrics()
    tabs = st.tabs(["總覽", "模型圖表", "電影推薦統計", "單句測試"])
    with tabs[0]:
        show_overview(metrics)
    with tabs[1]:
        show_model_analysis(metrics)
    with tabs[2]:
        show_movie_recommendations()
    with tabs[3]:
        show_live_prediction()


if __name__ == "__main__":
    main()
