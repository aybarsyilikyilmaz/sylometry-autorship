"""
Interactive Authorship Predictor.

Takes any raw text snippet, extracts its 156-dimensional stylometric features,
and predicts which of the 7 authors wrote it using both Burrows' Delta and Logistic Regression.
"""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.corpus import TextCleaner, TextChunk
from src.delta import BurrowsDelta
from src.features import StylometricExtractor


def load_or_train_models(
    models_dir: Path = Path("models"),
    features_csv: Path = Path("data/processed/features_1000.csv"),
):
    """Load pre-trained models from disk or train on the fly if missing."""
    delta_path = models_dir / "burrows_delta.joblib"
    lr_path = models_dir / "logistic_regression.joblib"
    scaler_path = models_dir / "scaler.joblib"
    meta_path = models_dir / "metadata.json"

    if delta_path.exists() and lr_path.exists() and scaler_path.exists() and meta_path.exists():
        delta_clf = joblib.load(delta_path)
        lr_clf = joblib.load(lr_path)
        scaler = joblib.load(scaler_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = meta["feature_names"]
        vocab = meta["vocabulary"]
        return delta_clf, lr_clf, scaler, feature_cols, vocab

    # Fallback to on-the-fly training
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in ["author", "book"]]
    X = df[feature_cols].values
    y = df["author"].values

    delta_clf = BurrowsDelta()
    delta_clf.fit(X, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr_clf = LogisticRegression(max_iter=1500, random_state=42)
    lr_clf.fit(X_scaled, y)

    vocab = [c[3:] for c in feature_cols if c.startswith("fw_")]
    return delta_clf, lr_clf, scaler, feature_cols, vocab


def predict_snippet(text: str, features_csv: Path = Path("data/processed/features_1000.csv")):
    """Predict author for a raw text snippet."""
    if not features_csv.exists():
        features_csv = Path("data/processed/features_500.csv")

    cleaner = TextCleaner()
    cleaned = cleaner.clean(text)
    word_count = len(cleaned.split())

    if word_count < 30:
        print(f"[Uyarı] Metin çok kısa ({word_count} kelime). Güvenilir stilometri için en az 100-500 kelime önerilir.")

    delta_clf, lr_clf, scaler, feature_cols, vocab = load_or_train_models(
        models_dir=Path("models"), features_csv=features_csv
    )

    extractor = StylometricExtractor(top_n_words=150)
    extractor.vocabulary = vocab

    dummy_chunk = TextChunk(
        text=cleaned,
        author="Unknown",
        book="Snippet",
        chunk_id=0,
        token_count=word_count,
    )
    feat_dict = extractor.extract_chunk_features(dummy_chunk)
    feat_vector = np.array([[feat_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float64)

    # Predictions
    delta_pred = delta_clf.predict(feat_vector)[0]
    dist_df = delta_clf.predict_distance_matrix(feat_vector)
    delta_probs = delta_clf.predict_proba(feat_vector)[0]

    scaled_vector = scaler.transform(feat_vector)
    lr_pred = lr_clf.predict(scaled_vector)[0]
    lr_probs = lr_clf.predict_proba(scaled_vector)[0]

    print("\n" + "=" * 75)
    print(f"METİN YAZARLIK TAHMİNİ ({word_count} Kelime)")
    print("=" * 75)
    print(f"🎯 Burrows' Delta Tahmini:      👉 {delta_pred}")
    print(f"🎯 Lojistik Regresyon Tahmini:  👉 {lr_pred}")
    print("-" * 75)
    print(f"{'Yazar':<24} | {'Burrows Delta Farkı':<20} | {'LR Güven Skoru':<16}")
    print("-" * 75)

    classes = delta_clf.classes_
    # Sort by Delta distance (smallest distance = best match)
    distances = dist_df.iloc[0].to_dict()
    sorted_authors = sorted(distances.keys(), key=lambda a: distances[a])

    for author in sorted_authors:
        idx = list(classes).index(author)
        dist = distances[author]
        prob = lr_probs[idx] * 100.0
        bar = "█" * int(prob / 5)
        print(f"{author:<24} | {dist:>8.4f} (Manhattan)  | %{prob:>5.1f} {bar}")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict author of any text")
    parser.add_argument("text", nargs="?", type=str, help="Raw text string to analyze")
    parser.add_argument("--file", type=str, help="Path to text file to analyze")

    args = parser.parse_args()

    sample_text = ""
    if args.file:
        sample_text = Path(args.file).read_text(encoding="utf-8", errors="replace")
    elif args.text:
        sample_text = args.text
    else:
        # Default interesting test quote
        sample_text = (
            "It is a truth universally acknowledged, that a single man in possession of a good fortune, "
            "must be in want of a wife. However little known the feelings or views of such a man may be "
            "on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding "
            "families, that he is considered as the rightful property of some one or other of their daughters."
        )

    predict_snippet(sample_text)
