"""
Model Training and Persistence Module.

Trains final production stylometry models (Burrows' Delta & Logistic Regression)
on the complete 56-book balanced corpus and saves serialized model artifacts to `models/`.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

from src.corpus import CorpusLoader
from src.delta import BurrowsDelta
from src.features import StylometricExtractor


def train_and_save(
    chunk_size: int = 1000,
    processed_dir: Path = Path("data/processed"),
    models_dir: Path = Path("models"),
    raw_dir: Path = Path("data/raw"),
    random_state: int = 42,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    features_csv = processed_dir / f"features_{chunk_size}.csv"

    print("=" * 80)
    print("STİLOMETRİK YAZARLIK TESPİTİ MODEL EĞİTİMİ (PRODUCTION TRAINING)")
    print("=" * 80)

    # 1. Feature data loading / extraction
    if features_csv.exists():
        print(f"[1/5] 📂 Özellik verisi okunuyor: {features_csv}")
        df = pd.read_csv(features_csv)
    else:
        print(f"[1/5] ⚙️  Ham metinlerden {chunk_size} kelimelik parçalar ve özellikler çıkarılıyor...")
        loader = CorpusLoader(raw_dir=raw_dir)
        chunks = loader.load_chunks(chunk_size=chunk_size)
        extractor = StylometricExtractor(top_n_words=150)
        raw_df = extractor.extract_features(chunks, fit_vocab=True)
        df = extractor.balance_classes(raw_df, random_state=random_state)
        extractor.save_features(df, chunk_size=chunk_size, output_dir=processed_dir)

    feature_cols = [c for c in df.columns if c not in ["author", "book"]]
    vocab = [c[3:] for c in feature_cols if c.startswith("fw_")]
    X = df[feature_cols].values
    y = df["author"].values
    books = df["book"].values
    unique_authors = sorted(list(set(y)))

    print(f"      Toplam Parça Sayısı:  {len(df)}")
    print(f"      Özellik Vektörü Boyutu: {len(feature_cols)} özellik (150 kelime, 2 cümle, 1 TTR, 5 noktalama)")
    print(f"      Yazar Sayısı:          {len(unique_authors)} ({', '.join(unique_authors)})")

    # 2. Train Burrows' Delta Model
    print("\n[2/5] 🧠 1. Model Eğitiliyor: Sıfırdan Yazılmış Burrows' Delta...")
    delta_clf = BurrowsDelta()
    delta_clf.fit(X, y)
    delta_train_preds = delta_clf.predict(X)
    delta_train_acc = accuracy_score(y, delta_train_preds)
    print(f"      ✓ Burrows' Delta Eğitim Başarısı (Train Accuracy): %{delta_train_acc * 100:.2f}")

    # 3. Train Logistic Regression Pipeline (StandardScaler + LogisticRegression)
    print("\n[3/5] 🧠 2. Model Eğitiliyor: StandardScaler + Lojistik Regresyon...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr_clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=random_state,
    )
    lr_clf.fit(X_scaled, y)
    lr_train_preds = lr_clf.predict(X_scaled)
    lr_train_acc = accuracy_score(y, lr_train_preds)
    print(f"      ✓ Lojistik Regresyon Eğitim Başarısı (Train Accuracy): %{lr_train_acc * 100:.2f}")

    # 4. Print Detailed Classification Report
    print("\n[4/5] 📊 Sınıflandırma Metrikleri (Lojistik Regresyon):")
    print("-" * 80)
    print(classification_report(y, lr_train_preds, digits=4))
    print("-" * 80)

    # 5. Serialize and Save Artifacts
    print("\n[5/5] 💾 Eğitilmiş Modeller Kaydediliyor (models/)...")
    delta_path = models_dir / "burrows_delta.joblib"
    lr_path = models_dir / "logistic_regression.joblib"
    scaler_path = models_dir / "scaler.joblib"
    meta_path = models_dir / "metadata.json"

    joblib.dump(delta_clf, delta_path)
    joblib.dump(lr_clf, lr_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "chunk_size": chunk_size,
        "n_samples": len(df),
        "n_features": len(feature_cols),
        "classes": unique_authors,
        "feature_names": feature_cols,
        "vocabulary": vocab,
        "train_accuracies": {
            "burrows_delta": float(delta_train_acc),
            "logistic_regression": float(lr_train_acc),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"      ✓ {delta_path} ({delta_path.stat().st_size / 1024:.1f} KB)")
    print(f"      ✓ {lr_path} ({lr_path.stat().st_size / 1024:.1f} KB)")
    print(f"      ✓ {scaler_path} ({scaler_path.stat().st_size / 1024:.1f} KB)")
    print(f"      ✓ {meta_path}")

    print("\n" + "=" * 80)
    print("🎯 MODEL EĞİTİMİ BAŞARIYLA TAMAMLANDI!")
    print(f"   Modeller hazır. Artık `predict.py` anında bu modelleri kullanarak tahmin yapacak.")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train production stylometry models")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size in words (default: 1000)")
    parser.add_argument("--models-dir", type=str, default="models", help="Output directory for trained models")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    train_and_save(
        chunk_size=args.chunk_size,
        models_dir=Path(args.models_dir),
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
