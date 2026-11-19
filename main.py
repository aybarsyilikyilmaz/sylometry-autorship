"""
Main Execution Script for Stylometry and Authorship Attribution.

Orchestrates multi-chunk experiments across 500, 1000, 2000, and 5000 words:
1. Loads raw books or existing cached feature sets from `data/processed/`.
2. Extracts 158 features (156 stylometric + 2 metadata) and downsamples classes to N_min.
3. Evaluates Burrows' Delta vs. Logistic Regression via book-isolated GroupKFold CV.
4. Generates terminal reports, lists distinctive author words, and saves 300 DPI figures to `figures/`.
"""

import argparse
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Ensure headless matplotlib configuration
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd

from src.corpus import CorpusLoader
from src.downloader import BOOK_CATALOG, download_all_books
from src.evaluator import ExperimentComparison, ExperimentRunner
from src.features import StylometricExtractor
from src.viz import (
    plot_chunk_size_comparison,
    plot_confusion_matrix,
    plot_pca_clusters,
)


def ensure_corpus(raw_dir: Path, auto_download: bool = True) -> bool:
    """
    Check if raw corpus exists and contains texts. Optionally triggers automated download.
    """
    loader = CorpusLoader(raw_dir=raw_dir)
    discovered = loader.discover_books()
    total_books = sum(len(b) for b in discovered.values())

    if total_books >= 14:  # At least 2 books per author for 7 authors
        print(f"[Corpus] Found existing corpus with {len(discovered)} authors and {total_books} books.")
        return True

    print(f"[Corpus] Current raw directory '{raw_dir}' has {total_books} books.")
    if auto_download:
        print("[Corpus] Initiating automated download of 56 books from Project Gutenberg...")
        summary = download_all_books(raw_dir=raw_dir, max_workers=4)
        print(f"[Corpus] Download complete: {summary['success_count']}/{summary['total_catalog']} books successfully saved.")
        return summary["success_count"] > 0

    return False


def get_or_create_features(
    chunk_size: int,
    raw_dir: Path,
    processed_dir: Path,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Retrieve processed features CSV if present; otherwise extract, balance, and save.
    """
    csv_file = processed_dir / f"features_{chunk_size}.csv"

    if csv_file.exists():
        print(f"[Data] Loading cached features for chunk_size={chunk_size} from {csv_file}")
        return pd.read_csv(csv_file)

    print(f"[Data] Extracting features for chunk_size={chunk_size} from raw texts...")
    loader = CorpusLoader(raw_dir=raw_dir)
    chunks = loader.load_chunks(chunk_size=chunk_size)

    if not chunks:
        raise RuntimeError(
            f"No text chunks generated for chunk_size={chunk_size}. "
            f"Check that books are present in '{raw_dir}'."
        )

    extractor = StylometricExtractor(top_n_words=150)
    raw_df = extractor.extract_features(chunks, fit_vocab=True)
    balanced_df = extractor.balance_classes(raw_df, random_state=random_state)

    extractor.save_features(balanced_df, chunk_size=chunk_size, output_dir=processed_dir)
    print(
        f"[Data] Saved balanced features: {len(balanced_df)} chunks ({balanced_df.shape[1]} columns) -> {csv_file}"
    )
    return balanced_df


def print_comparison_table(results: Dict[int, ExperimentComparison]) -> None:
    """Print a formatted ASCII summary table of experimental results across chunk sizes."""
    print("\n" + "=" * 90)
    print(f"{'CHUNK SIZE':<12} | {'CHUNKS':<8} | {'BURROWS DELTA ACC (CV)':<26} | {'LOGISTIC REGRESSION ACC (CV)':<28}")
    print("=" * 90)

    for chunk_size, comp in sorted(results.items()):
        d_acc = f"{comp.delta_result.mean_accuracy * 100:.2f}% ± {comp.delta_result.std_accuracy * 100:.2f}%"
        lr_acc = f"{comp.lr_result.mean_accuracy * 100:.2f}% ± {comp.lr_result.std_accuracy * 100:.2f}%"
        print(f"{chunk_size:<6} words | {comp.n_samples:<8} | {d_acc:<26} | {lr_acc:<28}")

    print("=" * 90 + "\n")


def print_author_distinctive_features(
    distinctive_features: Dict[str, Dict[str, List[tuple]]],
    top_k: int = 5,
) -> None:
    """Print the most distinctive positive and negative stylistic markers per author."""
    print("=" * 90)
    print(f"TOP {top_k} DISTINCTIVE STYLOMETRIC MARKERS PER AUTHOR (Logistic Regression Weights)")
    print("=" * 90)

    for author, marks in sorted(distinctive_features.items()):
        print(f"\n✍️  Author: {author}")
        pos_str = ", ".join(f"{name} ({w:+.2f})" for name, w in marks["top_positive"][:top_k])
        neg_str = ", ".join(f"{name} ({w:+.2f})" for name, w in marks["top_negative"][:top_k])
        print(f"   ➕ Strongest Positive Signals: {pos_str}")
        print(f"   ➖ Strongest Negative Signals: {neg_str}")

    print("\n" + "=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stylometry Authorship Attribution Pipeline")
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2000, 5000],
        help="List of chunk sizes to evaluate (default: 500 1000 2000 5000)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for GroupKFold cross-validation (default: 5)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw corpus directory (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to processed features directory (default: data/processed)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures",
        help="Path to figures directory (default: figures)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable automatic downloading if raw books are missing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    processed_path = Path(args.processed_dir)
    figures_path = Path(args.figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    # 1. Corpus verification
    ensure_corpus(raw_path, auto_download=not args.no_download)

    # 2. Run multi-chunk experiments
    experiment_results: Dict[int, ExperimentComparison] = {}
    delta_scores: List[float] = []
    lr_scores: List[float] = []
    valid_chunk_sizes: List[int] = []

    for chunk_size in sorted(args.chunk_sizes):
        print(f"\n>>> Running Experiment for Chunk Size = {chunk_size} words <<<")
        try:
            df = get_or_create_features(
                chunk_size=chunk_size,
                raw_dir=raw_path,
                processed_dir=processed_path,
                random_state=args.random_state,
            )
        except Exception as e:
            print(f"[Warning] Skipping chunk_size={chunk_size}: {e}")
            continue

        runner = ExperimentRunner(n_splits=args.n_splits, random_state=args.random_state)
        comp = runner.run_experiment(df)
        experiment_results[chunk_size] = comp
        valid_chunk_sizes.append(chunk_size)
        delta_scores.append(comp.delta_result.mean_accuracy)
        lr_scores.append(comp.lr_result.mean_accuracy)

        print(
            f"   Burrows' Delta Accuracy:      {comp.delta_result.mean_accuracy * 100:.2f}% "
            f"(std: {comp.delta_result.std_accuracy * 100:.2f}%)"
        )
        print(
            f"   Logistic Regression Accuracy: {comp.lr_result.mean_accuracy * 100:.2f}% "
            f"(std: {comp.lr_result.std_accuracy * 100:.2f}%)"
        )

    if not experiment_results:
        print("[Error] No valid experiments could be executed. Exiting.")
        sys.exit(1)

    # 3. Print overall comparison summary
    print_comparison_table(experiment_results)

    # 4. Generate visualizations
    print("[Viz] Generating publication-ready figures in 'figures/'...")

    # A. Chunk Size Comparison Curve
    if len(valid_chunk_sizes) > 1:
        plot_chunk_size_comparison(
            chunk_sizes=valid_chunk_sizes,
            delta_scores=delta_scores,
            lr_scores=lr_scores,
            output_path=figures_path / "chunk_accuracy.png",
        )
        print(f"   Saved: {figures_path / 'chunk_accuracy.png'}")

    # B. Confusion Matrix for highest chunk size (e.g. 5000 or max available)
    max_chunk = max(valid_chunk_sizes)
    best_comp = experiment_results[max_chunk]

    plot_confusion_matrix(
        cm=best_comp.delta_result.confusion_matrix,
        classes=best_comp.delta_result.classes,
        model_name=f"Burrows' Delta ({max_chunk} words)",
        output_path=figures_path / "confusion_matrix.png",
    )
    print(f"   Saved: {figures_path / 'confusion_matrix.png'}")

    plot_confusion_matrix(
        cm=best_comp.lr_result.confusion_matrix,
        classes=best_comp.lr_result.classes,
        model_name=f"Logistic Regression ({max_chunk} words)",
        output_path=figures_path / "confusion_matrix_lr.png",
        cmap="Oranges",
    )
    print(f"   Saved: {figures_path / 'confusion_matrix_lr.png'}")

    # C. 2D PCA Cluster Scatter Plot for highest chunk size
    max_chunk_df = get_or_create_features(
        chunk_size=max_chunk,
        raw_dir=raw_path,
        processed_dir=processed_path,
        random_state=args.random_state,
    )
    plot_pca_clusters(
        df=max_chunk_df,
        output_path=figures_path / "pca_clusters.png",
        title=f"2D PCA Stylometric Clusters ({max_chunk}-word chunks)",
    )
    print(f"   Saved: {figures_path / 'pca_clusters.png'}")

    # 5. Distinctive stylistic words
    print_author_distinctive_features(best_comp.distinctive_features, top_k=5)

    print("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
