"""
Visualization Module for Stylometry and Authorship Attribution.

Generates 300 DPI publication-quality figures:
1. `plot_confusion_matrix`: Heatmap of author prediction confusion matrix.
2. `plot_chunk_size_comparison`: Line chart comparing Burrows' Delta vs Logistic Regression across chunk sizes (500, 1000, 2000, 5000 words).
3. `plot_pca_clusters`: 2D PCA projection scatter plot colored by author.
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

# Set temporary matplotlib config directory to prevent permission delays
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: Sequence[str],
    model_name: str = "Burrows' Delta",
    output_path: Union[str, Path] = "figures/confusion_matrix.png",
    normalize: bool = True,
    cmap: str = "Blues",
) -> Path:
    """
    Plot and save a stylized confusion matrix heatmap at 300 DPI.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        display_data = cm_norm
        fmt = ".2%"
        title = f"Normalized Confusion Matrix — {model_name}"
    else:
        display_data = cm
        fmt = "d"
        title = f"Confusion Matrix (Counts) — {model_name}"

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.heatmap(
        display_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Author", fontsize=11, labelpad=8)
    ax.set_ylabel("True Author", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_chunk_size_comparison(
    chunk_sizes: Sequence[int],
    delta_scores: Sequence[float],
    lr_scores: Sequence[float],
    output_path: Union[str, Path] = "figures/chunk_accuracy.png",
) -> Path:
    """
    Plot cross-validation accuracy versus chunk sizes (500, 1000, 2000, 5000 words).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # Convert to percentages for readability
    delta_pct = [s * 100.0 for s in delta_scores]
    lr_pct = [s * 100.0 for s in lr_scores]

    ax.plot(
        chunk_sizes,
        delta_pct,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#2b5c8f",
        label="Burrows' Delta (Custom)",
    )
    ax.plot(
        chunk_sizes,
        lr_pct,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="#d95f02",
        linestyle="--",
        label="Logistic Regression (Baseline)",
    )

    # Annotate points
    for x, y_d, y_l in zip(chunk_sizes, delta_pct, lr_pct):
        ax.annotate(f"{y_d:.1f}%", (x, y_d), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color="#2b5c8f", fontweight="bold")
        ax.annotate(f"{y_l:.1f}%", (x, y_l), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#d95f02", fontweight="bold")

    ax.set_title("Authorship Attribution Accuracy vs. Chunk Size", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Chunk Size (Tokens / Words)", fontsize=11, labelpad=8)
    ax.set_ylabel("GroupKFold CV Accuracy (%)", fontsize=11, labelpad=8)
    ax.set_xticks(list(chunk_sizes))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="lower right")
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_pca_clusters(
    df: pd.DataFrame,
    output_path: Union[str, Path] = "figures/pca_clusters.png",
    title: str = "2D PCA Projection of Stylometric Feature Space",
) -> Path:
    """
    Standardizes feature space, reduces to 2 dimensions via PCA, and plots clusters by author.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if "author" not in df.columns:
        raise ValueError("DataFrame must contain an 'author' column.")

    feature_cols = [c for c in df.columns if c not in ["author", "book"]]
    X = df[feature_cols].values
    y = df["author"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_ * 100.0

    pca_df = pd.DataFrame(
        {
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Author": y,
        }
    )

    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)

    unique_authors = sorted(list(set(y)))
    palette = sns.color_palette("tab10", n_colors=len(unique_authors))

    for idx, author in enumerate(unique_authors):
        subset = pca_df[pca_df["Author"] == author]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            label=author,
            color=palette[idx],
            alpha=0.75,
            edgecolors="none",
            s=45,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1f}% variance)", fontsize=11, labelpad=8)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1f}% variance)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(title="Author", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    plt.tight_layout()

    plt.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file
