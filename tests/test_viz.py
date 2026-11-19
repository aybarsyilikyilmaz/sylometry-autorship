"""
Unit tests for src/viz.py (Visualization module)
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.viz import (
    plot_chunk_size_comparison,
    plot_confusion_matrix,
    plot_pca_clusters,
)


class TestViz(unittest.TestCase):

    def test_plot_confusion_matrix(self) -> None:
        cm = np.array([[10, 2], [1, 11]])
        classes = ["Austen", "Dickens"]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "cm.png"
            res = plot_confusion_matrix(cm, classes, output_path=out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_plot_chunk_size_comparison(self) -> None:
        chunk_sizes = [500, 1000, 2000, 5000]
        delta_scores = [0.65, 0.72, 0.81, 0.88]
        lr_scores = [0.70, 0.78, 0.85, 0.91]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "chunk_acc.png"
            res = plot_chunk_size_comparison(chunk_sizes, delta_scores, lr_scores, output_path=out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)

    def test_plot_pca_clusters(self) -> None:
        np.random.seed(42)
        n = 30
        df = pd.DataFrame(
            {
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "feat3": np.random.randn(n),
                "author": ["Austen"] * 15 + ["Dickens"] * 15,
                "book": ["b1"] * 15 + ["b2"] * 15,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "pca.png"
            res = plot_pca_clusters(df, output_path=out_file)
            self.assertTrue(res.exists())
            self.assertGreater(res.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
