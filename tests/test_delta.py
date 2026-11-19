"""
Unit tests for src/delta.py (Burrows' Delta Classifier)
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.delta import BurrowsDelta


class TestBurrowsDelta(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        # Author A: features around 10
        # Author B: features around 50
        # Author C: features around 100
        n_samples_per_class = 15
        n_features = 8

        X_a = np.random.normal(loc=10.0, scale=1.0, size=(n_samples_per_class, n_features))
        X_b = np.random.normal(loc=50.0, scale=1.0, size=(n_samples_per_class, n_features))
        X_c = np.random.normal(loc=100.0, scale=1.0, size=(n_samples_per_class, n_features))

        self.X = np.vstack([X_a, X_b, X_c])
        self.y = np.array(
            ["Austen"] * n_samples_per_class
            + ["Dickens"] * n_samples_per_class
            + ["Twain"] * n_samples_per_class
        )

    def test_fit_and_predict(self) -> None:
        clf = BurrowsDelta()
        clf.fit(self.X, self.y)

        # Check properties
        self.assertEqual(len(clf.classes_), 3)
        self.assertEqual(clf.n_features_in_, 8)
        self.assertIn("Austen", clf.author_profiles_)
        self.assertIn("Dickens", clf.author_profiles_)
        self.assertIn("Twain", clf.author_profiles_)

        # Predictions on training data should be 100% accurate given clear separation
        preds = clf.predict(self.X)
        accuracy = np.mean(preds == self.y)
        self.assertEqual(accuracy, 1.0)

    def test_predict_distance_matrix(self) -> None:
        clf = BurrowsDelta()
        clf.fit(self.X, self.y)
        dist_df = clf.predict_distance_matrix(self.X[:5])
        self.assertEqual(dist_df.shape, (5, 3))
        # Austen samples should have smallest distance to Austen column
        for i in range(5):
            self.assertEqual(dist_df.iloc[i].idxmin(), "Austen")

    def test_predict_proba_sums_to_one(self) -> None:
        clf = BurrowsDelta()
        clf.fit(self.X, self.y)
        probs = clf.predict_proba(self.X[:5])
        self.assertEqual(probs.shape, (5, 3))
        for row in probs:
            self.assertAlmostEqual(float(np.sum(row)), 1.0)

    def test_zero_variance_feature_stability(self) -> None:
        # Add a column with identical values (zero variance)
        X_with_constant = np.hstack([self.X, np.zeros((self.X.shape[0], 1))])
        clf = BurrowsDelta()
        clf.fit(X_with_constant, self.y)
        # Should not raise ZeroDivisionError
        preds = clf.predict(X_with_constant)
        self.assertEqual(len(preds), len(self.y))

    def test_sklearn_cross_validation_compatibility(self) -> None:
        from sklearn.model_selection import StratifiedKFold

        clf = BurrowsDelta()
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, self.X, self.y, cv=cv)
        self.assertEqual(len(scores), 3)
        self.assertTrue(np.all(scores >= 0.9))


if __name__ == "__main__":
    unittest.main()
