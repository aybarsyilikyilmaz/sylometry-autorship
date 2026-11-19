"""
Custom Burrows' Delta Classifier built from scratch.

Fully compliant with scikit-learn BaseEstimator and ClassifierMixin interfaces.

Mathematical Formulation (Burrows, 2002):
1. Compute mean (mu_j) and standard deviation (sigma_j) for each feature j across the training set:
     z_ij = (X_ij - mu_j) / sigma_j
2. Compute author profile vector P_a as the mean z-score for each author across their training samples:
     P_aj = (1 / |S_a|) * sum_{i in S_a} z_ij
3. For an unseen query sample x, calculate its z-score using training mu and sigma:
     z_test_j = (x_j - mu_j) / sigma_j
4. Measure the Manhattan distance (average absolute difference) to each author's profile:
     Delta(z_test, P_a) = (1 / M) * sum_{j=1}^M |z_test_j - P_aj|
5. Predict the author yielding the minimum Delta distance:
     y_hat = argmin_a Delta(z_test, P_a)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class BurrowsDelta(BaseEstimator, ClassifierMixin):
    """
    Burrows' Delta Classifier for Authorship Attribution.

    Parameters
    ----------
    epsilon : float, default=1e-9
        Small constant added to standard deviations to prevent division by zero for invariant features.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series, List[Any]]) -> "BurrowsDelta":
        """
        Fit the Burrows' Delta model.

        Calculates feature-wise means and standard deviations on training data,
        computes z-scores, and creates an author profile centroid for each class.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError(f"X must be a 2D array, got ndim={X_arr.ndim}")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"X and y must have equal length: {len(X_arr)} vs {len(y_arr)}")

        self.classes_ = np.unique(y_arr)
        self.n_features_in_ = X_arr.shape[1]

        # 1. Feature means (mu) and standard deviations (sigma) across training set
        self.feature_means_ = np.mean(X_arr, axis=0)
        self.feature_stds_ = np.std(X_arr, axis=0, ddof=0)
        # Avoid zero division
        self.feature_stds_safe_ = np.where(self.feature_stds_ < self.epsilon, 1.0, self.feature_stds_)

        # 2. Convert training data to z-scores
        Z_train = (X_arr - self.feature_means_) / self.feature_stds_safe_

        # 3. Compute author centroid profiles
        self.author_profiles_: Dict[Any, np.ndarray] = {}
        for author in self.classes_:
            mask = y_arr == author
            if np.any(mask):
                author_centroid = np.mean(Z_train[mask], axis=0)
                self.author_profiles_[author] = author_centroid
            else:
                self.author_profiles_[author] = np.zeros(self.n_features_in_, dtype=np.float64)

        return self

    def _calculate_deltas(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute the Manhattan Delta distance matrix between test samples and author profiles.
        Shape: (n_samples, n_classes)
        """
        check_is_fitted(self, ["classes_", "feature_means_", "feature_stds_safe_", "author_profiles_"])
        X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Feature count mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}")

        # Compute z-scores for test samples using training statistics
        Z_test = (X_arr - self.feature_means_) / self.feature_stds_safe_

        # Matrix of distances: rows = samples, columns = classes
        deltas = np.zeros((Z_test.shape[0], len(self.classes_)), dtype=np.float64)

        for col_idx, author in enumerate(self.classes_):
            profile = self.author_profiles_[author]
            # Manhattan distance normalized by number of features (average absolute difference)
            abs_diff = np.abs(Z_test - profile)
            delta_to_author = np.mean(abs_diff, axis=1)
            deltas[:, col_idx] = delta_to_author

        return deltas

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict author labels for samples in X.
        Returns the author class with minimal Delta distance.
        """
        deltas = self._calculate_deltas(X)
        best_indices = np.argmin(deltas, axis=1)
        return self.classes_[best_indices]

    def predict_distance_matrix(self, X: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Return the raw Delta distance table (samples x authors).
        Useful for error analysis, confidence inspection, and heatmaps.
        """
        deltas = self._calculate_deltas(X)
        return pd.DataFrame(deltas, columns=[str(c) for c in self.classes_])

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Estimate pseudo-probabilities via softmax over negative Delta distances.
        Smaller Delta -> higher probability.
        """
        deltas = self._calculate_deltas(X)
        # Softmax on negative distances with numerical stability
        neg_deltas = -deltas
        max_neg = np.max(neg_deltas, axis=1, keepdims=True)
        exp_scores = np.exp(neg_deltas - max_neg)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probabilities
