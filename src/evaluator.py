"""
Experiment Runner and Cross-Validation Module.

Provides:
- ExperimentRunner: Implements strict book-grouped cross-validation using GroupKFold
  so that chunks from any individual book never cross the train/test barrier.
- Benchmarks our custom BurrowsDelta classifier against a standardized LogisticRegression
  (or LinearSVC) baseline across identical folds.
- Extracts per-fold and aggregated accuracy, confusion matrices, and the most discriminative
  words/features per author via logistic regression weights.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.delta import BurrowsDelta


@dataclass
class EvaluationResult:
    """Stores cross-validation scores, confusion matrices, and model weights."""

    model_name: str
    fold_accuracies: List[float]
    mean_accuracy: float
    std_accuracy: float
    confusion_matrix: np.ndarray
    classes: List[str]
    y_true: List[str]
    y_pred: List[str]
    classification_report_str: str


@dataclass
class ExperimentComparison:
    """Holds comparative evaluation between Burrows' Delta and Logistic Regression."""

    delta_result: EvaluationResult
    lr_result: EvaluationResult
    distinctive_features: Dict[str, Dict[str, List[Tuple[str, float]]]]
    n_splits: int
    n_samples: int
    n_features: int


class ExperimentRunner:
    """
    Executes cross-validation on stylometric feature DataFrames using GroupKFold.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Split DataFrame into features X, target y (author), groups (book), and feature names.
        """
        if "author" not in df.columns or "book" not in df.columns:
            raise ValueError("DataFrame must contain 'author' and 'book' columns.")

        feature_cols = [c for c in df.columns if c not in ["author", "book"]]
        X = df[feature_cols].copy()
        y = df["author"].astype(str)
        groups = df["book"].astype(str)
        return X, y, groups, feature_cols

    def run_experiment(self, df: pd.DataFrame) -> ExperimentComparison:
        """
        Execute GroupKFold cross-validation on df, comparing Burrows' Delta and Logistic Regression.
        """
        X, y, groups, feature_names = self.prepare_data(df)
        unique_classes = sorted(y.unique().tolist())
        unique_groups = groups.unique()

        # Adjust splits if there are fewer books than n_splits
        effective_splits = min(self.n_splits, len(unique_groups))
        if effective_splits < 2:
            raise ValueError(f"Need at least 2 distinct book groups for cross-validation, found {len(unique_groups)}")

        # Use StratifiedGroupKFold to enforce book isolation while ensuring all authors
        # are represented in both train and test partitions
        try:
            splitter = StratifiedGroupKFold(
                n_splits=effective_splits, shuffle=True, random_state=self.random_state
            )
            splits = list(splitter.split(X, y, groups=groups))
        except Exception:
            splitter = GroupKFold(n_splits=effective_splits)
            splits = list(splitter.split(X, y, groups=groups))

        delta_fold_accs: List[float] = []
        lr_fold_accs: List[float] = []

        all_y_true: List[str] = []
        delta_all_preds: List[str] = []
        lr_all_preds: List[str] = []

        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            train_books = set(groups.iloc[train_idx])
            test_books = set(groups.iloc[test_idx])

            # Integrity check: Ensure strict group isolation
            assert train_books.isdisjoint(test_books), f"Data leakage: {train_books & test_books} in both train and test!"

            # 1. Burrows' Delta
            delta_clf = BurrowsDelta()
            delta_clf.fit(X_train, y_train)
            delta_preds = delta_clf.predict(X_test)
            delta_acc = float(accuracy_score(y_test, delta_preds))
            delta_fold_accs.append(delta_acc)
            delta_all_preds.extend(delta_preds.tolist())

            # 2. Logistic Regression Baseline with StandardScaler
            lr_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=1500,
                            solver="lbfgs",
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            lr_pipe.fit(X_train, y_train)
            lr_preds = lr_pipe.predict(X_test)
            lr_acc = float(accuracy_score(y_test, lr_preds))
            lr_fold_accs.append(lr_acc)
            lr_all_preds.extend(lr_preds.tolist())

            all_y_true.extend(y_test.tolist())

        # Construct EvaluationResult for Delta
        cm_delta = confusion_matrix(all_y_true, delta_all_preds, labels=unique_classes)
        cr_delta = classification_report(all_y_true, delta_all_preds, labels=unique_classes, zero_division=0)
        delta_res = EvaluationResult(
            model_name="Burrows' Delta",
            fold_accuracies=delta_fold_accs,
            mean_accuracy=float(np.mean(delta_fold_accs)),
            std_accuracy=float(np.std(delta_fold_accs)),
            confusion_matrix=cm_delta,
            classes=unique_classes,
            y_true=all_y_true,
            y_pred=delta_all_preds,
            classification_report_str=cr_delta,
        )

        # Construct EvaluationResult for Logistic Regression
        cm_lr = confusion_matrix(all_y_true, lr_all_preds, labels=unique_classes)
        cr_lr = classification_report(all_y_true, lr_all_preds, labels=unique_classes, zero_division=0)
        lr_res = EvaluationResult(
            model_name="Logistic Regression",
            fold_accuracies=lr_fold_accs,
            mean_accuracy=float(np.mean(lr_fold_accs)),
            std_accuracy=float(np.std(lr_fold_accs)),
            confusion_matrix=cm_lr,
            classes=unique_classes,
            y_true=all_y_true,
            y_pred=lr_all_preds,
            classification_report_str=cr_lr,
        )

        # 3. Extract distinctive words/features from Logistic Regression weights
        distinctive_features = self.extract_distinctive_features(X, y, feature_names, top_k=10)

        return ExperimentComparison(
            delta_result=delta_res,
            lr_result=lr_res,
            distinctive_features=distinctive_features,
            n_splits=effective_splits,
            n_samples=len(X),
            n_features=len(feature_names),
        )

    def extract_distinctive_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        top_k: int = 10,
    ) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
        """
        Fits Logistic Regression on the complete feature set to discover the strongest
        positive and negative stylometric weights for each author.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=self.random_state,
        )
        lr.fit(X_scaled, y)

        classes = lr.classes_
        coefs = lr.coef_  # shape: (n_classes, n_features)

        result: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

        for class_idx, author in enumerate(classes):
            if len(classes) == 2:
                author_weights = coefs[0] if class_idx == 1 else -coefs[0]
            else:
                author_weights = coefs[class_idx]
            # Top positive
            sorted_indices = np.argsort(author_weights)
            top_pos_indices = sorted_indices[-top_k:][::-1]
            # Top negative
            top_neg_indices = sorted_indices[:top_k]

            pos_features = [
                (self._clean_feature_name(feature_names[i]), float(author_weights[i]))
                for i in top_pos_indices
            ]
            neg_features = [
                (self._clean_feature_name(feature_names[i]), float(author_weights[i]))
                for i in top_neg_indices
            ]

            result[str(author)] = {
                "top_positive": pos_features,
                "top_negative": neg_features,
            }

        return result

    @staticmethod
    def _clean_feature_name(raw_name: str) -> str:
        """Helper to format feature column names for human-readable reporting."""
        if raw_name.startswith("fw_"):
            return f"word: '{raw_name[3:]}'"
        if raw_name.startswith("punct_"):
            return f"punct: {raw_name[6:]}"
        if raw_name == "mean_sentence_length":
            return "Mean Sentence Length"
        if raw_name == "std_sentence_length":
            return "Sentence Length Std"
        if raw_name == "type_token_ratio":
            return "Vocabulary Diversity (TTR)"
        return raw_name
