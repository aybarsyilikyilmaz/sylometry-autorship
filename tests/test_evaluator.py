"""
Unit tests for src/evaluator.py (ExperimentRunner and GroupKFold evaluation)
"""

import unittest

import numpy as np
import pandas as pd

from src.evaluator import ExperimentRunner


class TestExperimentRunner(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(42)
        rows = []
        authors = ["Austen", "Dickens"]
        # 8 books total: 4 for Austen, 4 for Dickens
        books_by_author = {
            "Austen": ["pride_prejudice", "emma", "sense_sensibility", "persuasion"],
            "Dickens": ["oliver_twist", "bleak_house", "david_copperfield", "great_expectations"],
        }

        # 10 chunks per book = 40 chunks total
        for author in authors:
            for book in books_by_author[author]:
                # Signal: Austen has higher feature 1, Dickens has higher feature 2
                base = 10.0 if author == "Austen" else -10.0
                for _ in range(10):
                    row = {
                        "fw_the": np.random.normal(loc=base, scale=1.0),
                        "fw_and": np.random.normal(loc=-base, scale=1.0),
                        "mean_sentence_length": np.random.normal(loc=20.0, scale=2.0),
                        "std_sentence_length": np.random.normal(loc=5.0, scale=1.0),
                        "type_token_ratio": 0.45,
                        "punct_comma": 50.0,
                        "punct_semicolon": 10.0,
                        "punct_dash": 5.0,
                        "punct_exclamation": 2.0,
                        "punct_question": 1.0,
                        "author": author,
                        "book": book,
                    }
                    rows.append(row)

        self.df = pd.DataFrame(rows)

    def test_run_experiment_group_kfold(self) -> None:
        runner = ExperimentRunner(n_splits=2, random_state=42)
        comp = runner.run_experiment(self.df)

        self.assertEqual(comp.n_samples, 80)
        self.assertEqual(comp.n_features, 10)
        self.assertEqual(len(comp.delta_result.fold_accuracies), 2)
        self.assertEqual(len(comp.lr_result.fold_accuracies), 2)

        # Accuracy should be very high given synthetic separation
        self.assertGreater(comp.delta_result.mean_accuracy, 0.8)
        self.assertGreater(comp.lr_result.mean_accuracy, 0.8)

        # Confusion matrices should have shape (2, 2)
        self.assertEqual(comp.delta_result.confusion_matrix.shape, (2, 2))
        self.assertEqual(comp.lr_result.confusion_matrix.shape, (2, 2))

    def test_distinctive_features(self) -> None:
        runner = ExperimentRunner(n_splits=2, random_state=42)
        comp = runner.run_experiment(self.df)

        self.assertIn("Austen", comp.distinctive_features)
        self.assertIn("Dickens", comp.distinctive_features)
        austen_pos = comp.distinctive_features["Austen"]["top_positive"]
        self.assertTrue(len(austen_pos) > 0)


if __name__ == "__main__":
    unittest.main()
