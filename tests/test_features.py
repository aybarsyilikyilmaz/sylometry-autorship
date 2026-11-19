"""
Unit tests for src/features.py
"""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.corpus import TextChunk
from src.features import StylometricExtractor


class TestStylometricExtractor(unittest.TestCase):

    def setUp(self) -> None:
        # Create sample chunks from 2 authors
        self.chunk1 = TextChunk(
            text="It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife! However little known the feelings or views of such a man may be on his first entering a neighbourhood, a truth is fixed? Yes; indeed—it is so.",
            author="Austen",
            book="pride_and_prejudice",
            chunk_id=0,
            token_count=50,
        )
        self.chunk2 = TextChunk(
            text="A single man possessing fortune must have a wife. It is acknowledged universally; and no one doubts it! What a truth—truly magnificent? Indeed, yes.",
            author="Austen",
            book="pride_and_prejudice",
            chunk_id=1,
            token_count=27,
        )
        self.chunk3 = TextChunk(
            text="London. Michaelmas term lately over, and the Lord Chancellor sitting in Lincoln's Inn Hall. Implacable November weather! As much mud in the streets; as if the waters had but newly retired from the face of the earth? Dogs, undistinguishable in mire—horses, scarcely better.",
            author="Dickens",
            book="bleak_house",
            chunk_id=0,
            token_count=45,
        )
        self.chunk4 = TextChunk(
            text="Fog everywhere. Fog up the river, where it flows among green aits and meadows; fog down the river, where it rolls defiled among the tiers of shipping! Fog in the eyes and throats of ancient Greenwich pensioners? Truly—yes, foggy.",
            author="Dickens",
            book="bleak_house",
            chunk_id=1,
            token_count=40,
        )
        self.chunk5 = TextChunk(
            text="More fog and smoke in London streets! The Chancellor watches the dense fog; is it clear? No, not yet—never.",
            author="Dickens",
            book="bleak_house",
            chunk_id=2,
            token_count=22,
        )

        self.chunks = [self.chunk1, self.chunk2, self.chunk3, self.chunk4, self.chunk5]

    def test_feature_dimensions(self) -> None:
        extractor = StylometricExtractor(top_n_words=10)
        df = extractor.extract_features(self.chunks)

        # 10 words + 2 sentence + 1 TTR + 5 punct + 2 metadata = 20 columns
        self.assertEqual(df.shape[0], 5)
        self.assertEqual(df.shape[1], 20)
        self.assertIn("author", df.columns)
        self.assertIn("book", df.columns)
        self.assertIn("type_token_ratio", df.columns)
        self.assertIn("mean_sentence_length", df.columns)
        self.assertIn("std_sentence_length", df.columns)
        self.assertIn("punct_comma", df.columns)
        self.assertIn("punct_semicolon", df.columns)
        self.assertIn("punct_dash", df.columns)
        self.assertIn("punct_exclamation", df.columns)
        self.assertIn("punct_question", df.columns)

    def test_exact_158_features_plus_metadata(self) -> None:
        # With default top_n_words = 150:
        # 150 functional words + 2 sentence stats + 1 TTR + 5 punctuation = 158 features
        # Plus 2 metadata columns ('author', 'book') = 160 total DataFrame columns
        extractor = StylometricExtractor(top_n_words=150)
        df = extractor.extract_features(self.chunks)
        feature_cols = [c for c in df.columns if c not in ["author", "book"]]
        self.assertEqual(len(feature_cols), 158)
        self.assertEqual(df.shape[1], 160)
        self.assertEqual(df.columns[-2], "author")
        self.assertEqual(df.columns[-1], "book")

    def test_punctuation_rates(self) -> None:
        extractor = StylometricExtractor()
        # Text with exactly 10 words, 1 comma, 1 semicolon, 1 dash, 1 exclamation, 1 question
        test_text = "Word, word; word— word! word? word word word word word."
        words = extractor.tokenize_words(test_text)
        punct = extractor.compute_punctuation_frequencies(test_text, len(words))
        # Total words = 10, rate = (1/10) * 1000 = 100.0
        self.assertAlmostEqual(punct["punct_comma"], 100.0)
        self.assertAlmostEqual(punct["punct_semicolon"], 100.0)
        self.assertAlmostEqual(punct["punct_dash"], 100.0)
        self.assertAlmostEqual(punct["punct_exclamation"], 100.0)
        self.assertAlmostEqual(punct["punct_question"], 100.0)

    def test_class_balancing(self) -> None:
        extractor = StylometricExtractor(top_n_words=5)
        df = extractor.extract_features(self.chunks)
        # Austen has 2, Dickens has 3 -> balanced should have 2 each = 4 total
        balanced_df = extractor.balance_classes(df, random_state=42)
        self.assertEqual(len(balanced_df), 4)
        counts = balanced_df["author"].value_counts()
        self.assertEqual(counts["Austen"], 2)
        self.assertEqual(counts["Dickens"], 2)

    def test_save_features(self) -> None:
        extractor = StylometricExtractor(top_n_words=5)
        df = extractor.extract_features(self.chunks)
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = extractor.save_features(df, chunk_size=5000, output_dir=tmpdir)
            self.assertTrue(saved_path.exists())
            loaded = pd.read_csv(saved_path)
            self.assertEqual(loaded.shape, df.shape)


if __name__ == "__main__":
    unittest.main()
