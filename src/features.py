"""
Stylometric Feature Extraction and Dataset Balancing Module.

Extracts an exact 156-dimensional stylometric feature vector per chunk:
- 150 Functional / High-Frequency Word relative frequencies: count(w) / total_tokens
- 2 Sentence Structure features: mean and standard deviation of sentence lengths (in words)
- 1 Lexical Richness feature: Type-Token Ratio (TTR = unique_tokens / total_tokens)
- 5 Punctuation Mark frequencies per 1,000 words: comma (,), semicolon (;), dash (- / — / –),
  exclamation mark (!), question mark (?)
Plus 2 metadata columns: `author` (target label) and `book` (group identifier).
Total columns = 158.

Also includes downsampling to equalize chunk counts per author to N_min (random_state=42).
"""

from collections import Counter
from pathlib import Path
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.corpus import TextChunk


class StylometricExtractor:
    """
    Extracts stylometric features from text chunks and prepares balanced datasets.
    """

    # Target punctuation symbols to measure
    PUNCTUATION_TARGETS = {
        "comma": ",",
        "semicolon": ";",
        "dash": "-—–",  # covers ASCII hyphen, em-dash, en-dash
        "exclamation": "!",
        "question": "?",
    }

    def __init__(self, top_n_words: int = 150) -> None:
        self.top_n_words = top_n_words
        self.vocabulary: List[str] = []
        self._punct_regexes = {
            "comma": re.compile(r","),
            "semicolon": re.compile(r";"),
            "dash": re.compile(r"[-—–]"),
            "exclamation": re.compile(r"!"),
            "question": re.compile(r"\?"),
        }
        self._sentence_splitter = re.compile(r"[.!?]+(?:\s+|$)")
        self._clean_token_pattern = re.compile(r"[^\w\s]")

    @staticmethod
    def tokenize_words(text: str) -> List[str]:
        """
        Tokenize text into lowercased alphanumeric words, stripping trailing/leading punctuation.
        """
        # Split on whitespace, strip surrounding punctuation
        raw_tokens = text.split()
        words: List[str] = []
        for t in raw_tokens:
            cleaned = t.strip(string.punctuation + "“”‘’—–…").lower()
            if cleaned and any(c.isalnum() for c in cleaned):
                words.append(cleaned)
        return words

    def fit_vocabulary(self, chunks: Sequence[TextChunk]) -> "StylometricExtractor":
        """
        Scan all chunks across the corpus and identify the top N most frequent words.
        """
        corpus_word_counts: Counter[str] = Counter()
        for chunk in chunks:
            words = self.tokenize_words(chunk.text)
            corpus_word_counts.update(words)

        # Select top N most frequent words
        most_common = corpus_word_counts.most_common(self.top_n_words)
        self.vocabulary = [word for word, _ in most_common]

        # If corpus has fewer unique words than top_n_words, pad with placeholders
        if len(self.vocabulary) < self.top_n_words:
            for i in range(len(self.vocabulary) + 1, self.top_n_words + 1):
                self.vocabulary.append(f"pad_{i}")

        return self

    def compute_sentence_length_stats(self, text: str) -> Tuple[float, float]:
        """
        Calculate the mean and standard deviation of sentence lengths (measured in word count).
        Sentences are delimited by '.', '!', '?'.
        """
        raw_sentences = self._sentence_splitter.split(text)
        sentence_lengths: List[int] = []

        for s in raw_sentences:
            s_stripped = s.strip()
            if not s_stripped:
                continue
            words_in_sentence = self.tokenize_words(s_stripped)
            if words_in_sentence:
                sentence_lengths.append(len(words_in_sentence))

        if not sentence_lengths:
            return 0.0, 0.0

        if len(sentence_lengths) == 1:
            return float(sentence_lengths[0]), 0.0

        arr = np.array(sentence_lengths, dtype=np.float64)
        mean_len = float(np.mean(arr))
        # Use sample standard deviation with ddof=1
        std_len = float(np.std(arr, ddof=1))
        return mean_len, std_len

    def compute_ttr(self, words: List[str]) -> float:
        """
        Compute Type-Token Ratio (TTR): unique word count / total word count.
        """
        total = len(words)
        if total == 0:
            return 0.0
        unique = len(set(words))
        return unique / total

    def compute_punctuation_frequencies(self, text: str, total_words: int) -> Dict[str, float]:
        """
        Compute occurrences of target punctuation marks per 1,000 words:
        (count / total_words) * 1000
        """
        if total_words == 0:
            return {f"punct_{name}": 0.0 for name in self.PUNCTUATION_TARGETS}

        scale = 1000.0 / total_words
        rates: Dict[str, float] = {}
        for name, regex in self._punct_regexes.items():
            count = len(regex.findall(text))
            rates[f"punct_{name}"] = count * scale

        return rates

    def extract_chunk_features(self, chunk: TextChunk) -> Dict[str, Any]:
        """
        Extract the complete 156-feature vector plus metadata for a single text chunk.
        """
        words = self.tokenize_words(chunk.text)
        total_words = len(words)
        chunk_word_counts = Counter(words)

        row: Dict[str, Any] = {}

        # 1. Functional Words (150 features)
        for word in self.vocabulary:
            freq = (chunk_word_counts[word] / total_words) if total_words > 0 else 0.0
            row[f"fw_{word}"] = freq

        # 2. Sentence Structure (2 features)
        mean_sl, std_sl = self.compute_sentence_length_stats(chunk.text)
        row["mean_sentence_length"] = mean_sl
        row["std_sentence_length"] = std_sl

        # 3. Lexical Diversity (1 feature)
        row["type_token_ratio"] = self.compute_ttr(words)

        # 4. Punctuation Frequencies (5 features)
        punct_stats = self.compute_punctuation_frequencies(chunk.text, total_words)
        row.update(punct_stats)

        # 5. Metadata Columns (2 columns at the end)
        row["author"] = chunk.author
        row["book"] = chunk.book

        return row

    def extract_features(
        self, chunks: Sequence[TextChunk], fit_vocab: bool = True
    ) -> pd.DataFrame:
        """
        Extract features for all chunks into a pandas DataFrame (158 columns).
        """
        if fit_vocab or not self.vocabulary:
            self.fit_vocabulary(chunks)

        rows = [self.extract_chunk_features(chunk) for chunk in chunks]
        df = pd.DataFrame(rows)

        # Ensure correct column ordering: 150 fw_*, 2 sentence, 1 TTR, 5 punct, author, book
        feature_cols = (
            [f"fw_{w}" for w in self.vocabulary]
            + ["mean_sentence_length", "std_sentence_length", "type_token_ratio"]
            + [f"punct_{name}" for name in self.PUNCTUATION_TARGETS]
        )
        ordered_cols = feature_cols + ["author", "book"]

        # Ensure any missing columns exist with 0.0
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = 0.0

        return df[ordered_cols]

    @staticmethod
    def balance_classes(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
        """
        Equalize the number of chunks across all authors by downsampling to N_min.
        Guarantees unbiased training so the classifier does not favor prolific authors.
        """
        if df.empty or "author" not in df.columns:
            return df

        author_counts = df["author"].value_counts()
        min_chunks = author_counts.min()

        sampled_dfs: List[pd.DataFrame] = []
        for author in author_counts.index:
            author_df = df[df["author"] == author]
            sampled = author_df.sample(n=min_chunks, random_state=random_state)
            sampled_dfs.append(sampled)

        balanced_df = (
            pd.concat(sampled_dfs, ignore_index=True)
            .sample(frac=1.0, random_state=random_state)
            .reset_index(drop=True)
        )
        return balanced_df

    @staticmethod
    def save_features(
        df: pd.DataFrame,
        chunk_size: int,
        output_dir: Union[str, Path] = "data/processed",
    ) -> Path:
        """
        Save the feature table to `data/processed/features_{chunk_size}.csv`.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"features_{chunk_size}.csv"
        df.to_csv(file_path, index=False)
        return file_path
