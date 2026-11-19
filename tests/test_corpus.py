"""
Unit tests for src/corpus.py
"""

from pathlib import Path
import tempfile
import unittest

from src.corpus import CorpusLoader, TextCleaner, TextChunk


class TestTextCleaner(unittest.TestCase):

    def setUp(self) -> None:
        self.cleaner = TextCleaner()

    def test_strip_boilerplate(self) -> None:
        sample = (
            "The Project Gutenberg eBook of Sample Work\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "Chapter 1. It was the best of times, it was the worst of times!\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
            "End of Project Gutenberg's work."
        )
        cleaned = self.cleaner.strip_boilerplate(sample)
        self.assertIn("Chapter 1", cleaned)
        self.assertNotIn("*** START", cleaned)
        self.assertNotIn("*** END", cleaned)
        self.assertNotIn("The Project Gutenberg eBook", cleaned)

    def test_normalize_whitespace_preserves_punctuation(self) -> None:
        raw = "Hello,   world! \n\n Is this—truly; a  test?  \tYes!"
        normalized = self.cleaner.normalize_whitespace(raw)
        self.assertEqual(normalized, "Hello, world! Is this—truly; a test? Yes!")
        for char in [",", "!", "—", ";", "?"]:
            self.assertIn(char, normalized)

    def test_chunking_exact_size_and_discard_remainder(self) -> None:
        # 12 words
        text = "one two three four five six seven eight nine ten eleven twelve"
        chunks = self.cleaner.chunk_text(text, chunk_size=5)
        # Should produce 2 chunks of 5 words (total 10 words), discarding last 2
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "one two three four five")
        self.assertEqual(chunks[1], "six seven eight nine ten")

        # Zero or negative chunk size raises ValueError
        with self.assertRaises(ValueError):
            self.cleaner.chunk_text(text, chunk_size=0)


class TestCorpusLoader(unittest.TestCase):

    def test_load_and_discover(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            author_dir = tmppath / "Austen"
            author_dir.mkdir()
            book_file = author_dir / "pride_and_prejudice.txt"

            # Create 15 words
            book_content = (
                "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
                "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10 Word11 Word12 Word13 Word14 Word15\n"
                "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
            )
            book_file.write_text(book_content, encoding="utf-8")

            loader = CorpusLoader(raw_dir=tmppath)
            books = loader.discover_books()
            self.assertIn("Austen", books)
            self.assertEqual(len(books["Austen"]), 1)

            chunks = loader.load_chunks(chunk_size=7)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].author, "Austen")
            self.assertEqual(chunks[0].book, "pride_and_prejudice")
            self.assertEqual(chunks[0].token_count, 7)

            summary = loader.get_summary(chunk_size=7)
            self.assertEqual(summary["author_count"], 1)
            self.assertEqual(summary["total_books"], 1)
            self.assertEqual(summary["total_chunks"], 2)


if __name__ == "__main__":
    unittest.main()
