"""
Corpus Loading and Text Cleaning Module for Stylometry Authorship Attribution.

Provides:
- TextCleaner: Strips Project Gutenberg boilerplate headers/footers, normalizes whitespace,
  and slices text into word-based chunks while strictly preserving punctuation.
- CorpusLoader: Traverses `data/raw/{author}/{book}.txt` directory structures and produces
  standardized, tagged text chunks for stylometric feature extraction.
"""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Optional, Union


@dataclass
class TextChunk:
    """Represents a discrete text chunk belonging to an author and work."""

    text: str
    author: str
    book: str
    chunk_id: int
    token_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk metadata and content to a dictionary."""
        return {
            "text": self.text,
            "author": self.author,
            "book": self.book,
            "chunk_id": self.chunk_id,
            "token_count": self.token_count,
        }


class TextCleaner:
    """
    Cleans raw literary texts and segments them into fixed-size word chunks.

    Maintains all punctuation marks (commas, semicolons, em-dashes, exclamation,
    and question marks) intact, which are essential for stylometric signature extraction.
    """

    # Project Gutenberg start patterns
    GUTENBERG_START_PATTERNS = [
        r"\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\*START OF THE PROJECT GUTENBERG",
        r"\*END\*THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS",
        r"<<THIS PROJECT GUTENBERG EBOOK",
        r"START OF THE PROJECT GUTENBERG EBOOK",
    ]

    # Project Gutenberg end patterns
    GUTENBERG_END_PATTERNS = [
        r"\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*",
        r"\*\*\*END OF THE PROJECT GUTENBERG",
        r"End of the Project Gutenberg",
        r"End of Project Gutenberg",
        r"\*\*\* END OF THIS PROJECT GUTENBERG",
    ]

    def __init__(self) -> None:
        self.start_regex = re.compile(
            "|".join(f"({p})" for p in self.GUTENBERG_START_PATTERNS),
            re.IGNORECASE,
        )
        self.end_regex = re.compile(
            "|".join(f"({p})" for p in self.GUTENBERG_END_PATTERNS),
            re.IGNORECASE,
        )
        self.whitespace_regex = re.compile(r"\s+")

    def strip_boilerplate(self, raw_text: str) -> str:
        """
        Remove Project Gutenberg header and footer disclaimers/licenses.
        If no Gutenberg markers are detected, the original text is preserved.
        """
        text = raw_text

        # Strip header
        start_match = None
        for match in self.start_regex.finditer(text):
            start_match = match
        if start_match:
            text = text[start_match.end() :]

        # Strip footer
        end_match = self.end_regex.search(text)
        if end_match:
            text = text[: end_match.start()]

        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Collapse multiple newlines, carriage returns, tabs, and spaces into
        a single space, trimming surrounding whitespace. Punctuation is preserved.
        """
        return self.whitespace_regex.sub(" ", text).strip()

    def clean(self, raw_text: str) -> str:
        """Complete pipeline: strip Gutenberg boilerplate and normalize whitespace."""
        stripped = self.strip_boilerplate(raw_text)
        return self.normalize_whitespace(stripped)

    def chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
        """
        Divide text into chunks of exactly `chunk_size` words.
        Words are split by whitespace so that attached punctuation is preserved.
        Any trailing chunk with fewer than `chunk_size` words is discarded.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

        tokens = text.split()
        total_tokens = len(tokens)

        chunks: List[str] = []
        for start_idx in range(0, total_tokens, chunk_size):
            end_idx = start_idx + chunk_size
            if end_idx <= total_tokens:
                chunk_slice = tokens[start_idx:end_idx]
                chunks.append(" ".join(chunk_slice))

        return chunks


class CorpusLoader:
    """
    Manages loading, cleaning, and chunking of text files organized under:
    `data/raw/{author}/{book}.txt`.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw",
        cleaner: Optional[TextCleaner] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.cleaner = cleaner or TextCleaner()

    def discover_books(self) -> Dict[str, List[Path]]:
        """
        Scan `raw_dir` and return a mapping of author -> list of book file paths.
        Expected structure: raw_dir / author_name / book_file.txt
        """
        if not self.raw_dir.exists():
            return {}

        books_by_author: Dict[str, List[Path]] = {}
        for author_dir in sorted(self.raw_dir.iterdir()):
            if author_dir.is_dir() and not author_dir.name.startswith("."):
                book_files = sorted(
                    [
                        f
                        for f in author_dir.iterdir()
                        if f.is_file() and f.suffix.lower() == ".txt" and not f.name.startswith(".")
                    ]
                )
                if book_files:
                    books_by_author[author_dir.name] = book_files

        return books_by_author

    def read_book_file(self, file_path: Path) -> str:
        """Read a text file with UTF-8 encoding fallback."""
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback with error replacement for non-standard encodings
            return file_path.read_text(encoding="utf-8", errors="replace")

    def load_chunks(self, chunk_size: int = 5000) -> List[TextChunk]:
        """
        Load all books, clean, chunk into `chunk_size` words, and tag with author and title.
        Trailing chunks smaller than `chunk_size` are discarded.
        """
        books_by_author = self.discover_books()
        all_chunks: List[TextChunk] = []

        for author, book_paths in books_by_author.items():
            for book_path in book_paths:
                book_title = book_path.stem
                raw_content = self.read_book_file(book_path)
                cleaned_content = self.cleaner.clean(raw_content)
                text_slices = self.cleaner.chunk_text(cleaned_content, chunk_size=chunk_size)

                for idx, slice_text in enumerate(text_slices):
                    chunk = TextChunk(
                        text=slice_text,
                        author=author,
                        book=book_title,
                        chunk_id=idx,
                        token_count=chunk_size,
                    )
                    all_chunks.append(chunk)

        return all_chunks

    def get_summary(self, chunk_size: int = 5000) -> Dict[str, Any]:
        """
        Provide diagnostic information on discovered authors, books, and chunk counts.
        """
        chunks = self.load_chunks(chunk_size=chunk_size)
        books_by_author = self.discover_books()

        chunks_per_author: Dict[str, int] = {}
        for c in chunks:
            chunks_per_author[c.author] = chunks_per_author.get(c.author, 0) + 1

        books_count_per_author = {a: len(b) for a, b in books_by_author.items()}

        return {
            "author_count": len(books_by_author),
            "total_books": sum(books_count_per_author.values()),
            "books_per_author": books_count_per_author,
            "total_chunks": len(chunks),
            "chunks_per_author": chunks_per_author,
            "min_chunks_per_author": min(chunks_per_author.values()) if chunks_per_author else 0,
        }
