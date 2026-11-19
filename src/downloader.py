"""
Automated Downloader Module for the 56-Book Corpus (7 Authors x 8 Books).

Uses Project Gutenberg mirrors with retry logic, thread pooling, and user-agent headers.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

# Catalog of 7 authors x 8 books = 56 books
BOOK_CATALOG: Dict[str, Dict[str, int]] = {
    "Jane_Austen": {
        "pride_and_prejudice": 1342,
        "sense_and_sensibility": 161,
        "emma": 158,
        "mansfield_park": 141,
        "northanger_abbey": 121,
        "persuasion": 105,
        "lady_susan": 946,
        "love_and_freindship": 1212,
    },
    "Charlotte_Bronte": {
        "jane_eyre": 1260,
        "villette": 9182,
        "shirley": 30486,
        "the_professor": 1028,
        "poems_by_currer_ellis_and_acton_bell": 1018,
        "the_green_dwarf": 43301,
        "tales_of_the_islanders": 45781,
        "the_spell": 46278,
    },
    "Charles_Dickens": {
        "a_tale_of_two_cities": 98,
        "great_expectations": 1400,
        "oliver_twist": 730,
        "david_copperfield": 766,
        "bleak_house": 1023,
        "hard_times": 786,
        "a_christmas_carol": 46,
        "our_mutual_friend": 883,
    },
    "Arthur_Conan_Doyle": {
        "a_study_in_scarlet": 244,
        "the_sign_of_the_four": 2097,
        "the_hound_of_the_baskervilles": 2852,
        "the_valley_of_fear": 3289,
        "the_adventures_of_sherlock_holmes": 1661,
        "the_memoirs_of_sherlock_holmes": 834,
        "the_return_of_sherlock_holmes": 108,
        "the_lost_world": 139,
    },
    "Mark_Twain": {
        "the_adventures_of_tom_sawyer": 74,
        "adventures_of_huckleberry_finn": 76,
        "a_connecticut_yankee": 86,
        "the_prince_and_the_pauper": 1837,
        "life_on_the_mississippi": 245,
        "the_innocents_abroad": 3176,
        "roughing_it": 377,
        "the_gilded_age": 11942,
    },
    "HG_Wells": {
        "the_time_machine": 35,
        "the_war_of_the_worlds": 36,
        "the_invisible_man": 5230,
        "the_island_of_doctor_moreau": 159,
        "the_first_men_in_the_moon": 1013,
        "when_the_sleeper_wakes": 775,
        "the_food_of_the_gods": 11505,
        "kipps": 11777,
    },
    "Oscar_Wilde": {
        "the_picture_of_dorian_gray": 174,
        "the_importance_of_being_earnest": 844,
        "lady_windermeres_fan": 790,
        "an_ideal_husband": 882,
        "a_woman_of_no_importance": 854,
        "de_profundis": 921,
        "the_canterville_ghost": 14522,
        "the_happy_prince_and_other_tales": 902,
    },
}

GUTENBERG_MIRRORS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://gutenberg.readingroo.ms/cache/epub/{id}/pg{id}.txt",
]


def download_single_book(
    author: str,
    book_title: str,
    gutenberg_id: int,
    raw_dir: Path,
    retries: int = 3,
) -> Tuple[str, str, bool, str]:
    """
    Downloads a single book from Project Gutenberg and saves to raw_dir / author / book_title.txt.
    """
    target_dir = raw_dir / author
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"{book_title}.txt"

    if target_file.exists() and target_file.stat().st_size > 1000:
        return (author, book_title, True, "Already cached")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/plain,text/html,application/xhtml+xml",
    }

    last_error = ""
    for mirror_template in GUTENBERG_MIRRORS:
        url = mirror_template.format(id=gutenberg_id)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content_bytes = resp.read()
                    if len(content_bytes) > 2000:
                        # Decode utf-8 with fallback
                        try:
                            text = content_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = content_bytes.decode("latin-1", errors="replace")

                        target_file.write_text(text, encoding="utf-8")
                        return (author, book_title, True, f"Downloaded {len(text)} chars from {url}")
            except Exception as e:
                last_error = str(e)
                time.sleep(1.0)

    return (author, book_title, False, f"Failed after {retries} retries: {last_error}")


def download_all_books(
    raw_dir: Path = Path("data/raw"),
    max_workers: int = 4,
) -> Dict[str, Any]:
    """
    Downloads all 56 books in parallel across the 7 authors.
    """
    raw_dir = Path(raw_dir)
    tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for author, books in BOOK_CATALOG.items():
            for title, gid in books.items():
                future = executor.submit(
                    download_single_book,
                    author,
                    title,
                    gid,
                    raw_dir,
                )
                tasks.append(future)

        results = [t.result() for t in as_completed(tasks)]

    successes = [r for r in results if r[2]]
    failures = [r for r in results if not r[2]]

    return {
        "total_catalog": len(tasks),
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
    }
