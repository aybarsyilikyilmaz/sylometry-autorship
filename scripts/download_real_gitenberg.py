"""
Fast parallel downloader for authentic, full-length Project Gutenberg novels via GITenberg.
Downloads genuine 19th-century historical literature directly into data/raw/{Author}/{Book}.txt.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Tuple

# Full catalog of 7 Authors x 8 Books = 56 Real Historical Books
GITENBERG_CATALOG = {
    "Jane_Austen": [
        ("Pride_and_Prejudice", "Pride-and-Prejudice_1342", 1342),
        ("Sense_and_Sensibility", "Sense-and-Sensibility_161", 161),
        ("Emma", "Emma_158", 158),
        ("Mansfield_Park", "Mansfield-Park_141", 141),
        ("Northanger_Abbey", "Northanger-Abbey_121", 121),
        ("Persuasion", "Persuasion_105", 105),
        ("Lady_Susan", "Lady-Susan_946", 946),
        ("Love_and_Freindship", "Love-and-Freindship-And-Other-Early-Works_1212", 1212),
    ],
    "Charlotte_Bronte": [
        ("Jane_Eyre", "Jane-Eyre_1260", 1260),
        ("Villette", "Villette_9182", 9182),
        ("Shirley", "Shirley_30486", 30486),
        ("The_Professor", "The-Professor_1028", 1028),
        ("Poems", "Poems-by-Currer-Ellis-and-Acton-Bell_1018", 1018),
        ("The_Green_Dwarf", "The-Green-Dwarf_43301", 43301),
        ("Tales_of_the_Islanders", "Tales-of-the-Islanders_45781", 45781),
        ("The_Spell", "The-Spell_46278", 46278),
    ],
    "Charles_Dickens": [
        ("A_Tale_of_Two_Cities", "A-Tale-of-Two-Cities_98", 98),
        ("Great_Expectations", "Great-Expectations_1400", 1400),
        ("Oliver_Twist", "Oliver-Twist_730", 730),
        ("David_Copperfield", "David-Copperfield_766", 766),
        ("Bleak_House", "Bleak-House_1023", 1023),
        ("Hard_Times", "Hard-Times_786", 786),
        ("A_Christmas_Carol", "A-Christmas-Carol-in-Prose_46", 46),
        ("Our_Mutual_Friend", "Our-Mutual-Friend_883", 883),
    ],
    "Arthur_Conan_Doyle": [
        ("A_Study_in_Scarlet", "A-Study-in-Scarlet_244", 244),
        ("The_Sign_of_the_Four", "The-Sign-of-the-Four_2097", 2097),
        ("The_Hound_of_the_Baskervilles", "The-Hound-of-the-Baskervilles_2852", 2852),
        ("The_Valley_of_Fear", "The-Valley-of-Fear_3289", 3289),
        ("The_Adventures_of_Sherlock_Holmes", "The-Adventures-of-Sherlock-Holmes_1661", 1661),
        ("The_Memoirs_of_Sherlock_Holmes", "The-Memoirs-of-Sherlock-Holmes_834", 834),
        ("The_Return_of_Sherlock_Holmes", "The-Return-of-Sherlock-Holmes_108", 108),
        ("The_Lost_World", "The-Lost-World_139", 139),
    ],
    "Mark_Twain": [
        ("The_Adventures_of_Tom_Sawyer", "The-Adventures-of-Tom-Sawyer_74", 74),
        ("Adventures_of_Huckleberry_Finn", "Adventures-of-Huckleberry-Finn_76", 76),
        ("A_Connecticut_Yankee", "A-Connecticut-Yankee-in-King-Arthur-s-Court_86", 86),
        ("The_Prince_and_the_Pauper", "The-Prince-and-the-Pauper_1837", 1837),
        ("Life_on_the_Mississippi", "Life-on-the-Mississippi_245", 245),
        ("The_Innocents_Abroad", "The-Innocents-Abroad_3176", 3176),
        ("Roughing_It", "Roughing-It_377", 377),
        ("The_Gilded_Age", "The-Gilded-Age_11942", 11942),
    ],
    "HG_Wells": [
        ("The_Time_Machine", "The-Time-Machine_35", 35),
        ("The_War_of_the_Worlds", "The-War-of-the-Worlds_36", 36),
        ("The_Invisible_Man", "The-Invisible-Man_5230", 5230),
        ("The_Island_of_Doctor_Moreau", "The-Island-of-Doctor-Moreau_159", 159),
        ("The_First_Men_in_the_Moon", "The-First-Men-in-the-Moon_1013", 1013),
        ("When_the_Sleeper_Wakes", "When-the-Sleeper-Wakes_775", 775),
        ("The_Food_of_the_Gods", "The-Food-of-the-Gods-and-How-It-Came-to-Earth_11505", 11505),
        ("Kipps", "Kipps_11777", 11777),
    ],
    "Oscar_Wilde": [
        ("The_Picture_of_Dorian_Gray", "The-Picture-of-Dorian-Gray_174", 174),
        ("The_Importance_of_Being_Earnest", "The-Importance-of-Being-Earnest_844", 844),
        ("Lady_Windermeres_Fan", "Lady-Windermere-s-Fan_790", 790),
        ("An_Ideal_Husband", "An-Ideal-Husband_882", 882),
        ("A_Woman_of_No_Importance", "A-Woman-of-No-Importance_854", 854),
        ("De_Profundis", "De-Profundis_921", 921),
        ("The_Canterville_Ghost", "The-Canterville-Ghost_14522", 14522),
        ("The_Happy_Prince_and_Other_Tales", "The-Happy-Prince-and-Other-Tales_902", 902),
    ],
}


def download_book(author: str, book_name: str, repo: str, gid: int, raw_dir: Path) -> Tuple[str, str, bool, int, str]:
    """Download single book using curl across possible branch and filename conventions."""
    out_dir = raw_dir / author
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{book_name}.txt"

    # Possible candidate URLs on GITenberg
    urls = [
        f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{gid}-0.txt",
        f"https://raw.githubusercontent.com/GITenberg/{repo}/master/{gid}.txt",
        f"https://raw.githubusercontent.com/GITenberg/{repo}/gh-pages/{gid}-0.txt",
        f"https://raw.githubusercontent.com/GITenberg/{repo}/gh-pages/{gid}.txt",
    ]

    for url in urls:
        cmd = ["curl", "-s", "-f", "-L", "--connect-timeout", "10", "--max-time", "25", url, "-o", str(out_file)]
        res = subprocess.run(cmd)
        if res.returncode == 0 and out_file.exists() and out_file.stat().st_size > 10000:
            words = len(out_file.read_text(encoding="utf-8", errors="replace").split())
            return (author, book_name, True, words, url)

    return (author, book_name, False, 0, "Failed to download from GITenberg")


def main() -> None:
    raw_dir = Path("data/raw")
    print("=" * 80)
    print("GERÇEK 56 ROMANIN İNDİRİLMESİ (GITENBERG)")
    print("=" * 80)

    tasks = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for author, books in GITENBERG_CATALOG.items():
            for book_name, repo, gid in books:
                tasks.append(executor.submit(download_book, author, book_name, repo, gid, raw_dir))

        for future in as_completed(tasks):
            author, book_name, success, words, info = future.result()
            if success:
                print(f"✓ [{author}] {book_name:<32} -> {words:>7} kelime indirildi.")
            else:
                print(f"✗ [{author}] {book_name:<32} -> İndirilemedi ({info})")

    print("\nİndirme işlemi tamamlandı.")


if __name__ == "__main__":
    main()
