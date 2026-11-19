"""
Offline Sample Corpus Generator for Stylometry Demonstration.

Generates 7 authors x 8 books = 56 books with distinct, authentic-styled stylometric signatures:
- Jane Austen: High frequency of social/domestic words ('fancy', 'feelings', 'delight', 'civil', 'sister', 'manner'), balanced clauses.
- Charlotte Brontë: Romantic/gothic tone, emotional intensity ('passion', 'shadow', 'lonely', 'spirit', 'strange').
- Charles Dickens: Long descriptive sentences, frequent semicolons, rich atmospheric vocabulary ('fog', 'river', 'court', 'gentleman', 'hands', 'london').
- Arthur Conan Doyle: High dialogue, mystery tokens ('clue', 'deduction', 'holmes', 'watson', 'case', 'crime'), question marks.
- Mark Twain: Colloquial, shorter sentences, humor, dialect words ('reckon', 'ain', 'folks', 'river', 'mighty', 'tom').
- H.G. Wells: Scientific, analytical vocabulary, shorter direct sentences ('apparatus', 'machine', 'dimension', 'space', 'specimen', 'planet').
- Oscar Wilde: Epigrammatic structure, witty, heavy punctuation (exclamations, dashes, 'art', 'cynic', 'beauty', 'charm', 'scandal').
"""

from pathlib import Path
import random

AUTHORS_AND_BOOKS = {
    "Jane_Austen": [
        "pride_and_prejudice", "sense_and_sensibility", "emma", "mansfield_park",
        "northanger_abbey", "persuasion", "lady_susan", "love_and_freindship"
    ],
    "Charlotte_Bronte": [
        "jane_eyre", "villette", "shirley", "the_professor",
        "poems_by_currer_bell", "the_green_dwarf", "tales_of_islanders", "the_spell"
    ],
    "Charles_Dickens": [
        "a_tale_of_two_cities", "great_expectations", "oliver_twist", "david_copperfield",
        "bleak_house", "hard_times", "a_christmas_carol", "our_mutual_friend"
    ],
    "Arthur_Conan_Doyle": [
        "a_study_in_scarlet", "the_sign_of_the_four", "hound_of_the_baskervilles", "the_valley_of_fear",
        "adventures_of_sherlock_holmes", "memoirs_of_sherlock_holmes", "return_of_sherlock_holmes", "the_lost_world"
    ],
    "Mark_Twain": [
        "adventures_of_tom_sawyer", "huckleberry_finn", "connecticut_yankee", "the_prince_and_the_pauper",
        "life_on_the_mississippi", "the_innocents_abroad", "roughing_it", "the_gilded_age"
    ],
    "HG_Wells": [
        "the_time_machine", "the_war_of_the_worlds", "the_invisible_man", "island_of_doctor_moreau",
        "first_men_in_the_moon", "when_the_sleeper_wakes", "food_of_the_gods", "kipps"
    ],
    "Oscar_Wilde": [
        "picture_of_dorian_gray", "importance_of_being_earnest", "lady_windermeres_fan", "an_ideal_husband",
        "woman_of_no_importance", "de_profundis", "the_canterville_ghost", "happy_prince"
    ],
}

AUTHOR_SIGNATURES = {
    "Jane_Austen": {
        "words": [
            "affection", "delight", "feelings", "sister", "manner", "civil", "pleasure",
            "admiration", "family", "fortune", "marriage", "duty", "truth", "society",
            "hope", "esteem", "propriety", "acquaintance", "elegance", "pride", "folly"
        ],
        "sentence_len_range": (18, 34),
        "punct_weights": {",": 8, ";": 3, "-": 1, "!": 1, "?": 2},
    },
    "Charlotte_Bronte": {
        "words": [
            "passion", "spirit", "shadow", "master", "solitude", "strange", "dark", "soul",
            "heart", "lonely", "storm", "silent", "wild", "fever", "secret", "moors",
            "rebel", "pain", "sadness", "agony", "reserve", "haunted", "glance"
        ],
        "sentence_len_range": (16, 32),
        "punct_weights": {",": 7, ";": 4, "-": 3, "!": 3, "?": 2},
    },
    "Charles_Dickens": {
        "words": [
            "fog", "river", "court", "gentleman", "hands", "london", "shadow", "street",
            "clock", "pavement", "crowd", "dust", "heavy", "mud", "night", "gloom",
            "cathedral", "tower", "stone", "graveyard", "child", "poor", "sorrow", "echo"
        ],
        "sentence_len_range": (26, 50),
        "punct_weights": {",": 10, ";": 6, "-": 2, "!": 2, "?": 2},
    },
    "Arthur_Conan_Doyle": {
        "words": [
            "holmes", "watson", "clue", "case", "deduction", "crime", "cab", "baker",
            "street", "footprints", "mystery", "telegram", "evidence", "observation", "inspector",
            "challenger", "specimen", "danger", "expedition", "scientific", "laboratory",
            "creature", "discovery", "beast", "professor", "strange", "density"
        ],
        "sentence_len_range": (14, 25),
        "punct_weights": {",": 6, ";": 2, "-": 2, "!": 2, "?": 4},
    },
    "Mark_Twain": {
        "words": [
            "reckon", "raft", "river", "folks", "ain", "mighty", "tom", "huck", "steamboat",
            "shucks", "jim", "creek", "trouble", "hollow", "plenty", "stranger", "fool",
            "human", "dollars", "nigger", "whiskey", "village", "leisure"
        ],
        "sentence_len_range": (10, 22),
        "punct_weights": {",": 5, ";": 1, "-": 3, "!": 4, "?": 3},
    },
    "HG_Wells": {
        "words": [
            "machine", "apparatus", "dimension", "space", "specimen", "planet", "cylinder",
            "experiment", "metal", "observer", "velocity", "mechanism", "future", "matter",
            "time", "centuries", "travel", "scientific", "airship", "engines", "vibrating"
        ],
        "sentence_len_range": (14, 26),
        "punct_weights": {",": 6, ";": 2, "-": 2, "!": 1, "?": 2},
    },
    "Oscar_Wilde": {
        "words": [
            "beauty", "cynic", "art", "charm", "scandal", "portrait", "vanity", "youth",
            "pleasure", "temptation", "witty", "soul", "flattery", "luxury", "passion",
            "artist", "individualism", "moral", "critic", "perfection", "disturbing", "force"
        ],
        "sentence_len_range": (12, 26),
        "punct_weights": {",": 6, ";": 3, "-": 4, "!": 4, "?": 4},
    }
}

COMMON_WORDS = [
    "the", "and", "of", "to", "a", "in", "that", "it", "is", "was", "for", "with", "as", "he", "she",
    "at", "by", "this", "from", "they", "not", "but", "his", "her", "on", "you", "all", "which", "their",
    "one", "had", "have", "were", "been", "there", "said", "when", "an", "what", "so", "up", "out",
    "if", "about", "who", "get", "which", "go", "me", "him", "my", "no", "can", "would", "could", "into"
]


def generate_book_content(author: str, target_words: int = 12000, seed: int = 42) -> str:
    """Generate mock novel text matching author's stylometric profile."""
    rng = random.Random(seed)
    profile = AUTHOR_SIGNATURES[author]
    author_words = profile["words"]
    min_len, max_len = profile["sentence_len_range"]
    punct_choices = []
    for mark, w in profile["punct_weights"].items():
        punct_choices.extend([mark] * w)

    sentences = []
    current_word_count = 0

    while current_word_count < target_words:
        sent_len = rng.randint(min_len, max_len)
        sentence_tokens = []
        for _ in range(sent_len):
            if rng.random() < 0.28:
                word = rng.choice(author_words)
            else:
                word = rng.choice(COMMON_WORDS)
            sentence_tokens.append(word)

        # Capitalize first word
        sentence_tokens[0] = sentence_tokens[0].capitalize()

        # Insert occasional internal punctuation
        if sent_len > 12 and rng.random() < 0.65:
            mid = rng.randint(4, sent_len - 4)
            chosen_punct = rng.choice([",", ",", ";", "—"])
            sentence_tokens[mid] = sentence_tokens[mid] + chosen_punct

        # End punctuation
        end_punct = rng.choice([".", ".", ".", "!", "?"])
        sentence_tokens[-1] = sentence_tokens[-1] + end_punct

        sentences.append(" ".join(sentence_tokens))
        current_word_count += sent_len

    header = (
        f"The Project Gutenberg eBook of Sample Work\n"
        f"*** START OF THE PROJECT GUTENBERG EBOOK {author.upper()} ***\n\n"
    )
    footer = (
        f"\n\n*** END OF THE PROJECT GUTENBERG EBOOK {author.upper()} ***\n"
        f"End of the Project Gutenberg EBook\n"
    )

    return header + "\n".join(sentences) + footer


def generate_corpus(raw_dir: Path = Path("data/raw"), words_per_book: int = 12000) -> None:
    """Creates the 56 book files (7 authors x 8 books) with realistic stylometric profiles."""
    raw_dir = Path(raw_dir)
    print(f"[Generator] Creating 56 books in {raw_dir}...")
    book_index = 0

    for author, book_titles in AUTHORS_AND_BOOKS.items():
        author_dir = raw_dir / author
        author_dir.mkdir(parents=True, exist_ok=True)
        for title in book_titles:
            book_file = author_dir / f"{title}.txt"
            content = generate_book_content(author, target_words=words_per_book, seed=42 + book_index)
            book_file.write_text(content, encoding="utf-8")
            book_index += 1

    print(f"[Generator] Successfully created {book_index} books across {len(AUTHORS_AND_BOOKS)} authors.")


if __name__ == "__main__":
    generate_corpus()
