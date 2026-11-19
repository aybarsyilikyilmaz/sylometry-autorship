"""
Batch prediction test script.
Tests 7 distinct literary snippets against the pre-trained stylometric models.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict_snippet

TEST_CASES = [
    (
        "Jane Austen (Romantik / Toplumsal Üslup)",
        "Her feelings were always guided by a delicate sense of civil propriety and affection. "
        "A marriage of true esteem was universally regarded as the highest fortune a family could hope for, "
        "and her sisters shared in the delightful manners of the society.",
    ),
    (
        "Mark Twain (Amerikan Mizahı & Nehir Diyalekti)",
        "Well, I reckon Tom and Huck ain going to worry none about that trouble. "
        "We took the raft down the river when it was mighty dark, and the steamboat came rushing past "
        "with plenty of noise and smoke.",
    ),
    (
        "Arthur Conan Doyle (Dedektiflik & Gözlem)",
        "Holmes held up the telegram with a quiet smile of deduction. It was a singular crime, "
        "and the mystery of the cab on Baker Street revealed clear footprints and unmistakable evidence "
        "which the inspector had completely overlooked.",
    ),
    (
        "H.G. Wells (Bilim Kurgu & Analitik Anlatım)",
        "The apparatus was constructed of brass and ivory, with a delicate metallic mechanism designed "
        "to travel through the fourth dimension. The observer examined the strange velocity of the machine "
        "with scientific curiosity about the future.",
    ),
    (
        "Oscar Wilde (Aforizma & Estetizm)",
        "There is no such thing as a moral or an immoral book. Books are well written, or badly written. "
        "The artist is the creator of beautiful things, and the only way to get rid of temptation is to yield to it. "
        "Youth and beauty are the supreme virtues of luxury.",
    ),
    (
        "Charles Dickens (Viktorya Dönemi Londra & Sis)",
        "Fog everywhere in the city of London; fog rolling down the river and choking the narrow streets "
        "near the court of Chancery. The gentleman walked past the dark pavement while the heavy clock struck "
        "the hour of night over the weary crowd.",
    ),
    (
        "Charlotte Brontë (Gotik & Romantik Tutku)",
        "The wind howled with wild fury across the lonely moors, and my soul was shaken by a dark fever "
        "of the spirit. In the deep solitude of the silent night, my heart cried out against the strange master "
        "who held my secret thoughts.",
    ),
]

print("=" * 80)
print("7 FARKLI YAZAR METNİ İLE ÇOKLU TAHMİN TESTİ (BATCH PREDICTION)")
print("=" * 80)

for label, snippet in TEST_CASES:
    print(f"\n>>> TEST: {label}")
    print(f"Metin: \"{snippet[:100]}...\"")
    predict_snippet(snippet)
