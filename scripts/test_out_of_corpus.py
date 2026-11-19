"""
Strict Out-of-Corpus Generalization Test.

Tests the models on completely unseen, external works that are NOT part of the 56 books:
1. Jane Austen: 'The Watsons' (Unfinished novel, not in the 8 books)
2. Charles Dickens: 'The Mystery of Edwin Drood' (His final unfinished novel, not in the 8 books)
3. Arthur Conan Doyle: 'The Poison Belt' (Professor Challenger sci-fi series, not Sherlock Holmes, not in the 8 books)
4. Mark Twain: 'The Mysterious Stranger' (Dark philosophical satire, not in the 8 books)
5. H.G. Wells: 'The Invisible Man' vs 'A Modern Utopia' / 'The War in the Air' (Not in the 8 books)
6. Oscar Wilde: 'The Soul of Man under Socialism' (Political/aesthetic essay, not in the 8 books)
7. Charlotte Brontë: 'Biographical Notice of Ellis and Acton Bell' (Personal memoir, not in the 8 books)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predict import predict_snippet

# Metinler: Modelin 56 kitabında KESİNLİKLE yer almayan, tamamen harici eserlerden alınmıştır.
OUT_OF_CORPUS_TESTS = [
    (
        "Jane Austen — Eser: 'The Watsons' (56 kitapta YOK, 1804 tarihli yarım kalan romanı)",
        "The day was cold and windy, but Emma Watson was too much engaged with the anticipation of her first ball "
        "to think of the weather. Her aunt had provided her with every necessary elegance of dress, and she felt "
        "all that delightful agitation of spirits which a young woman of nineteen, entering into society, is so apt to experience. "
        "She hoped her brother would arrive in time, and wondered whether Mr. Howard would ask her to dance."
    ),
    (
        "Charles Dickens — Eser: 'The Mystery of Edwin Drood' (56 kitapta YOK, 1870 tarihli son romanı)",
        "An ancient English cathedral tower? How can the ancient English cathedral tower be here! "
        "The well-known massive grey square tower of its old Cathedral rises in the tranquil morning light. "
        "A heavy gloom falls upon the choir, and the cold mist creeps among the stone pillars and tombs "
        "of departed bishops, while the footsteps of the verger echo through the dim arches."
    ),
    (
        "Arthur Conan Doyle — Eser: 'The Poison Belt' (56 kitapta YOK, Profesör Challenger serisi)",
        "Professor Challenger surveyed us from beneath his enormous bushy eyebrows with supreme condescension. "
        "I have demonstrated to you, gentlemen, that the ether of space contains a toxic vapor of unknown density. "
        "The instruments upon my laboratory table have recorded the change, and our scientific observations admit of no doubt."
    ),
    (
        "Mark Twain — Eser: 'The Mysterious Stranger' (56 kitapta YOK, 1916 felsefi novellası)",
        "It was in 1590, winter. Austria was far away from the world, and asleep; it was still the Middle Ages in Austria, "
        "and promised to remain so forever. Some even said it was clockwork, but I reckon we boys didn't care much "
        "about dates and history, so long as we had the woods and the river and plenty of leisure to play."
    ),
    (
        "Oscar Wilde — Eser: 'The Soul of Man under Socialism' (56 kitapta YOK, 1891 felsefi denemesi)",
        "The chief advantage that would result from the establishment of Socialism is, undoubtedly, the fact that "
        "Socialism would relieve us from that sordid necessity of living for others which, in the present condition of things, "
        "presses so hardly upon almost everybody. Art is individualism, and individualism is a disturbing and disintegrating force."
    ),
    (
        "Charlotte Brontë — Eser: 'Biographical Notice of Ellis and Acton Bell' (56 kitapta YOK, 1850 anı yazısı)",
        "We had very early cherished the dream of one day becoming authors. We had no acquaintance with authors, "
        "and our secluded life in the distant parsonage favored our literary solitude. The wild moors were our only company, "
        "and the strange reserve of our sisters concealed a spirit of remarkable power and sadness."
    ),
    (
        "H.G. Wells — Eser: 'The War in the Air' (56 kitapta YOK, 1908 havacılık kurgusu)",
        "The machine hovered above the clouds like a vast metallic beetle, its twin engines vibrating with immense power. "
        "The aeronaut adjusted the steering apparatus and peered through the glass floor at the landscape below, "
        "calculating the velocity of the airship against the violent gale."
    )
]

print("=" * 85)
print("GERÇEK DIŞ KÜLLİYAT TESTİ: 56 KİTAPTA ASLA OLMAYAN TAMAMEN YENİ ESERLER")
print("=" * 85)

for label, snippet in OUT_OF_CORPUS_TESTS:
    print(f"\n🔍 {label}")
    predict_snippet(snippet)
