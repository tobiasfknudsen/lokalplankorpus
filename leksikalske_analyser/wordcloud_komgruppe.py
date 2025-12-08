import sqlite3
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# --------------------------------------------------------
# 0) Find repoets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Stier
# --------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
OUTPUT_DIR = os.path.join(ROOT_DIR, "leksikalske_analyser", "wordclouds", "kommunegruppe")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# 2) Indstillinger
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
COLUMN_TEXT = "tekst_renset_lemmma"
COLUMN_YEAR = "aar"
COLUMN_KOMTYPE = "komtype_5kat"  

YEAR_CUTOFF = 1975 # kun til farveskala
batch_size = 1000

KOMTYPE_LIST = [
    "Landkommuner",
    "Oplandskommuner",
    "Provinsbykommuner",
    "Storbykommuner",
    "Hovedstadskommuner"
]

# Stopord fra tekstfil i repo
STOPWORDS_FILE = os.path.join(ROOT_DIR, "stopord.txt")
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOP_WORDS = {w.strip() for w in f if w.strip()}

# --------------------------------------------------------
# 3) Hent & indlæs data
# --------------------------------------------------------
print("Henter data...")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {COLUMN_TEXT} IS NOT NULL")
total_rows = cur.fetchone()[0]

# Data struktur
word_counts_by_type = defaultdict(lambda: defaultdict(Counter))
total_words_by_type = defaultdict(lambda: Counter())

for offset in tqdm(range(0, total_rows, batch_size), desc="Indlæser rækker"):
    cur.execute(f"""
        SELECT {COLUMN_TEXT}, {COLUMN_YEAR}, {COLUMN_KOMTYPE}
        FROM {TABLE_NAME}
        WHERE {COLUMN_TEXT} IS NOT NULL AND {COLUMN_YEAR} IS NOT NULL
        LIMIT {batch_size} OFFSET {offset}
    """)
    rows = cur.fetchall()

    for text, year, komtype in rows:
        if komtype not in KOMTYPE_LIST:
            continue

        words = [w.lower() for w in text.split() if w.lower() not in STOP_WORDS]

        total_words_by_type[komtype][year] += len(words)
        word_counts_by_type[komtype][year].update(words)

conn.close()

# --------------------------------------------------------
# 4) Funktion til at generere wordcloud for én komtype
# --------------------------------------------------------
def generate_wordcloud_for_komtype(komtype):

    # Samlet ordliste
    global_counter = Counter()
    for year in word_counts_by_type[komtype]:
        global_counter.update(word_counts_by_type[komtype][year])

    # Mean-year beregning
    years_used = sorted([y for y in total_words_by_type[komtype] if y >= YEAR_CUTOFF])
    mean_years = {}

    for word in global_counter:
        weighted_sum = 0
        total_weight = 0
        for year in years_used:
            total_words = total_words_by_type[komtype][year]
            if total_words == 0:
                continue
            norm_freq = word_counts_by_type[komtype][year][word] / total_words * 1000
            weighted_sum += year * norm_freq
            total_weight += norm_freq
        mean_years[word] = weighted_sum / total_weight if total_weight > 0 else None

    valid_words = {w: c for w, c in global_counter.items() if mean_years.get(w) is not None}

    if len(valid_words) == 0:
        print(f"Ingen ord kunne bruges for {komtype}")
        return

    # Farveskala
    COLOR_OLD = "#FF1100"
    COLOR_MID = "#939393"
    COLOR_NEW = "#00FF04"

    min_year = min(mean_years[w] for w in valid_words)
    max_year = max(mean_years[w] for w in valid_words)

    def hex_to_rgb(hex_color):
        return np.array(matplotlib.colors.to_rgb(hex_color))

    RGB_OLD = hex_to_rgb(COLOR_OLD)
    RGB_MID = hex_to_rgb(COLOR_MID)
    RGB_NEW = hex_to_rgb(COLOR_NEW)

    def boost_t(t):
        t = (t - 0.5) * 1.4 + 0.5
        return max(0, min(1, t))

    def color_from_year(mean_year):
        t = (mean_year - min_year) / (max_year - min_year)
        t = boost_t(t)
        if t < 0.5:
            ratio = t / 0.5
            rgb = RGB_OLD * (1 - ratio) + RGB_MID * ratio
        else:
            ratio = (t - 0.5) / 0.5
            rgb = RGB_MID * (1 - ratio) + RGB_NEW * ratio
        return matplotlib.colors.to_hex(rgb)

    def color_func(word, *args, **kwargs):
        return color_from_year(mean_years[word])

    # WordCloud
    print(f"Genererer WordCloud for {komtype}...")

    wordcloud = WordCloud(
        width=2380,
        height=1000,
        background_color="white",
        color_func=color_func,
        relative_scaling=1,
        font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        prefer_horizontal=1.0,
        margin=10,
        max_words=100
    ).generate_from_frequencies(valid_words)

    fig = plt.figure(figsize=(23.8, 10), dpi=100)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Gem fil med små bogstaver
    pdf_path = os.path.join(OUTPUT_DIR, f"wordcloud_slet_{komtype.lower()}.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"WordCloud gemt: {pdf_path}")

# --------------------------------------------------------
# 5) Generér wordclouds for alle komtyper
# --------------------------------------------------------
for komtype in KOMTYPE_LIST:
    generate_wordcloud_for_komtype(komtype)

print("Alle wordclouds er genereret!")
