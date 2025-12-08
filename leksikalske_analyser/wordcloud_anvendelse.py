import sqlite3
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# -----------------------------------
# 0) Find repoets rodmappe
# -----------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -----------------------------------
# 1) Indstillinger
# -----------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
TABLE_NAME = "lokalplaner"
COLUMN_TEXT = "tekst_renset_lemmma"
COLUMN_YEAR = "aar"
COLUMN_ANV = "anvendelse"

STOPWORDS_FILE = os.path.join(ROOT_DIR, "stopord.txt")  # stopord-fil

YEAR_CUTOFF = 1975 # kun til farveskala
batch_size = 1000

# Dine labels
anvendelse_labels = {
    "11_boligomraade": "Boligomraade",
    "21_blandetboligogerhverv": "Blandet_bolig_og_erhverv",
    "31_erhvervsomraade": "Erhvervsomraade",
    "41_centeromraade": "Centeromraade",
    "51_rekreativtomraade": "Rekreativt_omraade",
    "61_sommerhusomraade": "Sommerhusomraade",
    "71_omraadetiloffentligeformaal": "Omraade_til_offentlige_formaal",
    "81_tekniskeanlaeg": "Tekniske_anlaeg",
    "91_landomraade": "Landomraade",
    "96_andet": "Andet"
}

ANV_LIST = list(anvendelse_labels.keys())

# -----------------------------------
# 2) Indlæs stopord fra fil
# -----------------------------------
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOP_WORDS = {line.strip().lower() for line in f if line.strip()}

# -----------------------------------
# 3) Opret mappe til output
# -----------------------------------
BASE_DIR = os.path.join(ROOT_DIR, "leksikalske_analyser", "wordclouds")
ANV_DIR = os.path.join(BASE_DIR, "anvendelse")
os.makedirs(ANV_DIR, exist_ok=True)

# -----------------------------------
# 4) Hent & indlæs data
# -----------------------------------
print("Henter data...")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {COLUMN_TEXT} IS NOT NULL")
total_rows = cur.fetchone()[0]

word_counts_by_type = defaultdict(lambda: defaultdict(Counter))
total_words_by_type = defaultdict(lambda: Counter())

for offset in tqdm(range(0, total_rows, batch_size), desc="Indlæser rækker"):
    cur.execute(f"""
        SELECT {COLUMN_TEXT}, {COLUMN_YEAR}, {COLUMN_ANV}
        FROM {TABLE_NAME}
        WHERE {COLUMN_TEXT} IS NOT NULL AND {COLUMN_YEAR} IS NOT NULL
        LIMIT {batch_size} OFFSET {offset}
    """)
    rows = cur.fetchall()

    for text, year, anv in rows:
        if anv not in ANV_LIST:
            continue

        words = [w.lower() for w in text.split() if w.lower() not in STOP_WORDS]

        total_words_by_type[anv][year] += len(words)
        word_counts_by_type[anv][year].update(words)

conn.close()

# -----------------------------------
# 5) Funktion: Beregn mean-year og wordcloud for én anvendelse
# -----------------------------------
def generate_wordcloud_for_anvendelse(anv_key):

    anv_label = anvendelse_labels[anv_key]

    global_counter = Counter()
    for year in word_counts_by_type[anv_key]:
        global_counter.update(word_counts_by_type[anv_key][year])

    years_used = sorted([y for y in total_words_by_type[anv_key] if y >= YEAR_CUTOFF])
    mean_years = {}

    for word in global_counter:
        weighted_sum = 0
        total_weight = 0
        for year in years_used:
            total_words = total_words_by_type[anv_key][year]
            if total_words == 0:
                continue
            norm_freq = word_counts_by_type[anv_key][year][word] / total_words * 1000
            weighted_sum += year * norm_freq
            total_weight += norm_freq
        mean_years[word] = weighted_sum / total_weight if total_weight > 0 else None

    valid_words = {w: c for w, c in global_counter.items() if mean_years.get(w) is not None}

    if len(valid_words) == 0:
        print(f"Ingen ord kunne bruges for {anv_label}")
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

    print(f"Genererer WordCloud for {anv_label}...")

    wordcloud = WordCloud(
        width=2380,
        height=1000,
        background_color="white",
        color_func=color_func,
        relative_scaling=1,
        font_path=None,  # matplotlib default font fungerer på Windows/Linux
        prefer_horizontal=1.0,
        margin=10,
        max_words=100
    ).generate_from_frequencies(valid_words)

    fig = plt.figure(figsize=(23.8, 10), dpi=100)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    pdf_path = os.path.join(ANV_DIR, f"wordcloud_{anv_label.lower()}.pdf")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"WordCloud gemt: {pdf_path}")

# -----------------------------------
# 6) Generér alle wordclouds
# -----------------------------------
for anv in ANV_LIST:
    generate_wordcloud_for_anvendelse(anv)

print("Alle wordclouds er genereret!")
