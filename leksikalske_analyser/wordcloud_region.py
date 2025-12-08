import sqlite3
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

# --------------------------------------------------------
# 0) Find repoets rodmappe (som på GitHub)
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Stier
# --------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
STOPWORDS_FILE = os.path.join(ROOT_DIR, "stopord.txt")
MASK_FOLDER = os.path.join(ROOT_DIR, "leksikalske_analyser", "masks")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "leksikalske_analyser", "wordclouds", "region")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------------
# 2) Indstillinger
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
COLUMN_TEXT = "tekst_renset_lemmma"
COLUMN_KOMNR = "komnr"

BATCH_SIZE = 100
MAX_WORDS = 250
MAX_FONT_GLOBAL = 100  # globalt største ord får 100 px

# Farver til figurer
REGION_SETTINGS = {
    "1081": {"name": "Nordjylland", "color": "#cd3323", "mask": "nordjylland.png"},
    "1082": {"name": "Midtjylland", "color": "#024e69", "mask": "midtjylland.png"},
    "1083": {"name": "Syddanmark", "color": "#54985d", "mask": "syddanmark.png"},
    "1084_1085": {"name": "Oestdanmark", "color": "#4A2055", "mask": "oestdanmark.png", "merge": ["1084", "1085"]},
}

# --------------------------------------------------------
# 3) Indlæs stopord
# --------------------------------------------------------
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOP_WORDS = set([line.strip().lower() for line in f])

# --------------------------------------------------------
# 4) Konvertering fra kommunekode til regionkode
# --------------------------------------------------------
kom_region_mapping = {
    "101":"1084","173":"1084","621":"1083","167":"1084","607":"1083","450":"1083",
    "756":"1082","306":"1085","820":"1081","350":"1085","706":"1082","230":"1084",
    "340":"1085","316":"1085","707":"1082","269":"1085","846":"1081","575":"1083",
    "550":"1083","183":"1084","210":"1084","370":"1085","169":"1084","240":"1084",
    "265":"1085","580":"1083","185":"1084","157":"1084","223":"1084","573":"1083",
    "671":"1082","360":"1085","540":"1083","400":"1084","482":"1083","326":"1085",
    "710":"1082","746":"1082","849":"1081","151":"1084","461":"1083","161":"1084",
    "155":"1084","430":"1083","766":"1082","751":"1082","219":"1084","259":"1085",
    "320":"1085","163":"1084","217":"1084","165":"1084","201":"1084","159":"1084",
    "329":"1085","250":"1084","336":"1085","376":"1085","270":"1084","187":"1084",
    "253":"1085","791":"1082","779":"1082","840":"1081","615":"1082","730":"1082",
    "175":"1084","410":"1083","727":"1082","851":"1081","630":"1083","190":"1084",
    "661":"1082","420":"1083","760":"1082","825":"1081","561":"1083","773":"1081",
    "260":"1084","440":"1083","492":"1083","740":"1082","657":"1082","787":"1081",
    "153":"1084","390":"1085","330":"1085","147":"1084","480":"1083","510":"1083",
    "479":"1083","810":"1081","813":"1081","860":"1081","530":"1083","741":"1082",
    "563":"1083","411":"1084","665":"1082"
}

# --------------------------------------------------------
# 5) Indsaml og normaliser ord for hver region
# --------------------------------------------------------
all_norm_counts = {}  # {REGION_CODE: {word: norm_freq}}

for REGION_CODE, settings in REGION_SETTINGS.items():
    print(f"Bearbejder region {REGION_CODE} ({settings['name']}) ...")

    # ✔ Håndterer merged region Øst
    if "merge" in settings:
        merge_codes = settings["merge"]
        kommuner_i_region = [k for k, r in kom_region_mapping.items() if r in merge_codes]
    else:
        kommuner_i_region = [k for k, r in kom_region_mapping.items() if r == REGION_CODE]

    if not kommuner_i_region:
        print(f"  Ingen kommuner fundet for region {REGION_CODE}, springer over.")
        continue

    word_counts = Counter()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Find antal rækker
    total_rows_query = f"""
        SELECT COUNT(*) FROM {TABLE_NAME}
        WHERE {COLUMN_TEXT} IS NOT NULL
          AND {COLUMN_KOMNR} IN ({','.join('?' for _ in kommuner_i_region)})
    """
    cur.execute(total_rows_query, kommuner_i_region)
    total_rows = cur.fetchone()[0]
    print(f"  Antal rækker i region {REGION_CODE}: {total_rows}")

    # Læs i batches
    for offset in tqdm(range(0, total_rows, BATCH_SIZE),
                       desc=f"  Læser tekst (region {REGION_CODE})"):

        query = f"""
            SELECT {COLUMN_TEXT}
            FROM {TABLE_NAME}
            WHERE {COLUMN_TEXT} IS NOT NULL
              AND {COLUMN_KOMNR} IN ({','.join('?' for _ in kommuner_i_region)})
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """
        cur.execute(query, kommuner_i_region)
        rows = cur.fetchall()

        for (text,) in rows:
            words = [
                w.lower()
                for w in str(text).split()
                if w and w.lower() not in STOP_WORDS
            ]
            word_counts.update(words)

    conn.close()

    total_words = sum(word_counts.values())
    norm_counts = {word: (count / total_words) * 100.0 for word, count in word_counts.items()} if total_words > 0 else {}
    all_norm_counts[REGION_CODE] = norm_counts
    print(f"  Unikke ord i region {REGION_CODE}: {len(norm_counts)}")

# --------------------------------------------------------
# 6) Find global max
# --------------------------------------------------------
global_max = max((max(counts.values()) for counts in all_norm_counts.values() if counts), default=0)
if global_max == 0:
    raise ValueError("Global max frekvens er 0 - ingen ord fundet.")
print(f"\nGlobal max frekvens over alle regioner: {global_max:.6f}")

# --------------------------------------------------------
# 7) Generer WordCloud for hver region
# --------------------------------------------------------
for REGION_CODE, settings in REGION_SETTINGS.items():

    norm_counts = all_norm_counts.get(REGION_CODE, {})
    if not norm_counts:
        print(f"Region {REGION_CODE} har ingen ord, springer over.")
        continue

    print(f"\nGenererer wordcloud for region {REGION_CODE} ({settings['name']}) ...")

    mask_path = os.path.join(MASK_FOLDER, settings["mask"])
    if not os.path.exists(mask_path):
        print(f"  Maskefil findes ikke: {mask_path}, springer over.")
        continue

    mask_image = np.array(Image.open(mask_path))

    def color_func(word, *args, **kwargs):
        return settings["color"]

    region_max = max(norm_counts.values())
    region_max_font = MAX_FONT_GLOBAL * (region_max / global_max)
    region_max_font = max(5, int(region_max_font))

    wordcloud = WordCloud(
        width=2380,
        height=1000,
        background_color=None,
        max_words=MAX_WORDS,
        stopwords=STOP_WORDS,
        mask=mask_image,
        contour_width=0,
        relative_scaling=1,
        font_path=r"C:\Windows\Fonts\arial.ttf",
        prefer_horizontal=0.9,
        margin=0,
        mode="RGBA",
        font_step=0,
        color_func=color_func,
        max_font_size=region_max_font
    ).generate_from_frequencies(norm_counts)

    fig = plt.figure(figsize=(23.8, 10), dpi=100)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)

    # Gem fil med regionnavn i små bogstaver
    region_name_lower = settings["name"].replace(" ", "_").lower()
    output_path = os.path.join(
        OUTPUT_FOLDER,
        f"wordcloud_{region_name_lower}.png"
    )

    fig.savefig(output_path, format="png", dpi=1200,
                bbox_inches="tight", transparent=True)
    plt.close()

    print(f"  Gemte: {output_path}")
