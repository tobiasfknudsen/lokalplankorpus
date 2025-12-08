# -*- coding: utf-8 -*-
"""
Word2Vec ud fra årstal
"""

import sqlite3
import re
from collections import Counter
import numpy as np
import pandas as pd
import os
from gensim.models import Word2Vec
from tqdm import tqdm

# --------------------------------------------------------
# 0) Find repoets rodmappe
# --------------------------------------------------------
# Repoets rodmappe
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# model-output mappe
BASE_DIR = os.path.join(ROOT_DIR, "semantiske_analyser" ,"word2vec_modeller", "aarstal")
os.makedirs(BASE_DIR, exist_ok=True)

# databasefil i repoets rod
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")

# databasefil i repoets rod
STOPWORDS_FILE = os.path.join(ROOT_DIR, "stopord.txt")

# Indlæs stopord
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOPORD = {w.strip() for w in f if w.strip()}


# --------------------------------------------------------
# 1) Konfiguration
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
COLUMN_TEXT = "tekst_renset_lemmma"
COLUMN_YEAR = "aar"

YEAR_INTERVALS = [
    (1975, 2000),
    (2000, 2026)
]

# Indstillinger for visualisering
MIN_COUNT = 50
VECTOR_SIZE = 100
WINDOW = 5
EPOCHS = 10

# --------------------------------------------------------
# 2) Indlæs stopord
# --------------------------------------------------------
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOPORD = {w.strip() for w in f if w.strip()}

# --------------------------------------------------------
# 3) Funktion til tokenisering
# --------------------------------------------------------
def tokenize(txt):
    toks = re.findall(r"[a-zA-ZæøåÆØÅ]+", str(txt).lower())
    toks = [t for t in toks if len(t) > 2 and t not in STOPORD]
    return toks

# --------------------------------------------------------
# 4) Hent tekst + årstal fra database og træn modeller per interval
# --------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

for start_year, end_year in YEAR_INTERVALS:
    print(f"\nBehandler år {start_year}-{end_year}...")

    query = f"""
        SELECT {COLUMN_TEXT}, {COLUMN_YEAR}
        FROM {TABLE_NAME}
        WHERE {COLUMN_TEXT} IS NOT NULL
        AND {COLUMN_YEAR} >= ? AND {COLUMN_YEAR} < ?
    """
    rows = conn.execute(query, (start_year, end_year)).fetchall()
    print(f"Indlæste {len(rows)} dokumenter...")

    docs = [tokenize(r[0]) for r in tqdm(rows)]
    docs = [d for d in docs if d]
    print(f"Totalt antal ord: {sum(len(d) for d in docs):,}")

    # Fjern sjældne ord
    freq = Counter()
    for d in tqdm(docs):
        freq.update(d)
    docs = [[t for t in d if freq[t] >= MIN_COUNT] for d in docs]
    docs = [d for d in docs if d]

    # Træn Word2Vec
    print("Træner Word2Vec model...")
    model = Word2Vec(
        sentences=docs,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=4,
        sg=1,
        epochs=EPOCHS
    )
    print("Word2Vec trænet!")

    # Gem modellen med årinterval i filnavnet, altid små bogstaver
    interval_str = f"{start_year}_{end_year}"
    MODEL_PATH = os.path.join(BASE_DIR, f"word2vec_{interval_str}.model".lower())
    model.save(MODEL_PATH)
    print(f"✔ Modellen er gemt: {MODEL_PATH}")


conn.close()
