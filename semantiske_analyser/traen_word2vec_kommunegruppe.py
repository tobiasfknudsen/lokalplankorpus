# -*- coding: utf-8 -*-
"""
Word2Vec ud fra kommunegruppe
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
BASE_DIR = os.path.join(ROOT_DIR, "semantiske_analyser" ,"word2vec_modeller", "kommunegruppe")
os.makedirs(BASE_DIR, exist_ok=True)

# databasefil i repoets rod
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")

# stopord-fil i repoets rod
STOPWORDS_FILE = os.path.join(ROOT_DIR, "stopord.txt")

# --------------------------------------------------------
# 1) Konfiguration
# --------------------------------------------------------
TABLE_NAME     = "lokalplaner"
COLUMN_TEXT    = "tekst_renset_lemmma"
COLUMN_KOMTYPE = "komtype_5kat"

FILTER_KOMTYPE = ["Landkommuner"]   # fx ["Landkommuner"]

MIN_COUNT   = 50
VECTOR_SIZE = 100
WINDOW      = 5
EPOCHS      = 10

# --------------------------------------------------------
# 2) Load stopord
# --------------------------------------------------------
with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
    STOPORD = {w.strip() for w in f if w.strip()}

# --------------------------------------------------------
# 3) Hent tekst + kommunegruppe fra database
# --------------------------------------------------------
conn = sqlite3.connect(DB_PATH)

if FILTER_KOMTYPE:
    qmarks = ",".join("?" for _ in FILTER_KOMTYPE)
    query = f"""
        SELECT {COLUMN_TEXT}, {COLUMN_KOMTYPE}
        FROM {TABLE_NAME}
        WHERE {COLUMN_TEXT} IS NOT NULL
        AND {COLUMN_KOMTYPE} IN ({qmarks})
    """
    rows = conn.execute(query, FILTER_KOMTYPE).fetchall()
else:
    rows = conn.execute(
        f"SELECT {COLUMN_TEXT}, {COLUMN_KOMTYPE} FROM {TABLE_NAME} WHERE {COLUMN_TEXT} IS NOT NULL"
    ).fetchall()

conn.close()

print(f"Indlæste {len(rows)} dokumenter...")

# --------------------------------------------------------
# 4) Tokenisering 
# --------------------------------------------------------
def tokenize(txt):
    toks = re.findall(r"[a-zA-ZæøåÆØÅ]+", str(txt).lower())
    toks = [t for t in toks if len(t) > 2 and t not in STOPORD]
    return toks

print("Tokeniserer dokumenter...")
docs = [tokenize(r[0]) for r in tqdm(rows)]

docs = [d for d in docs if d]
print(f"Totalt antal ord: {sum(len(d) for d in docs):,}")

# Fjern sjældne ord før træning
print("Fjerner sjældne ord (min_count)...")
freq = Counter()
for d in tqdm(docs):
    freq.update(d)

docs = [[t for t in d if freq[t] >= MIN_COUNT] for d in docs]
docs = [d for d in docs if d]

# --------------------------------------------------------
# 5) Træn Word2Vec 
# --------------------------------------------------------
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

# Brug navnet på FILTER_KOMTYPE til filnavn (fjern evt. mellemrum)
komtype_str = "_".join(FILTER_KOMTYPE).replace(" ", "_") if FILTER_KOMTYPE else "alle_komtyper"

# Gem Word2Vec-modellen
MODEL_PATH = os.path.join(BASE_DIR, f"word2vec_model_{komtype_str}.model")
model.save(MODEL_PATH)
print(f"✔ Modellen er gemt: {MODEL_PATH}")
