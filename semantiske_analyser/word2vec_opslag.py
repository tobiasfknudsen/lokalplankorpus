# -*- coding: utf-8 -*-
"""
Opslag af specifikke ord i Word2Vec-modeller med cosine similarity og frekvens

Kør:
python word_lookup_all.py
"""

import os
from gensim.models import Word2Vec

# --------------------------------------------------------
# 0) Find repoets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Mappe med alle Word2Vec-modeller, eksempelvis for kommunegrupper
MODEL_DIR = os.path.join(ROOT_DIR, "semantiske_analyser", "word2vec_modeller", "kommunegruppe")

# --------------------------------------------------------
# 1) Find alle modeller i mappen automatisk
# --------------------------------------------------------
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".model")]

# Map fra modelnavn til filsti
KOMMUNE_MODELS = {}
for f in model_files:
    # Brug selve filnavnet som "kommunegruppe"-navn
    KOMMUNE_MODELS[f] = os.path.join(MODEL_DIR, f)

print("Fundne modeller:")
for k, v in KOMMUNE_MODELS.items():
    print(f"  {k}: {v}")

# --------------------------------------------------------
# 2) Indtast ord, som skal slås op
# --------------------------------------------------------
while True:
    word = input("\nIndtast ord (eller 'exit' for at afslutte): ").strip().lower()
    if word == "exit":
        break

    # --------------------------------------------------------
    # 3) Loop gennem alle modeller
    # --------------------------------------------------------
    for model_name, model_path in KOMMUNE_MODELS.items():
        try:
            model = Word2Vec.load(model_path)
            print(f"\n✔ Loaded model: {model_path}")
        except FileNotFoundError:
            print(f"\n⚠ Model '{model_name}' ikke fundet: {model_path}")
            continue

        # --------------------------------------------------------
        # 4) Opslag af ord
        # --------------------------------------------------------
        print(f"\n--- Model: {model_name} ---")
        if word in model.wv:
            word_freq = model.wv.get_vecattr(word, "count")
            print(f"'{word}' forekommer {word_freq} gange i træningsdataene.")
            print(f"10 mest beslægtede ord til '{word}' (Cosine similarity | Frekvens):\n")

            for w, sim in model.wv.most_similar(word, topn=10):
                freq = model.wv.get_vecattr(w, "count")
                print(f"  {w:<15} {sim:.2f} | {freq}")
        else:
            print(f"Ordet '{word}' findes ikke i modellen '{model_name}'.")
