import spacy
import pandas as pd
import sqlite3
import os
from spacy.tokens import Token
from spacy.language import Language

# --------------------------------------------------------
# 0) Find projektets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Stier
# --------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
LEMMALIST_FILE = os.path.join(ROOT_DIR, "text_preprocessing", "lemmaliste.csv")

# --------------------------------------------------------
# 2) Database-info
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
TEXT_COLUMN = "tekst_renset"
NEW_COLUMN = "tekst_renset_lemmma"

# --------------------------------------------------------
# 3) Indlæs lemmaliste
# --------------------------------------------------------
lemma_df = pd.read_csv(LEMMALIST_FILE, sep="\t")

custom_lemmas = {}
for _, row in lemma_df.iterrows():
    word = str(row["word"]).strip().lower()
    lemma = str(row["lemma"]).strip()
    pos = str(row["pos"]).strip().upper()
    custom_lemmas[(word, pos)] = lemma

# --------------------------------------------------------
# 4) Indlæs SpaCy
# --------------------------------------------------------
nlp = spacy.load("da_core_news_lg")

# --------------------------------------------------------
# 5) Definér custom lemmatizer ud fra lemmaliste 
# # Brug lemma fra lemmalisten kun hvis både ordet og ordtypen (POS) matcher; ellers SpaCy
# --------------------------------------------------------
@Language.component("custom_lemmatizer")
def custom_lemmatizer(doc):
    Token.set_extension("lemma_source", default="spacy", force=True)
    for token in doc:
        key = (token.text.lower(), token.pos_)
        if key in custom_lemmas:
            token.lemma_ = custom_lemmas[key]
            token._.lemma_source = "custom"
        else:
            token._.lemma_source = "spacy"
    return doc

nlp.add_pipe("custom_lemmatizer", last=True)

# --------------------------------------------------------
# 6) Forbind til SQLite
# --------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Opret ny kolonne, hvis den ikke findes
cur.execute(f"PRAGMA table_info({TABLE_NAME})")
columns = [col[1] for col in cur.fetchall()]
if NEW_COLUMN not in columns:
    cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {NEW_COLUMN} TEXT")
    conn.commit()

# --------------------------------------------------------
# 7) Hent kun rækker der mangler lemmatisering
# --------------------------------------------------------
cur.execute(f"""
    SELECT id, {TEXT_COLUMN}
    FROM {TABLE_NAME}
    WHERE {NEW_COLUMN} IS NULL OR {NEW_COLUMN} = ''
""")
rows = cur.fetchall()
print(f"Der er {len(rows):,} planer, der skal lemmatiseres...")

# --------------------------------------------------------
# 8) Lemmatize med batch
# --------------------------------------------------------
BATCH_SIZE = 10
for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i+BATCH_SIZE]
    ids = [r[0] for r in batch]
    texts = [r[1] if r[1] else "" for r in batch]

    # Brug nlp.pipe for hurtig batch-behandling
    docs = nlp.pipe(texts, batch_size=BATCH_SIZE)

    # Gem resultater i databasen
    for plan_id, doc in zip(ids, docs):
        lemmatized_text = " ".join([token.lemma_ for token in doc])
        cur.execute(f"UPDATE {TABLE_NAME} SET {NEW_COLUMN} = ? WHERE id = ?", 
                    (lemmatized_text, plan_id))

    conn.commit()
    print(f"Gemt {i + len(batch):,} af {len(rows):,} planer...")

conn.close()
print("Lemmatization færdig!")
