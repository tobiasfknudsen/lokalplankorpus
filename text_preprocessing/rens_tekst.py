import sqlite3
import re
import spacy
import os

# --------------------------------------------------------
# 0) Find projektets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Stier
# --------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
WHITELIST_FILE = os.path.join(ROOT_DIR, "lokalplankorpus", "text_preprocessing", "whitelist.txt")

# --------------------------------------------------------
# 2) Indlæs whitelist
# --------------------------------------------------------
with open(WHITELIST_FILE, encoding="utf-8") as f:
    WHITELIST = set(line.strip().lower() for line in f if line.strip())
print(f"Indlæst {len(WHITELIST):,} ord i whitelist")

# --------------------------------------------------------
# 3) Indlæs spaCy-model
# --------------------------------------------------------
nlp = spacy.load("da_core_news_lg")

# --------------------------------------------------------
# 4) Opsætning database
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
COLUMN_NAME = "tekst_renset"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Tilføj kolonnen kun hvis den ikke allerede findes
cur.execute(f"PRAGMA table_info({TABLE_NAME})")
existing_cols = [col[1] for col in cur.fetchall()]
if COLUMN_NAME not in existing_cols:
    cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {COLUMN_NAME} TEXT")
    conn.commit()
    print(f"Kolonne '{COLUMN_NAME}' tilføjet i tabellen '{TABLE_NAME}'")

# --------------------------------------------------------
# 5) Hjælpefunktion til tekst-rens
# --------------------------------------------------------
def custom_clean(text):
    # 1. Fjern linjer med "..." eller "…"
    lines = text.split("\n")
    lines = [line for line in lines if "..." not in line and "…" not in line]
    text = "\n".join(lines)

    # 2. Saml ord, der er delt med bindestreg over linjeskift
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

    # 3. Fjern alle linjeskift og ekstra mellemrum
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s+", " ", text)

    # 4. Lowercase alt
    text = text.lower()

    # 5. Fjern ord med tre ens bogstaver i træk
    text = re.sub(r"\b\w*(\w)\1\1\w*\b", " ", text)

    # 6. Fjern ord med ≤ 3 bogstaver, undtagen whitelist
    words = text.split()
    words = [w for w in words if len(w) > 3 or w in WHITELIST]

    # 7. Brug spaCy til kun at beholde alfabetiske tokens
    doc = nlp.make_doc(" ".join(words))
    text = " ".join([t.text for t in doc if t.is_alpha])

    return text.strip()

# --------------------------------------------------------
# 6) Hent alle tekster
# --------------------------------------------------------
cur.execute(f"SELECT id, tekst FROM {TABLE_NAME} WHERE tekst IS NOT NULL")
rows = cur.fetchall()
print(f"Der er {len(rows):,} tekster, der skal behandles...")

# --------------------------------------------------------
# 7) Rens og opdater database
# --------------------------------------------------------
BATCH_SIZE = 50
for i, (plan_id, text) in enumerate(rows, start=1):
    try:
        clean = custom_clean(text)
        cur.execute(f"UPDATE {TABLE_NAME} SET {COLUMN_NAME} = ? WHERE id = ?", (clean, plan_id))
    except Exception as e:
        print(f"Fejl ved ID {plan_id}: {e}")

    if i % BATCH_SIZE == 0:
        conn.commit()
        print(f"Gemt {i:,} af {len(rows):,} planer...")

conn.commit()
conn.close()

print("Tekst er nu renset, lowercased og kun alfabetiske ord er bevaret.")
