import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
import os

# --------------------------------------------------------
# 0) Find projektets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Stier
# --------------------------------------------------------
DB_PATH = os.path.join(ROOT_DIR, "lokalplankorpus.db")
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "leksikalske_analyser" , "ordfrekvens")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------------------------------------------------------
# 2) Konfiguration
# --------------------------------------------------------
TABLE_NAME = "lokalplaner"
COL_TEXT = "tekst_renset_lemmma"
COL_YEAR = "aar"

WORDS_TO_ANALYZE = ["vindmølle", "solcelleanlæg", "natur", "parkering", "støj"] # Tilpas ord her
colors = ["#cd3323", "#ef8a62", "#c6eacb", "#67a9cf", "#024e69"]

# --------------------------------------------------------
# 3) Læs data
# --------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(f"SELECT {COL_YEAR}, {COL_TEXT} FROM {TABLE_NAME}", conn)
conn.close()

df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
df = df.dropna(subset=[COL_YEAR, COL_TEXT])
df[COL_YEAR] = df[COL_YEAR].astype(int)

# --------------------------------------------------------
# 4) Beregn pct. af planer hvor ordet optræder
# --------------------------------------------------------
results_all = {word: {} for word in WORDS_TO_ANALYZE}
years_grouped = list(df.groupby(COL_YEAR))

with tqdm(total=len(WORDS_TO_ANALYZE) * len(years_grouped),
          desc="Beregner pct. af planer") as pbar:

    for word in WORDS_TO_ANALYZE:
        for year, df_year in years_grouped:
            total_plans = len(df_year)
            plans_with_word = df_year[COL_TEXT].str.contains(rf"\b{word}\b").sum()
            pct = (plans_with_word / total_plans * 100) if total_plans > 0 else 0
            results_all[word][year] = pct
            pbar.update(1)

# --------------------------------------------------------
# 5) Interpolation i år hvor der ingen planer er
# --------------------------------------------------------
ALL_YEARS = list(range(1928, 2026))
INTERP_PERIODS = [
    range(1929, 1933),  # 1929–1932
    range(1933, 1940)   # 1933–1939
]

interp_results = {}
for word in WORDS_TO_ANALYZE:
    s = pd.Series(results_all[word])
    s = s.reindex(ALL_YEARS)
    s_interp = s.interpolate(method="linear")
    interp_results[word] = s_interp

# --------------------------------------------------------
# 6) Plot
# --------------------------------------------------------
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(15, 8))

for word, color in zip(WORDS_TO_ANALYZE, colors):
    ax.plot(ALL_YEARS, interp_results[word], label=word, color=color, linewidth=4)

# Overlay: stiplet linje for interpolerede perioder
for period in INTERP_PERIODS:
    for word in WORDS_TO_ANALYZE:
        seg = interp_results[word].loc[list(period)]
        ax.plot(period, seg, color="#ffffff", linestyle="dotted", linewidth=3)

# X-ticks: alle år, store hvert 12. år
manual_labels = list(range(1928, 2026, 12))
ax.set_xticks(ALL_YEARS)
ax.set_xticklabels([str(y) if y in manual_labels else "" for y in ALL_YEARS],
                   fontsize=16, rotation=0)

# Juster tick-længde
ticks = ax.xaxis.get_ticklines()
for i, tick in enumerate(ticks):
    index = i // 2
    year = ALL_YEARS[index]
    if year in manual_labels:
        tick.set_markersize(7)
    else:
        tick.set_markersize(3)

ax.yaxis.set_major_formatter(PercentFormatter())  # procent

# X-akse
ax.set_xlim(1928, 2025)

# Y-akse: procent
ax.set_ylim(0, 100)
ax.set_yticks(range(10, 101, 10))
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel("Andel af planer med ordet", fontsize=16)

# Fjern top og højre kant
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=16)

# Grid
ax.yaxis.grid(True, linestyle=':', linewidth=0.6, color='black')

# Legend
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.08),
    ncol=5,
    frameon=False,
    fontsize=14
)

plt.tight_layout()

# Gem fil
output_path = os.path.join(OUTPUT_FOLDER, "ordfrekvens_planfrekvens.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot gemt i: {output_path}")
