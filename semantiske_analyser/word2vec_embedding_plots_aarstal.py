# -*- coding: utf-8 -*-
"""
Interaktiv visualisering af allerede trænet Word2Vec-model ud fra seed-ord
"""

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import plotly.graph_objects as go

# --------------------------------------------------------
# 0) Find repoets rodmappe
# --------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# --------------------------------------------------------
# 1) Mappe med modeller for årstal
# --------------------------------------------------------
MODEL_DIR = os.path.join(ROOT_DIR, "semantiske_analyser", "word2vec_modeller", "aarstal")
BASE_DIR = os.path.join(ROOT_DIR, "semantiske_analyser" ,"word2vec_embedding_plots", "aarstal")
os.makedirs(BASE_DIR, exist_ok=True)

# --------------------------------------------------------
# 2) Vælg år / filtrering
# --------------------------------------------------------
FILTER_YEAR = (2000, 2026)  # Vælg år
year_str = f"{FILTER_YEAR[0]}_{FILTER_YEAR[1]}"
MODEL_PATH = os.path.join(MODEL_DIR, f"word2vec_{year_str}.model".lower())

# --------------------------------------------------------
# 3) Load gemt Word2Vec-model
# --------------------------------------------------------
model = Word2Vec.load(MODEL_PATH)
print(f"✔ Modellen er indlæst: {MODEL_PATH}")

# --------------------------------------------------------
# 4) Definer seed-ord
# --------------------------------------------------------
CATEGORIES = {
    "Bebyggelse": ["bebyggelse", "byggefelt", "etage", "bebyggelsesprocent"],  
    "Anvendelse": ["bolig", "butik", "kontor", "offentlig"],
    "Mobilitet": ["vej", "sti", "parkering", "trafik"],
    "Rekreativ": ["beplantning", "rekreativ", "landskab", "natur"],
    "Teknik": ["støj", "forurening", "miljø", "påvirkning"],
    "Bevaring/Kultur": ["bevaring", "facade", "byrum", "arkitektonisk"]
}

NEIGHBORS_PER_SEED = 8
SIM_THRESHOLD = 0.45

selected_words = []
word_category = {}
word_seed = {}

for cat, seeds in CATEGORIES.items():
    for seed in seeds:
        if seed not in model.wv:
            continue
        if seed not in word_category:
            selected_words.append(seed)
            word_category[seed] = cat
            word_seed[seed] = seed
        for w, sim in model.wv.most_similar(seed, topn=NEIGHBORS_PER_SEED):
            if sim >= SIM_THRESHOLD and w not in word_category:
                selected_words.append(w)
                word_category[w] = cat
                word_seed[w] = seed

vectors = np.vstack([model.wv[w] for w in selected_words])

# --------------------------------------------------------
# 5) Dimensionreduktion (UMAP / t-SNE) - først UMAP, hvis ikke, så t-SNE
# --------------------------------------------------------
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(vectors)
    method = "UMAP"
except Exception:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca = PCA(n_components=min(30, vectors.shape[1]), random_state=42)
    X_pca = pca.fit_transform(vectors)
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        learning_rate="auto",
        init="random"
    )
    coords = tsne.fit_transform(X_pca)
    method = "t-SNE (via PCA)"

# --------------------------------------------------------
# 6) Cosine similarity til seed-ord
# --------------------------------------------------------
cos_sim_to_seed = [1.0 if w == word_seed[w] else float(model.wv.similarity(w, word_seed[w])) for w in selected_words]

# --------------------------------------------------------
# 7) DataFrame til visualisering
# --------------------------------------------------------
df = pd.DataFrame({
    "Y": coords[:, 0],
    "X": coords[:, 1],
    "Ord": selected_words,
    "Kategori": [word_category[w] for w in selected_words],
    "Seed": [word_seed[w] for w in selected_words],
    "Cosine_sim": cos_sim_to_seed
})

# Farver pr. kategori
color_map = {
    "Bebyggelse": "#1f77b4",
    "Anvendelse": "#ff7f0e",
    "Mobilitet": "#cb2e23",
    "Rekreativ": "#2ca02c",
    "Teknik": "#919191",
    "Bevaring/Kultur": "#9467bd"
}

title_txt = f"Word-embeddings ud fra seed-ord - {FILTER_YEAR[0]} - {FILTER_YEAR[1]}"


# --------------------------------------------------------
# 8) Seed-ord størrelse og labels
# --------------------------------------------------------
seed_words = [seed for seeds in CATEGORIES.values() for seed in seeds]
marker_sizes = [12 if w in seed_words else 6 for w in df["Ord"]]
text_labels = [w if w in seed_words else "" for w in df["Ord"]]

# --------------------------------------------------------
# 9) Plot med go.Figure med "halo" på seed-ord
# --------------------------------------------------------
fig = go.Figure()

for cat in df["Kategori"].unique():
    cat_df = df[df["Kategori"] == cat]
    
    fig.add_trace(
        go.Scatter(
            x=cat_df["X"],
            y=cat_df["Y"],
            mode="text",
            text=[w if w in seed_words else "" for w in cat_df["Ord"]],
            textposition="top center",
            textfont=dict(
                color="white",
                size=[12 if w in seed_words else 6 for w in cat_df["Ord"]],
                family="Arial"
            ),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=cat_df["X"],
            y=cat_df["Y"],
            mode="text+markers",
            text=[w if w in seed_words else "" for w in cat_df["Ord"]],
            textposition="top center",
            hovertemplate=(
                "Ord = %{customdata[0]}<br>"
                "Kategori = %{customdata[3]}<br>"
                "Seed = %{customdata[1]}<br>"
                "Cosine similarity = %{customdata[2]:.2f}<extra></extra>"
            ),
            marker=dict(
                size=[12 if w in seed_words else 6 for w in cat_df["Ord"]],
                color=color_map[cat],
                line=dict(width=0),
                opacity=1
            ),
            textfont=dict(
                color="black",
                size=[14 if w in seed_words else 6 for w in cat_df["Ord"]],
                family="Arial"
            ),
            name=cat,
            customdata=np.stack(
                [cat_df["Ord"], cat_df["Seed"], cat_df["Cosine_sim"], cat_df["Kategori"]], axis=-1
            )
        )
    )

fig.update_layout(
    template="plotly_white",
    autosize=True,
    title=dict(
        text=title_txt,
        font=dict(size=20, color="black")
    ),
    legend=dict(
        font=dict(size=14, color="black"),
        orientation="h",
        x=0.5,
        y=0,
        xanchor="center",
        yanchor="top",
        bgcolor="rgba(255,255,255,0)",
        bordercolor="rgba(0,0,0,0)",
    )
)

fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

# --------------------------------------------------------
# 10) Gem HTML
# --------------------------------------------------------
plot_name = f"word2vec_plot_{year_str.lower()}"  # fx word2vec_plot_2000_2026
VIS_PATH = os.path.join(BASE_DIR, f"{plot_name}.html")
fig.write_html(VIS_PATH, include_plotlyjs="cdn")
print(f"✔ Gemte visualisering som HTML: {VIS_PATH}")
