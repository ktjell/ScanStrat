"""Page 1 — Screening: ranked universe table."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from app_cache import get_service, get_ranker, get_universe, get_company_names

st.set_page_config(page_title="Screening · ScanStrat", page_icon="🔍", layout="wide")
st.title("🔍 Screening")

# ---------------------------------------------------------------
# Sidebar — filtre
# ---------------------------------------------------------------
st.sidebar.header("Filtre")

min_score = st.sidebar.slider("Min. score", 0, 100, 0, step=5)
only_golden = st.sidebar.checkbox("Kun golden cross (ingen death cross)", value=True)
top_n_display = st.sidebar.slider("Vis top N", 10, 200, 50, step=10)

st.sidebar.divider()
region = st.sidebar.selectbox(
    "Univers",
    ["US (S&P 500)", "EU (Euro Stoxx 50+)", "US + EU"],
    index=0,
)
region_map = {"US (S&P 500)": "US", "EU (Euro Stoxx 50+)": "EU", "US + EU": "US+EU"}
universe = get_universe(region_map[region])


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Henter kursdata…")
def load_ranked(universe: tuple[str, ...]) -> pd.DataFrame:
    service = get_service()
    ranker = get_ranker()
    end = date.today()
    start = end - timedelta(days=365 * 2)
    data = service.get_batch(list(universe), start, end)
    return ranker.rank(data)


with st.spinner(f"Ranker {len(universe)} aktier…"):
    ranked = load_ranked(tuple(universe))  # tuple = hashable for cache

if ranked.empty:
    st.warning("Ingen data tilgængeligt.")
    st.stop()

# ---------------------------------------------------------------
# Anvend filtre
# ---------------------------------------------------------------
df = ranked.copy()

if only_golden and "death_cross" in df.columns:
    df = df[df["death_cross"] != 1.0]

df = df[df["score"] >= min_score]
df = df.head(top_n_display)

# ---------------------------------------------------------------
# Vis tabel
# ---------------------------------------------------------------
st.markdown(f"**{len(df)} aktier** efter filtrering (univers: {len(universe)})")

# Formater kolonner pænt
format_map: dict[str, str] = {}
for col in df.columns:
    if col in ("rank", "ticker", "score"):
        continue
    if col == "death_cross":
        continue
    format_map[col] = "{:.2%}" if "momentum" in col or "dist" in col else "{:.2f}"

# Tilføj firmanavn
company_names = get_company_names(tuple(universe))
df.insert(df.columns.get_loc("ticker") + 1, "navn", df["ticker"].map(company_names))

display_cols = ["rank", "ticker", "navn", "score"] + [
    c for c in df.columns if c not in ("rank", "ticker", "navn", "score", "death_cross")
]
display_df = df[display_cols].set_index("rank")

st.dataframe(
    display_df.style.format(format_map, na_rep="—").background_gradient(
        subset=["score"], cmap="RdYlGn", vmin=0, vmax=100
    ),
    use_container_width=True,
    height=600,
)

# ---------------------------------------------------------------
# Klik til aktiedetalje
# ---------------------------------------------------------------
st.markdown("---")
selected = st.selectbox(
    "Gå til aktiedetalje",
    options=df["ticker"].tolist(),
    index=0,
)
if st.button("Vis detalje →"):
    st.session_state["selected_ticker"] = selected
    st.switch_page("pages/2_Aktiedetalje.py")
