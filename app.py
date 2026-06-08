"""ScanStrat — Streamlit entry point.

Run with:
    uv run streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ScanStrat",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 ScanStrat")
st.markdown(
    """
Vælg en side i sidemenuen til venstre:

| Side | Beskrivelse |
|------|-------------|
| **Screening** | Rangerede aktier fra hele universet — filtrer og sorter |
| **Aktiedetalje** | Kurs, SMA, RSI og feature-snapshot for én aktie |

Backtests og parameterstudier køres via `research/`-scripts i terminalen.
"""
)
