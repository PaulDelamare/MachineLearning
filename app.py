# ─────────────────────────────────────────────────────────────────────────────
# app.py — Point d'entrée principal de VisionIA
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from pages import page_analyse, page_jeu, page_visages, page_007

st.set_page_config(
    page_title="VisionIA – Reconnaissance d'images",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 VisionIA – Reconnaissance d'Images")
st.caption(
    "Détection automatique : Humains · Personnages Fictifs · Animaux · Plantes · "
    "Véhicules · Nourriture · Sport · Objets · Nature"
)
st.markdown("---")

tab_analyse, tab_jeu, tab_visages, tab_007 = st.tabs([
    "🔍 Analyse d'image",
    "🎮 Jeu de détection (10 manches)",
    "👥 Reconnaissance de visages",
    "🔫 007 Duel",
])

with tab_analyse:
    page_analyse.render()

with tab_jeu:
    page_jeu.render()

with tab_visages:
    page_visages.render()

with tab_007:
    page_007.render()
