# ─────────────────────────────────────────────────────────────────────────────
# modules/model.py — Chargement du modèle ViT (unique instance mise en cache)
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline


@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")


# Instance globale partagée par tous les modules
classifier = load_model()
