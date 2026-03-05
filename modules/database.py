# ─────────────────────────────────────────────────────────────────────────────
# modules/database.py — Connexion MongoDB et collection principale
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pymongo


@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")


client     = init_connection()
mongo_db   = client["visionai_db"]
collection = mongo_db["images"]
