# ─────────────────────────────────────────────────────────────────────────────
# modules/gestures.py — Reconnaissance de gestes via embeddings ViT
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle

import numpy as np
import torch

from config import GESTURES_DB_FILE, GESTURES_CONFIG
from modules.model import classifier


# ── Persistence ───────────────────────────────────────────────────────────────

def charger_db_gestes() -> dict:
    if os.path.exists(GESTURES_DB_FILE):
        with open(GESTURES_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def sauvegarder_db_gestes(db: dict) -> None:
    with open(GESTURES_DB_FILE, "wb") as f:
        pickle.dump(db, f)


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_gesture_embedding(pil_image):
    """Extrait un embedding ViT 768-dim normalisé. Retourne None en cas d'échec."""
    try:
        proc   = getattr(classifier, "image_processor", None) or classifier.feature_extractor
        inputs = proc(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = classifier.model(**inputs, output_hidden_states=True)
        # CLS token du dernier bloc Transformer = représentation globale
        emb = outputs.hidden_states[-1][:, 0, :].squeeze().numpy()
        # Normalise sur la sphère unité (distance L2 ≈ distance cosine)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb   # 768 dims
    except Exception:
        return None


# ── Enregistrement / Reconnaissance ──────────────────────────────────────────

def enregistrer_geste(pil_image, geste_key: str) -> tuple[bool, str]:
    """Ajoute un exemple d'embedding pour un geste. Retourne (succès, message)."""
    emb = get_gesture_embedding(pil_image)
    if emb is None:
        return False, "⚠️ Impossible d'analyser l'image."
    db = charger_db_gestes()
    if geste_key not in db:
        db[geste_key] = []
    db[geste_key].append(emb)
    sauvegarder_db_gestes(db)
    nb  = len(db[geste_key])
    cfg = GESTURES_CONFIG[geste_key]
    return True, f"✅ Geste **{cfg['label']}** enregistré ! ({nb} photo(s))"


def reconnaitre_geste(pil_image) -> tuple:
    """Identifie le geste. Retourne (geste_key, confiance_%) ou (None, None)."""
    db = charger_db_gestes()
    if not db:
        return None, None
    emb = get_gesture_embedding(pil_image)
    if emb is None:
        return None, None
    best_key, best_dist = None, float("inf")
    for key, embeddings in db.items():
        for known_emb in embeddings:
            dist = float(np.linalg.norm(emb - known_emb))
            if dist < best_dist:
                best_dist = dist
                best_key  = key
    # dist < 0.55 ≈ cosine > 0.85 = bonne correspondance
    if best_dist < 0.55:
        confiance = max(0, int((1 - best_dist / 0.55) * 100))
        return best_key, f"{confiance}%"
    # Si tous les gestes sont en base, retourne le plus proche (faible confiance)
    if best_key and len(db) == 3:
        return best_key, "??%"
    return None, None


def reconnaitre_geste_vote(frames: list) -> tuple:
    """
    Vote majoritaire sur plusieurs frames pour une détection plus robuste.
    Prend jusqu'à 9 frames réparties uniformément, retourne (geste, score).
    """
    if not frames:
        return None, "aucune frame"
    step    = max(1, len(frames) // 9)
    samples = list(frames)[::step][:9]
    votes   = {}
    for pil in samples:
        g, _ = reconnaitre_geste(pil)
        if g:
            votes[g] = votes.get(g, 0) + 1
    if not votes:
        return None, "non reconnu ⚠️"
    geste_final = max(votes, key=votes.get)
    n_vote      = votes[geste_final]
    n_total     = len(samples)
    return geste_final, f"{n_vote}/{n_total} frames ✓"
