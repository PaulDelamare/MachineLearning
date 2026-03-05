# ─────────────────────────────────────────────────────────────────────────────
# modules/faces.py — Reconnaissance faciale via FaceNet (facenet-pytorch)
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle

import numpy as np
import streamlit as st
import torch

from config import FACES_DB_FILE


@st.cache_resource
def load_face_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn  = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet


mtcnn_model, resnet_model = load_face_models()


# ── Persistence ──────────────────────────────────────────────────────────────

def charger_db_visages() -> dict:
    if os.path.exists(FACES_DB_FILE):
        with open(FACES_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def sauvegarder_db_visages(db: dict) -> None:
    with open(FACES_DB_FILE, "wb") as f:
        pickle.dump(db, f)


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(pil_image):
    """Extrait l'embedding facial (512-dim). Retourne None si aucun visage détecté."""
    try:
        face_tensor = mtcnn_model(pil_image)
        if face_tensor is None:
            return None
        with torch.no_grad():
            emb = resnet_model(face_tensor.unsqueeze(0))
        return emb[0].numpy()
    except Exception:
        return None


# ── Enregistrement / Reconnaissance ──────────────────────────────────────────

def enregistrer_visage(pil_image, nom: str) -> tuple[bool, str]:
    """Ajoute un visage au registre. Retourne (succès, message)."""
    emb = get_embedding(pil_image)
    if emb is None:
        return False, "⚠️ Aucun visage détecté dans l'image. Réessaie avec un visage bien visible."
    db = charger_db_visages()
    if nom not in db:
        db[nom] = []
    db[nom].append(emb)
    sauvegarder_db_visages(db)
    nb = len(db[nom])
    return True, f"✅ Visage de **{nom}** enregistré ! ({nb} photo(s) au total)"


def reconnaitre_visage(pil_image) -> tuple:
    """Tente d'identifier la personne. Retourne (nom, confiance%) ou (None, None)."""
    db = charger_db_visages()
    if not db:
        return None, None
    emb = get_embedding(pil_image)
    if emb is None:
        return None, None
    best_name, best_dist = None, float("inf")
    for nom, embeddings in db.items():
        for known_emb in embeddings:
            dist = float(np.linalg.norm(emb - known_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = nom
    # Seuil empirique : < 0.9 = bonne correspondance
    if best_dist < 0.9:
        confiance = max(0, int((1 - best_dist / 0.9) * 100))
        return best_name, f"{confiance}%"
    return None, None
