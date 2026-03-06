import re
import threading
import time
import random
import pickle
from collections import deque
import os
import numpy as np
import torch
import streamlit as st
import pymongo
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from bson.objectid import ObjectId
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# CONFIGURATION DE LA PAGE
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
st.set_page_config(
    page_title="VisionIA ÔÇô Reconnaissance d'images",
    page_icon="­ƒöì",
    layout="wide"
)

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# CHARGEMENT DU MOD├êLE (mis en cache)
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# CHARGEMENT MOD├êLE TENSORFLOW ÔÇô MobileNetV2
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
@st.cache_resource
def load_tf_model():
    try:
        import tensorflow as tf
        model      = tf.keras.applications.MobileNetV2(weights="imagenet")
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        decode     = tf.keras.applications.mobilenet_v2.decode_predictions
        return model, preprocess, decode
    except Exception:
        return None, None, None

_tf_model, _tf_preprocess, _tf_decode = load_tf_model()

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# TENSORFLOW FEATURE EXTRACTOR ÔÇô embeddings (visages + gestes)
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
@st.cache_resource
def load_tf_extractor():
    try:
        import tensorflow as tf
        model      = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, pooling="avg"
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        return model, preprocess
    except Exception:
        return None, None

_tf_extractor, _tf_ext_prep = load_tf_extractor()

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# CONNEXION MONGODB
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")

client = init_connection()
db = client["visionai_db"]
collection = db["images"]

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# RECONNAISSANCE FACIALE (FaceNet via facenet-pytorch)
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
FACES_DB_FILE    = "faces_db.pkl"
FACES_DB_TF_FILE = "faces_db_tf.pkl"

@st.cache_resource
def load_face_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet

mtcnn_model, resnet_model = load_face_models()

def charger_db_visages():
    if os.path.exists(FACES_DB_FILE):
        with open(FACES_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_db_visages(db):
    with open(FACES_DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_embedding(pil_image):
    """Extrait l'embedding facial d'une image PIL. Retourne None si pas de visage."""
    try:
        face_tensor = mtcnn_model(pil_image)
        if face_tensor is None:
            return None
        with torch.no_grad():
            emb = resnet_model(face_tensor.unsqueeze(0))
        return emb[0].numpy()
    except Exception:
        return None

def enregistrer_visage(pil_image, nom):
    """Ajoute un visage au registre (FaceNet + TF MobileNetV2)."""
    emb = get_embedding(pil_image)
    if emb is None:
        return False, "ÔÜá´©Å Aucun visage d├®tect├® dans l'image. R├®essaie avec un visage bien visible."
    faces_db = charger_db_visages()
    if nom not in faces_db:
        faces_db[nom] = []
    faces_db[nom].append(emb)
    sauvegarder_db_visages(faces_db)
    # Aussi enregistrer avec TF MobileNetV2
    emb_tf = get_embedding_tf_face(pil_image)
    if emb_tf is not None:
        fdb_tf = charger_db_visages_tf()
        if nom not in fdb_tf:
            fdb_tf[nom] = []
        fdb_tf[nom].append(emb_tf)
        sauvegarder_db_visages_tf(fdb_tf)
    nb = len(faces_db[nom])
    return True, f"Ô£à Visage de **{nom}** enregistr├® ! ({nb} photo(s) au total)"

def reconnaitre_visage(pil_image):
    """Identifie la personne via vote FaceNet + TF MobileNetV2."""
    faces_db = charger_db_visages()
    if not faces_db:
        return None, None
    emb = get_embedding(pil_image)
    if emb is None:
        return None, None
    # ÔöÇÔöÇ FaceNet ÔöÇÔöÇ
    best_name_fn, best_dist_fn = None, float("inf")
    for nom, embeddings in faces_db.items():
        for known_emb in embeddings:
            dist = float(np.linalg.norm(emb - known_emb))
            if dist < best_dist_fn:
                best_dist_fn = dist
                best_name_fn = nom
    nom_facenet = best_name_fn if best_dist_fn < 0.9 else None
    # ÔöÇÔöÇ TF MobileNetV2 ÔöÇÔöÇ
    nom_tf = None
    fdb_tf = charger_db_visages_tf()
    if fdb_tf:
        emb_tf = get_embedding_tf_face(pil_image)
        if emb_tf is not None:
            best_name_tf, best_dist_tf = None, float("inf")
            for nom, embeddings in fdb_tf.items():
                for known_emb in embeddings:
                    dist = float(np.linalg.norm(emb_tf - known_emb))
                    if dist < best_dist_tf:
                        best_dist_tf = dist
                        best_name_tf = nom
            nom_tf = best_name_tf if best_dist_tf < 0.9 else None
    # ÔöÇÔöÇ Vote majoritaire ÔöÇÔöÇ
    if nom_facenet and nom_tf:
        if nom_facenet == nom_tf:
            conf = max(0, int((1 - best_dist_fn / 0.9) * 100))
            return nom_facenet, f"{conf}% Ô£ô (FaceNet+TF)"
        conf = max(0, int((1 - best_dist_fn / 0.9) * 100))
        return nom_facenet, f"{conf}% (FaceNet)"
    elif nom_facenet:
        conf = max(0, int((1 - best_dist_fn / 0.9) * 100))
        return nom_facenet, f"{conf}% (FaceNet)"
    elif nom_tf:
        return nom_tf, "~% (TF)"
    return None, None

def charger_db_visages_tf():
    if os.path.exists(FACES_DB_TF_FILE):
        with open(FACES_DB_TF_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_db_visages_tf(db):
    with open(FACES_DB_TF_FILE, "wb") as f:
        pickle.dump(db, f)

def get_embedding_tf_face(pil_image):
    """Embedding facial MobileNetV2 TF (1280-dim), d├®tection visage via MTCNN."""
    if _tf_extractor is None:
        return None
    try:
        face_tensor = mtcnn_model(pil_image)
        if face_tensor is None:
            return None
        face_np = ((face_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        face_pil = Image.fromarray(face_np).resize((224, 224))
        img_arr  = np.expand_dims(np.array(face_pil, dtype=np.float32), axis=0)
        img_arr  = _tf_ext_prep(img_arr)
        emb      = _tf_extractor.predict(img_arr, verbose=0)[0]
        emb      = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception:
        return None

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# JEU 007 ÔÇô RECONNAISSANCE DE GESTES
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
GESTURES_DB_FILE    = "gestures_db.pkl"
GESTURES_DB_TF_FILE = "gestures_db_tf.pkl"
GESTURES_NB_MIN     = 3  # photos minimum par geste pour jouer

GESTURES_CONFIG = {
    "recharger": {
        "emoji": "­ƒñÖ", "label": "Recharger",
        "desc": "2 doigts point├®s ├á c├┤t├® de la t├¬te (tempe)",
        "couleur": "#1a73e8",
    },
    "tirer": {
        "emoji": "­ƒö½", "label": "Tirer",
        "desc": "Main en forme de pistolet, doigt point├®",
        "couleur": "#e8341a",
    },
    "proteger": {
        "emoji": "­ƒøí´©Å", "label": "Se prot├®ger",
        "desc": "Bras crois├®s devant toi en bouclier",
        "couleur": "#2e7d32",
    },
}

# R├¿gles du duel :
# Tirer > Recharger  (si tu as des balles)
# Prot├®ger bloque Tirer
# Recharger + Se prot├®ger = neutre
JEU007_VIES_MAX   = 3
JEU007_BALLES_MAX = 3

def charger_db_gestes():
    if os.path.exists(GESTURES_DB_FILE):
        with open(GESTURES_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_db_gestes(db):
    with open(GESTURES_DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_gesture_embedding(pil_image):
    """Extrait un embedding ViT 768-dim normalis├® depuis le mod├¿le d├®j├á charg├®."""
    try:
        # R├®utilise le processeur et le mod├¿le du pipeline classifier
        proc = getattr(classifier, "image_processor", None) or classifier.feature_extractor
        inputs = proc(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = classifier.model(**inputs, output_hidden_states=True)
        # CLS token du dernier bloc Transformer = repr├®sentation globale
        emb = outputs.hidden_states[-1][:, 0, :].squeeze().numpy()
        # Normalise sur la sph├¿re unit├® pour utiliser la distance L2 comme cosine
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb  # 768 dims
    except Exception:
        return None

def enregistrer_geste(pil_image, geste_key):
    """Ajoute une photo d'exemple pour un geste (ViT + TF MobileNetV2)."""
    emb = get_gesture_embedding(pil_image)
    if emb is None:
        return False, "ÔÜá´©Å Impossible d'analyser l'image."
    db = charger_db_gestes()
    if geste_key not in db:
        db[geste_key] = []
    db[geste_key].append(emb)
    sauvegarder_db_gestes(db)
    # TF MobileNetV2 embedding
    emb_tf = get_gesture_embedding_tf(pil_image)
    if emb_tf is not None:
        db_tf = charger_db_gestes_tf()
        if geste_key not in db_tf:
            db_tf[geste_key] = []
        db_tf[geste_key].append(emb_tf)
        sauvegarder_db_gestes_tf(db_tf)
    nb = len(db[geste_key])
    cfg = GESTURES_CONFIG[geste_key]
    return True, f"Ô£à Geste **{cfg['label']}** enregistr├® ! ({nb} photo(s))"

def reconnaitre_geste(pil_image):
    """Identifie le geste via vote ViT + TF MobileNetV2."""
    db = charger_db_gestes()
    if not db:
        return None, None
    emb = get_gesture_embedding(pil_image)
    if emb is None:
        return None, None
    # ÔöÇÔöÇ ViT ÔöÇÔöÇ
    best_key_vit, best_dist_vit = None, float("inf")
    for key, embeddings in db.items():
        for known_emb in embeddings:
            dist = float(np.linalg.norm(emb - known_emb))
            if dist < best_dist_vit:
                best_dist_vit = dist
                best_key_vit  = key
    geste_vit = best_key_vit if best_dist_vit < 0.55 else None
    # ÔöÇÔöÇ TF MobileNetV2 ÔöÇÔöÇ
    geste_tf = None
    db_tf = charger_db_gestes_tf()
    if db_tf:
        emb_tf = get_gesture_embedding_tf(pil_image)
        if emb_tf is not None:
            best_key_tf, best_dist_tf = None, float("inf")
            for key, embeddings in db_tf.items():
                for known_emb in embeddings:
                    dist = float(np.linalg.norm(emb_tf - known_emb))
                    if dist < best_dist_tf:
                        best_dist_tf = dist
                        best_key_tf  = key
            geste_tf = best_key_tf if best_dist_tf < 0.6 else None
    # ÔöÇÔöÇ Vote ÔöÇÔöÇ
    if geste_vit and geste_tf:
        if geste_vit == geste_tf:
            conf = max(0, int((1 - best_dist_vit / 0.55) * 100))
            return geste_vit, f"{conf}%"
        return geste_vit, "??%"
    elif geste_vit:
        conf = max(0, int((1 - best_dist_vit / 0.55) * 100))
        return geste_vit, f"{conf}%"
    elif geste_tf:
        return geste_tf, "??%"
    if best_key_vit and len(db) == 3:
        return best_key_vit, "??%"
    return None, None

def charger_db_gestes_tf():
    if os.path.exists(GESTURES_DB_TF_FILE):
        with open(GESTURES_DB_TF_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_db_gestes_tf(db):
    with open(GESTURES_DB_TF_FILE, "wb") as f:
        pickle.dump(db, f)

def get_gesture_embedding_tf(pil_image):
    """Embedding de geste MobileNetV2 TF (1280-dim normalis├®)."""
    if _tf_extractor is None:
        return None
    try:
        img     = pil_image.resize((224, 224)).convert("RGB")
        img_arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        img_arr = _tf_ext_prep(img_arr)
        emb     = _tf_extractor.predict(img_arr, verbose=0)[0]
        emb     = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception:
        return None


def reconnaitre_geste_vote(frames: list):
    """
    Vote majoritaire sur plusieurs frames pour une d├®tection plus robuste.
    Prend jusqu'├á 9 frames r├®parties uniform├®ment dans la liste fournie,
    classifie chacune, et retourne le geste le plus vot├® avec un r├®sum├®.
    """
    if not frames:
        return None, "aucune frame"
    # ├ëchantillonnage uniforme : max 9 frames
    step     = max(1, len(frames) // 9)
    samples  = list(frames)[::step][:9]
    votes    = {}
    for pil in samples:
        g, _ = reconnaitre_geste(pil)
        if g:
            votes[g] = votes.get(g, 0) + 1
    if not votes:
        return None, "non reconnu ÔÜá´©Å"
    geste_final = max(votes, key=votes.get)
    n_vote      = votes[geste_final]
    n_total     = len(samples)
    return geste_final, f"{n_vote}/{n_total} frames Ô£ô"

# ÔöÇÔöÇ Q-LEARNING 007 ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
Q007_FILE    = "q_007.pkl"
Q007_LR      = 0.3    # learning rate
Q007_GAMMA   = 0.85   # discount
Q007_EPSILON = 0.20   # taux d'exploration
GESTES_KEYS  = ["recharger", "tirer", "proteger"]  # index 0,1,2

def charger_qtable():
    if os.path.exists(Q007_FILE):
        with open(Q007_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_qtable(qt):
    with open(Q007_FILE, "wb") as f:
        pickle.dump(qt, f)

def etat_007(j_balles, ia_balles, j_vies, ia_vies):
    """Tuple discret repr├®sentant l'├®tat courant."""
    return (
        min(j_balles,  JEU007_BALLES_MAX),
        min(ia_balles, JEU007_BALLES_MAX),
        max(0, j_vies),
        max(0, ia_vies),
    )

def ia_choisit_geste(ia_balles, ia_vies, joueur_vies, state=None, qtable=None):
    """Choisit le geste via Q-learning epsilon-greedy."""
    can_shoot = (ia_balles > 0)
    if qtable is not None and state is not None and random.random() > Q007_EPSILON:
        # Exploitation : meilleure action connue
        q_vals = qtable.get(state, np.zeros(3)).copy()
        if not can_shoot:
            q_vals[1] = -9999  # bloquer "tirer" si pas de balles
        return GESTES_KEYS[int(np.argmax(q_vals))]
    # Exploration al├®atoire
    choices = ["recharger", "proteger"]
    if can_shoot:
        choices += ["tirer", "tirer"]  # l├®g├¿re pr├®f├®rence tirer
    return random.choice(choices)

def ia_apprendre(qtable, state, action_idx, reward, next_state):
    """Mise ├á jour Bellman de la Q-table."""
    if state not in qtable:
        qtable[state] = np.zeros(3)
    if next_state not in qtable:
        qtable[next_state] = np.zeros(3)
    cur = qtable[state][action_idx]
    qtable[state][action_idx] = cur + Q007_LR * (
        reward + Q007_GAMMA * np.max(qtable[next_state]) - cur
    )
    return qtable

def calculer_reward_ia(ia_geste, j_geste, ia_balles_avant, j_balles_avant,
                        ia_touche, j_touche, j_vies_new, ia_vies_new):
    """
    ia_touche : l'IA a ├®t├® touch├®e par le joueur (mauvais pour l'IA).
    j_touche  : le joueur a ├®t├® touch├® par l'IA (bon pour l'IA).
    """
    r = 0

    # ÔöÇ Combat ÔöÇ
    if j_touche:   r += 25   # toucher l'adversaire : priorit├® max
    if ia_touche:  r -= 25   # se faire toucher : sym├®trique

    # ÔöÇ Tirer (initiative offensive) ÔöÇ
    if ia_geste == "tirer" and ia_balles_avant > 0:
        r += 2   # petit bonus : tirer est une action courageuse

    # ÔöÇ Tour compl├¿tement inutile ÔöÇ
    if not j_touche and not ia_touche:
        r -= 3  # chaque tour sans d├®cision co├╗te

    # ÔöÇ Protection ÔöÇ
    if j_geste == "tirer" and j_balles_avant > 0 and ia_geste == "proteger":
        r += 12  # bloquer un vrai tir : bien r├®compens├®
    elif ia_geste == "proteger":
        r -= 8   # se prot├®ger sans raison : tour perdu, punit fortement

    # ÔöÇ Gestion munitions ÔöÇ
    if ia_geste == "recharger":
        if ia_balles_avant >= JEU007_BALLES_MAX:
            r -= 14  # recharger quand d├®j├á plein : tr├¿s inutile
        else:
            r += 3   # recharger quand en manque : utile

    # ÔöÇ Tirer ├á vide ÔöÇ
    if ia_geste == "tirer" and ia_balles_avant == 0:
        r -= 8

    # ÔöÇ Fin de partie ÔöÇ
    if ia_vies_new <= 0:  r -= 40
    if j_vies_new  <= 0:  r += 40
    return r


def s_entrainer_007(nb_parties: int = None, duree_s: float = None):
    """
    Entra├«ne le bot par auto-jeu ASYM├ëTRIQUE pour ├®viter la convergence vers les nuls.
    Alterne 3 modes de match ├á chaque partie :
      - QvQ   (40%) : les deux agents exploitent la Q-table
      - QvRand(35%) : agent Q affronte un opposant purement al├®atoire
      - QvAgro(25%) : agent Q affronte un opposant qui tire d├¿s qu'il a des balles
    Cela force le bot ├á apprendre contre des strat├®gies impr├®visibles/agressives
    plut├┤t que de converger vers un ├®quilibre d├®fensif sym├®trique.
    """
    if nb_parties is None and duree_s is None:
        nb_parties = 500

    qt     = charger_qtable()
    stats  = {"victoires": 0, "defaites": 0, "nuls": 0, "tours_total": 0,
              "parties": 0, "modes": {"QvQ": 0, "QvRand": 0, "QvAgro": 0}}
    t_debut    = time.time()
    partie_idx = 0

    def geste_random(balles, *_):
        c = ["recharger", "proteger"]
        if balles > 0: c.append("tirer")
        return random.choice(c)

    def geste_agro(balles, *_):
        return "tirer" if balles > 0 else "recharger"

    while True:
        if nb_parties is not None and stats["parties"] >= nb_parties:
            break
        if duree_s is not None and (time.time() - t_debut) >= duree_s:
            break

        # S├®lection du mode : 40% QvQ / 35% QvRand / 25% QvAgro
        r_mod = partie_idx % 20
        if   r_mod < 8:   mode = "QvQ"
        elif r_mod < 15:  mode = "QvRand"
        else:             mode = "QvAgro"
        stats["modes"][mode] += 1
        partie_idx += 1

        jb = 0; ib = 0
        jv = JEU007_VIES_MAX
        iv = JEU007_VIES_MAX
        ig = jg = "recharger"

        for _ in range(60):
            if jv <= 0 or iv <= 0:
                break
            stats["tours_total"] += 1

            s_ia = etat_007(jb, ib, jv, iv)
            s_j  = etat_007(ib, jb, iv, jv)

            ig = ia_choisit_geste(ib, iv, jv, s_ia, qt)  # IA exploite toujours

            if   mode == "QvQ":   jg = ia_choisit_geste(jb, jv, iv, s_j, qt)
            elif mode == "QvRand": jg = geste_random(jb)
            else:                  jg = geste_agro(jb)

            jb_av = jb; ib_av = ib
            jb, ib, jv, iv, _, jt, it = resoudre_duel(jg, ig, jb, ib, jv, iv)

            ns_ia = etat_007(jb, ib, jv, iv)
            ns_j  = etat_007(ib, jb, iv, jv)

            r_ia = calculer_reward_ia(ig, jg, ib_av, jb_av, it, jt, jv, iv)
            qt   = ia_apprendre(qt, s_ia, GESTES_KEYS.index(ig), r_ia, ns_ia)

            # L'opposant apprend uniquement en QvQ
            if mode == "QvQ":
                r_j = calculer_reward_ia(jg, ig, jb_av, ib_av, jt, it, iv, jv)
                qt  = ia_apprendre(qt, s_j, GESTES_KEYS.index(jg), r_j, ns_j)

        if iv > jv:
            stats["victoires"] += 1
        elif jv > iv:
            stats["defaites"] += 1
        else:
            stats["nuls"] += 1
            # P├®nalit├® nul plus forte
            s_ia_fin = etat_007(jb, ib, jv, iv)
            qt = ia_apprendre(qt, s_ia_fin, GESTES_KEYS.index(ig), -20, s_ia_fin)
            if mode == "QvQ":
                s_j_fin = etat_007(ib, jb, iv, jv)
                qt = ia_apprendre(qt, s_j_fin, GESTES_KEYS.index(jg), -20, s_j_fin)

        stats["parties"] += 1

    stats["duree_reelle_s"] = time.time() - t_debut
    sauvegarder_qtable(qt)
    return qt, stats

def resoudre_duel(j_geste, ia_geste, j_balles, ia_balles, j_vies, ia_vies):
    """Applique les r├¿gles 007 et retourne le nouvel ├®tat + messages."""
    msgs = []
    j_touche  = False
    ia_touche = False

    # ÔöÇÔöÇ Le joueur tire ÔöÇÔöÇ
    if j_geste == "tirer":
        if j_balles > 0:
            j_balles -= 1
            if ia_geste != "proteger":
                ia_vies -= 1
                ia_touche = True
                msgs.append("­ƒö½ **Tu as tir├®** ÔåÆ IA **touch├®e** ! (-1 vie)")
            else:
                msgs.append("­ƒö½ **Tu as tir├®** ÔåÆ IA s'est **prot├®g├®e**. Rat├® !")
        else:
            msgs.append("ÔØî **Tu as tir├®** mais tu n'as plus de balles !")
    elif j_geste == "recharger":
        j_balles = min(JEU007_BALLES_MAX, j_balles + 1)
        msgs.append(f"­ƒñÖ **Tu recharges** ÔåÆ {j_balles} balle(s)")
    elif j_geste == "proteger":
        msgs.append("­ƒøí´©Å **Tu te prot├¿ges**")

    # ÔöÇÔöÇ L'IA tire ÔöÇÔöÇ
    if ia_geste == "tirer":
        if ia_balles > 0:
            ia_balles -= 1
            if j_geste != "proteger":
                j_vies -= 1
                j_touche = True
                msgs.append("­ƒÆÇ **L'IA a tir├®** ÔåÆ Tu es **touch├®(e)** ! (-1 vie)")
            else:
                msgs.append("­ƒøí´©Å **L'IA a tir├®** ÔåÆ Tu t'es **prot├®g├®(e)** ! Bloqu├®.")
        else:
            msgs.append("ÔØî L'IA a tir├® sans balles !")
    elif ia_geste == "recharger":
        ia_balles = min(JEU007_BALLES_MAX, ia_balles + 1)
        msgs.append(f"­ƒñÖ **L'IA recharge** ÔåÆ {ia_balles} balle(s)")
    elif ia_geste == "proteger":
        msgs.append("­ƒøí´©Å **L'IA se prot├¿ge**")

    return j_balles, ia_balles, j_vies, ia_vies, msgs, j_touche, ia_touche


# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# MAPPING : labels ImageNet ÔåÆ 5 cat├®gories TP
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
CATEGORY_MAPPER = {
    # ÔöÇÔöÇ V├®hicules EN PREMIER (priorit├® haute pour ├®viter les faux positifs) ÔöÇÔöÇ
    "sports car": "V├®hicule", "sport car": "V├®hicule",
    "race car": "V├®hicule", "racing car": "V├®hicule",
    "car": "V├®hicule", "truck": "V├®hicule", "bus": "V├®hicule",
    "bicycle": "V├®hicule", "motorcycle": "V├®hicule", "airplane": "V├®hicule",
    "boat": "V├®hicule", "train": "V├®hicule", "ambulance": "V├®hicule",
    "taxi": "V├®hicule", "van": "V├®hicule", "tractor": "V├®hicule",
    "helicopter": "V├®hicule", "spacecraft": "V├®hicule", "submarine": "V├®hicule",
    "jeep": "V├®hicule", "minivan": "V├®hicule", "convertible": "V├®hicule",
    "racer": "V├®hicule", "go-kart": "V├®hicule", "streetcar": "V├®hicule",
    "scooter": "V├®hicule", "limousine": "V├®hicule", "fire engine": "V├®hicule",
    "police van": "V├®hicule", "cab": "V├®hicule", "minibus": "V├®hicule",

    # ÔöÇÔöÇ Personnages fictifs ÔöÇÔöÇ
    "puppet": "Personnage Fictif",
    "teddy": "Personnage Fictif",
    "teddy bear": "Personnage Fictif",
    "ocarina": "Personnage Fictif", "doll": "Personnage Fictif",
    "figurine": "Personnage Fictif", "action figure": "Personnage Fictif",
    "costume": "Personnage Fictif", "cloak": "Personnage Fictif",
    "robe": "Personnage Fictif",
    # BD, manga, super-h├®ros : retir├®s de la blacklist pour ├¬tre captur├®s ici
    "comic book": "Personnage Fictif", "comic strip": "Personnage Fictif",
    "mask": "Personnage Fictif", "cartoon": "Personnage Fictif",

    # ÔöÇÔöÇ Animaux ÔöÇÔöÇ
    "dog": "Animal", "cat": "Animal", "bird": "Animal", "fish": "Animal",
    "horse": "Animal", "cow": "Animal", "elephant": "Animal", "bear": "Animal",
    "zebra": "Animal", "giraffe": "Animal", "lion": "Animal", "tiger": "Animal",
    "wolf": "Animal", "fox": "Animal", "rabbit": "Animal", "hamster": "Animal",
    "duck": "Animal", "eagle": "Animal", "penguin": "Animal", "frog": "Animal",
    "snake": "Animal", "lizard": "Animal", "turtle": "Animal", "shark": "Animal",
    "whale": "Animal", "bee": "Animal", "butterfly": "Animal", "spider": "Animal",
    "crab": "Animal", "lobster": "Animal",

    # ÔöÇÔöÇ Plantes ÔöÇÔöÇ
    "flower": "Plante", "rose": "Plante", "daisy": "Plante", "tulip": "Plante",
    "sunflower": "Plante", "dandelion": "Plante", "tree": "Plante",
    "mushroom": "Plante", "cactus": "Plante", "fern": "Plante", "moss": "Plante",
    "leaf": "Plante", "grass": "Plante", "corn": "Plante", "banana": "Plante",
    "apple": "Plante", "orange": "Plante", "strawberry": "Plante",
    "broccoli": "Plante", "carrot": "Plante",

    # ÔöÇÔöÇ Humains EN DERNIER (v├¬tements, accessoires ÔåÆ visibles sur une personne) ÔöÇÔöÇ
    "suit": "Humain", "bow tie": "Humain", "trench coat": "Humain",
    "jersey": "Humain", "lab coat": "Humain", "groom": "Humain",
    "face powder": "Humain", "lipstick": "Humain", "swimming cap": "Humain",
    "wig": "Humain",
    "soccer ball": "Humain", "football helmet": "Humain",
    "basketball": "Humain", "tennis ball": "Humain", "baseball": "Humain",
    "volleyball": "Humain", "rugby ball": "Humain", "golf ball": "Humain",
    "running shoe": "Humain", "sneaker": "Humain",
    "uniform": "Humain", "sweatshirt": "Humain", "t-shirt": "Humain",
    "overskirt": "Humain", "miniskirt": "Humain", "bikini": "Humain",
    "brassiere": "Humain", "maillot": "Humain", "stole": "Humain",
    "vestment": "Humain", "apron": "Humain", "abaya": "Humain",
    "cardigan": "Humain", "jean": "Humain", "mitten": "Humain",
    "sunglasses": "Humain", "glasses": "Humain",
    "mortarboard": "Humain", "cowboy hat": "Humain", "bonnet": "Humain",
    # Labels ViT suppl├®mentaires fr├®quents pour une personne
    "polo shirt": "Humain", "dress shirt": "Humain", "jacket": "Humain",
    "hoodie": "Humain", "pullover": "Humain", "turtleneck": "Humain",
    "fur coat": "Humain", "overcoat": "Humain", "raincoat": "Humain",
    "parka": "Humain", "jean jacket": "Humain", "blazer": "Humain",
    "tank top": "Humain", "crop top": "Humain",
    "jeans": "Humain", "shorts": "Humain", "trousers": "Humain",
    "leggings": "Humain", "tracksuit": "Humain",
    "scarf": "Humain", "bandana": "Humain", "balaclava": "Humain",
    "baseball cap": "Humain", "beanie": "Humain", "cap": "Humain",
    "neck brace": "Humain",
    "bow": "Humain", "tie": "Humain", "suspenders": "Humain",

    # ÔöÇÔöÇ Nourriture ÔöÇÔöÇ
    "pizza": "Nourriture", "burger": "Nourriture", "cheeseburger": "Nourriture",
    "hot dog": "Nourriture", "sandwich": "Nourriture", "burrito": "Nourriture",
    "sushi": "Nourriture", "guacamole": "Nourriture", "pretzel": "Nourriture",
    "bagel": "Nourriture", "bread": "Nourriture", "croissant": "Nourriture",
    "waffle": "Nourriture", "pancake": "Nourriture", "cake": "Nourriture",
    "ice cream": "Nourriture", "chocolate": "Nourriture", "candy": "Nourriture",
    "cheese": "Nourriture", "egg": "Nourriture", "soup": "Nourriture",
    "pasta": "Nourriture", "noodle": "Nourriture", "rice": "Nourriture",
    "steak": "Nourriture", "meat": "Nourriture", "bacon": "Nourriture",
    "lemon": "Nourriture", "fig": "Nourriture", "pomegranate": "Nourriture",
    "taco": "Nourriture", "french loaf": "Nourriture",

    # ÔöÇÔöÇ Sport ÔöÇÔöÇ
    "tennis racket": "Sport", "baseball bat": "Sport", "cricket bat": "Sport",
    "golf club": "Sport", "ski": "Sport", "snowboard": "Sport",
    "surfboard": "Sport", "skateboard": "Sport", "parachute": "Sport",
    "bow": "Sport", "dumbbell": "Sport", "barbell": "Sport",
    "swimming": "Sport", "balance beam": "Sport", "horizontal bar": "Sport",
    "ping-pong ball": "Sport", "boxing glove": "Sport",

    # ÔöÇÔöÇ Objet du quotidien ÔöÇÔöÇ
    "bottle": "Objet", "wine bottle": "Objet", "beer bottle": "Objet",
    "water bottle": "Objet", "perfume": "Objet",
    "cup": "Objet", "coffee mug": "Objet", "teapot": "Objet",
    "bowl": "Objet", "plate": "Objet", "fork": "Objet",
    "knife": "Objet", "spoon": "Objet", "ladle": "Objet",
    "chair": "Objet", "table": "Objet", "desk": "Objet",
    "sofa": "Objet", "bed": "Objet", "pillow": "Objet",
    "lamp": "Objet", "clock": "Objet", "mirror": "Objet",
    "backpack": "Objet", "suitcase": "Objet", "handbag": "Objet",
    "umbrella": "Objet", "wallet": "Objet",
    "laptop": "Objet", "keyboard": "Objet", "mouse": "Objet",
    "phone": "Objet", "remote control": "Objet", "camera": "Objet",
    "book": "Objet", "pencil": "Objet", "scissors": "Objet",
    "hammer": "Objet", "wrench": "Objet", "screwdriver": "Objet",
    "gun": "Objet", "sword": "Objet",
    "candle": "Objet", "vase": "Objet", "pot": "Objet",
    "bucket": "Objet", "broom": "Objet", "toilet": "Objet",
    "bathtub": "Objet", "toaster": "Objet", "microwave": "Objet",
    "refrigerator": "Objet", "washing machine": "Objet",
    # Objets du quotidien suppl├®mentaires
    "sock": "Objet", "stocking": "Objet",
    "toilet tissue": "Objet", "paper towel": "Objet", "toilet paper": "Objet",
    "toothbrush": "Objet", "toothpaste": "Objet",
    "hair dryer": "Objet", "comb": "Objet",
    "ballpoint pen": "Objet", "crayon": "Objet", "ruler": "Objet",
    "nail": "Objet", "ping-pong paddle": "Objet",
    "sunscreen": "Objet", "lotion": "Objet",
    "bandage": "Objet", "pill bottle": "Objet",
    "mousetrap": "Objet", "padlock": "Objet", "key": "Objet",
    "cellular telephone": "Objet", "television": "Objet",
    "headphone": "Objet", "earphone": "Objet",
    # Note: jersey et running shoe sont d├®finis dans la section Humain ÔÇö ne pas red├®finir ici

    # ÔöÇÔöÇ Nature / Paysage ÔöÇÔöÇ
    "mountain": "Nature", "volcano": "Nature", "valley": "Nature",
    "ocean": "Nature", "lake": "Nature", "river": "Nature",
    "waterfall": "Nature", "beach": "Nature", "desert": "Nature",
    "cliff": "Nature", "coral reef": "Nature", "geyser": "Nature",
    "cloud": "Nature", "sky": "Nature", "rainbow": "Nature",
    "ice berg": "Nature", "glacier": "Nature", "cave": "Nature",
}

# Ic├┤nes et messages de gamification par cat├®gorie
CATEGORY_CONFIG = {
    "Humain":            {"emoji": "­ƒæñ", "message": "­ƒæñ Humain rep├®r├® ! Vous n'├¬tes pas seul..."},
    "Personnage Fictif": {"emoji": "­ƒºÖ", "message": "­ƒÄ¼ Cr├®ature de l├®gende d├®tect├®e ! Sortez le popcorn !"},
    "Animal":            {"emoji": "­ƒÉ¥", "message": "­ƒÉ¥ B├¬te sauvage rep├®r├®e ! Ne bougez plus..."},
    "Plante":            {"emoji": "­ƒî┐", "message": "­ƒî┐ La nature s'invite ! Pensez ├á arroser."},
    "V├®hicule":          {"emoji": "­ƒÜù", "message": "­ƒÜù Bolide en approche ! Attachez vos ceintures !"},
    "Nourriture":        {"emoji": "­ƒìò", "message": "­ƒìò Repas d├®tect├® ! J'ai faim maintenant..."},
    "Sport":             {"emoji": "­ƒÅå", "message": "­ƒÅå ├Ç vos marques, pr├¬ts, partez !"},
    "Objet":             {"emoji": "­ƒôª", "message": "­ƒôª Objet du quotidien identifi├® !"},
    "Nature":            {"emoji": "­ƒîì", "message": "­ƒîì Splendeur naturelle d├®tect├®e !"},
    "Inconnu":           {"emoji": "ÔØô", "message": "­ƒñö L'IA ne reconna├«t pas de cat├®gorie connue.  \nCela peut ├¬tre une ┼ôuvre d'art, un paysage ou un objet non classifiable."},
}

# Labels ImageNet qui indiquent une image plate (peinture, affiche, livre...)
# ÔåÆ forcer "Inconnu" directement, sans passer par le mapper
BLACKLIST_FLAT_IMAGE = {
    "book jacket", "dust cover", "dust jacket", "dust wrapper",
    "jigsaw puzzle", "envelope", "packet",
    "menu", "web site", "screen", "monitor", "television",
    "poster", "album", "cd",
    # Note: "comic book" retir├® ÔåÆ mappe vers Personnage Fictif
}

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# D├ëFIS POUR LE JEU (chasse aux objets)
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# keywords=None  ÔåÆ n'importe quel objet de la cat├®gorie
# keywords=[...] ÔåÆ un label pr├®cis doit ├¬tre d├®tect├®
DEFIS_POOL = [
    {
        "texte": "­ƒì¥ Trouve une BOUTEILLE !",
        "categorie": "Objet", "keywords": ["bottle", "wine bottle", "beer bottle", "water bottle"],
        "temps": 30, "points_max": 200, "emoji": "­ƒì¥",
        "conseil": "Cherche dans ta cuisine ou sur ton bureau !",
    },
    {
        "texte": "­ƒÉÂ Trouve un ANIMAL !",
        "categorie": "Animal", "keywords": None,
        "temps": 25, "points_max": 250, "emoji": "­ƒÉÂ",
        "conseil": "Un vrai animal, une peluche... sois cr├®atif !",
    },
    {
        "texte": "­ƒî▒ Trouve une PLANTE !",
        "categorie": "Plante", "keywords": None,
        "temps": 30, "points_max": 200, "emoji": "­ƒî▒",
        "conseil": "Une fleur, une plante d'int├®rieur, un arbre par la fen├¬tre !",
    },
    {
        "texte": "­ƒìò Trouve de la NOURRITURE !",
        "categorie": "Nourriture", "keywords": None,
        "temps": 25, "points_max": 200, "emoji": "­ƒìò",
        "conseil": "Direction la cuisine ! Frigo, placards...",
    },
    {
        "texte": "Ôÿò Trouve une TASSE ou un BOL !",
        "categorie": "Objet", "keywords": ["cup", "bowl", "coffee mug"],
        "temps": 20, "points_max": 300, "emoji": "Ôÿò",
        "conseil": "Sur ton bureau ? Dans la cuisine ?",
    },
    {
        "texte": "­ƒæñ Montre un HUMAIN !",
        "categorie": "Humain", "keywords": None,
        "temps": 20, "points_max": 300, "emoji": "­ƒæñ",
        "conseil": "Montre-toi, appelle quelqu'un, ou trouve une photo !",
    },
    {
        "texte": "­ƒôÜ Trouve un LIVRE !",
        "categorie": "Objet", "keywords": ["book"],
        "temps": 25, "points_max": 200, "emoji": "­ƒôÜ",
        "conseil": "Dans ta biblioth├¿que ou sur ta table !",
    },
    {
        "texte": "Ô£é´©Å Trouve des CISEAUX !",
        "categorie": "Objet", "keywords": ["scissors"],
        "temps": 35, "points_max": 250, "emoji": "Ô£é´©Å",
        "conseil": "Tiroir de bureau, trousse scolaire...",
    },
    {
        "texte": "­ƒÅå Trouve un OBJET DE SPORT !",
        "categorie": "Sport", "keywords": None,
        "temps": 35, "points_max": 250, "emoji": "­ƒÅå",
        "conseil": "Raquette, ballon, halt├¿res... cherche bien !",
    },
    {
        "texte": "­ƒî© Trouve une FLEUR !",
        "categorie": "Plante", "keywords": ["flower", "rose", "daisy", "tulip", "sunflower", "dandelion"],
        "temps": 30, "points_max": "200", "emoji": "­ƒî©",
        "conseil": "Dehors, sur une photo, ou dans un vase !",
    },
    {
        "texte": "­ƒÆ╗ Trouve un ORDINATEUR ou un T├ëL├ëPHONE !",
        "categorie": "Objet", "keywords": ["laptop", "keyboard", "phone", "mouse"],
        "temps": 15, "points_max": 350, "emoji": "­ƒÆ╗",
        "conseil": "Facile... tu dois en avoir un pr├¿s de toi !",
    },
    {
        "texte": "­ƒöæ Trouve des CL├ëS ou un SAC !",
        "categorie": "Objet", "keywords": ["backpack", "handbag", "suitcase", "wallet"],
        "temps": 30, "points_max": 250, "emoji": "­ƒöæ",
        "conseil": "Pr├¿s de l'entr├®e ou sur ton bureau !",
    },
    {
        "texte": "­ƒôÀ Trouve une LAMPE ou une HORLOGE !",
        "categorie": "Objet", "keywords": ["lamp", "clock"],
        "temps": 25, "points_max": 250, "emoji": "­ƒôÀ",
        "conseil": "Regarde autour de toi dans la pi├¿ce !",
    },
    {
        "texte": "­ƒº╗ Trouve du PAPIER TOILETTE !",
        "categorie": "Objet", "keywords": ["toilet tissue", "paper towel", "toilet paper"],
        "temps": 25, "points_max": 350, "emoji": "­ƒº╗",
        "conseil": "Check les toilettes ou la r├®serve !",
    },
    {
        "texte": "­ƒºª Trouve une CHAUSSETTE !",
        "categorie": "Objet", "keywords": ["sock", "stocking"],
        "temps": 30, "points_max": 300, "emoji": "­ƒºª",
        "conseil": "Dans ta chambre, sur le sol ou dans un tiroir !",
    },
    {
        "texte": "­ƒº¥a Trouve une BROSSE ├Ç DENTS !",
        "categorie": "Objet", "keywords": ["toothbrush"],
        "temps": 30, "points_max": 300, "emoji": "­ƒº¥a",
        "conseil": "Direction la salle de bain !",
    },
    {
        "texte": "­ƒô║ Trouve une T├ëL├ëCOMMANDE !",
        "categorie": "Objet", "keywords": ["remote control", "television"],
        "temps": 25, "points_max": 300, "emoji": "­ƒô║",
        "conseil": "Sur le canap├® ou pr├¿s de la t├®l├® !",
    },
    {
        "texte": "Ô£Å´©Å Trouve un STYLO ou un CRAYON !",
        "categorie": "Objet", "keywords": ["ballpoint pen", "pencil", "crayon"],
        "temps": 20, "points_max": 250, "emoji": "Ô£Å´©Å",
        "conseil": "Sur ton bureau ou dans ta trousse !",
    },
    {
        "texte": "­ƒøÅ´©Å Trouve un COUSSIN ou un OREILLER !",
        "categorie": "Objet", "keywords": ["pillow", "cushion"],
        "temps": 25, "points_max": 250, "emoji": "­ƒøÅ´©Å",
        "conseil": "Sur le canap├® ou dans ta chambre !",
    },
]
# S'assurer que points_max est toujours un int
for _d in DEFIS_POOL:
    _d["points_max"] = int(_d["points_max"])

# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# FONCTION D'ANALYSE
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
def analyser_image(pil_image):
    # R├®cup├¿re le top-5 pour maximiser les chances de trouver une cat├®gorie
    resultats = classifier(pil_image, top_k=5)
    meilleur = resultats[0]
    label_brut = meilleur["label"].lower()
    score_raw = meilleur["score"]

    # Parcourt le top-5 pour trouver la premi├¿re cat├®gorie connue
    # On utilise \b (word boundary) pour ├®viter les faux positifs
    # ex: "car" ne doit PAS matcher "ocarina"
    categorie = "Inconnu"
    label_reconnu = label_brut
    score_reconnu = score_raw  # score du label ayant d├®clench├® la cat├®gorie
    for res in resultats:
        lbl = res["label"].lower()
        # Si le label est dans la liste noire "image plate", on force Inconnu
        if any(bl in lbl for bl in BLACKLIST_FLAT_IMAGE):
            continue
        for mot_cle, cat in CATEGORY_MAPPER.items():
            pattern = r'\b' + re.escape(mot_cle) + r'\b'
            if re.search(pattern, lbl):
                categorie = cat
                label_reconnu = lbl
                score_reconnu = res["score"]
                break
        if categorie != "Inconnu":
            break

    score_pct = f"{score_reconnu * 100:.2f}%"

    # ­ƒÑÜ Easter egg : Baby Yoda (figurine verte avec cloak/ocarina = Baby Yoda tr├¿s probable)
    easter_egg = None
    baby_yoda_signals = ["yoda", "puppet", "teddy", "ocarina", "cloak", "robe"]
    if any(m in label_brut for m in baby_yoda_signals) or \
       any(m in label_reconnu for m in baby_yoda_signals):
        easter_egg = "­ƒƒó BABY YODA D├ëTECT├ë ! La Force est avec vous !"

    return {
        "label_brut": label_brut,
        "label_reconnu": label_reconnu,
        "score_pct": score_pct,
        "score_raw": score_reconnu,
        "categorie": categorie,
        "easter_egg": easter_egg,
    }


def analyser_image_tf(pil_image):
    """Analyse une image avec MobileNetV2 (TensorFlow/Keras) et retourne la cat├®gorie TP."""
    if _tf_model is None:
        return None
    import numpy as np
    img        = pil_image.resize((224, 224)).convert("RGB")
    img_array  = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    img_array  = _tf_preprocess(img_array)
    preds      = _tf_model.predict(img_array, verbose=0)
    top5       = _tf_decode(preds, top=5)[0]   # [(id, nom, score), ...]

    label_brut = top5[0][1].lower().replace("_", " ")
    categorie     = "Inconnu"
    label_reconnu = label_brut
    score_reconnu = float(top5[0][2])

    for _, class_name, score in top5:
        lbl = class_name.lower().replace("_", " ")
        if any(bl in lbl for bl in BLACKLIST_FLAT_IMAGE):
            continue
        for mot_cle, cat in CATEGORY_MAPPER.items():
            pattern = r'\b' + re.escape(mot_cle) + r'\b'
            if re.search(pattern, lbl):
                categorie     = cat
                label_reconnu = lbl
                score_reconnu = float(score)
                break
        if categorie != "Inconnu":
            break

    return {
        "label_brut":     label_brut,
        "label_reconnu":  label_reconnu,
        "score_pct":      f"{score_reconnu * 100:.2f}%",
        "score_raw":      score_reconnu,
        "categorie":      categorie,
    }


# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# PROCESSEUR VID├ëO TEMPS R├ëEL (Webcam)
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# Dictionnaire partag├® entre le thread principal et VideoProcessor pour l'overlay 007
_g007_overlay = {"active": False, "text": "", "color": (255, 200, 0)}


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result    = None
        self.result_tf = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.last_frame_pil = None   # derni├¿re frame (pour d├®tection cam├®ra pr├¬te)
        self.frame_buffer   = deque(maxlen=30)  # ~1s ├á 30fps pour vote multi-frames

    def recv(self, frame):
        img_array = frame.to_ndarray(format="rgb24")
        self.frame_count += 1

        pil_current = Image.fromarray(img_array)
        # Toujours stocker la derni├¿re frame et alimenter le buffer multi-frames
        with self.lock:
            self.last_frame_pil = pil_current
            self.frame_buffer.append(pil_current)

        # Analyse 1 frame sur 60 (~toutes les 2s ├á 30fps)
        if self.frame_count % 60 == 0:
            result    = analyser_image(pil_current)
            result_tf = analyser_image_tf(pil_current)
            if result["categorie"] == "Humain":
                nom_v, conf_v = reconnaitre_visage(pil_current)
                result["nom_visage"] = nom_v
                result["conf_visage"] = conf_v
            else:
                result["nom_visage"] = None
                result["conf_visage"] = None
            with self.lock:
                self.result    = result
                self.result_tf = result_tf

        # ÔöÇÔöÇ Overlay r├®sultat cat├®gorie ÔöÇÔöÇ
        with self.lock:
            current_result = self.result

        if current_result and not _g007_overlay["active"]:
            pil_draw = Image.fromarray(img_array)
            draw = ImageDraw.Draw(pil_draw)
            cfg = CATEGORY_CONFIG.get(current_result["categorie"], CATEGORY_CONFIG["Inconnu"])
            h, w = img_array.shape[:2]
            draw.rectangle([(0, h - 65), (w, h)], fill=(0, 0, 0))
            display_name = current_result.get("nom_visage") or current_result["categorie"]
            prefix = f"­ƒæï {display_name}" if current_result.get("nom_visage") else f"{cfg['emoji']}  {display_name}"
            draw.text((12, h - 52), prefix, fill=(100, 255, 100) if current_result.get("nom_visage") else (255, 255, 255))
            draw.text((12, h - 28), f"Confiance : {current_result['score_pct']}  |  {current_result['label_reconnu']}", fill=(180, 180, 180))
            img_array = np.array(pil_draw)

        # ÔöÇÔöÇ Overlay compte ├á rebours 007 ÔöÇÔöÇ
        if _g007_overlay["active"] and _g007_overlay["text"]:
            pil_draw = Image.fromarray(img_array)
            draw = ImageDraw.Draw(pil_draw)
            h, w = img_array.shape[:2]
            txt = _g007_overlay["text"]
            col = _g007_overlay["color"]
            try:
                font_size = {1: 160, 2: 150}.get(len(txt), 130)
                font_big = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font_big = ImageFont.load_default()
            # Ombre + texte centr├®
            try:
                bb = draw.textbbox((0, 0), txt, font=font_big)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                tw, th = 100, 100
            x, y = (w - tw) // 2, (h - th) // 2 - 20
            draw.text((x + 5, y + 5), txt, fill=(0, 0, 0), font=font_big)
            draw.text((x, y), txt, fill=col, font=font_big)
            img_array = np.array(pil_draw)

        return av.VideoFrame.from_ndarray(img_array, format="rgb24")


# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# INTERFACE PRINCIPALE
# ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
st.title("­ƒöì VisionIA ÔÇô Reconnaissance d'Images")
st.caption("D├®tection automatique : Humains ┬À Personnages Fictifs ┬À Animaux ┬À Plantes ┬À V├®hicules ┬À Nourriture ┬À Sport ┬À Objets ┬À Nature")
st.markdown("---")

tab_analyse, tab_jeu, tab_visages, tab_007 = st.tabs([
    "­ƒöì Analyse d'image",
    "­ƒÄ« Jeu de d├®tection (10 manches)",
    "­ƒæÑ Reconnaissances de visages",
    "­ƒö½ 007 Duel",
])

# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
# ONGLET 1 ÔÇô ANALYSE
# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
with tab_analyse:
    # ÔöÇÔöÇ Deux colonnes principales : upload | r├®sultat ÔöÇÔöÇ
    zone_upload, zone_resultat = st.columns([1, 1], gap="large")

    with zone_upload:
        st.subheader("­ƒô© Capture ou Upload")
        mode = st.radio("Source de l'image :", ["­ƒôÀ Webcam (temps r├®el)", "­ƒôü Fichier"], horizontal=True)

        image_source = None
        uploaded_file = None
        ctx = None

        if mode == "­ƒôÀ Webcam (temps r├®el)":
            st.caption("ÔÜí La cat├®gorie s'affiche sur la vid├®o. Cliquez START pour lancer.")
            ctx = webrtc_streamer(
                key="visionai-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            uploaded_file = st.file_uploader("Choisissez un fichier image", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                pil_image = Image.open(uploaded_file).convert("RGB")
                image_source = {"name": uploaded_file.name, "size": uploaded_file.size}
                st.image(pil_image, caption=uploaded_file.name, use_container_width=True)
                st.caption(f"Taille : {image_source['size']} octets")

    with zone_resultat:
        st.subheader("­ƒºá R├®sultat de l'Analyse")

        # ÔöÇÔöÇ MODE WEBCAM TEMPS R├ëEL ÔöÇÔöÇ
        if mode == "­ƒôÀ Webcam (temps r├®el)":
            if ctx and ctx.state.playing and ctx.video_processor:
                with ctx.video_processor.lock:
                    live_result    = ctx.video_processor.result
                    live_result_tf = ctx.video_processor.result_tf

                if live_result:
                    cfg = CATEGORY_CONFIG.get(live_result["categorie"], CATEGORY_CONFIG["Inconnu"])

                    # ÔöÇÔöÇ Nom reconnu ? ÔöÇÔöÇ
                    nom_reconnu = live_result.get("nom_visage")
                    conf_visage = live_result.get("conf_visage")
                    if nom_reconnu:
                        st.markdown(f"""
                        <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:12px;
                                    padding:12px 20px; text-align:center; margin-bottom:8px;'>
                            <span style='font-size:1.8em'>­ƒæï</span>
                            <h3 style='color:#4caf50; margin:4px 0;'>Bonjour <b>{nom_reconnu}</b> !</h3>
                            <p style='color:#aaa; margin:0;'>Confiance visage : {conf_visage}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"## {cfg['emoji']} {live_result['categorie']}")
                        if live_result["categorie"] == "Humain":
                            st.caption("­ƒæñ Visage non reconnu ÔÇö enregistre-toi dans **­ƒæÑ Reconnaissances de visages** !")

                    c1, c2 = st.columns(2)
                    c1.metric("Cat├®gorie d├®tect├®e", live_result["categorie"])
                    c2.metric("Confiance", live_result["score_pct"])
                    st.info(f"Label : `{live_result['label_reconnu']}`")
                    if live_result["easter_egg"]:
                        st.balloons()
                        st.warning(live_result["easter_egg"])
                    else:
                        st.success(cfg["message"])

                    if live_result_tf:
                        cfg_tf_l = CATEGORY_CONFIG.get(live_result_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                        accord   = live_result["categorie"] == live_result_tf["categorie"]
                        st.caption(
                            f"­ƒöÀ **TF MobileNetV2** ÔåÆ {cfg_tf_l['emoji']} {live_result_tf['categorie']} "
                            f"({live_result_tf['score_pct']}) {'✅ accord ViT' if accord else 'ÔÜá´©Å diverge de ViT'}"
                        )

                    st.markdown("---")
                    if st.button("­ƒÆ¥ Sauvegarder cette d├®tection dans MongoDB", type="primary"):
                        try:
                            collection.insert_one({
                                "date": datetime.now(),
                                "nom": "webcam_live.jpg",
                                "taille": 0,
                                "analyse": {
                                    "taux_reussite": live_result["score_pct"],
                                    "type_reconnu": live_result["categorie"],
                                    "label_brut": live_result["label_brut"],
                                    "label_reconnu": live_result["label_reconnu"],
                                }
                            })
                            st.success("Ô£à Sauvegard├® dans MongoDB !")
                        except Exception as e:
                            st.error(f"Erreur MongoDB : {e}")
                else:
                    st.info("ÔÅ│ En attente de la premi├¿re analyse... (environ 2 secondes apr├¿s START)")
            else:
                st.info("ÔûÂ´©Å Cliquez sur START dans le flux vid├®o pour lancer la d├®tection en temps r├®el.")

        # ÔöÇÔöÇ MODE FICHIER ÔöÇÔöÇ
        elif image_source is not None:
            with st.spinner("Analyse en cours par le mod├¿le ViT..."):
                resultat = analyser_image(pil_image)

            cfg = CATEGORY_CONFIG.get(resultat["categorie"], CATEGORY_CONFIG["Inconnu"])

            st.markdown(f"## {cfg['emoji']} {resultat['categorie']}")

            # ÔöÇÔöÇ Sauvegarde MongoDB (imm├®diatement visible) ÔöÇÔöÇ
            try:
                document = {
                    "date": datetime.now(),
                    "nom": image_source["name"],
                    "taille": image_source["size"],
                    "analyse": {
                        "taux_reussite": resultat["score_pct"],
                        "type_reconnu": resultat["categorie"],
                        "label_brut": resultat["label_brut"],
                        "label_reconnu": resultat["label_reconnu"],
                    }
                }
                collection.insert_one(document)
                st.success("Ô£à R├®sultat enregistr├® dans MongoDB !")
            except Exception as e:
                st.error(f"Erreur MongoDB : {e}")

            # ÔöÇÔöÇ Reconnaissance faciale si Humain ÔöÇÔöÇ
            nom_reconnu, conf_visage = None, None
            if resultat["categorie"] == "Humain":
                with st.spinner("­ƒæÑ Recherche du visage dans le registre..."):
                    nom_reconnu, conf_visage = reconnaitre_visage(pil_image)
            if nom_reconnu:
                st.markdown(f"""
                <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:12px;
                            padding:16px 24px; text-align:center; margin-bottom:12px;'>
                    <span style='font-size:2em'>­ƒæï</span>
                    <h3 style='color:#4caf50; margin:6px 0;'>Bonjour <b>{nom_reconnu}</b> !</h3>
                    <p style='color:#aaa; margin:0;'>Confiance : {conf_visage}</p>
                </div>
                """, unsafe_allow_html=True)
            elif resultat["categorie"] == "Humain":
                st.caption("­ƒæñ Visage non reconnu ÔÇö enregistre-toi dans l'onglet **­ƒæÑ Reconnaissances de visages** !")

            c1, c2 = st.columns(2)
            c1.metric("Cat├®gorie d├®tect├®e", resultat["categorie"])
            c2.metric("Taux de r├®ussite", resultat["score_pct"])
            st.info(f"Label principal du mod├¿le : `{resultat['label_brut']}`  \nLabel ayant d├®clench├® la cat├®gorie : `{resultat['label_reconnu']}`")

            if resultat["easter_egg"]:
                st.balloons()
                st.warning(resultat["easter_egg"])
            else:
                st.success(cfg["message"])

            # ÔöÇÔöÇ Comparaison TensorFlow MobileNetV2 ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
            st.markdown("---")
            st.subheader("­ƒåÜ Comparaison des mod├¿les")
            with st.spinner("Analyse TensorFlow / MobileNetV2 en cours..."):
                resultat_tf = analyser_image_tf(pil_image)

            if resultat_tf is None:
                st.warning("ÔÜá´©Å TensorFlow n'est pas disponible dans cet environnement.")
            else:
                cfg_tf  = CATEGORY_CONFIG.get(resultat_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                cfg_vit = CATEGORY_CONFIG.get(resultat["categorie"],    CATEGORY_CONFIG["Inconnu"])

                col_vit, col_tf = st.columns(2)
                with col_vit:
                    st.markdown("""
                    <div style='background:#1a1a2e; border:2px solid #4f8ef7;
                                border-radius:10px; padding:14px; text-align:center;'>
                        <p style='color:#4f8ef7; font-weight:bold; margin:0 0 6px;'>­ƒñû ViT ÔÇô HuggingFace / PyTorch</p>
                    """, unsafe_allow_html=True)
                    st.metric("Cat├®gorie", f"{cfg_vit['emoji']} {resultat['categorie']}")
                    st.metric("Confiance", resultat["score_pct"])
                    st.caption(f"Label : `{resultat['label_reconnu']}`")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_tf:
                    st.markdown("""
                    <div style='background:#1a2e1a; border:2px solid #f7a24f;
                                border-radius:10px; padding:14px; text-align:center;'>
                        <p style='color:#f7a24f; font-weight:bold; margin:0 0 6px;'>­ƒöÀ MobileNetV2 ÔÇô TensorFlow / Keras</p>
                    """, unsafe_allow_html=True)
                    st.metric("Cat├®gorie", f"{cfg_tf['emoji']} {resultat_tf['categorie']}")
                    st.metric("Confiance", resultat_tf["score_pct"])
                    st.caption(f"Label : `{resultat_tf['label_reconnu']}`")
                    st.markdown("</div>", unsafe_allow_html=True)

                if resultat["categorie"] == resultat_tf["categorie"]:
                    st.success(f"Ô£à Les deux mod├¿les sont **d'accord** : **{resultat['categorie']}**")
                else:
                    st.warning(f"ÔÜá´©Å Les mod├¿les **divergent** : ViT ÔåÆ **{resultat['categorie']}** | TensorFlow ÔåÆ **{resultat_tf['categorie']}**")
        else:
            st.info("Prenez une photo ou uploadez une image ├á gauche pour lancer l'analyse.")

    # ÔöÇÔöÇ Historique ÔöÇÔöÇ
    st.markdown("---")
    st.header("­ƒôï Historique des Analyses")

    historique_docs = list(collection.find().sort("date", -1))

    if historique_docs:
        historique_a_afficher = []
        for doc in historique_docs:
            analyse = doc.get("analyse", {})
            categorie = analyse.get("type_reconnu", "Inconnu")
            cfg = CATEGORY_CONFIG.get(categorie, CATEGORY_CONFIG["Inconnu"])
            historique_a_afficher.append({
                "ID": str(doc["_id"]),
                "Date": doc["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "Nom du fichier": doc.get("nom", "N/A"),
                "Taille (octets)": doc.get("taille", "N/A"),
                "Cat├®gorie": f"{cfg['emoji']} {categorie}",
                "Taux de r├®ussite": analyse.get("taux_reussite", "N/A"),
                "Label brut": analyse.get("label_brut", "N/A"),
            })

        df = pd.DataFrame(historique_a_afficher)
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

        st.markdown("### ­ƒôè R├®partition des cat├®gories d├®tect├®es")
        comptage = df["Cat├®gorie"].value_counts().reset_index()
        comptage.columns = ["Cat├®gorie", "Nombre d'analyses"]
        st.bar_chart(data=comptage.set_index("Cat├®gorie"))

        st.markdown("### ­ƒùæ´©Å Supprimer un enregistrement")
        options = {
            f"{r['Nom du fichier']} ÔÇô {r['Date']} ({r['Cat├®gorie']})": r["ID"]
            for r in historique_a_afficher
        }
        element = st.selectbox("S├®lectionner l'image ├á supprimer :", list(options.keys()))

        c_del, c_all = st.columns([1, 2])
        with c_del:
            if st.button("­ƒùæ´©Å Supprimer cet enregistrement", type="primary"):
                try:
                    collection.delete_one({"_id": ObjectId(options[element])})
                    st.success("Enregistrement supprim├® !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
        with c_all:
            if st.button("­ƒº╣ Vider tout l'historique"):
                collection.delete_many({})
                st.warning("Historique enti├¿rement vid├®.")
                st.rerun()
    else:
        st.info("Aucune analyse enregistr├®e pour le moment. Uploadez une image ci-dessus !")


# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
# ONGLET 2 ÔÇô JEU DE D├ëTECTION (Chasse aux Objets)
# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
with tab_jeu:
    NB_MANCHES = 10

    # ÔöÇÔöÇ Initialisation session state ÔöÇÔöÇ
    for _key, _val in [
        ("game_active", False), ("game_over", False), ("game_round", 1),
        ("game_score", 0), ("game_history", []), ("game_defi_order", []),
        ("game_start_time", 0.0), ("game_round_won", False),
    ]:
        if _key not in st.session_state:
            st.session_state[_key] = _val

    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    # ├ëCRAN D'ACCUEIL
    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    if not st.session_state.game_active and not st.session_state.game_over:
        st.markdown("""
        <div style='text-align:center; padding: 50px 20px;'>
            <h1 style='font-size:3em;'>­ƒÄ« Chasse aux Objets !</h1>
            <p style='font-size:1.3em; color: #aaa;'>
                Un d├®fi s'affiche ÔÇö tu as un <b>temps limit├®</b> pour rapporter l'objet devant la cam├®ra !<br>
                Plus tu es rapide, plus tu gagnes de points. ÔÜí<br><br>
                <b>10 manches &nbsp;┬À&nbsp; Chrono &nbsp;┬À&nbsp; Bonus vitesse &nbsp;┬À&nbsp; Classement final</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("­ƒÜÇ LANCER LA PARTIE !", type="primary", use_container_width=True):
                st.session_state.game_active = True
                st.session_state.game_round = 1
                st.session_state.game_score = 0
                st.session_state.game_history = []
                st.session_state.game_over = False
                st.session_state.game_round_won = False
                st.session_state.game_defi_order = random.sample(DEFIS_POOL, NB_MANCHES)
                st.session_state.game_start_time = time.time()
                st.rerun()

    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    # ├ëCRAN FIN DE PARTIE
    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    elif st.session_state.game_over:
        score_final = st.session_state.game_score
        score_max = sum(d["points_max"] for d in st.session_state.game_defi_order)
        pct = int(score_final / score_max * 100) if score_max > 0 else 0

        if pct >= 80:
            titre, medal = "CHAMPION ABSOLU !", "­ƒÅå"
        elif pct >= 60:
            titre, medal = "Tr├¿s bien jou├® !", "­ƒÑç"
        elif pct >= 40:
            titre, medal = "Pas mal du tout !", "­ƒÑê"
        else:
            titre, medal = "Continue de t'entra├«ner !", "­ƒÆ¬"

        st.markdown(f"""
        <div style='text-align:center; padding: 30px 0;'>
            <h1 style='font-size:3.5em;'>{medal}</h1>
            <h2>{titre}</h2>
            <h3 style='color:#e94560;'>Score final : {score_final} / {score_max} pts &nbsp;({pct}%)</h3>
        </div>
        """, unsafe_allow_html=True)

        if pct >= 60:
            st.balloons()

        st.markdown("### ­ƒôï R├®capitulatif des 10 manches")
        recap_data = []
        for i, h in enumerate(st.session_state.game_history):
            recap_data.append({
                "Manche": f"#{i + 1}",
                "D├®fi": h["defi"],
                "R├®sultat": "Ô£à Gagn├®e" if h["won"] else "ÔØî Perdue",
                "Points": h["points"],
                "Temps": f"{h['temps_pris']:.1f}s" if h["won"] else "ÔÇö",
            })
        st.dataframe(pd.DataFrame(recap_data), use_container_width=True, hide_index=True)

        _, col_r, _ = st.columns([1, 2, 1])
        with col_r:
            if st.button("­ƒöä Rejouer une nouvelle partie !", type="primary", use_container_width=True):
                for k in ["game_active", "game_over", "game_round", "game_score",
                          "game_history", "game_defi_order", "game_round_won"]:
                    del st.session_state[k]
                st.rerun()

    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    # PARTIE EN COURS
    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    else:
        round_idx = st.session_state.game_round - 1
        defi = st.session_state.game_defi_order[round_idx]
        elapsed = time.time() - st.session_state.game_start_time
        remaining = max(0.0, defi["temps"] - elapsed)
        time_ratio = remaining / defi["temps"]

        # ÔöÇÔöÇ Header : manche / score / chrono ÔöÇÔöÇ
        hc1, hc2, hc3 = st.columns([2, 1, 1])
        with hc1:
            st.markdown(f"### Manche **{st.session_state.game_round}** / {NB_MANCHES}")
        with hc2:
            st.metric("­ƒÅà Score", f"{st.session_state.game_score} pts")
        with hc3:
            color = "­ƒƒó" if time_ratio > 0.5 else ("­ƒƒí" if time_ratio > 0.2 else "­ƒö┤")
            st.metric(f"{color} Temps", f"{int(remaining)}s")

        st.progress(time_ratio)

        # ÔöÇÔöÇ Grand encart du d├®fi ÔöÇÔöÇ
        border_color = "#e94560" if time_ratio > 0.25 else "#ff0000"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border: 3px solid {border_color}; border-radius: 20px;
                    padding: 40px 30px; text-align: center; margin: 15px 0;'>
            <div style='font-size: 4em; margin-bottom: 10px;'>{defi["emoji"]}</div>
            <h2 style='color: white; font-size: 2.2em; margin: 0 0 12px 0;'>{defi["texte"]}</h2>
            <p style='color: #aaa; font-size: 1.1em; margin: 0;'>­ƒÆí {defi["conseil"]}</p>
            <p style='color: #e94560; font-size: 1em; margin-top: 10px;'>
                ÔÅ▒ {defi["temps"]}s max &nbsp;┬À&nbsp; ­ƒÄ» jusqu'├á {defi["points_max"]} pts
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ÔöÇÔöÇ Webcam jeu | R├®sultat c├┤te ├á c├┤te ÔöÇÔöÇ
        gcol1, gcol2 = st.columns([1, 1], gap="large")

        with gcol1:
            st.caption("­ƒôÀ Lance la cam├®ra et pointe-la vers l'objet !")
            ctx_game = webrtc_streamer(
                key="game-webcam",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with gcol2:
            st.subheader("­ƒÄ» D├®tection en cours...")
            game_result    = None
            game_result_tf = None
            if ctx_game and ctx_game.state.playing and ctx_game.video_processor:
                with ctx_game.video_processor.lock:
                    game_result    = ctx_game.video_processor.result
                    game_result_tf = ctx_game.video_processor.result_tf

            if game_result:
                detected_cat = game_result["categorie"]
                cfg_det = CATEGORY_CONFIG.get(detected_cat, CATEGORY_CONFIG["Inconnu"])

                # V├®rifier si le r├®sultat correspond au d├®fi
                kws = defi.get("keywords")
                won = False
                if detected_cat == defi["categorie"]:
                    if kws:
                        for kw in kws:
                            if kw in game_result["label_reconnu"] or kw in game_result["label_brut"]:
                                won = True
                                break
                    else:
                        won = True

                if won:
                    st.markdown(f"### Ô£à {cfg_det['emoji']} {detected_cat}")
                    st.success(f"**C'est bien ├ºa !** Label : `{game_result['label_reconnu']}` ÔÇô {game_result['score_pct']}")
                else:
                    st.markdown(f"**D├®tect├® :** {cfg_det['emoji']} {detected_cat}")
                    st.caption(f"Label : `{game_result['label_reconnu']}` ÔÇö {game_result['score_pct']}")
                    st.warning(f"Ce n'est pas ├ºa... cherche un(e) **{defi['categorie']}** !")

                if game_result_tf:
                    cfg_tf_g = CATEGORY_CONFIG.get(game_result_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                    accord_g = game_result["categorie"] == game_result_tf["categorie"]
                    st.caption(
                        f"­ƒöÀ TF MobileNetV2 : {cfg_tf_g['emoji']} **{game_result_tf['categorie']}** "
                        f"({game_result_tf['score_pct']}) {'✅' if accord_g else 'ÔÜá´©Å diverge'}"
                    )

                # ÔöÇÔöÇ Victoire ÔöÇÔöÇ
                if won and not st.session_state.game_round_won:
                    temps_pris = elapsed
                    if temps_pris < defi["temps"] * 0.25:
                        bonus = int(defi["points_max"] * 0.5)
                        bonus_txt = f"ÔÜí Bonus ├ëCLAIR +{bonus} pts !"
                    elif temps_pris < defi["temps"] * 0.5:
                        bonus = int(defi["points_max"] * 0.25)
                        bonus_txt = f"­ƒÜÇ Bonus RAPIDE +{bonus} pts !"
                    else:
                        bonus, bonus_txt = 0, ""

                    pts = defi["points_max"] + bonus
                    st.session_state.game_score += pts
                    st.session_state.game_round_won = True
                    st.session_state.game_history.append({
                        "defi": defi["texte"], "won": True,
                        "points": pts, "temps_pris": temps_pris,
                    })
                    st.balloons()
                    st.success(f"­ƒÄë TROUV├ë en {temps_pris:.1f}s ! **+{pts} pts**" +
                               (f"  \n{bonus_txt}" if bonus_txt else ""))
                    time.sleep(2.5)
                    if st.session_state.game_round >= NB_MANCHES:
                        st.session_state.game_over = True
                        st.session_state.game_active = False
                    else:
                        st.session_state.game_round += 1
                        st.session_state.game_start_time = time.time()
                        st.session_state.game_round_won = False
                    st.rerun()
            else:
                st.info("ÔÅ│ Lance la cam├®ra et pointe-la vers l'objet !")

        # ÔöÇÔöÇ Temps ├®coul├® ÔöÇÔöÇ
        if remaining <= 0 and not st.session_state.game_round_won:
            st.error(f"ÔÅ░ TEMPS ├ëCOUL├ë ! Il fallait trouver : **{defi['texte']}**")
            st.session_state.game_history.append({
                "defi": defi["texte"], "won": False, "points": 0, "temps_pris": 0,
            })
            time.sleep(2.5)
            if st.session_state.game_round >= NB_MANCHES:
                st.session_state.game_over = True
                st.session_state.game_active = False
            else:
                st.session_state.game_round += 1
                st.session_state.game_start_time = time.time()
                st.session_state.game_round_won = False
            st.rerun()

        # ÔöÇÔöÇ Barre de progression des manches ÔöÇÔöÇ
        st.markdown("---")
        manche_cols = st.columns(NB_MANCHES)
        for i, col in enumerate(manche_cols):
            mn = i + 1
            if mn < st.session_state.game_round:
                h = st.session_state.game_history[i] if i < len(st.session_state.game_history) else None
                icon = "Ô£à" if (h and h["won"]) else "ÔØî"
            elif mn == st.session_state.game_round:
                icon = "­ƒÄ»"
            else:
                icon = "Ô¼£"
            col.markdown(
                f"<div style='text-align:center'>{icon}<br><small style='color:#aaa'>#{mn}</small></div>",
                unsafe_allow_html=True,
            )

        # ÔöÇÔöÇ Bouton passer ÔöÇÔöÇ
        if st.button("ÔÅ¡´©Å Passer cette manche (0 pts)"):
            st.session_state.game_history.append({
                "defi": defi["texte"], "won": False, "points": 0, "temps_pris": 0,
            })
            if st.session_state.game_round >= NB_MANCHES:
                st.session_state.game_over = True
                st.session_state.game_active = False
            else:
                st.session_state.game_round += 1
                st.session_state.game_start_time = time.time()
                st.session_state.game_round_won = False
            st.rerun()

        # ÔöÇÔöÇ Auto-refresh chrono (toutes les 0.8s) ÔöÇÔöÇ
        if st.session_state.game_active and not st.session_state.game_round_won:
            time.sleep(0.8)
            st.rerun()


# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
# ONGLET 3 ÔÇô GESTION DES VISAGES
# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
with tab_visages:
    st.header("­ƒæÑ Registre des visages")
    st.caption("Enregistre ton visage ici ÔÇö l'IA te reconna├«tra dans l'onglet Analyse ou pendant le jeu !")
    st.markdown("---")

    vcol1, vcol2 = st.columns([1, 1], gap="large")

    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # ENREGISTREMENT
    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    with vcol1:
        st.subheader("Ô×ò Ajouter un visage")
        nom_input = st.text_input("­ƒôØ Pr├®nom ou nom ├á associer :", placeholder="ex: Paul, Marie...")
        reg_mode = st.radio("Source :", ["­ƒôÀ Webcam", "­ƒôé Fichier"], horizontal=True)

        reg_img = None
        if reg_mode == "­ƒôÀ Webcam":
            reg_photo = st.camera_input("Prends une photo de ton visage")
            if reg_photo:
                reg_img = Image.open(reg_photo).convert("RGB")
                st.image(reg_img, caption="Photo captur├®e", use_container_width=True)
        else:
            reg_file = st.file_uploader("Importe une photo", type=["jpg", "jpeg", "png"],
                                        key="reg_face_upload")
            if reg_file:
                reg_img = Image.open(reg_file).convert("RGB")
                st.image(reg_img, caption=reg_file.name, use_container_width=True)

        if st.button("­ƒÆ¥ Enregistrer ce visage", type="primary", disabled=(not nom_input or reg_img is None)):
            ok, msg = enregistrer_visage(reg_img, nom_input.strip())
            if ok:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)

        st.markdown("")
        st.markdown("""
        **Conseils pour un bon enregistrement :**
        - Visage face ├á la cam├®ra, bien ├®clair├®
        - Ajoute 2-3 photos sous des angles diff├®rents pour am├®liorer la pr├®cision
        - ├ëvite les lunettes de soleil ou le masque
        """)

    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # REGISTRE ACTUEL
    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    with vcol2:
        st.subheader("­ƒôï Personnes enregistr├®es")
        faces_db = charger_db_visages()

        if not faces_db:
            st.info("Aucun visage enregistr├® pour l'instant. Ajoute-toi ├á gauche !")
        else:
            for nom, embeddings in faces_db.items():
                col_n, col_d = st.columns([3, 1])
                col_n.markdown(f"­ƒæñ **{nom}** ÔÇö {len(embeddings)} photo(s)")
                if col_d.button("Supprimer", key=f"del_{nom}"):
                    del faces_db[nom]
                    sauvegarder_db_visages(faces_db)
                    st.success(f"{nom} supprim├® du registre.")
                    st.rerun()

            st.markdown("---")
            if st.button("­ƒùæ´©Å Effacer tout le registre"):
                sauvegarder_db_visages({})
                st.warning("Registre vid├®.")
                st.rerun()

    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    # TEST DE RECONNAISSANCE
    # ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
    st.markdown("---")
    st.subheader("­ƒºÉ Tester la reconnaissance")
    test_mode = st.radio("Source du test :", ["­ƒôÀ Webcam", "­ƒôé Fichier"], horizontal=True, key="test_mode")

    test_img = None
    if test_mode == "­ƒôÀ Webcam":
        test_snap = st.camera_input("Prends une photo pour tester")
        if test_snap:
            test_img = Image.open(test_snap).convert("RGB")
    else:
        test_file = st.file_uploader("Importe une photo test", type=["jpg", "jpeg", "png"],
                                     key="test_face_upload")
        if test_file:
            test_img = Image.open(test_file).convert("RGB")

    if test_img:
        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            st.image(test_img, caption="Image test", use_container_width=True)
        with tcol2:
            with st.spinner("­ƒöì Analyse du visage..."):
                nom_test, conf_test = reconnaitre_visage(test_img)
            if nom_test:
                st.markdown(f"""
                <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:16px;
                            padding:24px; text-align:center; margin-top:20px;'>
                    <div style='font-size:3em'>­ƒæï</div>
                    <h2 style='color:#4caf50;'>Bonjour <b>{nom_test}</b> !</h2>
                    <p style='color:#aaa;'>Similarit├® : <b>{conf_test}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                db_check = charger_db_visages()
                if not db_check:
                    st.warning("ÔÜá´©Å Aucun visage enregistr├®. Ajoute-toi d'abord !")
                else:
                    st.error("ÔØî Visage non reconnu dans le registre.")
                    st.caption("Essaie d'ajouter plus de photos sous diff├®rents angles.")


# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
# ONGLET 4 ÔÇô JEU 007
# ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
with tab_007:
    st.header("­ƒö½ 007 ÔÇô Duel contre l'IA")
    st.caption("Apprends ├á l'IA tes gestes, puis affronte-la en duel !")
    st.markdown("---")

    sub_appren, sub_jeu007 = st.tabs(["­ƒôÜ 1. Apprendre les gestes", "­ƒÄ« 2. Jouer"])

    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    # SOUS-ONGLET A ÔÇô APPRENTISSAGE DES GESTES
    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    with sub_appren:
        st.subheader("­ƒô© Enregistre tes gestes")
        st.markdown(
            f"Minimum **{GESTURES_NB_MIN} photos** par geste pour jouer ÔÇö mais plus tu en ajoutes, "
            "plus la d├®tection sera pr├®cise. **Pas besoin d'appuyer pendant le geste** : "
            "clique sur le bouton, puis pose les deux mains et attends le d├®compte !"
        )
        st.markdown("---")

        # ÔöÇÔöÇ Init session state d├®compte ÔöÇÔöÇ
        for _k, _dv in [
            ("glearn_geste_sel", list(GESTURES_CONFIG.keys())[0]),
            ("glearn_cd_start",  None),   # float timestamp ou None
            ("glearn_msg",       ""),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _dv

        db_gestes = charger_db_gestes()

        # ÔöÇÔöÇ Barre de progression globale ÔöÇÔöÇ
        total_done   = sum(len(db_gestes.get(k, [])) for k in GESTURES_CONFIG)
        total_needed = len(GESTURES_CONFIG) * GESTURES_NB_MIN
        pct_done     = min(1.0, total_done / total_needed)
        st.progress(pct_done, text=f"Progression minimum : {total_done}/{total_needed} photos enregistr├®es")
        if total_done >= total_needed:
            st.success("Ô£à Minimum atteint ! Va dans **­ƒÄ« 2. Jouer** pour d├®marrer. Tu peux continuer ├á ajouter des photos pour am├®liorer la pr├®cision.")
        st.markdown("")

        # ÔöÇÔöÇ Compteurs par geste ÔöÇÔöÇ
        cnt_cols = st.columns(len(GESTURES_CONFIG))
        for col, (gk, gcfg) in zip(cnt_cols, GESTURES_CONFIG.items()):
            nb = len(db_gestes.get(gk, []))
            ok = nb >= GESTURES_NB_MIN
            col.markdown(
                f"<div style='background:{'#1a3a1a' if ok else '#2a1a1a'}; "
                f"border:2px solid {'#4caf50' if ok else '#555'}; border-radius:12px; "
                f"padding:14px; text-align:center;'>"
                f"<div style='font-size:2em'>{gcfg['emoji']}</div>"
                f"<b style='color:{'#4caf50' if ok else '#ddd'};'>{gcfg['label']}</b><br>"
                f"<span style='color:#aaa; font-size:0.95em;'>{nb} photo(s)"
                f"{' Ô£à' if ok else f' / {GESTURES_NB_MIN} min'}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ÔöÇÔöÇ S├®lecteur de geste ÔöÇÔöÇ
        geste_labels = {k: f"{v['emoji']} {v['label']}" for k, v in GESTURES_CONFIG.items()}
        geste_sel = st.radio(
            "Quel geste veux-tu capturer ?",
            options=list(GESTURES_CONFIG.keys()),
            format_func=lambda k: geste_labels[k],
            horizontal=True,
            key="glearn_geste_sel",
        )
        gcfg_sel = GESTURES_CONFIG[geste_sel]
        st.caption(f"­ƒÆí {gcfg_sel['desc']}")
        st.markdown("")

        # ÔöÇÔöÇ Webcam unique + d├®compte ÔöÇÔöÇ
        lcol, rcol = st.columns([1.2, 1])

        with lcol:
            ctx_learn = webrtc_streamer(
                key="learn_cam",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with rcol:
            cd_box    = st.empty()
            msg_box   = st.empty()

            cd_start = st.session_state.glearn_cd_start

            # ÔöÇÔöÇ D├®compte en cours ÔöÇÔöÇ
            if cd_start is not None:
                elapsed_cd = time.time() - cd_start
                remaining  = 3.0 - elapsed_cd

                if remaining > 0:
                    # Affichage du d├®compte
                    step = int(remaining) + 1   # 3 ÔåÆ 2 ÔåÆ 1
                    cd_box.markdown(
                        f"<div style='text-align:center; padding:30px; "
                        f"border:3px solid #ffa500; border-radius:16px;'>"
                        f"<p style='color:#aaa; margin:0;'>Pr├®pare ton geste :</p>"
                        f"<h1 style='font-size:5em; margin:4px 0; color:#ffa500;'>{step}</h1>"
                        f"<p style='color:#aaa; font-size:1.1em'>{gcfg_sel['emoji']} {gcfg_sel['label']}</p>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.25)
                    st.rerun()

                else:
                    # ÔöÇÔöÇ Capture ! ÔöÇÔöÇ
                    st.session_state.glearn_cd_start = None
                    captured_frames = []
                    if ctx_learn and ctx_learn.video_processor:
                        with ctx_learn.video_processor.lock:
                            captured_frames = list(ctx_learn.video_processor.frame_buffer)

                    if captured_frames:
                        # Vote sur les frames pour choisir la meilleure frame centrale
                        mid = len(captured_frames) // 2
                        frame_cap = captured_frames[mid]
                        ok_g, msg_g = enregistrer_geste(frame_cap, geste_sel)
                        st.session_state.glearn_msg = msg_g if ok_g else f"ÔØî {msg_g}"
                    else:
                        st.session_state.glearn_msg = "ÔØî Cam├®ra inactive ÔÇö lance la webcam d'abord !"
                    st.rerun()

            # ÔöÇÔöÇ Bouton d├®marrer d├®compte ÔöÇÔöÇ
            else:
                nb_sel = len(db_gestes.get(geste_sel, []))
                cd_box.markdown(
                    f"<div style='text-align:center; padding:30px; "
                    f"border:2px dashed #444; border-radius:16px;'>"
                    f"<div style='font-size:3em'>{gcfg_sel['emoji']}</div>"
                    f"<p style='color:#aaa; margin:8px 0;'>{gcfg_sel['label']}</p>"
                    f"<p style='color:#777; font-size:0.9em;'>{nb_sel} photo(s) enregistr├®e(s)</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if st.button(
                    f"­ƒô© Capturer dans 3 secondes ÔÇö {gcfg_sel['emoji']} {gcfg_sel['label']}",
                    key="btn_capture_geste",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.glearn_cd_start = time.time()
                    st.session_state.glearn_msg = ""
                    st.rerun()

                # Message du dernier enregistrement
                if st.session_state.glearn_msg:
                    if "Ô£à" in st.session_state.glearn_msg:
                        msg_box.success(st.session_state.glearn_msg)
                    else:
                        msg_box.error(st.session_state.glearn_msg)

                # Bouton supprimer ce geste
                nb_sel = len(db_gestes.get(geste_sel, []))
                if nb_sel > 0:
                    if st.button(
                        f"­ƒùæ´©Å Supprimer toutes les photos de {gcfg_sel['label']} ({nb_sel})",
                        key="btn_del_geste"
                    ):
                        db_g2 = charger_db_gestes()
                        db_g2[geste_sel] = []
                        sauvegarder_db_gestes(db_g2)
                        st.session_state.glearn_msg = ""
                        st.rerun()

        st.markdown("---")
        if st.button("­ƒùæ´©Å R├®initialiser TOUS les gestes"):
            sauvegarder_db_gestes({})
            st.session_state.glearn_msg = ""
            st.warning("Tous les gestes ont ├®t├® effac├®s.")
            st.rerun()

    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    # SOUS-ONGLET B ÔÇô JEU
    # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
    with sub_jeu007:

        # ÔöÇÔöÇ Init session state 007 ÔöÇÔöÇ
        for _k, _v in [
            ("g007_active",  False),
            ("g007_over",    False),
            ("g007_j_vies",  JEU007_VIES_MAX),
            ("g007_ia_vies", JEU007_VIES_MAX),
            ("g007_j_balles",  0),
            ("g007_ia_balles", 0),
            ("g007_manche",  1),
            ("g007_history", []),
            ("g007_last",    None),
            ("g007_phase",   "idle"),   # idle | c0a | c0b | c7 | result
            ("g007_phase_t", 0.0),
            ("g007_ia_pre",  None),
            ("g007_prev_state",  None),
            ("g007_prev_action", None),
            ("g007_pending",     None),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        db_g = charger_db_gestes()
        gestes_prets = all(len(db_g.get(k, [])) >= GESTURES_NB_MIN for k in GESTURES_CONFIG)

        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        # ├ëCRAN D'ACCUEIL
        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        if not st.session_state.g007_active and not st.session_state.g007_over:
            _g007_overlay["active"] = False

            if not gestes_prets:
                manquants = [
                    f"{GESTURES_CONFIG[k]['emoji']} {GESTURES_CONFIG[k]['label']} ({len(db_g.get(k,[]))}/{GESTURES_NB_MIN})"
                    for k in GESTURES_CONFIG if len(db_g.get(k, [])) < GESTURES_NB_MIN
                ]
                st.warning("ÔÜá´©Å Enregistre d'abord tous tes gestes dans **­ƒôÜ 1. Apprendre les gestes** !")
                for m in manquants:
                    st.markdown(f"- {m}")
            else:
                qt = charger_qtable()
                nb_etats = len(qt)
                st.markdown(f"""
                <div style='text-align:center; padding:30px 20px;'>
                    <h1 style='font-size:3em;'>­ƒö½ 007 DUEL !</h1>
                    <p style='font-size:1.2em; color:#aaa;'>
                        Affronte l'IA au jeu <b>007</b> !<br>
                        Le compte ├á rebours s'affiche sur la cam├®ra ÔÇö tiens ton geste au <b style='color:#e94560;'>7</b> !<br><br>
                        ­ƒñÖ <b>Recharger</b> &nbsp;┬À&nbsp; ­ƒö½ <b>Tirer</b> &nbsp;┬À&nbsp; ­ƒøí´©Å <b>Se prot├®ger</b><br><br>
                        <b>3 vies chacun ÔÇö le premier ├á 0 perd !</b><br>
                        <small style='color:#666;'>­ƒºá Q-table IA : {nb_etats} ├®tats appris</small>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                _, cbtn, _ = st.columns([1, 2, 1])
                with cbtn:
                    if st.button("­ƒÜÇ LANCER LE DUEL !", type="primary", use_container_width=True):
                        st.session_state.g007_active    = True
                        st.session_state.g007_over      = False
                        st.session_state.g007_j_vies    = JEU007_VIES_MAX
                        st.session_state.g007_ia_vies   = JEU007_VIES_MAX
                        st.session_state.g007_j_balles  = 0
                        st.session_state.g007_ia_balles = 0
                        st.session_state.g007_manche    = 1
                        st.session_state.g007_history   = []
                        st.session_state.g007_last      = None
                        st.session_state.g007_phase     = "c0a"
                        st.session_state.g007_phase_t   = time.time()
                        st.session_state.g007_pending   = None
                        st.rerun()

                # ÔöÇÔöÇ Section entra├«nement auto-jeu ÔöÇÔöÇ
                st.markdown("---")
                st.markdown("#### ­ƒºá Entra├«ner le bot (auto-jeu)")
                with st.expander("­ƒÅô Lancer une session d'auto-entra├«nement", expanded=False):
                    st.caption(
                        "Le bot joue contre lui-m├¬me et met ├á jour sa Q-table. "
                        "L'entra├«nement est **cumulatif** : relance autant que tu veux."
                    )
                    mode_train = st.radio(
                        "Mode", ["Par nombre de parties", "Par dur├®e"],
                        horizontal=True, key="train007_mode"
                    )
                    if mode_train == "Par nombre de parties":
                        nb_parties_train = st.slider(
                            "Nombre de parties", 100, 20000, 2000, step=100,
                            key="train007_nb"
                        )
                        duree_train_s = None
                        label_spinner = f"­ƒÅï´©Å Entra├«nement sur {nb_parties_train} parties..."
                    else:
                        duree_min = st.slider(
                            "Dur├®e (minutes)", 1, 30, 5, step=1,
                            key="train007_min"
                        )
                        nb_parties_train = None
                        duree_train_s    = duree_min * 60
                        label_spinner    = f"­ƒÅï´©Å Entra├«nement pendant {duree_min} min..."

                    if st.button("ÔûÂ´©Å D├®marrer l'entra├«nement", key="train007_go", use_container_width=True):
                        with st.spinner(label_spinner):
                            qt_new, stats_t = s_entrainer_007(
                                nb_parties=nb_parties_train,
                                duree_s=duree_train_s
                            )
                        nb_faites = stats_t["parties"]
                        duree_r   = stats_t["duree_reelle_s"]
                        st.session_state["train007_result"] = {
                            "nb":        nb_faites,
                            "etats":     len(qt_new),
                            "victoires": stats_t["victoires"],
                            "defaites":  stats_t["defaites"],
                            "nuls":      stats_t["nuls"],
                            "tours":     stats_t["tours_total"],
                            "duree_s":   duree_r,
                        }

                    # Affichage persistant du dernier r├®sultat
                    res_t = st.session_state.get("train007_result")
                    if res_t:
                        nb_t  = res_t["nb"]
                        pct_v = res_t["victoires"] / max(nb_t, 1) * 100
                        pct_d = res_t["defaites"]  / max(nb_t, 1) * 100
                        pct_n = res_t["nuls"]      / max(nb_t, 1) * 100
                        dur_r = res_t["duree_s"]
                        vps   = nb_t / dur_r if dur_r > 0 else 0
                        moy_tours = res_t["tours"] / max(nb_t, 1)

                        # Dur├®e lisible
                        if dur_r >= 60:
                            dur_str = f"{int(dur_r // 60)}min {int(dur_r % 60)}s"
                        else:
                            dur_str = f"{dur_r:.1f}s"

                        # Barre de progression victoires/nuls/d├®faites (CSS)
                        bar_v = f"width:{pct_v:.1f}%"
                        bar_n = f"width:{pct_n:.1f}%"
                        bar_d = f"width:{pct_d:.1f}%"

                        st.markdown(f"""
<div style='background:#0e1117; border:1px solid #2a2a3a; border-radius:16px;
            padding:22px 26px; margin-top:14px;'>
  <h4 style='margin:0 0 16px 0; color:#eee;'>­ƒôè Rapport d'entra├«nement</h4>

  <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px;'>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#a78bfa;'>{nb_t:,}</div>
      <div style='color:#888; font-size:.85em;'>parties jou├®es</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#60a5fa;'>{vps:,.0f}</div>
      <div style='color:#888; font-size:.85em;'>parties / seconde</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#f0abfc;'>{dur_str}</div>
      <div style='color:#888; font-size:.85em;'>dur├®e totale</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#34d399;'>{res_t["tours"]:,}</div>
      <div style='color:#888; font-size:.85em;'>tours simul├®s</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#fbbf24;'>{moy_tours:.1f}</div>
      <div style='color:#888; font-size:.85em;'>tours / partie (moy.)</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#38bdf8;'>{res_t["etats"]}</div>
      <div style='color:#888; font-size:.85em;'>├®tats Q-table</div>
    </div>
  </div>

  <div style='margin-bottom:6px; color:#aaa; font-size:.85em;'>R├®sultats IA (sym├®trie auto-jeu)</div>
  <div style='display:flex; border-radius:8px; overflow:hidden; height:28px; margin-bottom:6px;'>
    <div style='{bar_v}; background:#16a34a; display:flex; align-items:center;
                justify-content:center; font-size:.8em; font-weight:bold; color:#fff;
                min-width:30px; overflow:hidden;'>{pct_v:.0f}%</div>
    <div style='{bar_n}; background:#6b7280; display:flex; align-items:center;
                justify-content:center; font-size:.8em; font-weight:bold; color:#fff;
                min-width:24px; overflow:hidden;'>{pct_n:.0f}%</div>
    <div style='{bar_d}; background:#dc2626; display:flex; align-items:center;
                justify-content:center; font-size:.8em; font-weight:bold; color:#fff;
                min-width:30px; overflow:hidden;'>{pct_d:.0f}%</div>
  </div>
  <div style='display:flex; gap:18px; font-size:.82em; color:#888;'>
    <span>­ƒƒ® Victoires {res_t["victoires"]:,}</span>
    <span>Ô¼£ Nuls {res_t["nuls"]:,}</span>
    <span>­ƒƒÑ D├®faites {res_t["defaites"]:,}</span>
  </div>
</div>
                        """, unsafe_allow_html=True)

                        st.markdown("")
                        if st.button("­ƒùæ´©Å Effacer le rapport", key="train007_clear"):
                            del st.session_state["train007_result"]
                            st.rerun()

        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        # ├ëCRAN FIN DE PARTIE
        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        elif st.session_state.g007_over:
            _g007_overlay["active"] = False
            j_vies  = st.session_state.g007_j_vies
            ia_vies = st.session_state.g007_ia_vies

            if j_vies > ia_vies:
                titre, medal, couleur = "TU AS GAGN├ë !",   "­ƒÅå", "#4caf50"
                st.balloons()
            elif ia_vies > j_vies:
                titre, medal, couleur = "L'IA A GAGN├ë...", "­ƒñû", "#e94560"
            else:
                titre, medal, couleur = "├ëGALIT├ë !",       "­ƒñØ", "#ff9800"

            hearts_j  = 'ÔØñ´©Å' * max(0, j_vies)  + '­ƒûñ' * max(0, JEU007_VIES_MAX - j_vies)
            hearts_ia = 'ÔØñ´©Å' * max(0, ia_vies) + '­ƒûñ' * max(0, JEU007_VIES_MAX - ia_vies)
            st.markdown(f"""
            <div style='text-align:center; padding:30px 0;'>
                <h1 style='font-size:3.5em;'>{medal}</h1>
                <h2 style='color:{couleur};'>{titre}</h2>
                <p style='color:#aaa;'>Toi : {hearts_j} &nbsp;&nbsp;|&nbsp;&nbsp; IA : {hearts_ia}</p>
            </div>
            """, unsafe_allow_html=True)

            qt_fin = charger_qtable()
            st.caption(f"­ƒºá Q-table IA mise ├á jour : **{len(qt_fin)} ├®tats**")

            st.markdown("### ­ƒôï Historique des manches")
            hist_data = []
            for i, h in enumerate(st.session_state.g007_history):
                ia_cfg = GESTURES_CONFIG.get(h["ia_geste"], {})
                j_cfg  = GESTURES_CONFIG.get(h["j_geste"],  {})
                hist_data.append({
                    "#":         i + 1,
                    "Ton geste": f"{j_cfg.get('emoji','')} {j_cfg.get('label','?')} ({h.get('j_conf','?')})",
                    "Geste IA":  f"{ia_cfg.get('emoji','')} {ia_cfg.get('label','?')}",
                    "R├®sultat":  h["res_txt"],
                })
            if hist_data:
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)

            _, cr, _ = st.columns([1, 2, 1])
            with cr:
                if st.button("­ƒöä Rejouer !", type="primary", use_container_width=True):
                    for _k in ["g007_active", "g007_over", "g007_j_vies", "g007_ia_vies",
                               "g007_j_balles", "g007_ia_balles", "g007_manche",
                               "g007_history", "g007_last", "g007_phase", "g007_phase_t",
                               "g007_ia_pre", "g007_prev_state", "g007_prev_action", "g007_pending"]:
                        if _k in st.session_state:
                            del st.session_state[_k]
                    st.rerun()

        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        # PARTIE EN COURS
        # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ
        else:
            j_vies    = st.session_state.g007_j_vies
            ia_vies   = st.session_state.g007_ia_vies
            j_balles  = st.session_state.g007_j_balles
            ia_balles = st.session_state.g007_ia_balles
            manche    = st.session_state.g007_manche
            phase     = st.session_state.g007_phase
            elapsed   = time.time() - st.session_state.g007_phase_t

            # ÔöÇÔöÇ Header vies / balles ÔöÇÔöÇ
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("ÔØñ´©Å Tes vies",   'ÔØñ´©Å' * j_vies  + '­ƒûñ' * (JEU007_VIES_MAX - j_vies))
            hc2.metric("­ƒö½ Tes balles", f"{j_balles} / {JEU007_BALLES_MAX}")
            hc3.metric("­ƒñû Vies IA",    'ÔØñ´©Å' * ia_vies + '­ƒûñ' * (JEU007_VIES_MAX - ia_vies))
            hc4.metric("­ƒö½ Balles IA",  f"{ia_balles} / {JEU007_BALLES_MAX}")

            # ÔöÇÔöÇ Rappel des gestes ÔöÇÔöÇ
            gc1, gc2, gc3 = st.columns(3)
            for _col, (_key, _gcfg) in zip([gc1, gc2, gc3], GESTURES_CONFIG.items()):
                _col.markdown(
                    f"<div style='background:{_gcfg['couleur']}22; border:1px solid {_gcfg['couleur']};"
                    f"border-radius:10px; padding:8px; text-align:center;'>"
                    f"<span style='font-size:1.6em'>{_gcfg['emoji']}</span> "
                    f"<b style='color:{_gcfg['couleur']};'>{_gcfg['label']}</b><br>"
                    f"<small style='color:#888;'>{_gcfg['desc']}</small></div>",
                    unsafe_allow_html=True
                )
            st.markdown("")

            # ÔöÇÔöÇ Webcam + r├®sultat c├┤te ├á c├┤te ÔöÇÔöÇ
            wc1, wc2 = st.columns([1, 1], gap="large")

            with wc1:
                ctx_007 = webrtc_streamer(
                    key="game007",
                    video_processor_factory=VideoProcessor,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

            with wc2:
                res_box = st.empty()

                # ÔöÇÔöÇ Phase R├ëSULTAT : affichage du dernier duel ÔöÇÔöÇ
                pending = st.session_state.g007_pending
                if phase == "result" and pending:
                    j_cfg_r  = GESTURES_CONFIG.get(pending["j_geste"],  {})
                    ia_cfg_r = GESTURES_CONFIG.get(pending["ia_geste"], {})
                    j_touche_r  = pending["j_touche"]
                    ia_touche_r = pending["ia_touche"]

                    if ia_touche_r and not j_touche_r:
                        bg_r, titre_r = "#1a3a1a", "Ô£à IA TOUCH├ëE !"
                        border_r = "#4caf50"
                    elif j_touche_r and not ia_touche_r:
                        bg_r, titre_r = "#3a1a1a", "­ƒÆÑ TU ES TOUCH├ë(E) !"
                        border_r = "#e94560"
                    elif j_touche_r and ia_touche_r:
                        bg_r, titre_r = "#3a2a00", "­ƒÆÑ DOUBLE TOUCHE !"
                        border_r = "#ff9800"
                    else:
                        bg_r, titre_r = "#1a1a2e", "= NEUTRE"
                        border_r = "#666"

                    t_res  = 3.0
                    prog_r = max(0.0, 1.0 - elapsed / t_res)

                    res_box.markdown(f"""
                    <div style='background:{bg_r}; border:3px solid {border_r}; border-radius:20px;
                                padding:24px; text-align:center;'>
                        <h2 style='color:{border_r}; margin:0 0 12px 0;'>{titre_r}</h2>
                        <div style='display:flex; justify-content:space-around; margin:16px 0;'>
                            <div>
                                <div style='font-size:2.5em'>{j_cfg_r.get('emoji','?')}</div>
                                <b style='color:#ddd;'>TOI</b><br>
                                <span style='color:#aaa'>{j_cfg_r.get('label','?')}</span><br>
                                <small style='color:#666'>{pending['j_conf']}</small>
                            </div>
                            <div style='font-size:2em; align-self:center;'>ÔÜö´©Å</div>
                            <div>
                                <div style='font-size:2.5em'>{ia_cfg_r.get('emoji','?')}</div>
                                <b style='color:#ddd;'>IA</b><br>
                                <span style='color:#aaa'>{ia_cfg_r.get('label','?')}</span>
                            </div>
                        </div>
                        <hr style='border-color:#444; margin:10px 0;'>
                        {''.join(f"<p style='color:#ccc; margin:4px 0;'>{m}</p>" for m in pending['msgs'])}
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(prog_r)

                elif phase in ("c0a", "c0b", "c7"):
                    label_phase = {
                        "c0a": ("0",   "­ƒÄ» Pr├®pare ton geste..."),
                        "c0b": ("00",  "­ƒÄ» Pr├®pare ton geste..."),
                        "c7":  ("007", "­ƒô© Tiens ton geste !"),
                    }[phase]
                    col_chiffre = "#ff4444" if phase == "c7" else "#ffa500"
                    res_box.markdown(f"""
                    <div style='text-align:center; padding:40px 20px;
                                border:2px solid #333; border-radius:16px;'>
                        <p style='color:#aaa; font-size:1.1em;'>{label_phase[1]}</p>
                        <h1 style='font-size:4em; margin:0; color:{col_chiffre};
                                   letter-spacing:0.12em;'>{label_phase[0]}</h1>
                        <p style='color:#555;'>Manche {manche}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    res_box.markdown("""
                    <div style='text-align:center; padding:50px 20px;
                                border:2px dashed #333; border-radius:16px;'>
                        <p style='color:#555;'>En attente...</p>
                    </div>
                    """, unsafe_allow_html=True)

            # ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ Logique des phases ÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉÔòÉ

            # ÔöÇÔöÇ c0a : premier "0" ÔÇö attend que la cam├®ra soit pr├¬te ÔöÇÔöÇ
            if phase == "c0a":
                # V├®rifier si la cam├®ra a d├®j├á fourni un frame
                cam_prete = (ctx_007 and ctx_007.video_processor
                             and ctx_007.video_processor.last_frame_pil is not None)
                if not cam_prete:
                    # Cam├®ra pas encore pr├¬te : r├®initialiser le chrono
                    st.session_state.g007_phase_t = time.time()
                    elapsed = 0.0
                _g007_overlay.update({"active": True, "text": "0", "color": (255, 165, 0)})
                if cam_prete and elapsed >= 0.7:
                    st.session_state.g007_phase   = "c0b"
                    st.session_state.g007_phase_t = time.time()
                    st.rerun()
                time.sleep(0.2)
                st.rerun()

            # ÔöÇÔöÇ c0b : deuxi├¿me "00" + pr├®-choix IA (0.7s) ÔöÇÔöÇ
            elif phase == "c0b":
                _g007_overlay.update({"active": True, "text": "00", "color": (255, 165, 0)})
                if elapsed >= 0.7:
                    qt_now = charger_qtable()
                    st_now = etat_007(j_balles, ia_balles, j_vies, ia_vies)
                    ia_pre = ia_choisit_geste(ia_balles, ia_vies, j_vies, st_now, qt_now)
                    st.session_state.g007_ia_pre      = ia_pre
                    st.session_state.g007_prev_state  = st_now
                    st.session_state.g007_prev_action = GESTES_KEYS.index(ia_pre)
                    st.session_state.g007_phase   = "c7"
                    st.session_state.g007_phase_t = time.time()
                    st.rerun()
                time.sleep(0.2)
                st.rerun()

            # ÔöÇÔöÇ c7 : "007" + capture automatique (0.8s) ÔöÇÔöÇ
            elif phase == "c7":
                _g007_overlay.update({"active": True, "text": "007", "color": (255, 50, 50)})
                if elapsed >= 0.8:
                    # Vote multi-frames sur le buffer de la webcam (~30 frames = 1s)
                    captured_frames = []
                    if ctx_007 and ctx_007.video_processor:
                        with ctx_007.video_processor.lock:
                            captured_frames = list(ctx_007.video_processor.frame_buffer)

                    if captured_frames:
                        j_geste, j_conf = reconnaitre_geste_vote(captured_frames)
                    else:
                        j_geste, j_conf = None, "cam├®ra inactive"

                    if j_geste is None:
                        j_geste = random.choice(GESTES_KEYS)
                        j_conf  = "non reconnu ÔÜá´©Å"

                    ia_geste = st.session_state.g007_ia_pre or random.choice(GESTES_KEYS)

                    # R├®solution duel
                    j_balles_new, ia_balles_new, j_vies_new, ia_vies_new, msgs, j_touche, ia_touche = \
                        resoudre_duel(j_geste, ia_geste, j_balles, ia_balles, j_vies, ia_vies)

                    # Q-learning : r├®compense + mise ├á jour
                    reward_ia = calculer_reward_ia(
                        ia_geste, j_geste, ia_balles, j_balles,
                        ia_touche, j_touche, j_vies_new, ia_vies_new
                    )
                    qt_up   = charger_qtable()
                    next_st = etat_007(j_balles_new, ia_balles_new, j_vies_new, ia_vies_new)
                    if st.session_state.g007_prev_state is not None:
                        qt_up = ia_apprendre(
                            qt_up,
                            st.session_state.g007_prev_state,
                            st.session_state.g007_prev_action,
                            reward_ia, next_st
                        )
                        sauvegarder_qtable(qt_up)

                    # R├®sultat texte
                    if ia_touche and not j_touche:   res_txt = "­ƒñû IA touch├®e"
                    elif j_touche and not ia_touche: res_txt = "­ƒÆÑ Joueur touch├®"
                    elif j_touche and ia_touche:     res_txt = "­ƒÆÑ Double touche"
                    else:                             res_txt = "= Neutre"

                    # Historique
                    st.session_state.g007_history.append({
                        "j_geste":   j_geste,  "ia_geste":  ia_geste,
                        "j_conf":    j_conf,   "res_txt":   res_txt,
                        "j_touche":  j_touche, "ia_touche": ia_touche,
                        "msgs":      msgs,
                    })
                    st.session_state.g007_pending = {
                        "j_geste":   j_geste,  "ia_geste":  ia_geste,
                        "j_conf":    j_conf,   "msgs":      msgs,
                        "j_touche":  j_touche, "ia_touche": ia_touche,
                    }
                    # Mise ├á jour ├®tat
                    st.session_state.g007_j_vies    = j_vies_new
                    st.session_state.g007_ia_vies   = ia_vies_new
                    st.session_state.g007_j_balles  = j_balles_new
                    st.session_state.g007_ia_balles = ia_balles_new
                    st.session_state.g007_manche    = manche + 1
                    st.session_state.g007_phase     = "result"
                    st.session_state.g007_phase_t   = time.time()
                    _g007_overlay["active"] = False
                    st.rerun()
                time.sleep(0.2)
                st.rerun()

            # ÔöÇÔöÇ result : affichage r├®sultat (3s) ÔöÇÔöÇ
            elif phase == "result":
                _g007_overlay["active"] = False
                if elapsed >= 3.0:
                    if (st.session_state.g007_j_vies  <= 0 or
                            st.session_state.g007_ia_vies <= 0):
                        st.session_state.g007_active = False
                        st.session_state.g007_over   = True
                        st.rerun()
                    else:
                        st.session_state.g007_pending = None
                        st.session_state.g007_phase   = "c0a"
                        st.session_state.g007_phase_t = time.time()
                        st.rerun()
                time.sleep(0.3)
                st.rerun()

