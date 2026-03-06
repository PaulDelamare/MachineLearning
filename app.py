import re
import threading
import time
import random
import pickle
from collections import deque
import os
import numpy as np
import streamlit as st
import pymongo
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline
from bson.objectid import ObjectId
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import mediapipe as mp
from streamlit_autorefresh import st_autorefresh

# ─────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VisionIA – Reconnaissance d'images",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────────────────────
# MODÈLE PRINCIPAL – ViT HuggingFace (meilleur)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

classifier = load_model()

# ─────────────────────────────────────────────
# MODÈLE COMPARAISON – MobileNetV2 TensorFlow
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# TENSORFLOW FEATURE EXTRACTOR – embeddings (visages + gestes)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# MEDIAPIPE HANDS – reconnaissance de gestes
# ─────────────────────────────────────────────
@st.cache_resource
def load_mp_hands():
    try:
        return mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
    except Exception:
        return None

_mp_hands = load_mp_hands()

# ─────────────────────────────────────────────
# CONNEXION MONGODB
# ─────────────────────────────────────────────
@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")

client = init_connection()
db = client["visionai_db"]
collection = db["images"]

# ─────────────────────────────────────────────
# RECONNAISSANCE FACIALE (MTCNN TensorFlow + MobileNetV2 TF)
# ─────────────────────────────────────────────
FACES_DB_FILE = "faces_db.pkl"

@st.cache_resource
def load_face_models():
    """Charge le détecteur de visages MTCNN (TensorFlow)."""
    from mtcnn import MTCNN
    return MTCNN()

mtcnn_model = load_face_models()

def charger_db_visages():
    if os.path.exists(FACES_DB_FILE):
        with open(FACES_DB_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def sauvegarder_db_visages(db):
    with open(FACES_DB_FILE, "wb") as f:
        pickle.dump(db, f)

def get_embedding(pil_image):
    """Détecte le visage via MTCNN (TF) et calcule l'embedding MobileNetV2 (TF)."""
    if _tf_extractor is None:
        return None
    try:
        img_rgb = np.array(pil_image.convert("RGB"))
        results = mtcnn_model.detect_faces(img_rgb)
        if not results:
            return None
        x, y, w, h = results[0]['box']
        x, y = max(0, x), max(0, y)
        face_crop = img_rgb[y:y+h, x:x+w]
        if face_crop.size == 0:
            return None
        face_pil = Image.fromarray(face_crop).resize((224, 224))
        img_arr  = np.expand_dims(np.array(face_pil, dtype=np.float32), axis=0)
        img_arr  = _tf_ext_prep(img_arr)
        emb      = _tf_extractor.predict(img_arr, verbose=0)[0]
        emb      = emb / (np.linalg.norm(emb) + 1e-8)
        return emb
    except Exception:
        return None

def enregistrer_visage(pil_image, nom):
    """Ajoute un visage au registre (MTCNN + MobileNetV2 TF)."""
    emb = get_embedding(pil_image)
    if emb is None:
        return False, "⚠️ Aucun visage détecté dans l'image. Réessaie avec un visage bien visible."
    faces_db = charger_db_visages()
    if nom not in faces_db:
        faces_db[nom] = []
    faces_db[nom].append(emb)
    sauvegarder_db_visages(faces_db)
    nb = len(faces_db[nom])
    return True, f"✅ Visage de **{nom}** enregistré ! ({nb} photo(s) au total)"

def reconnaitre_visage(pil_image):
    """Identifie la personne via MTCNN (TF) + MobileNetV2 (TF)."""
    faces_db = charger_db_visages()
    if not faces_db:
        return None, None
    emb = get_embedding(pil_image)
    if emb is None:
        return None, None
    best_name, best_dist = None, float("inf")
    for nom, embeddings in faces_db.items():
        for known_emb in embeddings:
            if known_emb.shape != emb.shape:  # ignore embeddings d'une ancienne version
                continue
            dist = float(np.linalg.norm(emb - known_emb))
            if dist < best_dist:
                best_dist = dist
                best_name = nom
    if best_name and best_dist < 0.9:
        conf = max(0, int((1 - best_dist / 0.9) * 100))
        return best_name, f"{conf}% (TF)"
    return None, None

# ─────────────────────────────────────────────
# JEU 007 – RECONNAISSANCE DE GESTES
# ─────────────────────────────────────────────
GESTURES_DB_FILE = "gestures_db.pkl"
GESTURES_NB_MIN  = 3  # photos minimum par geste pour jouer

GESTURES_CONFIG = {
    "recharger": {
        "emoji": "🤙", "label": "Recharger",
        "desc": "2 doigts pointés à côté de la tête (tempe)",
        "couleur": "#1a73e8",
    },
    "tirer": {
        "emoji": "🔫", "label": "Tirer",
        "desc": "Main en forme de pistolet, doigt pointé",
        "couleur": "#e8341a",
    },
    "proteger": {
        "emoji": "🛡️", "label": "Se protéger",
        "desc": "Bras croisés devant toi en bouclier",
        "couleur": "#2e7d32",
    },
}

# Règles du duel :
# Tirer > Recharger  (si tu as des balles)
# Protéger bloque Tirer
# Recharger + Se protéger = neutre
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

def _landmark_to_vector(lm_list):
    """
    Convertit 21 landmarks MediaPipe en vecteur 63-dim normalisé.
    Invariant en translation (centrée sur le poignet) et en échelle
    (divisée par la distance poignet → base du majeur).
    """
    coords = np.array([[p.x, p.y, p.z] for p in lm_list], dtype=np.float32)
    wrist  = coords[0].copy()
    coords -= wrist                                        # centrage sur le poignet
    scale   = float(np.linalg.norm(coords[9])) + 1e-8    # poignet → base du majeur
    return (coords / scale).flatten()                     # 63 dims


def get_gesture_landmark(pil_image):
    """
    Extrait les landmarks MediaPipe d'une main visible dans l'image.
    Retourne un vecteur NumPy 63-dim normalisé, ou None si aucune main détectée.
    Thread-safe grâce à _mp_lock.
    """
    if _mp_hands is None:
        return None
    try:
        img = np.array(pil_image.resize((256, 256)).convert("RGB"))
        with _mp_lock:
            results = _mp_hands.process(img)
        if not results.multi_hand_landmarks:
            return None
        return _landmark_to_vector(results.multi_hand_landmarks[0].landmark)
    except Exception:
        return None


# Alias de compatibilité (si du code existant appelle get_gesture_embedding)
get_gesture_embedding = get_gesture_landmark


def enregistrer_geste(pil_image, geste_key):
    """Ajoute une photo d'exemple pour un geste (landmarks MediaPipe)."""
    vec = get_gesture_landmark(pil_image)
    if vec is None:
        return False, "⚠️ Aucune main détectée ! Montre ta main bien visible à la caméra, doigts déployés."
    db = charger_db_gestes()
    if geste_key not in db:
        db[geste_key] = []
    db[geste_key].append(vec)
    sauvegarder_db_gestes(db)
    nb  = len(db[geste_key])
    cfg = GESTURES_CONFIG[geste_key]
    return True, f"✅ Geste **{cfg['label']}** enregistré ! ({nb} photo(s))"


def reconnaitre_geste(pil_image):
    """
    Identifie le geste via les landmarks MediaPipe + distance L2.
    Seuil empirique sur vecteurs normalisés : < 0.45 → bonne confiance.
    """
    db = charger_db_gestes()
    if not db:
        return None, None
    vec = get_gesture_landmark(pil_image)
    if vec is None:
        return None, None   # main non visible
    best_key, best_dist = None, float("inf")
    for key, vecs in db.items():
        for known in vecs:
            if known.shape != vec.shape:
                continue
            d = float(np.linalg.norm(vec - known))
            if d < best_dist:
                best_dist = d
                best_key  = key
    if best_key and best_dist < 0.45:
        conf = max(0, int((1 - best_dist / 0.45) * 100))
        return best_key, f"{conf}%"
    if best_key:       # fallback : retourne le plus proche même hors seuil
        return best_key, "??%"
    return None, None


def reconnaitre_geste_vote(frames: list):
    """
    Vote majoritaire sur N frames avec MediaPipe (CPU, très rapide).
    Chaque frame est analysée séparément ; on retourne le geste majoritaire.
    """
    if not frames:
        return None, "aucune frame"
    db = charger_db_gestes()
    if not db:
        return None, "base vide"

    step    = max(1, len(frames) // 7)
    samples = list(frames)[::step][:7]

    votes = {}
    for pil in samples:
        key, _ = reconnaitre_geste(pil)
        if key:
            votes[key] = votes.get(key, 0) + 1

    if not votes:
        return None, "main non détectée ⚠️"
    geste_final = max(votes, key=votes.get)
    n_vote  = votes[geste_final]
    n_total = len(samples)
    return geste_final, f"{n_vote}/{n_total} ✓"

# ── Q-LEARNING 007 ──────────────────────────────────────────
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
    """Tuple discret représentant l'état courant."""
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
    # Exploration aléatoire
    choices = ["recharger", "proteger"]
    if can_shoot:
        choices += ["tirer", "tirer"]  # légère préférence tirer
    return random.choice(choices)

def ia_apprendre(qtable, state, action_idx, reward, next_state):
    """Mise à jour Bellman de la Q-table."""
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
    ia_touche : l'IA a été touchée par le joueur (mauvais pour l'IA).
    j_touche  : le joueur a été touché par l'IA (bon pour l'IA).
    """
    r = 0

    # ─ Combat ─
    if j_touche:   r += 25   # toucher l'adversaire : priorité max
    if ia_touche:  r -= 25   # se faire toucher : symétrique

    # ─ Tirer (initiative offensive) ─
    if ia_geste == "tirer" and ia_balles_avant > 0:
        r += 2   # petit bonus : tirer est une action courageuse

    # ─ Tour complètement inutile ─
    if not j_touche and not ia_touche:
        r -= 3  # chaque tour sans décision coûte

    # ─ Protection ─
    if j_geste == "tirer" and j_balles_avant > 0 and ia_geste == "proteger":
        r += 12  # bloquer un vrai tir : bien récompensé
    elif ia_geste == "proteger":
        r -= 8   # se protéger sans raison : tour perdu, punit fortement

    # ─ Gestion munitions ─
    if ia_geste == "recharger":
        if ia_balles_avant >= JEU007_BALLES_MAX:
            r -= 14  # recharger quand déjà plein : très inutile
        else:
            r += 3   # recharger quand en manque : utile

    # ─ Tirer à vide ─
    if ia_geste == "tirer" and ia_balles_avant == 0:
        r -= 8

    # ─ Fin de partie ─
    if ia_vies_new <= 0:  r -= 40
    if j_vies_new  <= 0:  r += 40
    return r


def s_entrainer_007(nb_parties: int = None, duree_s: float = None):
    """
    Entraîne le bot par auto-jeu ASYMÉTRIQUE pour éviter la convergence vers les nuls.
    Alterne 3 modes de match à chaque partie :
      - QvQ   (40%) : les deux agents exploitent la Q-table
      - QvRand(35%) : agent Q affronte un opposant purement aléatoire
      - QvAgro(25%) : agent Q affronte un opposant qui tire dès qu'il a des balles
    Cela force le bot à apprendre contre des stratégies imprévisibles/agressives
    plutôt que de converger vers un équilibre défensif symétrique.
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

        # Sélection du mode : 40% QvQ / 35% QvRand / 25% QvAgro
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
            # Pénalité nul plus forte
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
    """Applique les règles 007 et retourne le nouvel état + messages."""
    msgs = []
    j_touche  = False
    ia_touche = False

    # ── Le joueur tire ──
    if j_geste == "tirer":
        if j_balles > 0:
            j_balles -= 1
            if ia_geste != "proteger":
                ia_vies -= 1
                ia_touche = True
                msgs.append("🔫 **Tu as tiré** → IA **touchée** ! (-1 vie)")
            else:
                msgs.append("🔫 **Tu as tiré** → IA s'est **protégée**. Raté !")
        else:
            msgs.append("❌ **Tu as tiré** mais tu n'as plus de balles !")
    elif j_geste == "recharger":
        j_balles = min(JEU007_BALLES_MAX, j_balles + 1)
        msgs.append(f"🤙 **Tu recharges** → {j_balles} balle(s)")
    elif j_geste == "proteger":
        msgs.append("🛡️ **Tu te protèges**")

    # ── L'IA tire ──
    if ia_geste == "tirer":
        if ia_balles > 0:
            ia_balles -= 1
            if j_geste != "proteger":
                j_vies -= 1
                j_touche = True
                msgs.append("💀 **L'IA a tiré** → Tu es **touché(e)** ! (-1 vie)")
            else:
                msgs.append("🛡️ **L'IA a tiré** → Tu t'es **protégé(e)** ! Bloqué.")
        else:
            msgs.append("❌ L'IA a tiré sans balles !")
    elif ia_geste == "recharger":
        ia_balles = min(JEU007_BALLES_MAX, ia_balles + 1)
        msgs.append(f"🤙 **L'IA recharge** → {ia_balles} balle(s)")
    elif ia_geste == "proteger":
        msgs.append("🛡️ **L'IA se protège**")

    return j_balles, ia_balles, j_vies, ia_vies, msgs, j_touche, ia_touche


# ─────────────────────────────────────────────
# MAPPING : labels ImageNet → 5 catégories TP
# ─────────────────────────────────────────────
CATEGORY_MAPPER = {
    # ── Véhicules EN PREMIER (priorité haute pour éviter les faux positifs) ──
    "sports car": "Véhicule", "sport car": "Véhicule",
    "race car": "Véhicule", "racing car": "Véhicule",
    "car": "Véhicule", "truck": "Véhicule", "bus": "Véhicule",
    "bicycle": "Véhicule", "motorcycle": "Véhicule", "airplane": "Véhicule",
    "boat": "Véhicule", "train": "Véhicule", "ambulance": "Véhicule",
    "taxi": "Véhicule", "van": "Véhicule", "tractor": "Véhicule",
    "helicopter": "Véhicule", "spacecraft": "Véhicule", "submarine": "Véhicule",
    "jeep": "Véhicule", "minivan": "Véhicule", "convertible": "Véhicule",
    "racer": "Véhicule", "go-kart": "Véhicule", "streetcar": "Véhicule",
    "scooter": "Véhicule", "limousine": "Véhicule", "fire engine": "Véhicule",
    "police van": "Véhicule", "cab": "Véhicule", "minibus": "Véhicule",

    # ── Personnages fictifs ──
    "puppet": "Personnage Fictif",
    "teddy": "Personnage Fictif",
    "teddy bear": "Personnage Fictif",
    "ocarina": "Personnage Fictif", "doll": "Personnage Fictif",
    "figurine": "Personnage Fictif", "action figure": "Personnage Fictif",
    "costume": "Personnage Fictif", "cloak": "Personnage Fictif",
    "robe": "Personnage Fictif",
    # BD, manga, super-héros : retirés de la blacklist pour être capturés ici
    "comic book": "Personnage Fictif", "comic strip": "Personnage Fictif",
    "mask": "Personnage Fictif", "cartoon": "Personnage Fictif",

    # ── Animaux ──
    "dog": "Animal", "cat": "Animal", "bird": "Animal", "fish": "Animal",
    "horse": "Animal", "cow": "Animal", "elephant": "Animal", "bear": "Animal",
    "zebra": "Animal", "giraffe": "Animal", "lion": "Animal", "tiger": "Animal",
    "wolf": "Animal", "fox": "Animal", "rabbit": "Animal", "hamster": "Animal",
    "duck": "Animal", "eagle": "Animal", "penguin": "Animal", "frog": "Animal",
    "snake": "Animal", "lizard": "Animal", "turtle": "Animal", "shark": "Animal",
    "whale": "Animal", "bee": "Animal", "butterfly": "Animal", "spider": "Animal",
    "crab": "Animal", "lobster": "Animal",

    # ── Plantes ──
    "flower": "Plante", "rose": "Plante", "daisy": "Plante", "tulip": "Plante",
    "sunflower": "Plante", "dandelion": "Plante", "tree": "Plante",
    "mushroom": "Plante", "cactus": "Plante", "fern": "Plante", "moss": "Plante",
    "leaf": "Plante", "grass": "Plante", "corn": "Plante", "banana": "Plante",
    "apple": "Plante", "orange": "Plante", "strawberry": "Plante",
    "broccoli": "Plante", "carrot": "Plante",

    # ── Humains EN DERNIER (vêtements, accessoires → visibles sur une personne) ──
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
    # Labels ViT supplémentaires fréquents pour une personne
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

    # ── Nourriture ──
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

    # ── Sport ──
    "tennis racket": "Sport", "baseball bat": "Sport", "cricket bat": "Sport",
    "golf club": "Sport", "ski": "Sport", "snowboard": "Sport",
    "surfboard": "Sport", "skateboard": "Sport", "parachute": "Sport",
    "bow": "Sport", "dumbbell": "Sport", "barbell": "Sport",
    "swimming": "Sport", "balance beam": "Sport", "horizontal bar": "Sport",
    "ping-pong ball": "Sport", "boxing glove": "Sport",

    # ── Objet du quotidien ──
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
    # Objets du quotidien supplémentaires
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
    # Note: jersey et running shoe sont définis dans la section Humain — ne pas redéfinir ici

    # ── Nature / Paysage ──
    "mountain": "Nature", "volcano": "Nature", "valley": "Nature",
    "ocean": "Nature", "lake": "Nature", "river": "Nature",
    "waterfall": "Nature", "beach": "Nature", "desert": "Nature",
    "cliff": "Nature", "coral reef": "Nature", "geyser": "Nature",
    "cloud": "Nature", "sky": "Nature", "rainbow": "Nature",
    "ice berg": "Nature", "glacier": "Nature", "cave": "Nature",
}

# Icônes et messages de gamification par catégorie
CATEGORY_CONFIG = {
    "Humain":            {"emoji": "👤", "message": "👤 Humain repéré ! Vous n'êtes pas seul..."},
    "Personnage Fictif": {"emoji": "🧙", "message": "🎬 Créature de légende détectée ! Sortez le popcorn !"},
    "Animal":            {"emoji": "🐾", "message": "🐾 Bête sauvage repérée ! Ne bougez plus..."},
    "Plante":            {"emoji": "🌿", "message": "🌿 La nature s'invite ! Pensez à arroser."},
    "Véhicule":          {"emoji": "🚗", "message": "🚗 Bolide en approche ! Attachez vos ceintures !"},
    "Nourriture":        {"emoji": "🍕", "message": "🍕 Repas détecté ! J'ai faim maintenant..."},
    "Sport":             {"emoji": "🏆", "message": "🏆 À vos marques, prêts, partez !"},
    "Objet":             {"emoji": "📦", "message": "📦 Objet du quotidien identifié !"},
    "Nature":            {"emoji": "🌍", "message": "🌍 Splendeur naturelle détectée !"},
    "Inconnu":           {"emoji": "❓", "message": "🤔 L'IA ne reconnaît pas de catégorie connue.  \nCela peut être une œuvre d'art, un paysage ou un objet non classifiable."},
}

# Labels ImageNet qui indiquent une image plate (peinture, affiche, livre...)
# → forcer "Inconnu" directement, sans passer par le mapper
BLACKLIST_FLAT_IMAGE = {
    "book jacket", "dust cover", "dust jacket", "dust wrapper",
    "jigsaw puzzle", "envelope", "packet",
    "menu", "web site", "screen", "monitor", "television",
    "poster", "album", "cd",
    # Note: "comic book" retiré → mappe vers Personnage Fictif
}

# ─────────────────────────────────────────────
# DÉFIS POUR LE JEU (chasse aux objets)
# ─────────────────────────────────────────────
# keywords=None  → n'importe quel objet de la catégorie
# keywords=[...] → un label précis doit être détecté
DEFIS_POOL = [
    {
        "texte": "🍾 Trouve une BOUTEILLE !",
        "categorie": "Objet", "keywords": ["bottle", "wine bottle", "beer bottle", "water bottle"],
        "temps": 30, "points_max": 200, "emoji": "🍾",
        "conseil": "Cherche dans ta cuisine ou sur ton bureau !",
    },
    {
        "texte": "🐶 Trouve un ANIMAL !",
        "categorie": "Animal", "keywords": None,
        "temps": 25, "points_max": 250, "emoji": "🐶",
        "conseil": "Un vrai animal, une peluche... sois créatif !",
    },
    {
        "texte": "🌱 Trouve une PLANTE !",
        "categorie": "Plante", "keywords": None,
        "temps": 30, "points_max": 200, "emoji": "🌱",
        "conseil": "Une fleur, une plante d'intérieur, un arbre par la fenêtre !",
    },
    {
        "texte": "🍕 Trouve de la NOURRITURE !",
        "categorie": "Nourriture", "keywords": None,
        "temps": 25, "points_max": 200, "emoji": "🍕",
        "conseil": "Direction la cuisine ! Frigo, placards...",
    },
    {
        "texte": "☕ Trouve une TASSE ou un BOL !",
        "categorie": "Objet", "keywords": ["cup", "bowl", "coffee mug"],
        "temps": 20, "points_max": 300, "emoji": "☕",
        "conseil": "Sur ton bureau ? Dans la cuisine ?",
    },
    {
        "texte": "👤 Montre un HUMAIN !",
        "categorie": "Humain", "keywords": None,
        "temps": 20, "points_max": 300, "emoji": "👤",
        "conseil": "Montre-toi, appelle quelqu'un, ou trouve une photo !",
    },
    {
        "texte": "📚 Trouve un LIVRE !",
        "categorie": "Objet", "keywords": ["book"],
        "temps": 25, "points_max": 200, "emoji": "📚",
        "conseil": "Dans ta bibliothèque ou sur ta table !",
    },
    {
        "texte": "✂️ Trouve des CISEAUX !",
        "categorie": "Objet", "keywords": ["scissors"],
        "temps": 35, "points_max": 250, "emoji": "✂️",
        "conseil": "Tiroir de bureau, trousse scolaire...",
    },
    {
        "texte": "🏆 Trouve un OBJET DE SPORT !",
        "categorie": "Sport", "keywords": None,
        "temps": 35, "points_max": 250, "emoji": "🏆",
        "conseil": "Raquette, ballon, haltères... cherche bien !",
    },
    {
        "texte": "🌸 Trouve une FLEUR !",
        "categorie": "Plante", "keywords": ["flower", "rose", "daisy", "tulip", "sunflower", "dandelion"],
        "temps": 30, "points_max": "200", "emoji": "🌸",
        "conseil": "Dehors, sur une photo, ou dans un vase !",
    },
    {
        "texte": "💻 Trouve un ORDINATEUR ou un TÉLÉPHONE !",
        "categorie": "Objet", "keywords": ["laptop", "keyboard", "phone", "mouse"],
        "temps": 15, "points_max": 350, "emoji": "💻",
        "conseil": "Facile... tu dois en avoir un près de toi !",
    },
    {
        "texte": "🔑 Trouve des CLÉS ou un SAC !",
        "categorie": "Objet", "keywords": ["backpack", "handbag", "suitcase", "wallet"],
        "temps": 30, "points_max": 250, "emoji": "🔑",
        "conseil": "Près de l'entrée ou sur ton bureau !",
    },
    {
        "texte": "📷 Trouve une LAMPE ou une HORLOGE !",
        "categorie": "Objet", "keywords": ["lamp", "clock"],
        "temps": 25, "points_max": 250, "emoji": "📷",
        "conseil": "Regarde autour de toi dans la pièce !",
    },
    {
        "texte": "🧻 Trouve du PAPIER TOILETTE !",
        "categorie": "Objet", "keywords": ["toilet tissue", "paper towel", "toilet paper"],
        "temps": 25, "points_max": 350, "emoji": "🧻",
        "conseil": "Check les toilettes ou la réserve !",
    },
    {
        "texte": "🧦 Trouve une CHAUSSETTE !",
        "categorie": "Objet", "keywords": ["sock", "stocking"],
        "temps": 30, "points_max": 300, "emoji": "🧦",
        "conseil": "Dans ta chambre, sur le sol ou dans un tiroir !",
    },
    {
        "texte": "🧾a Trouve une BROSSE À DENTS !",
        "categorie": "Objet", "keywords": ["toothbrush"],
        "temps": 30, "points_max": 300, "emoji": "🧾a",
        "conseil": "Direction la salle de bain !",
    },
    {
        "texte": "📺 Trouve une TÉLÉCOMMANDE !",
        "categorie": "Objet", "keywords": ["remote control", "television"],
        "temps": 25, "points_max": 300, "emoji": "📺",
        "conseil": "Sur le canapé ou près de la télé !",
    },
    {
        "texte": "✏️ Trouve un STYLO ou un CRAYON !",
        "categorie": "Objet", "keywords": ["ballpoint pen", "pencil", "crayon"],
        "temps": 20, "points_max": 250, "emoji": "✏️",
        "conseil": "Sur ton bureau ou dans ta trousse !",
    },
    {
        "texte": "🛏️ Trouve un COUSSIN ou un OREILLER !",
        "categorie": "Objet", "keywords": ["pillow", "cushion"],
        "temps": 25, "points_max": 250, "emoji": "🛏️",
        "conseil": "Sur le canapé ou dans ta chambre !",
    },
]
# S'assurer que points_max est toujours un int
for _d in DEFIS_POOL:
    _d["points_max"] = int(_d["points_max"])

# ─────────────────────────────────────────────
# FONCTION D'ANALYSE
# ─────────────────────────────────────────────
def analyser_image(pil_image):
    # Inférence ViT HuggingFace – top-5
    resultats = classifier(pil_image, top_k=5)
    meilleur = resultats[0]
    label_brut = meilleur["label"].lower()
    score_raw = meilleur["score"]

    # Parcourt le top-5 pour trouver la première catégorie connue
    # On utilise \b (word boundary) pour éviter les faux positifs
    # ex: "car" ne doit PAS matcher "ocarina"
    categorie = "Inconnu"
    label_reconnu = label_brut
    score_reconnu = score_raw  # score du label ayant déclenché la catégorie
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

    # 🥚 Easter egg : Baby Yoda (figurine verte avec cloak/ocarina = Baby Yoda très probable)
    easter_egg = None
    baby_yoda_signals = ["yoda", "puppet", "teddy", "ocarina", "cloak", "robe"]
    if any(m in label_brut for m in baby_yoda_signals) or \
       any(m in label_reconnu for m in baby_yoda_signals):
        easter_egg = "🟢 BABY YODA DÉTECTÉ ! La Force est avec vous !"

    return {
        "label_brut": label_brut,
        "label_reconnu": label_reconnu,
        "score_pct": score_pct,
        "score_raw": score_reconnu,
        "categorie": categorie,
        "easter_egg": easter_egg,
    }


def analyser_image_tf(pil_image):
    """Analyse une image avec MobileNetV2 (TensorFlow/Keras) et retourne la catégorie TP."""
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


# ─────────────────────────────────────────────
# PROCESSEUR VIDÉO TEMPS RÉEL (Webcam)
# ─────────────────────────────────────────────
# Dictionnaire partagé entre le thread principal et VideoProcessor pour l'overlay 007
_g007_overlay = {"active": False, "text": "", "color": (255, 200, 0)}

# Lock thread-safe pour MediaPipe (non thread-safe nativement)
_mp_lock = threading.Lock()

# ──────────────────────────────────────────────
# LIGHT VIDEO PROCESSOR  (gestes + 007 game)
# Pas d'inférence ViT/TF → démarre instantanément
# Dessine les landmarks MediaPipe en direct (feedback visuel)
# ──────────────────────────────────────────────
class LightVideoProcessor(VideoProcessorBase):
    """
    Processeur léger pour la webcam de capture de gestes et le jeu 007.
    - Aucune inférence IA lourde en background
    - Dessine les landmarks MediaPipe Hands en temps réel (~1 ms/frame)
    - Buffer de frames pour la capture et le vote 007
    """
    def __init__(self):
        self.lock           = threading.Lock()
        self.last_frame_pil = None
        self.frame_buffer   = deque(maxlen=60)  # ~2s à 30fps
        # Instance MediaPipe propre à ce processeur (static_image_mode=False = mode vidéo rapide)
        try:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._mp_draw = mp.solutions.drawing_utils
            self._mp_styles = mp.solutions.drawing_styles
        except Exception:
            self._hands = None

    def recv(self, frame):
        img_rgb = frame.to_ndarray(format="rgb24")
        pil_img = Image.fromarray(img_rgb)

        with self.lock:
            self.last_frame_pil = pil_img
            self.frame_buffer.append(pil_img)

        # ── Overlay compte à rebours 007 ──
        if _g007_overlay["active"] and _g007_overlay["text"]:
            pil_draw = pil_img.copy()
            draw = ImageDraw.Draw(pil_draw)
            h, w = img_rgb.shape[:2]
            txt  = _g007_overlay["text"]
            col  = _g007_overlay["color"]
            try:
                font_size = {1: 160, 2: 150}.get(len(txt), 130)
                font_big  = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font_big = ImageFont.load_default()
            try:
                bb = draw.textbbox((0, 0), txt, font=font_big)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                tw, th = 100, 100
            x, y = (w - tw) // 2, (h - th) // 2 - 20
            draw.text((x + 4, y + 4), txt, fill=(0, 0, 0), font=font_big)
            draw.text((x, y),         txt, fill=col,       font=font_big)
            return av.VideoFrame.from_ndarray(np.array(pil_draw), format="rgb24")

        # ── Overlay MediaPipe Hands (mode capture gestes) ──
        if self._hands is not None:
            try:
                results = self._hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    # Dessin du squelette via MediaPipe (travaille en numpy RGB)
                    annotated = img_rgb.copy()
                    for hl in results.multi_hand_landmarks:
                        self._mp_draw.draw_landmarks(
                            annotated, hl,
                            mp.solutions.hands.HAND_CONNECTIONS,
                            self._mp_styles.get_default_hand_landmarks_style(),
                            self._mp_styles.get_default_hand_connections_style(),
                        )
                    # Bandeau vert en bas via PIL pour garder RGB propre
                    pil_ann = Image.fromarray(annotated)
                    draw_ann = ImageDraw.Draw(pil_ann)
                    h_a, w_a = annotated.shape[:2]
                    draw_ann.rectangle([(0, h_a - 36), (w_a, h_a)], fill=(0, 120, 0))
                    draw_ann.text((10, h_a - 28), "✓ Main détectée",
                                  fill=(100, 255, 100))
                    return av.VideoFrame.from_ndarray(np.array(pil_ann), format="rgb24")
            except Exception:
                pass

        return frame  # aucune modification si pas de landmarks


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result    = None
        self.result_tf = None
        self.frame_count = 0
        self.lock = threading.Lock()
        self.last_frame_pil = None   # dernière frame (pour détection caméra prête)
        self.frame_buffer   = deque(maxlen=30)  # ~1s à 30fps pour vote multi-frames

    def _analyse_background(self, pil_image):
        """Lançé dans un thread pour ne pas bloquer recv()."""
        try:
            result    = analyser_image(pil_image)
            result_tf = analyser_image_tf(pil_image)
            if result["categorie"] == "Humain":
                nom_v, conf_v = reconnaitre_visage(pil_image)
                result["nom_visage"]  = nom_v
                result["conf_visage"] = conf_v
            else:
                result["nom_visage"]  = None
                result["conf_visage"] = None
            with self.lock:
                self.result    = result
                self.result_tf = result_tf
                self._analysing = False
        except Exception:
            with self.lock:
                self._analysing = False

    def recv(self, frame):
        img_array = frame.to_ndarray(format="rgb24")
        self.frame_count += 1

        pil_current = Image.fromarray(img_array)
        # Toujours stocker la dernière frame et alimenter le buffer multi-frames
        with self.lock:
            self.last_frame_pil = pil_current
            self.frame_buffer.append(pil_current)
            already_analysing = getattr(self, "_analysing", False)

        # Lance l’analyse en background (une seule à la fois) toutes les ~60 frames
        # Suspendue pendant le jeu 007 pour libérer le modèle TF
        if self.frame_count % 60 == 0 and not already_analysing and not _g007_overlay["active"]:
            with self.lock:
                self._analysing = True
            t = threading.Thread(target=self._analyse_background, args=(pil_current,), daemon=True)
            t.start()

        # ── Overlay résultat catégorie ──
        with self.lock:
            current_result = self.result

        if current_result and not _g007_overlay["active"]:
            pil_draw = Image.fromarray(img_array)
            draw = ImageDraw.Draw(pil_draw)
            cfg = CATEGORY_CONFIG.get(current_result["categorie"], CATEGORY_CONFIG["Inconnu"])
            h, w = img_array.shape[:2]
            draw.rectangle([(0, h - 65), (w, h)], fill=(0, 0, 0))
            display_name = current_result.get("nom_visage") or current_result["categorie"]
            prefix = f"👋 {display_name}" if current_result.get("nom_visage") else f"{cfg['emoji']}  {display_name}"
            draw.text((12, h - 52), prefix, fill=(100, 255, 100) if current_result.get("nom_visage") else (255, 255, 255))
            draw.text((12, h - 28), f"Confiance : {current_result['score_pct']}  |  {current_result['label_reconnu']}", fill=(180, 180, 180))
            img_array = np.array(pil_draw)

        # ── Overlay compte à rebours 007 ──
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
            # Ombre + texte centré
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


# ─────────────────────────────────────────────
# INTERFACE PRINCIPALE
# ─────────────────────────────────────────────
st.title("🔍 VisionIA – Reconnaissance d'Images")
st.caption("Détection automatique : Humains · Personnages Fictifs · Animaux · Plantes · Véhicules · Nourriture · Sport · Objets · Nature")
st.markdown("---")

tab_analyse, tab_jeu, tab_visages, tab_007 = st.tabs([
    "🔍 Analyse d'image",
    "🎮 Jeu de détection (10 manches)",
    "👥 Reconnaissances de visages",
    "🔫 007 Duel",
])

# ══════════════════════════════════════════════
# ONGLET 1 – ANALYSE
# ══════════════════════════════════════════════
with tab_analyse:
    # ── Deux colonnes principales : upload | résultat ──
    zone_upload, zone_resultat = st.columns([1, 1], gap="large")

    with zone_upload:
        st.subheader("📸 Capture ou Upload")
        mode = st.radio("Source de l'image :", ["📷 Webcam (temps réel)", "📁 Fichier"], horizontal=True)

        image_source = None
        uploaded_file = None
        ctx = None

        if mode == "📷 Webcam (temps réel)":
            st.caption("⚡ La catégorie s'affiche sur la vidéo. Cliquez START pour lancer.")
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
        st.subheader("🧠 Résultat de l'Analyse")

        # ── MODE WEBCAM TEMPS RÉEL ──
        if mode == "📷 Webcam (temps réel)":
            if ctx and ctx.state.playing and ctx.video_processor:
                with ctx.video_processor.lock:
                    live_result    = ctx.video_processor.result
                    live_result_tf = ctx.video_processor.result_tf

                if live_result:
                    cfg = CATEGORY_CONFIG.get(live_result["categorie"], CATEGORY_CONFIG["Inconnu"])

                    # ── Nom reconnu ? ──
                    nom_reconnu = live_result.get("nom_visage")
                    conf_visage = live_result.get("conf_visage")
                    if nom_reconnu:
                        st.markdown(f"""
                        <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:12px;
                                    padding:12px 20px; text-align:center; margin-bottom:8px;'>
                            <span style='font-size:1.8em'>👋</span>
                            <h3 style='color:#4caf50; margin:4px 0;'>Bonjour <b>{nom_reconnu}</b> !</h3>
                            <p style='color:#aaa; margin:0;'>Confiance visage : {conf_visage}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"## {cfg['emoji']} {live_result['categorie']}")
                        if live_result["categorie"] == "Humain":
                            st.caption("👤 Visage non reconnu — enregistre-toi dans **👥 Reconnaissances de visages** !")

                    c1, c2 = st.columns(2)
                    c1.metric("Catégorie détectée", live_result["categorie"])
                    c2.metric("Confiance", live_result["score_pct"])
                    st.info(f"Label : `{live_result['label_reconnu']}`")
                    if live_result["easter_egg"]:
                        st.balloons()
                        st.warning(live_result["easter_egg"])
                    else:
                        st.success(cfg["message"])

                    if live_result_tf:
                        cfg_tf_l = CATEGORY_CONFIG.get(live_result_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                        accord      = live_result["categorie"] == live_result_tf["categorie"]
                        accord_txt  = "✅ accord ViT" if accord else "⚠️ diverge de ViT"
                        st.caption(
                            f"🔷 **TF MobileNetV2** → {cfg_tf_l['emoji']} {live_result_tf['categorie']} "
                            f"({live_result_tf['score_pct']}) {accord_txt}"
                        )

                    st.markdown("---")
                    if st.button("💾 Sauvegarder cette détection dans MongoDB", type="primary"):
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
                            st.success("✅ Sauvegardé dans MongoDB !")
                        except Exception as e:
                            st.error(f"Erreur MongoDB : {e}")
                else:
                    st.info("⏳ En attente de la première analyse... (environ 2 secondes après START)")
            else:
                st.info("▶️ Cliquez sur START dans le flux vidéo pour lancer la détection en temps réel.")

        # ── MODE FICHIER ──
        elif image_source is not None:
            with st.spinner("Analyse en cours par le modèle ViT..."):
                resultat = analyser_image(pil_image)

            cfg = CATEGORY_CONFIG.get(resultat["categorie"], CATEGORY_CONFIG["Inconnu"])

            st.markdown(f"## {cfg['emoji']} {resultat['categorie']}")

            # ── Sauvegarde MongoDB (immédiatement visible) ──
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
                st.success("✅ Résultat enregistré dans MongoDB !")
            except Exception as e:
                st.error(f"Erreur MongoDB : {e}")

            # ── Reconnaissance faciale si Humain ──
            nom_reconnu, conf_visage = None, None
            if resultat["categorie"] == "Humain":
                with st.spinner("👥 Recherche du visage dans le registre..."):
                    nom_reconnu, conf_visage = reconnaitre_visage(pil_image)
            if nom_reconnu:
                st.markdown(f"""
                <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:12px;
                            padding:16px 24px; text-align:center; margin-bottom:12px;'>
                    <span style='font-size:2em'>👋</span>
                    <h3 style='color:#4caf50; margin:6px 0;'>Bonjour <b>{nom_reconnu}</b> !</h3>
                    <p style='color:#aaa; margin:0;'>Confiance : {conf_visage}</p>
                </div>
                """, unsafe_allow_html=True)
            elif resultat["categorie"] == "Humain":
                st.caption("👤 Visage non reconnu — enregistre-toi dans l'onglet **👥 Reconnaissances de visages** !")

            c1, c2 = st.columns(2)
            c1.metric("Catégorie détectée", resultat["categorie"])
            c2.metric("Taux de réussite", resultat["score_pct"])
            st.info(f"Label principal du modèle : `{resultat['label_brut']}`  \nLabel ayant déclenché la catégorie : `{resultat['label_reconnu']}`")

            if resultat["easter_egg"]:
                st.balloons()
                st.warning(resultat["easter_egg"])
            else:
                st.success(cfg["message"])

            # ── Comparaison TensorFlow MobileNetV2 ──────────────────────
            st.markdown("---")
            st.subheader("🆚 Comparaison des modèles")
            with st.spinner("Analyse TensorFlow / MobileNetV2 en cours..."):
                resultat_tf = analyser_image_tf(pil_image)

            if resultat_tf is None:
                st.warning("⚠️ TensorFlow n'est pas disponible dans cet environnement.")
            else:
                cfg_tf  = CATEGORY_CONFIG.get(resultat_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                cfg_vit = CATEGORY_CONFIG.get(resultat["categorie"],    CATEGORY_CONFIG["Inconnu"])

                col_vit, col_tf = st.columns(2)
                with col_vit:
                    st.markdown("""
                    <div style='background:#1a1a2e; border:2px solid #4f8ef7;
                                border-radius:10px; padding:14px; text-align:center;'>
                        <p style='color:#4f8ef7; font-weight:bold; margin:0 0 6px;'>🤖 ViT – HuggingFace / PyTorch</p>
                    """, unsafe_allow_html=True)
                    st.metric("Catégorie", f"{cfg_vit['emoji']} {resultat['categorie']}")
                    st.metric("Confiance", resultat["score_pct"])
                    st.caption(f"Label : `{resultat['label_reconnu']}`")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_tf:
                    st.markdown("""
                    <div style='background:#1a2e1a; border:2px solid #f7a24f;
                                border-radius:10px; padding:14px; text-align:center;'>
                        <p style='color:#f7a24f; font-weight:bold; margin:0 0 6px;'>🔷 MobileNetV2 – TensorFlow / Keras</p>
                    """, unsafe_allow_html=True)
                    st.metric("Catégorie", f"{cfg_tf['emoji']} {resultat_tf['categorie']}")
                    st.metric("Confiance", resultat_tf["score_pct"])
                    st.caption(f"Label : `{resultat_tf['label_reconnu']}`")
                    st.markdown("</div>", unsafe_allow_html=True)

                if resultat["categorie"] == resultat_tf["categorie"]:
                    st.success(f"✅ Les deux modèles sont **d'accord** : **{resultat['categorie']}**")
                else:
                    st.warning(f"⚠️ Les modèles **divergent** : ViT → **{resultat['categorie']}** | TensorFlow → **{resultat_tf['categorie']}**")
        else:
            st.info("Prenez une photo ou uploadez une image à gauche pour lancer l'analyse.")

    # ── Historique ──
    st.markdown("---")
    st.header("📋 Historique des Analyses")

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
                "Catégorie": f"{cfg['emoji']} {categorie}",
                "Taux de réussite": analyse.get("taux_reussite", "N/A"),
                "Label brut": analyse.get("label_brut", "N/A"),
            })

        df = pd.DataFrame(historique_a_afficher)
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

        st.markdown("### 📊 Répartition des catégories détectées")
        comptage = df["Catégorie"].value_counts().reset_index()
        comptage.columns = ["Catégorie", "Nombre d analyses"]
        st.bar_chart(data=comptage.set_index("Catégorie"))

        st.markdown("### 🗑️ Supprimer un enregistrement")
        options = {
            f"{r['Nom du fichier']} – {r['Date']} ({r['Catégorie']})": r["ID"]
            for r in historique_a_afficher
        }
        element = st.selectbox("Sélectionner l'image à supprimer :", list(options.keys()))

        c_del, c_all = st.columns([1, 2])
        with c_del:
            if st.button("🗑️ Supprimer cet enregistrement", type="primary"):
                try:
                    collection.delete_one({"_id": ObjectId(options[element])})
                    st.success("Enregistrement supprimé !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur : {e}")
        with c_all:
            if st.button("🧹 Vider tout l'historique"):
                collection.delete_many({})
                st.warning("Historique entièrement vidé.")
                st.rerun()
    else:
        st.info("Aucune analyse enregistrée pour le moment. Uploadez une image ci-dessus !")


# ══════════════════════════════════════════════════════
# ONGLET 2 – JEU DE DÉTECTION (Chasse aux Objets)
# ══════════════════════════════════════════════════════
with tab_jeu:
    NB_MANCHES = 10

    # ── Initialisation session state ──
    for _key, _val in [
        ("game_active", False), ("game_over", False), ("game_round", 1),
        ("game_score", 0), ("game_history", []), ("game_defi_order", []),
        ("game_start_time", 0.0), ("game_round_won", False),
    ]:
        if _key not in st.session_state:
            st.session_state[_key] = _val

    # ════════════════════════════════════════
    # ÉCRAN D'ACCUEIL
    # ════════════════════════════════════════
    if not st.session_state.game_active and not st.session_state.game_over:
        st.markdown("""
        <div style='text-align:center; padding: 50px 20px;'>
            <h1 style='font-size:3em;'>🎮 Chasse aux Objets !</h1>
            <p style='font-size:1.3em; color: #aaa;'>
                Un défi s'affiche — tu as un <b>temps limité</b> pour rapporter l'objet devant la caméra !<br>
                Plus tu es rapide, plus tu gagnes de points. ⚡<br><br>
                <b>10 manches &nbsp;·&nbsp; Chrono &nbsp;·&nbsp; Bonus vitesse &nbsp;·&nbsp; Classement final</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("🚀 LANCER LA PARTIE !", type="primary", use_container_width=True):
                st.session_state.game_active = True
                st.session_state.game_round = 1
                st.session_state.game_score = 0
                st.session_state.game_history = []
                st.session_state.game_over = False
                st.session_state.game_round_won = False
                st.session_state.game_defi_order = random.sample(DEFIS_POOL, NB_MANCHES)
                st.session_state.game_start_time = time.time()
                st.rerun()

    # ════════════════════════════════════════
    # ÉCRAN FIN DE PARTIE
    # ════════════════════════════════════════
    elif st.session_state.game_over:
        score_final = st.session_state.game_score
        score_max = sum(d["points_max"] for d in st.session_state.game_defi_order)
        pct = int(score_final / score_max * 100) if score_max > 0 else 0

        if pct >= 80:
            titre, medal = "CHAMPION ABSOLU !", "🏆"
        elif pct >= 60:
            titre, medal = "Très bien joué !", "🥇"
        elif pct >= 40:
            titre, medal = "Pas mal du tout !", "🥈"
        else:
            titre, medal = "Continue de t'entraîner !", "💪"

        st.markdown(f"""
        <div style='text-align:center; padding: 30px 0;'>
            <h1 style='font-size:3.5em;'>{medal}</h1>
            <h2>{titre}</h2>
            <h3 style='color:#e94560;'>Score final : {score_final} / {score_max} pts &nbsp;({pct}%)</h3>
        </div>
        """, unsafe_allow_html=True)

        if pct >= 60:
            st.balloons()

        st.markdown("### 📋 Récapitulatif des 10 manches")
        recap_data = []
        for i, h in enumerate(st.session_state.game_history):
            recap_data.append({
                "Manche": f"#{i + 1}",
                "Défi": h["defi"],
                "Résultat": "✅ Gagnée" if h["won"] else "❌ Perdue",
                "Points": h["points"],
                "Temps": f"{h['temps_pris']:.1f}s" if h["won"] else "—",
            })
        st.dataframe(pd.DataFrame(recap_data), use_container_width=True, hide_index=True)

        _, col_r, _ = st.columns([1, 2, 1])
        with col_r:
            if st.button("🔄 Rejouer une nouvelle partie !", type="primary", use_container_width=True):
                for k in ["game_active", "game_over", "game_round", "game_score",
                          "game_history", "game_defi_order", "game_round_won"]:
                    del st.session_state[k]
                st.rerun()

    # ════════════════════════════════════════
    # PARTIE EN COURS
    # ════════════════════════════════════════
    else:
        round_idx = st.session_state.game_round - 1
        defi = st.session_state.game_defi_order[round_idx]
        elapsed = time.time() - st.session_state.game_start_time
        remaining = max(0.0, defi["temps"] - elapsed)
        time_ratio = remaining / defi["temps"]

        # ── Header : manche / score / chrono ──
        hc1, hc2, hc3 = st.columns([2, 1, 1])
        with hc1:
            st.markdown(f"### Manche **{st.session_state.game_round}** / {NB_MANCHES}")
        with hc2:
            st.metric("🏅 Score", f"{st.session_state.game_score} pts")
        with hc3:
            color = "🟢" if time_ratio > 0.5 else ("🟡" if time_ratio > 0.2 else "🔴")
            st.metric(f"{color} Temps", f"{int(remaining)}s")

        st.progress(time_ratio)

        # ── Grand encart du défi ──
        border_color = "#e94560" if time_ratio > 0.25 else "#ff0000"
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    border: 3px solid {border_color}; border-radius: 20px;
                    padding: 40px 30px; text-align: center; margin: 15px 0;'>
            <div style='font-size: 4em; margin-bottom: 10px;'>{defi["emoji"]}</div>
            <h2 style='color: white; font-size: 2.2em; margin: 0 0 12px 0;'>{defi["texte"]}</h2>
            <p style='color: #aaa; font-size: 1.1em; margin: 0;'>💡 {defi["conseil"]}</p>
            <p style='color: #e94560; font-size: 1em; margin-top: 10px;'>
                ⏱ {defi["temps"]}s max &nbsp;·&nbsp; 🎯 jusqu'à {defi["points_max"]} pts
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Webcam jeu | Résultat côte à côte ──
        gcol1, gcol2 = st.columns([1, 1], gap="large")

        with gcol1:
            st.caption("📷 Lance la caméra et pointe-la vers l'objet !")
            ctx_game = webrtc_streamer(
                key="game-webcam",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with gcol2:
            st.subheader("🎯 Détection en cours...")
            game_result    = None
            game_result_tf = None
            if ctx_game and ctx_game.state.playing and ctx_game.video_processor:
                with ctx_game.video_processor.lock:
                    game_result    = ctx_game.video_processor.result
                    game_result_tf = ctx_game.video_processor.result_tf

            if game_result:
                detected_cat = game_result["categorie"]
                cfg_det = CATEGORY_CONFIG.get(detected_cat, CATEGORY_CONFIG["Inconnu"])

                # Vérifier si le résultat correspond au défi
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
                    st.markdown(f"### ✅ {cfg_det['emoji']} {detected_cat}")
                    st.success(f"**C'est bien ça !** Label : `{game_result['label_reconnu']}` – {game_result['score_pct']}")
                else:
                    st.markdown(f"**Détecté :** {cfg_det['emoji']} {detected_cat}")
                    st.caption(f"Label : `{game_result['label_reconnu']}` — {game_result['score_pct']}")
                    st.warning(f"Ce n'est pas ça... cherche un(e) **{defi['categorie']}** !")

                if game_result_tf:
                    cfg_tf_g   = CATEGORY_CONFIG.get(game_result_tf["categorie"], CATEGORY_CONFIG["Inconnu"])
                    accord_g   = game_result["categorie"] == game_result_tf["categorie"]
                    accord_g_txt = "✅" if accord_g else "⚠️ diverge"
                    st.caption(
                        f"🔷 TF MobileNetV2 : {cfg_tf_g['emoji']} **{game_result_tf['categorie']}** "
                        f"({game_result_tf['score_pct']}) {accord_g_txt}"
                    )

                # ── Victoire ──
                if won and not st.session_state.game_round_won:
                    temps_pris = elapsed
                    if temps_pris < defi["temps"] * 0.25:
                        bonus = int(defi["points_max"] * 0.5)
                        bonus_txt = f"⚡ Bonus ÉCLAIR +{bonus} pts !"
                    elif temps_pris < defi["temps"] * 0.5:
                        bonus = int(defi["points_max"] * 0.25)
                        bonus_txt = f"🚀 Bonus RAPIDE +{bonus} pts !"
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
                    st.success(f"🎉 TROUVÉ en {temps_pris:.1f}s ! **+{pts} pts**" +
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
                st.info("⏳ Lance la caméra et pointe-la vers l'objet !")

        # ── Temps écoulé ──
        if remaining <= 0 and not st.session_state.game_round_won:
            st.error(f"⏰ TEMPS ÉCOULÉ ! Il fallait trouver : **{defi['texte']}**")
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

        # ── Barre de progression des manches ──
        st.markdown("---")
        manche_cols = st.columns(NB_MANCHES)
        for i, col in enumerate(manche_cols):
            mn = i + 1
            if mn < st.session_state.game_round:
                h = st.session_state.game_history[i] if i < len(st.session_state.game_history) else None
                icon = "✅" if (h and h["won"]) else "❌"
            elif mn == st.session_state.game_round:
                icon = "🎯"
            else:
                icon = "⬜"
            col.markdown(
                f"<div style='text-align:center'>{icon}<br><small style='color:#aaa'>#{mn}</small></div>",
                unsafe_allow_html=True,
            )

        # ── Bouton passer ──
        if st.button("⏭️ Passer cette manche (0 pts)"):
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

        # ── Auto-refresh chrono (toutes les 0.8s) ──
        if st.session_state.game_active and not st.session_state.game_round_won:
            st_autorefresh(interval=800, key="game_chrono")


# ══════════════════════════════════════════════
# ONGLET 3 – GESTION DES VISAGES
# ══════════════════════════════════════════════
with tab_visages:
    st.header("👥 Registre des visages")
    st.caption("Enregistre ton visage ici — l'IA te reconnaîtra dans l'onglet Analyse ou pendant le jeu !")
    st.markdown("---")

    vcol1, vcol2 = st.columns([1, 1], gap="large")

    # ────────────────────
    # ENREGISTREMENT
    # ────────────────────
    with vcol1:
        st.subheader("➕ Ajouter un visage")
        nom_input = st.text_input("📝 Prénom ou nom à associer :", placeholder="ex: Paul, Marie...")
        reg_mode = st.radio("Source :", ["📷 Webcam", "📂 Fichier"], horizontal=True)

        reg_img = None
        if reg_mode == "📷 Webcam":
            reg_photo = st.camera_input("Prends une photo de ton visage")
            if reg_photo:
                reg_img = Image.open(reg_photo).convert("RGB")
                st.image(reg_img, caption="Photo capturée", use_container_width=True)
        else:
            reg_file = st.file_uploader("Importe une photo", type=["jpg", "jpeg", "png"],
                                        key="reg_face_upload")
            if reg_file:
                reg_img = Image.open(reg_file).convert("RGB")
                st.image(reg_img, caption=reg_file.name, use_container_width=True)

        if st.button("💾 Enregistrer ce visage", type="primary", disabled=(not nom_input or reg_img is None)):
            ok, msg = enregistrer_visage(reg_img, nom_input.strip())
            if ok:
                st.success(msg)
                st.balloons()
            else:
                st.error(msg)

        st.markdown("")
        st.markdown("""
        **Conseils pour un bon enregistrement :**
        - Visage face à la caméra, bien éclairé
        - Ajoute 2-3 photos sous des angles différents pour améliorer la précision
        - Évite les lunettes de soleil ou le masque
        """)

    # ────────────────────
    # REGISTRE ACTUEL
    # ────────────────────
    with vcol2:
        st.subheader("📋 Personnes enregistrées")
        faces_db = charger_db_visages()

        if not faces_db:
            st.info("Aucun visage enregistré pour l'instant. Ajoute-toi à gauche !")
        else:
            for nom, embeddings in faces_db.items():
                col_n, col_d = st.columns([3, 1])
                col_n.markdown(f"👤 **{nom}** — {len(embeddings)} photo(s)")
                if col_d.button("Supprimer", key=f"del_{nom}"):
                    del faces_db[nom]
                    sauvegarder_db_visages(faces_db)
                    st.success(f"{nom} supprimé du registre.")
                    st.rerun()

            st.markdown("---")
            if st.button("🗑️ Effacer tout le registre"):
                sauvegarder_db_visages({})
                st.warning("Registre vidé.")
                st.rerun()

    # ──────────────────────
    # TEST DE RECONNAISSANCE
    # ──────────────────────
    st.markdown("---")
    st.subheader("🧐 Tester la reconnaissance")
    test_mode = st.radio("Source du test :", ["📷 Webcam", "📂 Fichier"], horizontal=True, key="test_mode")

    test_img = None
    if test_mode == "📷 Webcam":
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
            with st.spinner("🔍 Analyse du visage..."):
                nom_test, conf_test = reconnaitre_visage(test_img)
            if nom_test:
                st.markdown(f"""
                <div style='background:#1a3a1a; border:2px solid #4caf50; border-radius:16px;
                            padding:24px; text-align:center; margin-top:20px;'>
                    <div style='font-size:3em'>👋</div>
                    <h2 style='color:#4caf50;'>Bonjour <b>{nom_test}</b> !</h2>
                    <p style='color:#aaa;'>Similarité : <b>{conf_test}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                db_check = charger_db_visages()
                if not db_check:
                    st.warning("⚠️ Aucun visage enregistré. Ajoute-toi d'abord !")
                else:
                    st.error("❌ Visage non reconnu dans le registre.")
                    st.caption("Essaie d'ajouter plus de photos sous différents angles.")


# ══════════════════════════════════════════════════════
# ONGLET 4 – JEU 007
# ══════════════════════════════════════════════════════
with tab_007:
    st.header("🔫 007 – Duel contre l'IA")
    st.caption("Apprends à l'IA tes gestes, puis affronte-la en duel !")
    st.markdown("---")

    sub_appren, sub_jeu007 = st.tabs(["📚 1. Apprendre les gestes", "🎮 2. Jouer"])

    # ══════════════════════════════════════════
    # SOUS-ONGLET A – APPRENTISSAGE DES GESTES
    # ══════════════════════════════════════════
    with sub_appren:
        st.subheader("📸 Enregistre tes gestes")
        st.markdown(
            f"Minimum **{GESTURES_NB_MIN} photos** par geste pour jouer — mais plus tu en ajoutes, "
            "plus la détection sera précise. **Pas besoin d'appuyer pendant le geste** : "
            "clique sur le bouton, puis pose les deux mains et attends le décompte !"
        )
        st.markdown("---")

        # ── Init session state ──
        for _k, _dv in [
            ("glearn_geste_sel", list(GESTURES_CONFIG.keys())[0]),
            ("glearn_msg",       ""),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _dv

        db_gestes = charger_db_gestes()

        # ── Barre de progression globale ──
        total_done   = sum(len(db_gestes.get(k, [])) for k in GESTURES_CONFIG)
        total_needed = len(GESTURES_CONFIG) * GESTURES_NB_MIN
        pct_done     = min(1.0, total_done / total_needed)
        st.progress(pct_done, text=f"Progression minimum : {total_done}/{total_needed} photos enregistrées")
        if total_done >= total_needed:
            st.success("✅ Minimum atteint ! Va dans **🎮 2. Jouer** pour démarrer. Tu peux continuer à ajouter des photos pour améliorer la précision.")
        st.markdown("")

        # ── Compteurs par geste ──
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
                f"{' ✅' if ok else f' / {GESTURES_NB_MIN} min'}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Sélecteur de geste ──
        geste_labels = {k: f"{v['emoji']} {v['label']}" for k, v in GESTURES_CONFIG.items()}
        geste_sel = st.radio(
            "Quel geste veux-tu capturer ?",
            options=list(GESTURES_CONFIG.keys()),
            format_func=lambda k: geste_labels[k],
            horizontal=True,
            key="glearn_geste_sel",
        )
        gcfg_sel = GESTURES_CONFIG[geste_sel]
        st.caption(f"💡 {gcfg_sel['desc']}")
        st.markdown("")

        # ── Capture par photo directe ──
        # st.camera_input() : pas de WebRTC, pas de STUN/TURN, démarre instantanément
        snap_col, info_col = st.columns([1.3, 1])

        with snap_col:
            nb_deja = len(db_gestes.get(geste_sel, []))
            snap = st.camera_input(
                f"📸 {gcfg_sel['emoji']} {gcfg_sel['label']} — positionne ta main et clique sur Take photo",
                key=f"cam_learn_{geste_sel}_{nb_deja}",  # clé unique → reset après chaque capture
            )

        with info_col:
            nb_sel = len(db_gestes.get(geste_sel, []))
            ok_nb  = nb_sel >= GESTURES_NB_MIN
            st.markdown(
                f"<div style='padding:22px; border:2px dashed {gcfg_sel['couleur']}; "
                f"border-radius:16px; text-align:center;'>"
                f"<div style='font-size:3em'>{gcfg_sel['emoji']}</div>"
                f"<h3 style='color:{gcfg_sel['couleur']}; margin:8px 0'>{gcfg_sel['label']}</h3>"
                f"<p style='color:#aaa; margin:0; font-size:0.95em;'>{gcfg_sel['desc']}</p>"
                f"<hr style='border-color:#333; margin:12px 0;'>"
                f"<p style='color:{'#4caf50' if ok_nb else '#888'}; margin:0;'>"
                f"{nb_sel} photo(s) {'✅' if ok_nb else ('/ ' + str(GESTURES_NB_MIN) + ' minimum')}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown("")

            if st.session_state.glearn_msg:
                if "✅" in st.session_state.glearn_msg:
                    st.success(st.session_state.glearn_msg)
                else:
                    st.error(st.session_state.glearn_msg)

            if nb_sel > 0:
                if st.button(
                    f"🗑️ Supprimer toutes les photos de {gcfg_sel['label']} ({nb_sel})",
                    key="btn_del_geste"
                ):
                    db_g2 = charger_db_gestes()
                    db_g2[geste_sel] = []
                    sauvegarder_db_gestes(db_g2)
                    st.session_state.glearn_msg = ""
                    st.rerun()

        # Traitement de la photo prise
        if snap is not None:
            pil_snap = Image.open(snap).convert("RGB")
            ok_g, msg_g = enregistrer_geste(pil_snap, geste_sel)
            st.session_state.glearn_msg = msg_g if ok_g else f"❌ {msg_g}"
            st.rerun()

        st.markdown("---")
        if st.button("🗑️ Réinitialiser TOUS les gestes"):
            sauvegarder_db_gestes({})
            st.session_state.glearn_msg = ""
            st.warning("Tous les gestes ont été effacés.")
            st.rerun()
            sauvegarder_db_gestes({})
            st.session_state.glearn_msg = ""
            st.warning("Tous les gestes ont été effacés.")
            st.rerun()

    # ══════════════════════════════════════════
    # SOUS-ONGLET B – JEU
    # ══════════════════════════════════════════
    with sub_jeu007:

        # ── Init session state 007 ──
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

        # ══════════════════════════════
        # ÉCRAN D'ACCUEIL
        # ══════════════════════════════
        if not st.session_state.g007_active and not st.session_state.g007_over:
            _g007_overlay["active"] = False

            if not gestes_prets:
                manquants = [
                    f"{GESTURES_CONFIG[k]['emoji']} {GESTURES_CONFIG[k]['label']} ({len(db_g.get(k,[]))}/{GESTURES_NB_MIN})"
                    for k in GESTURES_CONFIG if len(db_g.get(k, [])) < GESTURES_NB_MIN
                ]
                st.warning("⚠️ Enregistre d'abord tous tes gestes dans **📚 1. Apprendre les gestes** !")
                for m in manquants:
                    st.markdown(f"- {m}")
            else:
                qt = charger_qtable()
                nb_etats = len(qt)
                st.markdown(f"""
                <div style='text-align:center; padding:30px 20px;'>
                    <h1 style='font-size:3em;'>🔫 007 DUEL !</h1>
                    <p style='font-size:1.2em; color:#aaa;'>
                        Affronte l'IA au jeu <b>007</b> !<br>
                        Le compte à rebours s'affiche sur la caméra — tiens ton geste au <b style='color:#e94560;'>7</b> !<br><br>
                        🤙 <b>Recharger</b> &nbsp;·&nbsp; 🔫 <b>Tirer</b> &nbsp;·&nbsp; 🛡️ <b>Se protéger</b><br><br>
                        <b>3 vies chacun — le premier à 0 perd !</b><br>
                        <small style='color:#666;'>🧠 Q-table IA : {nb_etats} états appris</small>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                _, cbtn, _ = st.columns([1, 2, 1])
                with cbtn:
                    if st.button("🚀 LANCER LE DUEL !", type="primary", use_container_width=True):
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

                # ── Section entraînement auto-jeu ──
                st.markdown("---")
                st.markdown("#### 🧠 Entraîner le bot (auto-jeu)")
                with st.expander("🏓 Lancer une session d'auto-entraînement", expanded=False):
                    st.caption(
                        "Le bot joue contre lui-même et met à jour sa Q-table. "
                        "L'entraînement est **cumulatif** : relance autant que tu veux."
                    )
                    mode_train = st.radio(
                        "Mode", ["Par nombre de parties", "Par durée"],
                        horizontal=True, key="train007_mode"
                    )
                    if mode_train == "Par nombre de parties":
                        nb_parties_train = st.slider(
                            "Nombre de parties", 100, 20000, 2000, step=100,
                            key="train007_nb"
                        )
                        duree_train_s = None
                        label_spinner = f"🏋️ Entraînement sur {nb_parties_train} parties..."
                    else:
                        duree_min = st.slider(
                            "Durée (minutes)", 1, 30, 5, step=1,
                            key="train007_min"
                        )
                        nb_parties_train = None
                        duree_train_s    = duree_min * 60
                        label_spinner    = f"🏋️ Entraînement pendant {duree_min} min..."

                    if st.button("▶️ Démarrer l'entraînement", key="train007_go", use_container_width=True):
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

                    # Affichage persistant du dernier résultat
                    res_t = st.session_state.get("train007_result")
                    if res_t:
                        nb_t  = res_t["nb"]
                        pct_v = res_t["victoires"] / max(nb_t, 1) * 100
                        pct_d = res_t["defaites"]  / max(nb_t, 1) * 100
                        pct_n = res_t["nuls"]      / max(nb_t, 1) * 100
                        dur_r = res_t["duree_s"]
                        vps   = nb_t / dur_r if dur_r > 0 else 0
                        moy_tours = res_t["tours"] / max(nb_t, 1)

                        # Durée lisible
                        if dur_r >= 60:
                            dur_str = f"{int(dur_r // 60)}min {int(dur_r % 60)}s"
                        else:
                            dur_str = f"{dur_r:.1f}s"

                        # Barre de progression victoires/nuls/défaites (CSS)
                        bar_v = f"width:{pct_v:.1f}%"
                        bar_n = f"width:{pct_n:.1f}%"
                        bar_d = f"width:{pct_d:.1f}%"

                        st.markdown(f"""
<div style='background:#0e1117; border:1px solid #2a2a3a; border-radius:16px;
            padding:22px 26px; margin-top:14px;'>
  <h4 style='margin:0 0 16px 0; color:#eee;'>📊 Rapport d'entraînement</h4>

  <div style='display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px;'>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#a78bfa;'>{nb_t:,}</div>
      <div style='color:#888; font-size:.85em;'>parties jouées</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#60a5fa;'>{vps:,.0f}</div>
      <div style='color:#888; font-size:.85em;'>parties / seconde</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#f0abfc;'>{dur_str}</div>
      <div style='color:#888; font-size:.85em;'>durée totale</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#34d399;'>{res_t["tours"]:,}</div>
      <div style='color:#888; font-size:.85em;'>tours simulés</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#fbbf24;'>{moy_tours:.1f}</div>
      <div style='color:#888; font-size:.85em;'>tours / partie (moy.)</div>
    </div>
    <div style='background:#1a1a2e; border-radius:10px; padding:12px; text-align:center;'>
      <div style='font-size:1.8em; font-weight:bold; color:#38bdf8;'>{res_t["etats"]}</div>
      <div style='color:#888; font-size:.85em;'>états Q-table</div>
    </div>
  </div>

  <div style='margin-bottom:6px; color:#aaa; font-size:.85em;'>Résultats IA (symétrie auto-jeu)</div>
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
    <span>🟩 Victoires {res_t["victoires"]:,}</span>
    <span>⬜ Nuls {res_t["nuls"]:,}</span>
    <span>🟥 Défaites {res_t["defaites"]:,}</span>
  </div>
</div>
                        """, unsafe_allow_html=True)

                        st.markdown("")
                        if st.button("🗑️ Effacer le rapport", key="train007_clear"):
                            del st.session_state["train007_result"]
                            st.rerun()

        # ══════════════════════════════
        # ÉCRAN FIN DE PARTIE
        # ══════════════════════════════
        elif st.session_state.g007_over:
            _g007_overlay["active"] = False
            j_vies  = st.session_state.g007_j_vies
            ia_vies = st.session_state.g007_ia_vies

            if j_vies > ia_vies:
                titre, medal, couleur = "TU AS GAGNÉ !",   "🏆", "#4caf50"
                st.balloons()
            elif ia_vies > j_vies:
                titre, medal, couleur = "L'IA A GAGNÉ...", "🤖", "#e94560"
            else:
                titre, medal, couleur = "ÉGALITÉ !",       "🤝", "#ff9800"

            hearts_j  = '❤️' * max(0, j_vies)  + '🖤' * max(0, JEU007_VIES_MAX - j_vies)
            hearts_ia = '❤️' * max(0, ia_vies) + '🖤' * max(0, JEU007_VIES_MAX - ia_vies)
            st.markdown(f"""
            <div style='text-align:center; padding:30px 0;'>
                <h1 style='font-size:3.5em;'>{medal}</h1>
                <h2 style='color:{couleur};'>{titre}</h2>
                <p style='color:#aaa;'>Toi : {hearts_j} &nbsp;&nbsp;|&nbsp;&nbsp; IA : {hearts_ia}</p>
            </div>
            """, unsafe_allow_html=True)

            qt_fin = charger_qtable()
            st.caption(f"🧠 Q-table IA mise à jour : **{len(qt_fin)} états**")

            st.markdown("### 📋 Historique des manches")
            hist_data = []
            for i, h in enumerate(st.session_state.g007_history):
                ia_cfg = GESTURES_CONFIG.get(h["ia_geste"], {})
                j_cfg  = GESTURES_CONFIG.get(h["j_geste"],  {})
                hist_data.append({
                    "#":         i + 1,
                    "Ton geste": f"{j_cfg.get('emoji','')} {j_cfg.get('label','?')} ({h.get('j_conf','?')})",
                    "Geste IA":  f"{ia_cfg.get('emoji','')} {ia_cfg.get('label','?')}",
                    "Résultat":  h["res_txt"],
                })
            if hist_data:
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True, hide_index=True)

            _, cr, _ = st.columns([1, 2, 1])
            with cr:
                if st.button("🔄 Rejouer !", type="primary", use_container_width=True):
                    for _k in ["g007_active", "g007_over", "g007_j_vies", "g007_ia_vies",
                               "g007_j_balles", "g007_ia_balles", "g007_manche",
                               "g007_history", "g007_last", "g007_phase", "g007_phase_t",
                               "g007_ia_pre", "g007_prev_state", "g007_prev_action", "g007_pending"]:
                        if _k in st.session_state:
                            del st.session_state[_k]
                    st.rerun()

        # ══════════════════════════════
        # PARTIE EN COURS
        # ══════════════════════════════
        else:
            # st.camera_input() → pas de WebRTC, pas de STUN/TURN, démarre instantanément
            # st_autorefresh pilote le compte à rebours sans bloquer l'UI
            st_autorefresh(interval=400, key="g007_tick")

            j_vies    = st.session_state.g007_j_vies
            ia_vies   = st.session_state.g007_ia_vies
            j_balles  = st.session_state.g007_j_balles
            ia_balles = st.session_state.g007_ia_balles
            manche    = st.session_state.g007_manche
            phase     = st.session_state.g007_phase
            elapsed   = time.time() - st.session_state.g007_phase_t

            # ── Header vies / balles ──
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("❤️ Tes vies",   '❤️' * j_vies  + '🖤' * (JEU007_VIES_MAX - j_vies))
            hc2.metric("🔫 Tes balles", f"{j_balles} / {JEU007_BALLES_MAX}")
            hc3.metric("🤖 Vies IA",    '❤️' * ia_vies + '🖤' * (JEU007_VIES_MAX - ia_vies))
            hc4.metric("🔫 Balles IA",  f"{ia_balles} / {JEU007_BALLES_MAX}")

            # ── Rappel des gestes ──
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

            # ══════════ Affichage selon la phase ══════════
            pending = st.session_state.g007_pending

            # ── Phase RÉSULTAT ──
            if phase == "result" and pending:
                j_cfg_r  = GESTURES_CONFIG.get(pending["j_geste"],  {})
                ia_cfg_r = GESTURES_CONFIG.get(pending["ia_geste"], {})
                j_touche_r  = pending["j_touche"]
                ia_touche_r = pending["ia_touche"]

                if ia_touche_r and not j_touche_r:
                    bg_r, titre_r, border_r = "#1a3a1a", "✅ IA TOUCHÉE !",      "#4caf50"
                elif j_touche_r and not ia_touche_r:
                    bg_r, titre_r, border_r = "#3a1a1a", "💥 TU ES TOUCHÉ(E) !", "#e94560"
                elif j_touche_r and ia_touche_r:
                    bg_r, titre_r, border_r = "#3a2a00", "💥 DOUBLE TOUCHE !",   "#ff9800"
                else:
                    bg_r, titre_r, border_r = "#1a1a2e", "= NEUTRE",             "#666"

                prog_r = max(0.0, 1.0 - elapsed / 3.5)
                st.markdown(f"""
                <div style='background:{bg_r}; border:3px solid {border_r}; border-radius:20px;
                            padding:24px; text-align:center; margin:10px 0;'>
                    <h2 style='color:{border_r}; margin:0 0 16px 0;'>{titre_r}</h2>
                    <div style='display:flex; justify-content:space-around; margin:16px 0;'>
                        <div>
                            <div style='font-size:2.5em'>{j_cfg_r.get('emoji','?')}</div>
                            <b style='color:#ddd;'>TOI</b><br>
                            <span style='color:#aaa'>{j_cfg_r.get('label','?')}</span><br>
                            <small style='color:#666'>{pending['j_conf']}</small>
                        </div>
                        <div style='font-size:2em; align-self:center;'>⚔️</div>
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

            # ── Phase COMPTE À REBOURS (0 / 00) ──
            elif phase in ("c0a", "c0b"):
                chiffre = "0" if phase == "c0a" else "00"
                _, cd_center, _ = st.columns([1, 2, 1])
                cd_center.markdown(f"""
                <div style='text-align:center; padding:50px 20px;
                            border:3px solid #ffa500; border-radius:24px;
                            background:#1a1200;'>
                    <p style='color:#aaa; font-size:1.2em; margin:0 0 8px 0;'>🎯 Prépare ton geste...</p>
                    <h1 style='font-size:9em; margin:0; color:#ffa500;
                               letter-spacing:0.15em; font-weight:900;'>{chiffre}</h1>
                    <p style='color:#555; margin:8px 0 0 0;'>Manche {manche}</p>
                </div>
                """, unsafe_allow_html=True)

            # ── Phase CAPTURE (007) ──
            elif phase == "c7":
                cam_col, txt_col = st.columns([1, 1], gap="large")
                with cam_col:
                    shot = st.camera_input(
                        "📸 Montre ton geste puis clique sur **Take photo** !",
                        key=f"cam007_m{manche}",
                    )
                with txt_col:
                    st.markdown(f"""
                    <div style='text-align:center; padding:30px 20px;
                                border:3px solid #ff4444; border-radius:20px;
                                background:#200000; margin-top:10px;'>
                        <p style='color:#ff8888; font-size:1.1em; margin:0 0 8px 0;'>📸 CAPTURE !</p>
                        <h1 style='font-size:6em; margin:0; color:#ff4444;
                                   letter-spacing:0.15em; font-weight:900;'>007</h1>
                        <p style='color:#888; margin:10px 0 0 0;'>Manche {manche}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if elapsed >= 8.0:
                        st.warning("⏰ Temps écoulé — geste aléatoire joué !")

                # Traitement : photo prise OU timeout 10s
                if shot is not None or elapsed >= 10.0:
                    if shot is not None:
                        pil_shot        = Image.open(shot).convert("RGB")
                        j_geste, j_conf = reconnaitre_geste(pil_shot)
                        if j_geste is None:
                            j_geste = random.choice(GESTES_KEYS)
                            j_conf  = "non reconnu ⚠️"
                    else:
                        j_geste = random.choice(GESTES_KEYS)
                        j_conf  = "⏰ timeout"

                    ia_geste = st.session_state.g007_ia_pre or random.choice(GESTES_KEYS)

                    j_balles_new, ia_balles_new, j_vies_new, ia_vies_new, msgs, j_touche, ia_touche = \
                        resoudre_duel(j_geste, ia_geste, j_balles, ia_balles, j_vies, ia_vies)

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

                    if ia_touche and not j_touche:   res_txt = "🤖 IA touchée"
                    elif j_touche and not ia_touche: res_txt = "💥 Joueur touché"
                    elif j_touche and ia_touche:     res_txt = "💥 Double touche"
                    else:                             res_txt = "= Neutre"

                    st.session_state.g007_history.append({
                        "j_geste":  j_geste,  "ia_geste": ia_geste,
                        "j_conf":   j_conf,   "res_txt":  res_txt,
                        "j_touche": j_touche, "ia_touche": ia_touche, "msgs": msgs,
                    })
                    st.session_state.g007_pending = {
                        "j_geste":  j_geste,  "ia_geste": ia_geste,
                        "j_conf":   j_conf,   "msgs":     msgs,
                        "j_touche": j_touche, "ia_touche": ia_touche,
                    }
                    st.session_state.g007_j_vies    = j_vies_new
                    st.session_state.g007_ia_vies   = ia_vies_new
                    st.session_state.g007_j_balles  = j_balles_new
                    st.session_state.g007_ia_balles = ia_balles_new
                    st.session_state.g007_manche    = manche + 1
                    st.session_state.g007_phase     = "result"
                    st.session_state.g007_phase_t   = time.time()
                    st.rerun()

            # ══════════ Transitions automatiques de phases ══════════

            # c0a → c0b après 1.5s
            if phase == "c0a" and elapsed >= 1.5:
                qt_now = charger_qtable()
                st_now = etat_007(j_balles, ia_balles, j_vies, ia_vies)
                ia_pre = ia_choisit_geste(ia_balles, ia_vies, j_vies, st_now, qt_now)
                st.session_state.g007_ia_pre      = ia_pre
                st.session_state.g007_prev_state  = st_now
                st.session_state.g007_prev_action = GESTES_KEYS.index(ia_pre)
                st.session_state.g007_phase   = "c0b"
                st.session_state.g007_phase_t = time.time()
                st.rerun()

            # c0b → c7 après 1.5s
            elif phase == "c0b" and elapsed >= 1.5:
                st.session_state.g007_phase   = "c7"
                st.session_state.g007_phase_t = time.time()
                st.rerun()

            # result → manche suivante après 3.5s
            elif phase == "result" and elapsed >= 3.5:
                if (st.session_state.g007_j_vies  <= 0 or
                        st.session_state.g007_ia_vies <= 0):
                    st.session_state.g007_active = False
                    st.session_state.g007_over   = True
                else:
                    st.session_state.g007_pending = None
                    st.session_state.g007_phase   = "c0a"
                    st.session_state.g007_phase_t = time.time()
                st.rerun()
