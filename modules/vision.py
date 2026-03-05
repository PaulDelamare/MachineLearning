# ─────────────────────────────────────────────────────────────────────────────
# modules/vision.py — Analyse d'image ViT + VideoProcessor (webcam temps réel)
# ─────────────────────────────────────────────────────────────────────────────
import re
import threading
from collections import deque

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import VideoProcessorBase

from config import CATEGORY_MAPPER, CATEGORY_CONFIG, BLACKLIST_FLAT_IMAGE
from modules.model import classifier
from modules.faces import reconnaitre_visage


# ── Dictionnaire partagé pour l'overlay 007 (thread principal ↔ VideoProcessor)
_g007_overlay = {"active": False, "text": "", "color": (255, 200, 0)}


# ── Analyse d'image ───────────────────────────────────────────────────────────

def analyser_image(pil_image) -> dict:
    """
    Classifie une image PIL via ViT et retourne un dict :
    {label_brut, label_reconnu, score_pct, score_raw, categorie, easter_egg}
    """
    resultats   = classifier(pil_image, top_k=5)
    meilleur    = resultats[0]
    label_brut  = meilleur["label"].lower()
    score_raw   = meilleur["score"]

    categorie      = "Inconnu"
    label_reconnu  = label_brut
    score_reconnu  = score_raw

    for res in resultats:
        lbl = res["label"].lower()
        if any(bl in lbl for bl in BLACKLIST_FLAT_IMAGE):
            continue
        for mot_cle, cat in CATEGORY_MAPPER.items():
            pattern = r'\b' + re.escape(mot_cle) + r'\b'
            if re.search(pattern, lbl):
                categorie     = cat
                label_reconnu = lbl
                score_reconnu = res["score"]
                break
        if categorie != "Inconnu":
            break

    score_pct = f"{score_reconnu * 100:.2f}%"

    # Easter egg — Baby Yoda
    easter_egg = None
    baby_yoda_signals = ["yoda", "puppet", "teddy", "ocarina", "cloak", "robe"]
    if any(m in label_brut for m in baby_yoda_signals) or \
       any(m in label_reconnu for m in baby_yoda_signals):
        easter_egg = "🟢 BABY YODA DÉTECTÉ ! La Force est avec vous !"

    return {
        "label_brut":    label_brut,
        "label_reconnu": label_reconnu,
        "score_pct":     score_pct,
        "score_raw":     score_reconnu,
        "categorie":     categorie,
        "easter_egg":    easter_egg,
    }


# ── VideoProcessor ────────────────────────────────────────────────────────────

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.result       = None
        self.frame_count  = 0
        self.lock         = threading.Lock()
        self.last_frame_pil = None          # pour détecter que la caméra est prête
        self.frame_buffer   = deque(maxlen=30)  # ~1s à 30fps (vote multi-frames)

    def recv(self, frame):
        img_array = frame.to_ndarray(format="rgb24")
        self.frame_count += 1

        pil_current = Image.fromarray(img_array)
        with self.lock:
            self.last_frame_pil = pil_current
            self.frame_buffer.append(pil_current)

        # Analyse ViT + reconn. faciale toutes les ~2s (1 frame / 60)
        if self.frame_count % 60 == 0:
            result = analyser_image(pil_current)
            if result["categorie"] == "Humain":
                nom_v, conf_v = reconnaitre_visage(pil_current)
                result["nom_visage"]  = nom_v
                result["conf_visage"] = conf_v
            else:
                result["nom_visage"]  = None
                result["conf_visage"] = None
            with self.lock:
                self.result = result

        # ── Overlay catégorie (bande info bas de frame) ──
        with self.lock:
            current_result = self.result

        if current_result and not _g007_overlay["active"]:
            pil_draw = Image.fromarray(img_array)
            draw     = ImageDraw.Draw(pil_draw)
            cfg      = CATEGORY_CONFIG.get(current_result["categorie"], CATEGORY_CONFIG["Inconnu"])
            h, w     = img_array.shape[:2]
            draw.rectangle([(0, h - 65), (w, h)], fill=(0, 0, 0))
            display_name = current_result.get("nom_visage") or current_result["categorie"]
            prefix = (
                f"👋 {display_name}"
                if current_result.get("nom_visage")
                else f"{cfg['emoji']}  {display_name}"
            )
            draw.text(
                (12, h - 52), prefix,
                fill=(100, 255, 100) if current_result.get("nom_visage") else (255, 255, 255),
            )
            draw.text(
                (12, h - 28),
                f"Confiance : {current_result['score_pct']}  |  {current_result['label_reconnu']}",
                fill=(180, 180, 180),
            )
            img_array = np.array(pil_draw)

        # ── Overlay compte à rebours 007 ──
        if _g007_overlay["active"] and _g007_overlay["text"]:
            pil_draw = Image.fromarray(img_array)
            draw     = ImageDraw.Draw(pil_draw)
            h, w     = img_array.shape[:2]
            txt      = _g007_overlay["text"]
            col      = _g007_overlay["color"]
            try:
                font_size = {1: 160, 2: 150}.get(len(txt), 130)
                font_big  = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                font_big = ImageFont.load_default()
            try:
                bb       = draw.textbbox((0, 0), txt, font=font_big)
                tw, th   = bb[2] - bb[0], bb[3] - bb[1]
            except Exception:
                tw, th = 100, 100
            x, y = (w - tw) // 2, (h - th) // 2 - 20
            draw.text((x + 5, y + 5), txt, fill=(0, 0, 0), font=font_big)
            draw.text((x, y), txt, fill=col, font=font_big)
            img_array = np.array(pil_draw)

        return av.VideoFrame.from_ndarray(img_array, format="rgb24")
