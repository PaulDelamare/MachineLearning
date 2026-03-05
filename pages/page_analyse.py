# ─────────────────────────────────────────────────────────────────────────────
# pages/page_analyse.py — Onglet "Analyse d'image"
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime

import pandas as pd
import streamlit as st
from bson.objectid import ObjectId
from PIL import Image
from streamlit_webrtc import webrtc_streamer

from config import CATEGORY_CONFIG
from modules.database import collection
from modules.faces import reconnaitre_visage
from modules.vision import VideoProcessor, analyser_image


def render() -> None:
    zone_upload, zone_resultat = st.columns([1, 1], gap="large")

    # ── Zone upload / webcam ──────────────────────────────────────────────────
    with zone_upload:
        st.subheader("📸 Capture ou Upload")
        mode = st.radio(
            "Source de l'image :",
            ["📷 Webcam (temps réel)", "📁 Fichier"],
            horizontal=True,
        )

        image_source   = None
        uploaded_file  = None
        ctx            = None

        if mode == "📷 Webcam (temps réel)":
            st.caption("⚡ La catégorie s'affiche sur la vidéo. Cliquez START pour lancer.")
            ctx = webrtc_streamer(
                key="visionai-live",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            uploaded_file = st.file_uploader(
                "Choisissez un fichier image", type=["png", "jpg", "jpeg", "webp"]
            )
            if uploaded_file is not None:
                pil_image    = Image.open(uploaded_file).convert("RGB")
                image_source = {"name": uploaded_file.name, "size": uploaded_file.size}
                st.image(pil_image, caption=uploaded_file.name, use_container_width=True)
                st.caption(f"Taille : {image_source['size']} octets")

    # ── Zone résultat ─────────────────────────────────────────────────────────
    with zone_resultat:
        st.subheader("🧠 Résultat de l'Analyse")

        # MODE WEBCAM TEMPS RÉEL
        if mode == "📷 Webcam (temps réel)":
            if ctx and ctx.state.playing and ctx.video_processor:
                with ctx.video_processor.lock:
                    live_result = ctx.video_processor.result

                if live_result:
                    cfg         = CATEGORY_CONFIG.get(live_result["categorie"], CATEGORY_CONFIG["Inconnu"])
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
                    c2.metric("Confiance",           live_result["score_pct"])
                    st.info(f"Label : `{live_result['label_reconnu']}`")
                    if live_result["easter_egg"]:
                        st.balloons()
                        st.warning(live_result["easter_egg"])
                    else:
                        st.success(cfg["message"])

                    st.markdown("---")
                    if st.button("💾 Sauvegarder cette détection dans MongoDB", type="primary"):
                        try:
                            collection.insert_one({
                                "date": datetime.now(),
                                "nom":  "webcam_live.jpg",
                                "taille": 0,
                                "analyse": {
                                    "taux_reussite": live_result["score_pct"],
                                    "type_reconnu":  live_result["categorie"],
                                    "label_brut":    live_result["label_brut"],
                                    "label_reconnu": live_result["label_reconnu"],
                                },
                            })
                            st.success("✅ Sauvegardé dans MongoDB !")
                        except Exception as e:
                            st.error(f"Erreur MongoDB : {e}")
                else:
                    st.info("⏳ En attente de la première analyse... (environ 2 secondes après START)")
            else:
                st.info("▶️ Cliquez sur START dans le flux vidéo pour lancer la détection en temps réel.")

        # MODE FICHIER
        elif image_source is not None:
            with st.spinner("Analyse en cours par le modèle ViT..."):
                resultat = analyser_image(pil_image)

            cfg = CATEGORY_CONFIG.get(resultat["categorie"], CATEGORY_CONFIG["Inconnu"])
            st.markdown(f"## {cfg['emoji']} {resultat['categorie']}")

            try:
                collection.insert_one({
                    "date":   datetime.now(),
                    "nom":    image_source["name"],
                    "taille": image_source["size"],
                    "analyse": {
                        "taux_reussite": resultat["score_pct"],
                        "type_reconnu":  resultat["categorie"],
                        "label_brut":    resultat["label_brut"],
                        "label_reconnu": resultat["label_reconnu"],
                    },
                })
                st.success("✅ Résultat enregistré dans MongoDB !")
            except Exception as e:
                st.error(f"Erreur MongoDB : {e}")

            nom_reconnu = conf_visage = None
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
            c1.metric("Catégorie détectée",   resultat["categorie"])
            c2.metric("Taux de réussite",     resultat["score_pct"])
            st.info(
                f"Label principal du modèle : `{resultat['label_brut']}`  \n"
                f"Label ayant déclenché la catégorie : `{resultat['label_reconnu']}`"
            )
            if resultat["easter_egg"]:
                st.balloons()
                st.warning(resultat["easter_egg"])
            else:
                st.success(cfg["message"])
        else:
            st.info("Prenez une photo ou uploadez une image à gauche pour lancer l'analyse.")

    # ── Historique ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.header("📋 Historique des Analyses")

    historique_docs = list(collection.find().sort("date", -1))
    if historique_docs:
        historique_a_afficher = []
        for doc in historique_docs:
            analyse   = doc.get("analyse", {})
            categorie = analyse.get("type_reconnu", "Inconnu")
            cfg       = CATEGORY_CONFIG.get(categorie, CATEGORY_CONFIG["Inconnu"])
            historique_a_afficher.append({
                "ID":             str(doc["_id"]),
                "Date":           doc["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "Nom du fichier": doc.get("nom", "N/A"),
                "Taille (octets)": doc.get("taille", "N/A"),
                "Catégorie":      f"{cfg['emoji']} {categorie}",
                "Taux de réussite": analyse.get("taux_reussite", "N/A"),
                "Label brut":     analyse.get("label_brut", "N/A"),
            })

        df = pd.DataFrame(historique_a_afficher)
        st.dataframe(df.drop(columns=["ID"]), use_container_width=True, hide_index=True)

        st.markdown("### 📊 Répartition des catégories détectées")
        comptage = df["Catégorie"].value_counts().reset_index()
        comptage.columns = ["Catégorie", "Nombre d'analyses"]
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
