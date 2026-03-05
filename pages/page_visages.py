# ─────────────────────────────────────────────────────────────────────────────
# pages/page_visages.py — Onglet "Reconnaissances de visages"
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
from PIL import Image

from modules.faces import (
    charger_db_visages,
    enregistrer_visage,
    reconnaitre_visage,
    sauvegarder_db_visages,
)


def render() -> None:
    st.header("👥 Registre des visages")
    st.caption("Enregistre ton visage ici — l'IA te reconnaîtra dans l'onglet Analyse ou pendant le jeu !")
    st.markdown("---")

    vcol1, vcol2 = st.columns([1, 1], gap="large")

    # ── Enregistrement ────────────────────────────────────────────────────────
    with vcol1:
        st.subheader("➕ Ajouter un visage")
        nom_input = st.text_input("📝 Prénom ou nom à associer :", placeholder="ex: Paul, Marie...")
        reg_mode  = st.radio("Source :", ["📷 Webcam", "📂 Fichier"], horizontal=True)

        reg_img = None
        if reg_mode == "📷 Webcam":
            reg_photo = st.camera_input("Prends une photo de ton visage")
            if reg_photo:
                reg_img = Image.open(reg_photo).convert("RGB")
                st.image(reg_img, caption="Photo capturée", use_container_width=True)
        else:
            reg_file = st.file_uploader(
                "Importe une photo", type=["jpg", "jpeg", "png"], key="reg_face_upload"
            )
            if reg_file:
                reg_img = Image.open(reg_file).convert("RGB")
                st.image(reg_img, caption=reg_file.name, use_container_width=True)

        if st.button(
            "💾 Enregistrer ce visage",
            type="primary",
            disabled=(not nom_input or reg_img is None),
        ):
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

    # ── Registre actuel ───────────────────────────────────────────────────────
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

    # ── Test de reconnaissance ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧐 Tester la reconnaissance")
    test_mode = st.radio(
        "Source du test :",
        ["📷 Webcam", "📂 Fichier"],
        horizontal=True,
        key="test_mode",
    )

    test_img = None
    if test_mode == "📷 Webcam":
        test_snap = st.camera_input("Prends une photo pour tester")
        if test_snap:
            test_img = Image.open(test_snap).convert("RGB")
    else:
        test_file = st.file_uploader(
            "Importe une photo test", type=["jpg", "jpeg", "png"], key="test_face_upload"
        )
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
