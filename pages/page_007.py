# ─────────────────────────────────────────────────────────────────────────────
# pages/page_007.py — Onglet "007 Duel" (apprentissage gestes + jeu)
# ─────────────────────────────────────────────────────────────────────────────
import random
import time

import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from config import (
    GESTURES_CONFIG, GESTURES_NB_MIN,
    GESTES_KEYS, JEU007_VIES_MAX, JEU007_BALLES_MAX,
)
from modules.gestures import (
    charger_db_gestes,
    enregistrer_geste,
    reconnaitre_geste_vote,
    sauvegarder_db_gestes,
)
from modules.rl_007 import (
    calculer_reward_ia,
    charger_qtable,
    etat_007,
    ia_apprendre,
    ia_choisit_geste,
    resoudre_duel,
    s_entrainer_007,
    sauvegarder_qtable,
)
from modules.vision import VideoProcessor, _g007_overlay


def render() -> None:
    st.header("🔫 007 – Duel contre l'IA")
    st.caption("Apprends à l'IA tes gestes, puis affronte-la en duel !")
    st.markdown("---")

    sub_appren, sub_jeu007 = st.tabs(["📚 1. Apprendre les gestes", "🎮 2. Jouer"])

    # ══════════════════════════════════════════════════════════════════
    # SOUS-ONGLET A — APPRENTISSAGE DES GESTES
    # ══════════════════════════════════════════════════════════════════
    with sub_appren:
        st.subheader("📸 Enregistre tes gestes")
        st.markdown(
            f"Minimum **{GESTURES_NB_MIN} photos** par geste pour jouer — mais plus tu en ajoutes, "
            "plus la détection sera précise. **Pas besoin d'appuyer pendant le geste** : "
            "clique sur le bouton, puis pose les deux mains et attends le décompte !"
        )
        st.markdown("---")

        # ── Init session state décompte ──
        for _k, _dv in [
            ("glearn_geste_sel", list(GESTURES_CONFIG.keys())[0]),
            ("glearn_cd_start",  None),
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
            st.success(
                "✅ Minimum atteint ! Va dans **🎮 2. Jouer** pour démarrer. "
                "Tu peux continuer à ajouter des photos pour améliorer la précision."
            )
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
                f"<span style='color:#aaa; font-size:.95em;'>{nb} photo(s)"
                f"{' ✅' if ok else f' / {GESTURES_NB_MIN} min'}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Sélecteur de geste ──
        geste_labels = {k: f"{v['emoji']} {v['label']}" for k, v in GESTURES_CONFIG.items()}
        geste_sel    = st.radio(
            "Quel geste veux-tu capturer ?",
            options=list(GESTURES_CONFIG.keys()),
            format_func=lambda k: geste_labels[k],
            horizontal=True,
            key="glearn_geste_sel",
        )
        gcfg_sel = GESTURES_CONFIG[geste_sel]
        st.caption(f"💡 {gcfg_sel['desc']}")
        st.markdown("")

        # ── Webcam + décompte ──
        lcol, rcol = st.columns([1.2, 1])

        with lcol:
            ctx_learn = webrtc_streamer(
                key="learn_cam",
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with rcol:
            cd_box  = st.empty()
            msg_box = st.empty()
            cd_start = st.session_state.glearn_cd_start

            if cd_start is not None:
                elapsed_cd = time.time() - cd_start
                remaining  = 3.0 - elapsed_cd

                if remaining > 0:
                    step = int(remaining) + 1   # 3 → 2 → 1
                    cd_box.markdown(
                        f"<div style='text-align:center; padding:30px; "
                        f"border:3px solid #ffa500; border-radius:16px;'>"
                        f"<p style='color:#aaa; margin:0;'>Prépare ton geste :</p>"
                        f"<h1 style='font-size:5em; margin:4px 0; color:#ffa500;'>{step}</h1>"
                        f"<p style='color:#aaa; font-size:1.1em'>{gcfg_sel['emoji']} {gcfg_sel['label']}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.25)
                    st.rerun()
                else:
                    # Capture !
                    st.session_state.glearn_cd_start = None
                    captured_frames = []
                    if ctx_learn and ctx_learn.video_processor:
                        with ctx_learn.video_processor.lock:
                            captured_frames = list(ctx_learn.video_processor.frame_buffer)

                    if captured_frames:
                        mid       = len(captured_frames) // 2
                        frame_cap = captured_frames[mid]
                        ok_g, msg_g = enregistrer_geste(frame_cap, geste_sel)
                        st.session_state.glearn_msg = msg_g if ok_g else f"❌ {msg_g}"
                    else:
                        st.session_state.glearn_msg = "❌ Caméra inactive — lance la webcam d'abord !"
                    st.rerun()
            else:
                nb_sel = len(db_gestes.get(geste_sel, []))
                cd_box.markdown(
                    f"<div style='text-align:center; padding:30px; "
                    f"border:2px dashed #444; border-radius:16px;'>"
                    f"<div style='font-size:3em'>{gcfg_sel['emoji']}</div>"
                    f"<p style='color:#aaa; margin:8px 0;'>{gcfg_sel['label']}</p>"
                    f"<p style='color:#777; font-size:.9em;'>{nb_sel} photo(s) enregistrée(s)</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    f"📸 Capturer dans 3 secondes — {gcfg_sel['emoji']} {gcfg_sel['label']}",
                    key="btn_capture_geste",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.glearn_cd_start = time.time()
                    st.session_state.glearn_msg      = ""
                    st.rerun()

                if st.session_state.glearn_msg:
                    if "✅" in st.session_state.glearn_msg:
                        msg_box.success(st.session_state.glearn_msg)
                    else:
                        msg_box.error(st.session_state.glearn_msg)

                nb_sel2 = len(db_gestes.get(geste_sel, []))
                if nb_sel2 > 0:
                    if st.button(
                        f"🗑️ Supprimer toutes les photos de {gcfg_sel['label']} ({nb_sel2})",
                        key="btn_del_geste",
                    ):
                        db_g2 = charger_db_gestes()
                        db_g2[geste_sel] = []
                        sauvegarder_db_gestes(db_g2)
                        st.session_state.glearn_msg = ""
                        st.rerun()

        st.markdown("---")
        if st.button("🗑️ Réinitialiser TOUS les gestes"):
            sauvegarder_db_gestes({})
            st.session_state.glearn_msg = ""
            st.warning("Tous les gestes ont été effacés.")
            st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # SOUS-ONGLET B — JEU
    # ══════════════════════════════════════════════════════════════════
    with sub_jeu007:
        # ── Init session state 007 ──
        for _k, _v in [
            ("g007_active",      False),
            ("g007_over",        False),
            ("g007_j_vies",      JEU007_VIES_MAX),
            ("g007_ia_vies",     JEU007_VIES_MAX),
            ("g007_j_balles",    0),
            ("g007_ia_balles",   0),
            ("g007_manche",      1),
            ("g007_history",     []),
            ("g007_last",        None),
            ("g007_phase",       "idle"),
            ("g007_phase_t",     0.0),
            ("g007_ia_pre",      None),
            ("g007_prev_state",  None),
            ("g007_prev_action", None),
            ("g007_pending",     None),
        ]:
            if _k not in st.session_state:
                st.session_state[_k] = _v

        db_g         = charger_db_gestes()
        gestes_prets = all(len(db_g.get(k, [])) >= GESTURES_NB_MIN for k in GESTURES_CONFIG)

        # ══════════════════════════════════════
        # ÉCRAN D'ACCUEIL
        # ══════════════════════════════════════
        if not st.session_state.g007_active and not st.session_state.g007_over:
            _g007_overlay["active"] = False

            if not gestes_prets:
                manquants = [
                    f"{GESTURES_CONFIG[k]['emoji']} {GESTURES_CONFIG[k]['label']} "
                    f"({len(db_g.get(k, []))}/{GESTURES_NB_MIN})"
                    for k in GESTURES_CONFIG if len(db_g.get(k, [])) < GESTURES_NB_MIN
                ]
                st.warning("⚠️ Enregistre d'abord tous tes gestes dans **📚 1. Apprendre les gestes** !")
                for m in manquants:
                    st.markdown(f"- {m}")
            else:
                qt       = charger_qtable()
                nb_etats = len(qt)
                st.markdown(f"""
                <div style='text-align:center; padding:30px 20px;'>
                    <h1 style='font-size:3em;'>🔫 007 DUEL !</h1>
                    <p style='font-size:1.2em; color:#aaa;'>
                        Affronte l'IA au jeu <b>007</b> !<br>
                        Le compte à rebours s'affiche sur la caméra — tiens ton geste au
                        <b style='color:#e94560;'>7</b> !<br><br>
                        🤙 <b>Recharger</b> &nbsp;·&nbsp; 🔫 <b>Tirer</b> &nbsp;·&nbsp;
                        🛡️ <b>Se protéger</b><br><br>
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

                # ── Section entraînement ──
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
                            "Nombre de parties", 100, 20000, 2000, step=100, key="train007_nb"
                        )
                        duree_train_s  = None
                        label_spinner  = f"🏋️ Entraînement sur {nb_parties_train} parties..."
                    else:
                        duree_min = st.slider(
                            "Durée (minutes)", 1, 30, 5, step=1, key="train007_min"
                        )
                        nb_parties_train = None
                        duree_train_s    = duree_min * 60
                        label_spinner    = f"🏋️ Entraînement pendant {duree_min} min..."

                    if st.button("▶️ Démarrer l'entraînement", key="train007_go", use_container_width=True):
                        with st.spinner(label_spinner):
                            qt_new, stats_t = s_entrainer_007(
                                nb_parties=nb_parties_train,
                                duree_s=duree_train_s,
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

                    res_t = st.session_state.get("train007_result")
                    if res_t:
                        nb_t  = res_t["nb"]
                        pct_v = res_t["victoires"] / max(nb_t, 1) * 100
                        pct_d = res_t["defaites"]  / max(nb_t, 1) * 100
                        pct_n = res_t["nuls"]      / max(nb_t, 1) * 100
                        dur_r = res_t["duree_s"]
                        vps   = nb_t / dur_r if dur_r > 0 else 0
                        moy_tours = res_t["tours"] / max(nb_t, 1)
                        dur_str = (
                            f"{int(dur_r // 60)}min {int(dur_r % 60)}s"
                            if dur_r >= 60 else f"{dur_r:.1f}s"
                        )
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
  <div style='margin-bottom:6px; color:#aaa; font-size:.85em;'>Résultats IA</div>
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

        # ══════════════════════════════════════
        # ÉCRAN FIN DE PARTIE
        # ══════════════════════════════════════
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

            hearts_j  = "❤️" * max(0, j_vies)  + "🖤" * max(0, JEU007_VIES_MAX - j_vies)
            hearts_ia = "❤️" * max(0, ia_vies) + "🖤" * max(0, JEU007_VIES_MAX - ia_vies)
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
                    for _k in [
                        "g007_active", "g007_over", "g007_j_vies", "g007_ia_vies",
                        "g007_j_balles", "g007_ia_balles", "g007_manche",
                        "g007_history", "g007_last", "g007_phase", "g007_phase_t",
                        "g007_ia_pre", "g007_prev_state", "g007_prev_action", "g007_pending",
                    ]:
                        if _k in st.session_state:
                            del st.session_state[_k]
                    st.rerun()

        # ══════════════════════════════════════
        # PARTIE EN COURS
        # ══════════════════════════════════════
        else:
            j_vies    = st.session_state.g007_j_vies
            ia_vies   = st.session_state.g007_ia_vies
            j_balles  = st.session_state.g007_j_balles
            ia_balles = st.session_state.g007_ia_balles
            manche    = st.session_state.g007_manche
            phase     = st.session_state.g007_phase
            elapsed   = time.time() - st.session_state.g007_phase_t

            # ── Header vies / balles ──
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("❤️ Tes vies",   "❤️" * j_vies  + "🖤" * (JEU007_VIES_MAX - j_vies))
            hc2.metric("🔫 Tes balles", f"{j_balles} / {JEU007_BALLES_MAX}")
            hc3.metric("🤖 Vies IA",    "❤️" * ia_vies + "🖤" * (JEU007_VIES_MAX - ia_vies))
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
                    unsafe_allow_html=True,
                )
            st.markdown("")

            # ── Webcam + résultat ──
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
                pending = st.session_state.g007_pending

                if phase == "result" and pending:
                    j_cfg_r    = GESTURES_CONFIG.get(pending["j_geste"],  {})
                    ia_cfg_r   = GESTURES_CONFIG.get(pending["ia_geste"], {})
                    j_touche_r  = pending["j_touche"]
                    ia_touche_r = pending["ia_touche"]

                    if ia_touche_r and not j_touche_r:
                        bg_r, titre_r, border_r = "#1a3a1a", "✅ IA TOUCHÉE !",   "#4caf50"
                    elif j_touche_r and not ia_touche_r:
                        bg_r, titre_r, border_r = "#3a1a1a", "💥 TU ES TOUCHÉ(E) !", "#e94560"
                    elif j_touche_r and ia_touche_r:
                        bg_r, titre_r, border_r = "#3a2a00", "💥 DOUBLE TOUCHE !", "#ff9800"
                    else:
                        bg_r, titre_r, border_r = "#1a1a2e", "= NEUTRE", "#666"

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

                elif phase in ("c0a", "c0b", "c7"):
                    label_phase = {
                        "c0a": ("0",   "🎯 Prépare ton geste..."),
                        "c0b": ("00",  "🎯 Prépare ton geste..."),
                        "c7":  ("007", "📸 Tiens ton geste !"),
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

            # ══════════ Logique des phases ══════════

            if phase == "c0a":
                cam_prete = (
                    ctx_007 and ctx_007.video_processor
                    and ctx_007.video_processor.last_frame_pil is not None
                )
                if not cam_prete:
                    st.session_state.g007_phase_t = time.time()
                    elapsed = 0.0
                _g007_overlay.update({"active": True, "text": "0", "color": (255, 165, 0)})
                if cam_prete and elapsed >= 0.7:
                    st.session_state.g007_phase   = "c0b"
                    st.session_state.g007_phase_t = time.time()
                    st.rerun()
                time.sleep(0.2)
                st.rerun()

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

            elif phase == "c7":
                _g007_overlay.update({"active": True, "text": "007", "color": (255, 50, 50)})
                if elapsed >= 0.8:
                    captured_frames = []
                    if ctx_007 and ctx_007.video_processor:
                        with ctx_007.video_processor.lock:
                            captured_frames = list(ctx_007.video_processor.frame_buffer)

                    if captured_frames:
                        j_geste, j_conf = reconnaitre_geste_vote(captured_frames)
                    else:
                        j_geste, j_conf = None, "caméra inactive"

                    if j_geste is None:
                        j_geste = random.choice(GESTES_KEYS)
                        j_conf  = "non reconnu ⚠️"

                    ia_geste = st.session_state.g007_ia_pre or random.choice(GESTES_KEYS)

                    (j_balles_new, ia_balles_new, j_vies_new, ia_vies_new,
                     msgs, j_touche, ia_touche) = resoudre_duel(
                        j_geste, ia_geste, j_balles, ia_balles, j_vies, ia_vies
                    )

                    reward_ia = calculer_reward_ia(
                        ia_geste, j_geste, ia_balles, j_balles,
                        ia_touche, j_touche, j_vies_new, ia_vies_new,
                    )
                    qt_up   = charger_qtable()
                    next_st = etat_007(j_balles_new, ia_balles_new, j_vies_new, ia_vies_new)
                    if st.session_state.g007_prev_state is not None:
                        qt_up = ia_apprendre(
                            qt_up,
                            st.session_state.g007_prev_state,
                            st.session_state.g007_prev_action,
                            reward_ia, next_st,
                        )
                        sauvegarder_qtable(qt_up)

                    if ia_touche and not j_touche:   res_txt = "🤖 IA touchée"
                    elif j_touche and not ia_touche: res_txt = "💥 Joueur touché"
                    elif j_touche and ia_touche:     res_txt = "💥 Double touche"
                    else:                             res_txt = "= Neutre"

                    st.session_state.g007_history.append({
                        "j_geste":   j_geste,  "ia_geste":  ia_geste,
                        "j_conf":    j_conf,   "res_txt":   res_txt,
                        "j_touche":  j_touche, "ia_touche": ia_touche,
                        "msgs":      msgs,
                    })
                    st.session_state.g007_pending = {
                        "j_geste":  j_geste,  "ia_geste":  ia_geste,
                        "j_conf":   j_conf,   "msgs":      msgs,
                        "j_touche": j_touche, "ia_touche": ia_touche,
                    }
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

            elif phase == "result":
                _g007_overlay["active"] = False
                if elapsed >= 3.0:
                    if st.session_state.g007_j_vies <= 0 or st.session_state.g007_ia_vies <= 0:
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
