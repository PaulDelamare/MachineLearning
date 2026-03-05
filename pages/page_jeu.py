# ─────────────────────────────────────────────────────────────────────────────
# pages/page_jeu.py — Onglet "Jeu de détection (Chasse aux Objets)"
# ─────────────────────────────────────────────────────────────────────────────
import random
import time

import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from config import CATEGORY_CONFIG, DEFIS_POOL
from modules.vision import VideoProcessor

NB_MANCHES = 10


def render() -> None:
    # ── Init session state ────────────────────────────────────────────────────
    for _key, _val in [
        ("game_active",     False),
        ("game_over",       False),
        ("game_round",      1),
        ("game_score",      0),
        ("game_history",    []),
        ("game_defi_order", []),
        ("game_start_time", 0.0),
        ("game_round_won",  False),
    ]:
        if _key not in st.session_state:
            st.session_state[_key] = _val

    # ════════════════════════════════════════════════════════════════════
    # ÉCRAN D'ACCUEIL
    # ════════════════════════════════════════════════════════════════════
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
                st.session_state.game_active     = True
                st.session_state.game_round      = 1
                st.session_state.game_score      = 0
                st.session_state.game_history    = []
                st.session_state.game_over       = False
                st.session_state.game_round_won  = False
                st.session_state.game_defi_order = random.sample(DEFIS_POOL, NB_MANCHES)
                st.session_state.game_start_time = time.time()
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # ÉCRAN FIN DE PARTIE
    # ════════════════════════════════════════════════════════════════════
    elif st.session_state.game_over:
        score_final = st.session_state.game_score
        score_max   = sum(d["points_max"] for d in st.session_state.game_defi_order)
        pct         = int(score_final / score_max * 100) if score_max > 0 else 0

        if pct >= 80:
            titre, medal = "CHAMPION ABSOLU !", "🏆"
        elif pct >= 60:
            titre, medal = "Très bien joué !",  "🥇"
        elif pct >= 40:
            titre, medal = "Pas mal du tout !",  "🥈"
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
        recap_data = [
            {
                "Manche":  f"#{i + 1}",
                "Défi":    h["defi"],
                "Résultat": "✅ Gagnée" if h["won"] else "❌ Perdue",
                "Points":  h["points"],
                "Temps":   f"{h['temps_pris']:.1f}s" if h["won"] else "—",
            }
            for i, h in enumerate(st.session_state.game_history)
        ]
        st.dataframe(pd.DataFrame(recap_data), use_container_width=True, hide_index=True)

        _, col_r, _ = st.columns([1, 2, 1])
        with col_r:
            if st.button("🔄 Rejouer une nouvelle partie !", type="primary", use_container_width=True):
                for k in ["game_active", "game_over", "game_round", "game_score",
                          "game_history", "game_defi_order", "game_round_won"]:
                    del st.session_state[k]
                st.rerun()

    # ════════════════════════════════════════════════════════════════════
    # PARTIE EN COURS
    # ════════════════════════════════════════════════════════════════════
    else:
        round_idx  = st.session_state.game_round - 1
        defi       = st.session_state.game_defi_order[round_idx]
        elapsed    = time.time() - st.session_state.game_start_time
        remaining  = max(0.0, defi["temps"] - elapsed)
        time_ratio = remaining / defi["temps"]

        hc1, hc2, hc3 = st.columns([2, 1, 1])
        with hc1:
            st.markdown(f"### Manche **{st.session_state.game_round}** / {NB_MANCHES}")
        with hc2:
            st.metric("🏅 Score", f"{st.session_state.game_score} pts")
        with hc3:
            color = "🟢" if time_ratio > 0.5 else ("🟡" if time_ratio > 0.2 else "🔴")
            st.metric(f"{color} Temps", f"{int(remaining)}s")
        st.progress(time_ratio)

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
            game_result = None
            if ctx_game and ctx_game.state.playing and ctx_game.video_processor:
                with ctx_game.video_processor.lock:
                    game_result = ctx_game.video_processor.result

            if game_result:
                detected_cat = game_result["categorie"]
                cfg_det      = CATEGORY_CONFIG.get(detected_cat, CATEGORY_CONFIG["Inconnu"])
                kws          = defi.get("keywords")
                won          = False
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

                if won and not st.session_state.game_round_won:
                    temps_pris = elapsed
                    if temps_pris < defi["temps"] * 0.25:
                        bonus     = int(defi["points_max"] * 0.5)
                        bonus_txt = f"⚡ Bonus ÉCLAIR +{bonus} pts !"
                    elif temps_pris < defi["temps"] * 0.5:
                        bonus     = int(defi["points_max"] * 0.25)
                        bonus_txt = f"🚀 Bonus RAPIDE +{bonus} pts !"
                    else:
                        bonus, bonus_txt = 0, ""

                    pts = defi["points_max"] + bonus
                    st.session_state.game_score     += pts
                    st.session_state.game_round_won  = True
                    st.session_state.game_history.append({
                        "defi": defi["texte"], "won": True,
                        "points": pts, "temps_pris": temps_pris,
                    })
                    st.balloons()
                    st.success(
                        f"🎉 TROUVÉ en {temps_pris:.1f}s ! **+{pts} pts**"
                        + (f"  \n{bonus_txt}" if bonus_txt else "")
                    )
                    time.sleep(2.5)
                    if st.session_state.game_round >= NB_MANCHES:
                        st.session_state.game_over   = True
                        st.session_state.game_active = False
                    else:
                        st.session_state.game_round     += 1
                        st.session_state.game_start_time = time.time()
                        st.session_state.game_round_won  = False
                    st.rerun()
            else:
                st.info("⏳ Lance la caméra et pointe-la vers l'objet !")

        # Temps écoulé
        if remaining <= 0 and not st.session_state.game_round_won:
            st.error(f"⏰ TEMPS ÉCOULÉ ! Il fallait trouver : **{defi['texte']}**")
            st.session_state.game_history.append({
                "defi": defi["texte"], "won": False, "points": 0, "temps_pris": 0,
            })
            time.sleep(2.5)
            if st.session_state.game_round >= NB_MANCHES:
                st.session_state.game_over   = True
                st.session_state.game_active = False
            else:
                st.session_state.game_round     += 1
                st.session_state.game_start_time = time.time()
                st.session_state.game_round_won  = False
            st.rerun()

        # Barre de progression des manches
        st.markdown("---")
        manche_cols = st.columns(NB_MANCHES)
        for i, col in enumerate(manche_cols):
            mn = i + 1
            if mn < st.session_state.game_round:
                h    = st.session_state.game_history[i] if i < len(st.session_state.game_history) else None
                icon = "✅" if (h and h["won"]) else "❌"
            elif mn == st.session_state.game_round:
                icon = "🎯"
            else:
                icon = "⬜"
            col.markdown(
                f"<div style='text-align:center'>{icon}<br><small style='color:#aaa'>#{mn}</small></div>",
                unsafe_allow_html=True,
            )

        if st.button("⏭️ Passer cette manche (0 pts)"):
            st.session_state.game_history.append({
                "defi": defi["texte"], "won": False, "points": 0, "temps_pris": 0,
            })
            if st.session_state.game_round >= NB_MANCHES:
                st.session_state.game_over   = True
                st.session_state.game_active = False
            else:
                st.session_state.game_round     += 1
                st.session_state.game_start_time = time.time()
                st.session_state.game_round_won  = False
            st.rerun()

        # Auto-refresh chrono
        if st.session_state.game_active and not st.session_state.game_round_won:
            time.sleep(0.8)
            st.rerun()
