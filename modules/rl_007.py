# ─────────────────────────────────────────────────────────────────────────────
# modules/rl_007.py — Q-learning, règles du duel 007, entraînement auto-jeu
# ─────────────────────────────────────────────────────────────────────────────
import os
import pickle
import random
import time

import numpy as np

from config import (
    Q007_FILE, Q007_LR, Q007_GAMMA, Q007_EPSILON,
    GESTES_KEYS, JEU007_VIES_MAX, JEU007_BALLES_MAX,
)


# ── Persistence Q-table ───────────────────────────────────────────────────────

def charger_qtable() -> dict:
    if os.path.exists(Q007_FILE):
        with open(Q007_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def sauvegarder_qtable(qt: dict) -> None:
    with open(Q007_FILE, "wb") as f:
        pickle.dump(qt, f)


# ── Représentation d'état ─────────────────────────────────────────────────────

def etat_007(my_balles: int, opp_balles: int, my_vies: int, opp_vies: int) -> tuple:
    """Encode l'état courant en tuple hashable."""
    return (
        min(my_balles,  JEU007_BALLES_MAX),
        min(opp_balles, JEU007_BALLES_MAX),
        my_vies,
        opp_vies,
    )


# ── Politiques ───────────────────────────────────────────────────────────────

def ia_choisit_geste(my_balles: int, my_vies: int, opp_vies: int,
                     state: tuple, qt: dict) -> str:
    """Stratégie epsilon-greedy : exploite la Q-table ou explore aléatoirement."""
    if random.random() < Q007_EPSILON or state not in qt:
        choices = ["recharger", "proteger"]
        if my_balles > 0:
            choices.append("tirer")
        return random.choice(choices)
    q_vals = qt[state]
    return GESTES_KEYS[int(np.argmax(q_vals))]


# ── Apprentissage Bellman ─────────────────────────────────────────────────────

def ia_apprendre(qt: dict, state: tuple, action_idx: int,
                 reward: float, next_state: tuple) -> dict:
    """Mise à jour Q(s,a) via l'équation de Bellman."""
    if state not in qt:
        qt[state]      = np.zeros(len(GESTES_KEYS))
    if next_state not in qt:
        qt[next_state] = np.zeros(len(GESTES_KEYS))
    old_q   = qt[state][action_idx]
    max_nxt = float(np.max(qt[next_state]))
    qt[state][action_idx] = old_q + Q007_LR * (reward + Q007_GAMMA * max_nxt - old_q)
    return qt


# ── Fonction de récompense ────────────────────────────────────────────────────

def calculer_reward_ia(ia_geste: str, j_geste: str,
                       ia_balles_avant: int, j_balles_avant: int,
                       ia_touche: bool, j_touche: bool,
                       j_vies_new: int, ia_vies_new: int) -> float:
    """Calcule la récompense pour l'IA après un tour."""
    r = 0.0

    # Combat
    if j_touche:   r += 25   # toucher l'adversaire
    if ia_touche:  r -= 25   # se faire toucher

    # Initiative : tirer
    if ia_geste == "tirer" and ia_balles_avant > 0:
        r += 2   # petit bonus pour l'initiative

    # Tour neutre
    if not j_touche and not ia_touche:
        r -= 3   # chaque tour sans décision coûte

    # Protection
    if j_geste == "tirer" and j_balles_avant > 0 and ia_geste == "proteger":
        r += 12  # bloquer un vrai tir
    elif ia_geste == "proteger":
        r -= 8   # se protéger sans raison : punit fortement

    # Munitions
    if ia_geste == "recharger":
        if ia_balles_avant >= JEU007_BALLES_MAX:
            r -= 14  # recharger quand plein
        else:
            r += 3   # recharger utile
    if ia_geste == "tirer" and ia_balles_avant == 0:
        r -= 8   # tirer sans munitions

    # Fin de partie
    if ia_vies_new <= 0:  r -= 40
    if j_vies_new  <= 0:  r += 40

    return r


# ── Règles du duel ────────────────────────────────────────────────────────────

def resoudre_duel(j_geste: str, ia_geste: str,
                  j_balles: int, ia_balles: int,
                  j_vies: int, ia_vies: int) -> tuple:
    """
    Résout un tour de duel et retourne :
    (j_balles, ia_balles, j_vies, ia_vies, messages, j_touche, ia_touche)
    """
    msgs      = []
    j_touche  = False
    ia_touche = False

    # ── Joueur TIRE ──
    if j_geste == "tirer":
        if j_balles > 0:
            j_balles -= 1
            if ia_geste != "proteger":
                ia_vies  -= 1
                ia_touche = True
                msgs.append("🔫 Tu tires → IA touchée !")
            else:
                msgs.append("🛡️ Tu tires → IA se protège !")
        else:
            msgs.append("⚠️ Tu tires mais n'as plus de balles !")

    # ── IA TIRE ──
    if ia_geste == "tirer":
        if ia_balles > 0:
            ia_balles -= 1
            if j_geste != "proteger":
                j_vies   -= 1
                j_touche  = True
                msgs.append("🤖 IA tire → tu es touché(e) !")
            else:
                msgs.append("🛡️ IA tire → tu te protèges !")
        else:
            msgs.append("⚠️ IA tire mais n'a plus de balles !")

    # ── Rechargements ──
    if j_geste == "recharger":
        j_balles = min(JEU007_BALLES_MAX, j_balles + 1)
        msgs.append("🤙 Tu recharges !")

    if ia_geste == "recharger":
        ia_balles = min(JEU007_BALLES_MAX, ia_balles + 1)
        msgs.append("🤖 IA recharge !")

    # ── Messages neutres ──
    if not msgs:
        msgs.append("= Neutre — aucune action décisive.")

    return j_balles, ia_balles, j_vies, ia_vies, msgs, j_touche, ia_touche


# ── Entraînement asymétrique ──────────────────────────────────────────────────

def s_entrainer_007(nb_parties: int = None, duree_s: float = None) -> tuple:
    """
    Entraîne le bot par auto-jeu ASYMÉTRIQUE pour éviter la convergence vers les nuls.
    Alterne 3 modes de match à chaque partie :
      - QvQ   (40%) : les deux agents exploitent la Q-table
      - QvRand(35%) : agent Q affronte un opposant purement aléatoire
      - QvAgro(25%) : agent Q affronte un opposant qui tire dès qu'il a des balles
    """
    if nb_parties is None and duree_s is None:
        nb_parties = 500

    qt     = charger_qtable()
    stats  = {
        "victoires": 0, "defaites": 0, "nuls": 0,
        "tours_total": 0, "parties": 0,
        "modes": {"QvQ": 0, "QvRand": 0, "QvAgro": 0},
    }
    t_debut    = time.time()
    partie_idx = 0

    def geste_random(balles, *_):
        c = ["recharger", "proteger"]
        if balles > 0:
            c.append("tirer")
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

            ig = ia_choisit_geste(ib, iv, jv, s_ia, qt)

            if   mode == "QvQ":    jg = ia_choisit_geste(jb, jv, iv, s_j, qt)
            elif mode == "QvRand": jg = geste_random(jb)
            else:                  jg = geste_agro(jb)

            jb_av = jb; ib_av = ib
            jb, ib, jv, iv, _, jt, it = resoudre_duel(jg, ig, jb, ib, jv, iv)

            ns_ia = etat_007(jb, ib, jv, iv)
            ns_j  = etat_007(ib, jb, iv, jv)

            r_ia = calculer_reward_ia(ig, jg, ib_av, jb_av, it, jt, jv, iv)
            qt   = ia_apprendre(qt, s_ia, GESTES_KEYS.index(ig), r_ia, ns_ia)

            if mode == "QvQ":
                r_j = calculer_reward_ia(jg, ig, jb_av, ib_av, jt, it, iv, jv)
                qt  = ia_apprendre(qt, s_j, GESTES_KEYS.index(jg), r_j, ns_j)

        if iv > jv:
            stats["victoires"] += 1
        elif jv > iv:
            stats["defaites"] += 1
        else:
            stats["nuls"] += 1
            s_ia_fin = etat_007(jb, ib, jv, iv)
            qt = ia_apprendre(qt, s_ia_fin, GESTES_KEYS.index(ig), -20, s_ia_fin)
            if mode == "QvQ":
                s_j_fin = etat_007(ib, jb, iv, jv)
                qt = ia_apprendre(qt, s_j_fin, GESTES_KEYS.index(jg), -20, s_j_fin)

        stats["parties"] += 1

    stats["duree_reelle_s"] = time.time() - t_debut
    sauvegarder_qtable(qt)
    return qt, stats
