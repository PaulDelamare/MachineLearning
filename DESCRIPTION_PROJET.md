# VisionIA — Description technique complète

---

## Le projet en une phrase

VisionIA est une application web temps réel qui utilise la **vision par ordinateur** et l'**intelligence artificielle** pour reconnaître ce qu'une caméra filme : objets, animaux, personnes, visages identifiés nominalement — avec en bonus un jeu de duel 007 dont l'adversaire IA apprend en jouant.

---

## Stack technologique

| Couche | Technologie |
|---|---|
| Interface web | **Streamlit** (Python) |
| Streaming webcam | **streamlit-webrtc** + **libav** (av) |
| Modèle principal de classification | **ViT** `google/vit-base-patch16-224` via **HuggingFace Transformers** |
| Modèle comparaison | **MobileNetV2** via **TensorFlow 2.20 / Keras** |
| Reconnaissance faciale | **MTCNN** (TF) + **MobileNetV2** feature extractor (TF) |
| Reconnaissance de gestes | **MediaPipe Hands** (landmarks CPU) |
| Bot jeu 007 | **Q-learning** (Reinforcement Learning, NumPy) |
| Base de données analyses | **MongoDB** local |
| Stockage embeddings | Fichiers `.pkl` (sérialisation Python) |

---

## Les 4 onglets de l'application

### 1. Analyse d'image

- Upload d'image **ou** webcam en temps réel
- Analyse par **ViT** (top-5 ImageNet) → catégorie principale affichée
- Analyse simultanée par **MobileNetV2 TF** → affichée en comparaison sous le résultat ViT
- Si la catégorie détectée est **"Humain"** → lancement automatique de la reconnaissance faciale (si des visages ont été enregistrés)
- Sauvegarde dans **MongoDB** avec date, nom du fichier, taux de réussite, catégorie
- Historique consultable avec graphique en barres et suppression au choix
- 🥚 Easter egg : si ViT détecte `ocarina`, `cloak`, `puppet`, `yoda` → message "BABY YODA DÉTECTÉ !"

### 2. Jeu de détection (10 manches)

- L'appli tire au sort un défi parmi un pool d'objets (chat, voiture, plante, nourriture...)
- Le joueur a un temps limité pour montrer l'objet à la webcam
- La détection ViT valide ou invalide la bonne catégorie
- Score cumulé sur 10 manches avec **bonus de rapidité** (trouver en moins de 25% du temps → +50% de points)
- Récapitulatif complet en fin de partie

### 3. Reconnaissance de visages

**Phase d'enregistrement :**
1. Le joueur prend une ou plusieurs photos de son visage
2. **MTCNN** (TF) localise et extrait la boîte englobante du visage
3. Le crop est redimensionné en 224×224 et passé dans **MobileNetV2 sans sa tête finale** (`include_top=False, pooling="avg"`) → vecteur de **1280 dimensions** = l'empreinte numérique du visage
4. Ce vecteur est normalisé (norme L2 = 1) et stocké dans `faces_db.pkl` associé au prénom

**Phase de reconnaissance :**
1. Même pipeline : MTCNN → MobileNetV2 → vecteur 1280-dim normalisé
2. Comparaison avec tous les vecteurs connus via la **distance euclidienne** : $d = \|emb_1 - emb_2\|_2$
3. Si $d < 0.9$ → identifié, confiance $= \max(0,\ (1 - d/0.9) \times 100)\%$
4. L'overlay webcam affiche le nom en vert en temps réel

### 4. Jeu 007 — Duel contre l'IA

Deux sous-onglets :

**Apprendre les gestes** : chaque geste est capturé via **MediaPipe Hands** qui extrait 21 landmarks 3D de la main (vecteur 63-dim normalisé). Un décompte automatique 3→ 2→ 1 s'affiche, le squelette de la main est dessiné en direct pour confirmer la détection, puis la capture se fait automatiquement :
- 🤙 **Recharger** — doigts pointés à la tempe
- 🔫 **Tirer** — main en pistolet
- 🛡️ **Se protéger** — bras croisés

**Jouer** : duel contre le bot avec affichage **0 → 00 → 007** en overlay sur la webcam, résolution des règles du jeu, mise à jour du Q-learning après chaque manche.

---

## Comment une machine reconnaît une image ?

### Étape 1 — L'image devient des nombres

Une image = une grille de pixels. Chaque pixel = 3 nombres (Rouge, Vert, Bleu) entre 0 et 255.
Une image 224×224 pixels = **150 528 nombres** en entrée du réseau.

### Étape 2 — Les deux modèles du projet

---

#### ViT — Vision Transformer (modèle principal)

**Origine :** Architecture inventée par Google en 2020 en transposant les Transformers du traitement du texte (BERT/GPT) aux images.

**Fonctionnement pas à pas :**

1. L'image 224×224 est découpée en **196 patches de 16×16 pixels** (comme des mots dans une phrase)
2. Chaque patch est aplati et projeté en un vecteur dense
3. Un token spécial `[CLS]` est ajouté en tête de la séquence
4. Les 197 vecteurs passent dans **12 couches Transformer** avec mécanisme d'**attention multi-têtes** : chaque patch peut "regarder" tous les autres patches et pondérer leur importance
5. Le token `[CLS]` final contient la représentation globale de toute l'image
6. Une couche linéaire finale produit **1000 scores** (une par classe ImageNet)
7. Un `softmax` convertit en probabilités → on retourne le top-5

**Pourquoi c'est puissant :** L'attention long-range permet de comprendre le contexte global. Un chien devant une voiture : l'attention connecte les deux zones et comprend la scène dans son ensemble.

**Pré-entraînement :** ViT a été entraîné sur **ImageNet-21k** (14 millions d'images, 21 841 classes) puis fine-tuné sur ImageNet-1k. Notre usage est en **zero-shot** : on réutilise directement sans ré-entraîner.

---

#### MobileNetV2 — CNN léger (comparaison + embeddings)

**Origine :** Réseau de convolutions conçu par Google en 2018 pour tourner sur mobile avec peu de ressources.

**Fonctionnement pas à pas :**

1. L'image 224×224 passe dans des **couches de convolution** successives
2. Chaque filtre de convolution détecte un pattern local (bord, texture, coin, forme)
3. Les premiers layers détectent des **features bas niveau** (contours, couleurs)
4. Les layers profonds détectent des **features haut niveau** (roues, yeux, fourrure, texte)
5. Un **GlobalAveragePooling** en fin de réseau compresse tout en un vecteur de **1280 dimensions** (quand on retire la tête de classification)
6. Ce vecteur = l'**embedding** = représentation compressée de l'image dans un espace mathématique

**Bottleneck inversé :** La particularité de MobileNetV2 — les blocs résiduels compressent puis expansent les features pour être légers en calcul tout en gardant une bonne précision.

**Pré-entraînement :** ImageNet-1k (1,28 million d'images, 1000 classes).

---

## L'apprentissage par transfert (Transfer Learning)

**Concept clé du projet :** On ne repart pas de zéro. Les modèles ont déjà appris à "voir" sur des millions d'images.

```
ImageNet (millions d'images) ──► Entraînement long ──► Poids sauvegardés
                                                               │
Notre image (webcam) ──────────────────────────────────────────► Inférence directe
```

Pour la **reconnaissance faciale**, on utilise le **feature extraction** :
- On coupe la tête de MobileNetV2 (la couche de classification finale)
- On garde uniquement l’extracteur de features → sortie 1280 dimensions
- Ce “projecteur” transforme toute image en un point dans un espace à 1280 dimensions
- Deux images similaires → deux points proches → distance euclidienne faible

---

## Reconnaissance de gestes — MediaPipe Hands

**Principe :** Contrairement à MobileNetV2 qui est un classifieur d’objets général, **MediaPipe Hands** est spécialisé dans la détection et le suivi des mains.

**Fonctionnement :**
1. Un modèle de détection de paumes localise la main dans l’image
2. Un deuxième modèle (régression) prédit les coordonnées 3D (x, y, z) de **21 landmarks** (poignet, jointures, bouts de doigts)
3. Ces 21 points $\times$ 3 coordonnées = **63 dimensions**
4. Le vecteur est **centré** sur le poignet (invariant en translation) et **normalisé** par la distance poignet → base du majeur (invariant en échelle)
5. Deux gestes identiques sous différentes distances caméra ou positions produiront des vecteurs proches

**En temps réel :** le squelette de la main (21 points + connexions) est dessiné sur le flux webcam avec un bandeau vert “✓ Main détectée” — le joueur sait instantanément si sa main est bien captée avant d’appuyer sur capture.

---

## Pipeline complet — Reconnaissance faciale

```
Photo webcam (RGB numpy array)
        │
        ▼
MTCNN (TF) ─────── Détecte les boîtes englobantes des visages
        │
        ▼
Crop du visage + resize 224×224
        │
        ▼
MobileNetV2 sans top (TF) ────── Sortie : vecteur 1280-dim
        │
        ▼
Normalisation L2 ──────── norme = 1 (tous les vecteurs sur la sphère unité)
        │
        ▼
Stockage dans faces_db.pkl  ←──  { "Prénom" : [vecteur1, vecteur2, ...] }

──── Reconnaissance ────

Nouvelle photo → même pipeline → vecteur 1280-dim
        │
        ▼
Distance euclidienne avec chaque vecteur connu
        │
        ▼
Si d_min < 0.9 ──► identifié  |  confiance = (1 - d/0.9) × 100%
Si d_min ≥ 0.9 ──► inconnu
```

**Pourquoi ça marche :** MobileNetV2 a appris que deux photos du même visage sous des angles différents ont des vecteurs proches, même sans avoir été entraîné spécifiquement sur des visages — les features visuelles générales suffisent.

---

## Jeu 007 — Q-Learning (Apprentissage par Renforcement)

### Principe

Ce n'est **pas** du Deep Learning. C'est du **Reinforcement Learning tabulaire** : le bot apprend par essai/erreur en jouant contre lui-même, sans aucune donnée étiquetée.

### L'état du jeu

À chaque tour, l'état est un tuple de 4 entiers :

$$s = (\text{balles\_joueur},\ \text{balles\_IA},\ \text{vies\_joueur},\ \text{vies\_IA})$$

Avec 3 balles max et 3 vies max → $4 \times 4 \times 4 \times 4 = 256$ états possibles.

### La Q-table

Un dictionnaire Python : `{ état → [Q(recharger), Q(tirer), Q(protéger)] }`

Chaque valeur Q représente la **récompense future espérée** si on choisit cette action dans cet état. Plus Q est élevé, meilleure est l'action.

### Mise à jour — Équation de Bellman

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Paramètre | Valeur | Rôle |
|---|---|---|
| $\alpha$ | 0.3 | Learning rate — vitesse d'apprentissage |
| $\gamma$ | 0.85 | Discount — importance du futur vs immédiat |
| $\varepsilon$ | 0.20 | 20% d'exploration aléatoire, 80% d'exploitation |

### Système de récompenses

| Action / Événement | Récompense IA |
|---|---|
| Toucher l'adversaire | **+25** |
| Se faire toucher | **-25** |
| Gagner la partie | **+40** |
| Perdre la partie | **-40** |
| Bloquer un vrai tir (protéger quand l'adversaire tire) | **+12** |
| Se protéger sans raison | **-8** |
| Recharger quand déjà à fond | **-14** |
| Recharger quand en manque | **+3** |
| Tirer (initiative offensive) | **+2** |
| Tour sans résultat (neutre) | **-3** |
| Tirer à vide | **-8** |

### Entraînement asymétrique — 3 modes

Pour éviter que le bot converge vers une stratégie défensive uniforme (problème classique en RL) :

| Mode | Proportion | Description |
|---|---|---|
| **QvQ** | 40% | Les deux joueurs exploitent la Q-table → apprentissage stratégique mutuel |
| **QvRand** | 35% | Le bot affronte un joueur aléatoire → apprend la robustesse |
| **QvAgro** | 25% | Le bot affronte un joueur qui tire dès qu'il a des balles → apprend à protéger |

En ~2000 parties (environ 1 seconde de calcul CPU), le bot explore la majorité des 256 états et construit une stratégie cohérente.

---

## Pipeline webcam temps réel — Sans freeze

**Problème :** l'inférence ViT + MobileNetV2 TF prend 500ms à 2 secondes. La webcam doit tourner à 30 fps → on ne peut pas bloquer `recv()`.

**Solution : thread daemon séparé**

```
Thread principal recv()  ── 30 fps, ne bloque JAMAIS
    │
    ├── Stocke chaque frame dans frame_buffer (deque 30 frames = ~1s)
    │
    └── Tous les 60 frames (~2s) : si aucune analyse en cours
        ET si le jeu 007 n'est pas actif
                │
                ▼
        Thread daemon _analyse_background()
            ├── ViT inference (300-500ms)
            ├── MobileNetV2 TF inference (100-200ms)
            ├── Si Humain → MTCNN + MobileNetV2 face embedding
            └── Met à jour self.result (thread-safe via lock)
                    │
                    ▼
            L'overlay webcam lit self.result sans jamais bloquer
```

**Optimization jeu 007 :** la reconnaissance de gestes utilise **MediaPipe Hands** qui tourne sur CPU en ~1 ms par frame. Un vote majoritaire est effectué sur 7 frames du buffer — rapide et sans appel de modèle lourd.

---

## Types d'apprentissage utilisés

| Fonctionnalité | Type d'apprentissage | Algorithme |
|---|---|---|
| Classification images (principal) | **Supervisé — zero-shot** | Vision Transformer (ViT) |
| Classification images (comparaison) | **Supervisé — zero-shot** | CNN MobileNetV2 |
| Reconnaissance faciale | **Supervisé — few-shot** | Transfer Learning + distance L2 |
| Reconnaissance gestes 007 | **Supervisé — few-shot** | MediaPipe Hands + distance L2 |
| Bot adversaire 007 | **Par renforcement** | Q-learning tabulaire (Bellman) |

- **Zero-shot** = pré-entraîné sur ImageNet, utilisé directement sans aucun réentraînement
- **Few-shot** = 3 à 5 exemples suffisent pour apprendre un nouveau visage ou geste
- **Par renforcement** = pas de données étiquetées, apprentissage uniquement par récompenses et punitions

---

## Contraintes rencontrées

| Problème | Solution apportée |
|---|---|
| Webcam freeze pendant l'inférence IA | Thread daemon `_analyse_background()` — `recv()` ne bloque plus |
| Incompatibilité de dimensions embeddings (512-dim FaceNet vs 1280-dim MobileNetV2) | Vérification de shape + reset des bases `.pkl` |
| TensorFlow non installé | `pip install tensorflow` (v2.20.0) |
| F-strings Python 3.11 avec backslash | Variables intermédiaires — bug Python < 3.12 |
| Apostrophe dans nom de colonne Vega-Lite | Renommage de la colonne — parseur Vega ne supporte pas les apostrophes dans les chemins de champs |
| Mauvaise reconnaissance de gestes (MobileNetV2 général) | Remplacement par **MediaPipe Hands** (landmarks spécialisés main) |
| Jeu 007 trop lent (N appels predict séquentiels) | Vote multi-frames MediaPipe (CPU, ~1 ms/frame) |
| Page qui s'assombrit pendant le jeu 007 | Suppression des `time.sleep()` → `st_autorefresh` pur JavaScript |
| Compétition sur le modèle TF pendant le jeu | Suspension du thread d'analyse quand l'overlay 007 est actif + `LightVideoProcessor` dédié |

## Réussites

- Webcam fluide à 30 fps avec inférence IA en arrière-plan non bloquante
- Double modèle côte à côte (ViT HuggingFace + MobileNetV2 TF) avec comparaison en temps réel
- Reconnaissance faciale avec seulement 3 à 5 photos par personne
- Reconnaissance gestuelle avec vote sur plusieurs frames pour plus de robustesse
- Bot 007 qui apprend une vraie stratégie en quelques secondes d'auto-jeu
- Entraînement asymétrique (3 modes) pour éviter la convergence défensive
- Easter egg Baby Yoda 🟢
