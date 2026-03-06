# Documentation technique — VisionIA & Jeu 007

## Table des matières

1. [Vue d'ensemble du projet](#1-vue-densemble-du-projet)
2. [Les modèles utilisés — comment ça marche vraiment](#2-les-modèles-utilisés--comment-ça-marche-vraiment)
   - [ViT — Vision Transformer (classification d'images & gestes)](#21-vit--vision-transformer-classification-dimages--gestes)
   - [FaceNet — Reconnaissance faciale](#22-facenet--reconnaissance-faciale)
3. [Comment on a intégré les modèles dans le projet](#3-comment-on-a-intégré-les-modèles-dans-le-projet)
4. [Système de reconnaissance des gestes (jeu 007)](#4-système-de-reconnaissance-des-gestes-jeu-007)
5. [Le Q-Learning — l'IA qui apprend à jouer](#5-le-q-learning--lia-qui-apprend-à-jouer)
6. [L'entraînement par auto-jeu](#6-lentraînement-par-auto-jeu)
7. [Base de données MongoDB — pourquoi et comment](#7-base-de-données-mongodb--pourquoi-et-comment)
8. [Architecture globale du code](#8-architecture-globale-du-code)
9. [Glossaire des termes clés](#9-glossaire-des-termes-clés)

---

## 1. Vue d'ensemble du projet

VisionIA est une application web interactive construite avec **Streamlit**. Elle regroupe trois fonctionnalités principales :

| Fonctionnalité | Technologie clé | Ce qu'elle fait |
|---|---|---|
| Analyse d'images | ViT (Vision Transformer) | Classifie une image uploadée dans une des 8 catégories (Animal, Véhicule, etc.) |
| Reconnaissance faciale | FaceNet (InceptionResNetV1 + MTCNN) | Identifie qui est sur une photo |
| Jeu 007 | ViT (embeddings) + Q-Learning | Jeu de duels par gestes contre une IA qui apprend |

---

## 2. Les modèles utilisés — comment ça marche vraiment

### 2.1 ViT — Vision Transformer (classification d'images & gestes)

**Modèle utilisé** : `google/vit-base-patch16-224` (téléchargé depuis HuggingFace)

#### Qu'est-ce qu'un Transformer ?

À l'origine, les Transformers ont été inventés pour le texte (ex : GPT). L'idée clé : au lieu de lire les mots un par un, le modèle regarde **tous les mots en même temps** et calcule des "relations d'attention" entre eux — certains mots comptent plus que d'autres selon le contexte.

ViT applique **exactement la même idée aux images**.

#### Comment ViT découpe et lit une image

```
Image 224×224 px
       │
       ▼
Découpage en 196 patchs de 16×16 px
(comme découper une photo en 196 petits carreaux)
       │
       ▼
Chaque patch → vecteur de 768 nombres (projection linéaire)
+ ajout d'un token spécial [CLS] en début de séquence
       │
       ▼
12 couches Transformer (chaque couche = attention multi-tête + réseau dense)
→ les patchs "se parlent" entre eux via l'attention
       │
       ▼
Token [CLS] final : vecteur de 768 nombres
= représentation globale de l'image
       │
       ▼
Couche de classification finale (linéaire)
→ 1000 scores (une par classe ImageNet)
→ softmax → probabilités
```

#### Qu'est-ce que l'attention ?

L'attention est le mécanisme central. Pour chaque patch, le modèle calcule combien il doit "regarder" les autres patchs. Par exemple, pour reconnaître un chien :

- Les patchs du museau regarderont fortement les patchs des oreilles et des yeux
- Les patchs du fond (herbe, ciel) seront ignorés

Mathématiquement : pour chaque patch, on calcule un score d'attention vers tous les autres patchs, on applique un softmax pour obtenir des poids, puis on fait une somme pondérée.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q** (Query) : "Qu'est-ce que je cherche ?"
- **K** (Key) : "Qu'est-ce que j'offre ?"
- **V** (Value) : "Quelle information je transmets ?"
- **d_k** : dimension des vecteurs (pour stabiliser les gradients)

#### Le token [CLS] — pourquoi c'est important pour nous

En sortie du dernier bloc Transformer, le token `[CLS]` contient un résumé de **toute l'image**. C'est ce vecteur de 768 dimensions qu'on appelle **embedding** — une empreinte numérique de l'image.

Dans le projet, on extrait cet embedding pour la reconnaissance de gestes :

```python
outputs = classifier.model(**inputs, output_hidden_states=True)
emb = outputs.hidden_states[-1][:, 0, :].squeeze().numpy()
# hidden_states[-1] = dernière couche
# [:, 0, :] = le token [CLS] (indice 0)
```

#### Pre-training sur ImageNet

Le modèle a été **pré-entraîné** sur ImageNet-21k (21 millions d'images, 21 841 classes), puis **fine-tuné** sur ImageNet-1k (1 000 classes). Cela veut dire qu'il a vu des millions d'images et appris à reconnaître tout type d'objet.

Nous, on l'utilise **sans le ré-entraîner** (transfer learning) : on profite de tout ce qu'il a appris pour extraire des embeddings intelligents.

---

### 2.2 FaceNet — Reconnaissance faciale

**Modèles utilisés** :
- `MTCNN` : détecte et recadre le visage dans une image
- `InceptionResNetV1` (pré-entraîné sur `vggface2`) : transforme un visage en vecteur de 512 nombres

#### Étape 1 — MTCNN (détection de visage)

MTCNN est un réseau de neurones en **3 étapes** (P-Net, R-Net, O-Net) qui :
1. Cherche des candidats "visage-like" à différentes échelles
2. Affine les boîtes englobantes
3. Aligne précisément le visage (yeux, nez, bouche)

En sortie : une image 160×160 px centrée sur le visage, normalisée.

```python
face_tensor = mtcnn_model(pil_image)  # → tensor 3×160×160, None si pas de visage
```

#### Étape 2 — InceptionResNetV1 (embedding facial)

Ce réseau profond (basé sur Inception + ResNet) transforme le visage recadré en un **vecteur de 512 dimensions** — l'empreinte faciale.

Ce vecteur est entraîné pour que :
- Deux photos du **même visage** → vecteurs très proches (distance L2 faible)
- Deux visages **différents** → vecteurs éloignés (distance L2 grande)

```python
with torch.no_grad():
    emb = resnet_model(face_tensor.unsqueeze(0))  # → vecteur 512-dim
```

#### Étape 3 — Identification par plus proche voisin

Quand on veut identifier quelqu'un :
1. On extrait l'embedding du visage inconnu
2. On calcule la distance L2 avec **tous** les embeddings enregistrés
3. Si la distance minimum est < 0.9, on retourne le nom correspondant

```python
dist = float(np.linalg.norm(emb - known_emb))
if best_dist < 0.9:
    confiance = max(0, int((1 - best_dist / 0.9) * 100))
```

**Pourquoi 0.9 ?** C'est un seuil empirique : en dessous, les deux visages sont "probablement la même personne". Au-delà, trop différent → "inconnu".

---

## 3. Comment on a intégré les modèles dans le projet

### Chargement en cache

Les modèles sont lourds (ViT ≈ 330 Mo, FaceNet ≈ 90 Mo). On utilise `@st.cache_resource` pour les charger **une seule fois** au démarrage, et les réutiliser pour toutes les requêtes :

```python
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="google/vit-base-patch16-224")

@st.cache_resource
def load_face_models():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet
```

Sans ce cache, l'application rechargerait le modèle depuis le disque à **chaque clic** — ce serait 10-30 secondes d'attente à chaque interaction.

### Pipeline HuggingFace

`pipeline("image-classification", ...)` est un wrapper pratique qui :
1. Charge le modèle et son tokenizer/processor
2. Gère le prétraitement (resize, normalisation)
3. Retourne les résultats sous forme de liste de dicts `[{"label": "...", "score": 0.95}]`

### Mapping ImageNet → catégories métier

ViT connaît 1 000 classes ImageNet avec des noms techniques ("golden retriever", "aircraft carrier"...). On a créé un dictionnaire `CATEGORY_MAPPER` qui traduit ces labels en 8 catégories lisibles :

```python
CATEGORY_MAPPER = {
    "dog": "Animal", "golden retriever": "Animal",
    "sports car": "Véhicule", "truck": "Véhicule",
    ...
}
```

Logique d'application :
1. ViT retourne le top-5 des prédictions
2. On parcourt ces prédictions par ordre de confiance décroissant
3. On cherche une correspondance partielle dans le dictionnaire
4. Premier match trouvé = catégorie retournée

---

## 4. Système de reconnaissance des gestes (jeu 007)

Le jeu 007 nécessite de reconnaître 3 gestes de la main :
- 🤙 **Recharger** : deux doigts pointés vers la tempe
- 🔫 **Tirer** : main en forme de pistolet
- 🛡️ **Protéger** : bras croisés en bouclier

### Pourquoi utiliser ViT pour les gestes ?

On **ne ré-entraîne pas** ViT pour les gestes. On exploite le fait que ses **embeddings (vecteur CLS 768-dim)** capturent les caractéristiques visuelles importantes de n'importe quelle image — y compris des mains.

Deux images qui montrent le même geste → embeddings proches dans l'espace à 768 dimensions.

### Apprentissage par l'exemple (few-shot / k-NN)

Quand l'utilisateur enregistre un geste (ex : "Tirer"), le système :
1. Prend la photo webcam
2. Extrait l'embedding ViT (768-dim)
3. **Normalise** ce vecteur sur la sphère unité (norme = 1)
4. Stocke ce vecteur dans un fichier pickle `gestures_db.pkl`

```python
emb = emb / (np.linalg.norm(emb) + 1e-8)  # normalisation L2
```

**Pourquoi normaliser ?** Sur la sphère unité, la distance L2 entre deux vecteurs est directement liée à leur similarité cosinus :

$$\text{distance L2} = \sqrt{2 - 2\cos(\theta)}$$

Donc comparer des vecteurs normalisés avec L2 revient à mesurer l'angle entre eux — c'est une mesure de similarité plus robuste que la distance brute.

### Reconnaissance (inférence)

Quand l'utilisateur joue :
1. Photo → embedding ViT normalisé
2. Distance L2 avec **tous** les embeddings enregistrés
3. Geste le plus proche si distance < 0.55

```python
if best_dist < 0.55:
    confiance = max(0, int((1 - best_dist / 0.55) * 100))
    return best_key, f"{confiance}%"
```

**Seuil 0.55** : empirique. Sur embeddings normalisés, correspond à une similarité cosinus ≈ 0.85. En dessous → gestes suffisamment différents pour être fiables.

### Vote majoritaire (optionnel)

Pour plus de robustesse, `reconnaitre_geste_vote()` prend plusieurs frames et fait un vote :

```
9 frames → [tirer, tirer, proteger, tirer, tirer, tirer, tirer, tirer, tirer]
                                                         ↓
                                               "tirer" (8/9 frames)
```

---

## 5. Le Q-Learning — l'IA qui apprend à jouer

### Concept du Q-Learning

Le Q-Learning est un algorithme d'**apprentissage par renforcement** (Reinforcement Learning). L'idée : un agent apprend à maximiser ses récompenses futures en jouant et en observant les conséquences de ses actions.

**Les 4 éléments clés** :

| Élément | Dans notre jeu |
|---|---|
| **État** (state) | (balles joueur, balles IA, vies joueur, vies IA) |
| **Action** | recharger / tirer / protéger |
| **Récompense** (reward) | +25 si IA touche le joueur, -25 si IA est touchée... |
| **Q-Table** | Dictionnaire état → [Q(recharger), Q(tirer), Q(protéger)] |

### La Q-Table

La Q-Table stocke, pour chaque état observé, la **valeur espérée** de chaque action :

```
état (3,2,3,1) → [Q(recharger)=2.1, Q(tirer)=8.7, Q(protéger)=-1.2]
                                             ↑
                                  → l'IA choisit "tirer"
```

Format réel en Python :
```python
# qtable = dict{état_tuple → np.array(3)}
qtable[(3, 2, 3, 1)] = np.array([2.1, 8.7, -1.2])
```

La Q-Table est sauvegardée dans `q_007.pkl` et se remplit progressivement à chaque partie jouée.

### L'équation de mise à jour (Bellman)

Après chaque action, on met à jour la Q-Table avec la formule de Bellman :

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s, a)\right]$$

Dans le code :
```python
cur = qtable[state][action_idx]
qtable[state][action_idx] = cur + Q007_LR * (
    reward + Q007_GAMMA * np.max(qtable[next_state]) - cur
)
```

**Paramètres utilisés** :
- `α = 0.3` (learning rate) : à quel point on fait confiance à la nouvelle information
- `γ = 0.85` (gamma/discount) : à quel point on valorise les récompenses futures vs immédiates
- `ε = 0.20` (epsilon) : 20% du temps, l'IA explore au hasard plutôt qu'exploiter

### Stratégie epsilon-greedy

L'IA doit trouver l'équilibre entre **exploration** (essayer des actions inconnues) et **exploitation** (utiliser ce qu'elle sait).

```python
if random.random() > Q007_EPSILON:  # 80% du temps
    # Exploitation : prendre la meilleure action connue
    q_vals = qtable.get(state, np.zeros(3)).copy()
    if not can_shoot:
        q_vals[1] = -9999  # bloquer "tirer" si pas de balles
    return GESTES_KEYS[int(np.argmax(q_vals))]
else:  # 20% du temps
    # Exploration : action aléatoire
    return random.choice(choices)
```

Sans exploration, l'IA resterait bloquée sur les premières stratégies qu'elle a testées.

### Le système de récompenses

Les récompenses sont soigneusement calibrées pour guider l'apprentissage :

```python
# Combat
if j_touche:   r += 25   # toucher l'adversaire = priorité max
if ia_touche:  r -= 25   # se faire toucher = symétrique

# Actions offensives
if ia_geste == "tirer" and ia_balles > 0:
    r += 2   # encourager l'initiative

# Tours inutiles
if not j_touche and not ia_touche:
    r -= 3   # pénaliser l'inaction prolongée

# Protection intelligente
if j_geste == "tirer" and j_balles > 0 and ia_geste == "proteger":
    r += 12  # bloquer un vrai tir = bien récompensé
elif ia_geste == "proteger":
    r -= 8   # se protéger sans raison = pénalisé

# Gestion des munitions
if ia_geste == "recharger" and ia_balles >= MAX:
    r -= 14  # recharger à plein = très inutile
if ia_geste == "tirer" and ia_balles == 0:
    r -= 8   # tirer à vide = pénalisé

# Fin de partie
if ia_vies <= 0:  r -= 40
if j_vies  <= 0:  r += 40
```

**Équilibrage important** : les pénalités pour actions inutiles (protéger sans raison, recharger à plein) forcent l'IA à prendre des décisions *pertinentes* et pas juste "rester en vie".

---

## 6. L'entraînement par auto-jeu

### Le problème de l'entraînement naïf

Si deux agents Q-learning identiques jouent l'un contre l'autre, ils **convergent vers un équilibre défensif** : les deux rechargent, se protègent, et personne ne tire. Les Q-values se stabilisent sur des stratégies "sûres" mais pas optimales.

### Solution : auto-jeu asymétrique (3 modes)

La fonction `s_entrainer_007()` alterne 3 types d'adversaires :

| Mode | Fréquence | Description |
|---|---|---|
| **QvQ** | 40% | Les deux agents exploitent la Q-table (classique) |
| **QvRand** | 35% | L'agent Q affronte un adversaire **purement aléatoire** |
| **QvAgro** | 25% | L'agent Q affronte un adversaire **agressif** (tire dès qu'il peut) |

```python
r_mod = partie_idx % 20
if   r_mod < 8:   mode = "QvQ"    # 8/20 = 40%
elif r_mod < 15:  mode = "QvRand" # 7/20 = 35%
else:             mode = "QvAgro" # 5/20 = 25%
```

**Pourquoi ça marche** :
- Contre un adversaire aléatoire, l'IA apprend à protéger quand c'est risqué, recharger intelligemment
- Contre l'adversaire agressif, l'IA apprend impérativement à se protéger et gérer ses balles
- En QvQ, les deux agents s'affinent mutuellement

### Pénalité sur les nuls

Un match nul (même nombre de vies à la fin) reçoit une pénalité supplémentaire de -20 :

```python
stats["nuls"] += 1
qt = ia_apprendre(qt, s_ia_fin, GESTES_KEYS.index(ig), -20, s_ia_fin)
```

Cela pousse activement l'IA vers des stratégies décisives plutôt que des parties qui s'éternisent.

### Boucle d'entraînement

```
Pour chaque partie (jusqu'à nb_parties ou duree_s) :
    ├─ Sélectionner le mode (QvQ / QvRand / QvAgro)
    ├─ Pour chaque tour (max 60 tours) :
    │   ├─ Calculer l'état courant s
    │   ├─ Choisir l'action a (epsilon-greedy)
    │   ├─ Appliquer les règles → nouvel état s', récompense r
    │   └─ Mettre à jour Q(s,a) avec Bellman
    └─ Si nul : pénalité -20
Sauvegarder la Q-table dans q_007.pkl
```

Une partie dure en moyenne 10-20 tours. En 500 parties, l'IA explore ~7 500 transitions état→action, couvrant la plupart des situations possibles.

---

## 7. Base de données MongoDB — pourquoi et comment

### Structure

MongoDB est une base **NoSQL orientée documents**. Chaque enregistrement est un document JSON-like (BSON). Dans ce projet, la collection `images` stocke l'historique des analyses :

```json
{
  "_id": ObjectId("..."),
  "Nom du fichier": "photo.jpg",
  "Catégorie": "Animal",
  "Résultat": "golden retriever (ImageNet)",
  "Confiance": "94.2%",
  "Date": "2026-03-06 12:30:00"
}
```

### Pourquoi MongoDB et pas SQLite ?

- **Schéma flexible** : pas besoin de définir les colonnes à l'avance
- **Documents JSON natifs** : naturel pour des résultats d'IA (structure variable)
- **Scalabilité** : peut stocker des millions d'entrées sans configuration

### Connexion mise en cache

```python
@st.cache_resource
def init_connection():
    return pymongo.MongoClient("mongodb://localhost:27017/")
```

Même principe que les modèles : connexion créée une fois, réutilisée partout.

### Opérations CRUD utilisées

```python
# Créer (Create)
collection.insert_one({...})

# Lire (Read) — tri par date décroissante, limite 50
collection.find({}, {"_id": 1, ...}).sort("Date", -1).limit(50)

# Supprimer (Delete)
collection.delete_one({"_id": ObjectId(id)})
collection.delete_many({})  # vider tout
```

---

## 8. Architecture globale du code

```
app.py
├── Imports & configuration page
├── load_model()          → ViT pipeline (cache)
├── load_face_models()    → MTCNN + InceptionResNetV1 (cache)
├── init_connection()     → MongoDB client (cache)
│
├── RECONNAISSANCE FACIALE
│   ├── get_embedding(pil)           → vecteur 512-dim FaceNet
│   ├── enregistrer_visage(pil, nom) → ajoute à faces_db.pkl
│   └── reconnaitre_visage(pil)      → (nom, confiance) par k-NN
│
├── JEU 007 — GESTES
│   ├── get_gesture_embedding(pil)   → vecteur 768-dim ViT (CLS token)
│   ├── enregistrer_geste(pil, key)  → ajoute à gestures_db.pkl
│   ├── reconnaitre_geste(pil)       → (geste_key, confiance) par k-NN
│   └── reconnaitre_geste_vote(frames) → vote majoritaire
│
├── Q-LEARNING 007
│   ├── etat_007(...)               → tuple état discret
│   ├── ia_choisit_geste(...)       → epsilon-greedy sur Q-table
│   ├── ia_apprendre(...)           → mise à jour Bellman
│   ├── calculer_reward_ia(...)     → fonction de récompense
│   ├── s_entrainer_007(...)        → auto-jeu asymétrique
│   └── resoudre_duel(...)          → applique les règles 007
│
├── CATEGORY_MAPPER                  → dict ImageNet → catégories
│
└── INTERFACE STREAMLIT
    ├── Tab 1 — Analyse d'images (upload + webcam)
    ├── Tab 2 — Historique MongoDB
    ├── Tab 3 — Reconnaissance faciale
    └── Tab 4 — Jeu 007
        ├── Apprentissage des gestes
        ├── Partie en cours (phases c0a → c0b → c7 → result)
        └── Entraînement IA / Q-table
```

---

## 9. Glossaire des termes clés

| Terme | Définition |
|---|---|
| **Embedding** | Représentation d'une image (ou d'un visage) sous forme de vecteur de nombres. Deux images similaires → embeddings proches. |
| **Token [CLS]** | Token spécial ajouté en début de séquence dans un Transformer. Après traitement, contient un résumé de toute la séquence. |
| **Attention** | Mécanisme permettant à chaque élément d'une séquence de "regarder" les autres éléments et de pondérer leur importance. |
| **Transfer learning** | Réutiliser un modèle pré-entraîné sur de nombreuses données pour une tâche différente, sans le ré-entraîner. |
| **K-NN (k plus proches voisins)** | Algorithme de classification : pour identifier un élément, on cherche les k éléments les plus proches dans la base et on prend leur label majoritaire. Ici k=1 (plus proche voisin unique). |
| **Distance L2 (euclidienne)** | Distance entre deux vecteurs : $\sqrt{\sum_i (a_i - b_i)^2}$. Plus petite = plus similaire. |
| **Similarité cosinus** | Mesure l'angle entre deux vecteurs. = 1 si identiques, 0 si perpendiculaires, -1 si opposés. Équivalent à L2 sur vecteurs normalisés. |
| **Q-Learning** | Algorithme de RL qui apprend une table Q(état, action) représentant la valeur espérée de faire une action dans un état donné. |
| **Epsilon-greedy** | Stratégie combinant exploitation (meilleure action connue) et exploration (action aléatoire). ε = probabilité d'explorer. |
| **Équation de Bellman** | Équation fondamentale du RL : la valeur d'une action = récompense immédiate + valeur future escomptée. |
| **Reward (récompense)** | Signal numérique indiquant à l'agent si son action était bonne (+) ou mauvaise (-). |
| **ImageNet** | Base de données de 14 millions d'images annotées en 21 841 catégories. Standard de référence pour l'évaluation des modèles de vision. |
| **Patch (ViT)** | Sous-image 16×16 pixels. ViT découpe l'image en 196 patchs (224/16 × 224/16). |
| **MTCNN** | Multi-Task Cascaded Convolutional Networks. Réseau en 3 étapes pour détecter et aligner précisément les visages. |
| **Few-shot learning** | Apprendre à reconnaître une classe à partir de très peu d'exemples (ici 3-5 photos par geste). Possible grâce aux embeddings pré-entraînés. |
| **Normalisation L2** | Diviser un vecteur par sa norme pour qu'il ait une longueur = 1. Permet de comparer les directions plutôt que les amplitudes. |
| **Softmax** | Fonction qui transforme un vecteur de scores en probabilités (somme = 1, tous positifs). |
| **@st.cache_resource** | Décorateur Streamlit qui exécute une fonction une seule fois et met le résultat en mémoire pour toute la session. |
