# ─────────────────────────────────────────────────────────────────────────────
# config.py — Toutes les constantes partagées de l'application VisionIA
# ─────────────────────────────────────────────────────────────────────────────

# ── Fichiers de données ────────────────────────────────────────────────────
FACES_DB_FILE    = "faces_db.pkl"
GESTURES_DB_FILE = "gestures_db.pkl"
Q007_FILE        = "q_007.pkl"

# ── Paramètres reconnaissance de gestes ───────────────────────────────────
GESTURES_NB_MIN = 3   # photos minimum par geste pour jouer

GESTURES_CONFIG = {
    "recharger": {
        "emoji":   "🤙",
        "label":   "Recharger",
        "desc":    "2 doigts pointés à côté de la tête (tempe)",
        "couleur": "#1a73e8",
    },
    "tirer": {
        "emoji":   "🔫",
        "label":   "Tirer",
        "desc":    "Main en forme de pistolet, doigt pointé",
        "couleur": "#e8341a",
    },
    "proteger": {
        "emoji":   "🛡️",
        "label":   "Se protéger",
        "desc":    "Bras croisés devant toi en bouclier",
        "couleur": "#2e7d32",
    },
}

# ── Paramètres jeu 007 ─────────────────────────────────────────────────────
JEU007_VIES_MAX   = 3
JEU007_BALLES_MAX = 3

# ── Paramètres Q-learning ──────────────────────────────────────────────────
Q007_LR      = 0.3    # learning rate
Q007_GAMMA   = 0.85   # discount
Q007_EPSILON = 0.20   # taux d'exploration
GESTES_KEYS  = ["recharger", "tirer", "proteger"]   # index 0, 1, 2

# ── Mapping labels ImageNet → catégories ──────────────────────────────────
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
    "puppet": "Personnage Fictif", "teddy": "Personnage Fictif",
    "teddy bear": "Personnage Fictif", "ocarina": "Personnage Fictif",
    "doll": "Personnage Fictif", "figurine": "Personnage Fictif",
    "action figure": "Personnage Fictif", "costume": "Personnage Fictif",
    "cloak": "Personnage Fictif", "robe": "Personnage Fictif",
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

    # ── Humains EN DERNIER ──
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
    "dumbbell": "Sport", "barbell": "Sport",
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

    # ── Nature / Paysage ──
    "mountain": "Nature", "volcano": "Nature", "valley": "Nature",
    "ocean": "Nature", "lake": "Nature", "river": "Nature",
    "waterfall": "Nature", "beach": "Nature", "desert": "Nature",
    "cliff": "Nature", "coral reef": "Nature", "geyser": "Nature",
    "cloud": "Nature", "sky": "Nature", "rainbow": "Nature",
    "ice berg": "Nature", "glacier": "Nature", "cave": "Nature",
}

# Labels ImageNet → forcer "Inconnu" (images plates : affiche, livre, pochet…)
BLACKLIST_FLAT_IMAGE = {
    "book jacket", "dust cover", "dust jacket", "dust wrapper",
    "jigsaw puzzle", "envelope", "packet",
    "menu", "web site", "screen", "monitor", "television",
    "poster", "album", "cd",
}

# Config par catégorie : emoji + message de gamification
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

# ── Défis pour le jeu de chasse aux objets ────────────────────────────────
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
        "temps": 30, "points_max": 200, "emoji": "🌸",
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
        "texte": "🧾 Trouve une BROSSE À DENTS !",
        "categorie": "Objet", "keywords": ["toothbrush"],
        "temps": 30, "points_max": 300, "emoji": "🧾",
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
