import os
from langchain_community.embeddings import FastEmbedEmbeddings

# DISH_ALIASES maps dish keywords to customer-friendly everyday language.
# When a dish name or description contains a keyword,
# alias words are appended to the embedding text.
# This improves semantic search without any infrastructure changes.
DISH_ALIASES = {
    # Wraps / Rolls / Flatbreads
    "doner":        "wrap roll pita pocket flatbread shawarma sandwich hand-held",
    "shawarma":     "wrap roll pita pocket flatbread doner sandwich hand-held",
    "wrap":         "roll sandwich pita pocket flatbread hand-held",
    "roll":         "wrap kathi roll frankie stuffed bread hand-held street food",
    "kathi":        "roll wrap frankie street food stuffed",
    "frankie":      "roll wrap kathi street food stuffed bread",
    "pita":         "wrap bread pocket flatbread mediterranean",
    "paratha":      "flatbread stuffed bread indian breakfast layered",
    "naan":         "bread flatbread indian baked tandoor",
    "kulcha":       "bread flatbread stuffed baked indian",
    "roti":         "bread chapati flatbread indian wheat",
    "chapati":      "roti flatbread bread indian wheat",
    "puri":         "fried bread indian puffed wheat",
    "bhatura":      "fried bread fluffy indian puffed",

    # Rice Dishes
    "biryani":      "rice dish pulao spiced aromatic one-pot meal",
    "pulao":        "rice dish biryani mild aromatic vegetable",
    "fried rice":   "rice chinese wok tossed schezwan",
    "khichdi":      "rice lentil comfort food mild simple",
    "jeera rice":   "rice cumin plain simple mild",

    # Grilled / Tandoor / Kebabs
    "tikka":        "grilled tandoor starter kebab smoky marinated chicken paneer",
    "kebab":        "grilled skewer starter smoky marinated seekh shami",
    "seekh":        "kebab minced grilled starter smoky",
    "shami":        "kebab patty fried starter minced",
    "tandoori":     "grilled smoky clay oven charred starter main",
    "malai":        "creamy mild white kebab grilled starter",
    "boti":         "grilled meat chunks kebab starter smoky",
    "galouti":      "kebab soft melt minced starter lucknowi",
    "satay":        "skewer grilled starter peanut sauce thai",

    # Curries / Gravies
    "curry":        "gravy sauce indian spiced main course",
    "masala":       "spiced gravy curry sauce indian",
    "butter chicken": "makhani creamy tomato mild chicken curry popular",
    "makhani":      "butter creamy tomato mild curry popular",
    "korma":        "mild creamy curry cashew nut rich",
    "rogan josh":   "lamb mutton red spicy curry kashmiri",
    "kadai":        "wok tossed dry semi-dry curry spiced",
    "handi":        "pot cooked curry slow smoky rich",
    "do pyaza":     "onion heavy curry semi-dry spiced",
    "saag":         "spinach leafy greens curry palak",
    "palak":        "spinach leafy greens saag curry",
    "methi":        "fenugreek leafy curry bitter aromatic",
    "nihari":       "slow cooked stew lamb mutton rich breakfast",
    "haleem":       "slow cooked wheat lentil meat stew",
    "vindaloo":     "spicy hot tangy goan curry pork chicken",
    "chettinad":    "south indian spicy aromatic chicken curry",

    # Lentils / Dals
    "dal":          "lentil soup indian protein comfort food",
    "daal":         "lentil soup indian protein comfort food",
    "dal makhani":  "black lentil creamy rich slow cooked",
    "dal tadka":    "yellow lentil tempered simple comfort",
    "sambhar":      "south indian lentil tamarind vegetable curry soup",
    "rasam":        "south indian thin tamarind pepper soup digestive",

    # Paneer Dishes
    "paneer":       "cottage cheese vegetarian indian protein",
    "shahi paneer": "rich creamy royal vegetarian curry paneer",
    "palak paneer": "spinach cottage cheese vegetarian curry green",
    "paneer tikka": "grilled cottage cheese starter vegetarian",
    "paneer butter masala": "creamy tomato vegetarian curry popular mild",

    # Street Food / Snacks / Starters
    "chaat":        "street food tangy sweet spicy snack popular indian",
    "pani puri":    "golgappa puchka street food snack hollow crispy",
    "golgappa":     "pani puri puchka street food hollow crispy snack",
    "puchka":       "pani puri golgappa street food snack",
    "bhel puri":    "puffed rice tangy street food snack",
    "sev puri":     "crispy flat street food snack tangy",
    "dahi puri":    "yogurt crispy street food snack cool",
    "samosa":       "fried pastry stuffed snack starter street food",
    "kachori":      "fried stuffed pastry snack street food",
    "vada":         "fried lentil snack south indian starter",
    "pakora":       "fritter fried snack starter onion",
    "bhajia":       "pakora fritter fried snack starter",
    "bonda":        "fried ball snack south indian starter",
    "aloo tikki":   "potato patty snack street food fried starter",
    "spring roll":  "fried stuffed roll starter chinese snack crispy",
    "falafel":      "chickpea fried balls veg starter middle eastern",
    "nachos":       "chips salsa starter snack mexican crispy",
    "fries":        "chips french fries potato finger chips snack fried",
    "wedges":       "potato snack thick fries starter",
    "popcorn chicken": "fried chicken bites snack starter crispy",
    "wings":        "chicken starter spicy crispy grilled fried",
    "momos":        "dumplings steamed fried stuffed tibetan snack starter",
    "dimsums":      "dumplings momos steamed fried chinese starter",
    "dumplings":    "momos dimsums steamed fried stuffed",
    "loaded fries": "fries cheese sauce toppings snack indulgent",

    # Burgers / Sandwiches
    "burger":       "sandwich bun patty grilled hand-held",
    "sandwich":     "bread stuffed hand-held snack meal",
    "club sandwich": "layered bread toast chicken snack meal",
    "sub":          "long sandwich bread stuffed cold cuts",
    "slider":       "mini burger small bun patty snack",

    # Pizza / Pasta / Italian
    "pizza":        "flatbread cheesy baked italian pie",
    "pasta":        "noodles italian baked saucy penne spaghetti",
    "risotto":      "rice creamy italian",
    "lasagna":      "layered pasta baked italian cheesy",
    "penne":        "pasta tubes italian",
    "spaghetti":    "pasta long noodles italian",

    # Chinese / Asian
    "noodles":      "chowmein hakka pasta chinese stir fry",
    "chowmein":     "noodles hakka chinese stir fry",
    "hakka":        "noodles chowmein chinese stir fry",
    "manchurian":   "fried balls gravy chinese indo-chinese starter",
    "schezwan":     "spicy chinese sauce noodles fried rice",
    "fried rice":   "chinese rice wok tossed chinese stir fry",
    "soup":         "broth liquid warm starter chinese thai",
    "hot and sour": "soup chinese tangy spicy",
    "sweet corn":   "soup mild cream starter",
    "tom yum":      "thai soup spicy sour aromatic",
    "ramen":        "japanese noodle soup broth",
    "sushi":        "japanese rice roll raw fish seaweed",
    "thai curry":   "coconut milk curry thai aromatic",

    # South Indian
    "dosa":         "south indian crepe rice batter crispy breakfast",
    "masala dosa":  "south indian stuffed crepe potato filling",
    "idli":         "south indian steamed rice cake soft breakfast",
    "vada":         "south indian fried lentil donut crispy",
    "uttapam":      "south indian thick pancake toppings",
    "appam":        "kerala rice pancake lacy coconut",
    "puttu":        "kerala steamed rice cylinder coconut",
    "fish curry":   "kerala goan coastal seafood tamarind",

    # Soups / Salads / Healthy
    "salad":        "fresh healthy greens raw tossed light",
    "caesar salad": "lettuce croutons dressing healthy light",
    "greek salad":  "olives feta cheese vegetable healthy",
    "soup":         "warm liquid comfort broth starter",
    "smoothie":     "blended drink fruit healthy cold",
    "juice":        "fresh fruit drink cold healthy",

    # Beverages / Drinks
    "lassi":        "yogurt drink sweet salty indian beverage cool",
    "chaas":        "buttermilk thin lassi digestive indian",
    "chai":         "tea indian spiced milk hot beverage",
    "coffee":       "hot cold beverage espresso cappuccino latte",
    "milkshake":    "thick drink blended milk sweet cold",
    "mocktail":     "non-alcoholic drink mixer fruit cold",
    "lemonade":     "cold drink lime nimbu pani fresh",

    # Desserts / Sweets
    "brownie":      "dessert chocolate cake sweet baked fudge",
    "cake":         "dessert sweet baked birthday celebration",
    "cheesecake":   "dessert creamy sweet cold baked",
    "ice cream":    "dessert cold sweet scoop frozen",
    "gelato":       "italian ice cream dessert cold sweet",
    "waffle":       "dessert sweet crispy batter breakfast",
    "pancake":      "dessert breakfast sweet fluffy batter",
    "gulab jamun":  "indian dessert sweet fried syrup soft",
    "rasgulla":     "indian dessert sweet soft white syrup bengali",
    "jalebi":       "indian dessert sweet fried spiral crispy",
    "halwa":        "indian dessert sweet cooked semolina carrot",
    "kheer":        "indian dessert rice pudding sweet milk",
    "kulfi":        "indian ice cream frozen dessert sweet",
    "rabri":        "indian dessert thickened milk sweet",
    "payasam":      "south indian dessert kheer sweet milk",
    "peda":         "indian sweet milk fudge mithai",
    "barfi":        "indian sweet fudge milk mithai",
    "ladoo":        "indian sweet round ball mithai festive",
    "tiramisu":     "italian dessert coffee cream layered",
    "mousse":       "dessert airy creamy chocolate sweet",
    "pudding":      "dessert creamy warm cold sweet",

    # Seafood / Meat
    "fish":         "seafood grilled fried fillet coastal",
    "prawn":        "seafood shrimp grilled fried starter",
    "shrimp":       "prawn seafood grilled fried starter",
    "crab":         "seafood coastal fried curry",
    "lobster":      "seafood premium coastal grilled",
    "mutton":       "lamb goat meat curry biryani",
    "lamb":         "mutton goat meat curry biryani",
    "chicken":      "poultry grilled fried curry starter",
    "pork":         "meat grilled fried bacon",
    "beef":         "meat grilled burger steak",
    "steak":        "beef grilled premium meat",

    # Veg / Vegan tags
    "veg":          "vegetarian plant-based no meat",
    "vegan":        "plant-based no dairy no egg vegetarian",
    "jain":         "no onion no garlic vegetarian strict",
}

def dish_to_text(dish: dict) -> str:
    # Build text representation for embedding
    allergens = ', '.join(dish.get('allergens', [])) or 'none'
    ingredients = ', '.join(dish.get('ingredients', [])) or 'not listed'
    veg_text = 'Vegetarian' if dish.get('is_veg') else 'Non-Vegetarian'

    name_lower = dish.get('name', '').lower()
    description_lower = dish.get('description', '').lower()

    # Apply aliases to improve semantic search
    matched_aliases = []
    for keyword, alias_text in DISH_ALIASES.items():
        if keyword in name_lower or keyword in description_lower:
            matched_aliases.append(alias_text)

    alias_line = ""
    if matched_aliases:
        alias_line = f" Also searchable as: {' '.join(matched_aliases)}."

    return (
        f"Dish: {dish.get('name', 'Unknown')}. "
        f"Description: {dish.get('description', 'No description')}. "
        f"Price: {dish.get('price', 'N/A')} rupees. "
        f"Cuisine: {dish.get('cuisine', 'General')}. "
        f"{veg_text}. "
        f"Spice Level: {dish.get('spice_level', 'low')}. "
        f"Allergens: {allergens}. "
        f"Ingredients: {ingredients}.{alias_line}"
    )

def load_embedding_model():
    # BAAI/bge-small-en-v1.5 is chosen for:
    # - better semantic retrieval
    # - ONNX speed via FastEmbed
    # - 384 dimensions matching current schema
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
