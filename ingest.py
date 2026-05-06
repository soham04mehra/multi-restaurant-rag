# REMINDER: The .env file contains sensitive keys and must NEVER be committed to version control!
import json
import config
from embeddings import dish_to_text, load_embedding_model
from database import insert_dish

def run_ingestion_pipeline(json_path: str, restaurant_id: str):
    print("\n--- 🟢 STARTING JSON INGESTION PIPELINE ---")
    
    print(f"Loading JSON file from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading JSON file: {e}")
        return
        
    if not isinstance(data, list) or len(data) == 0:
        print("❌ Error: JSON file must contain a non-empty list of dishes.")
        return
        
    required_fields = ["name", "description", "price", "category", "is_veg", "spice_level", "allergens", "ingredients"]
    
    print("Loading BGE-Small embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = load_embedding_model()
    
    # Database connection is now handled automatically by database.py
    
    print("\nProcessing dishes...")
    
    for dish in data:
        # Get the dish name, or use "Unknown"
        dish_name = dish.get("name", "Unknown Dish")
        
        # Build the data using .get() to provide defaults if fields are missing
        row_data = {
            "restaurant_id": restaurant_id,
            "name": dish_name,
            "description": dish.get("description", ""),
            "price": float(dish.get("price", 0)),
            "cuisine": dish.get("cuisine", "General"), # Updated column name
            "is_veg": bool(dish.get("is_veg", True)),
            "spice_level": str(dish.get("spice_level", "low")),
            "allergens": dish.get("allergens", []),
            "ingredients": dish.get("ingredients", []),
            "content": dish_to_text(dish),
            "embedding": embeddings.embed_query(dish_to_text(dish))
        }
        
        try:
            insert_dish(row_data)
            print(f"✅ Successfully inserted: {dish_name}")
        except Exception as e:
            print(f"❌ Failed to insert '{dish_name}': {e}")
        
    print("\n--- 🔴 PIPELINE COMPLETE ---")

if __name__ == "__main__":
    run_ingestion_pipeline("menu_1_updated_trimmed.json", "rest_delhi_01")
