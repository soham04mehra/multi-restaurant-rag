from supabase import create_async_client, AsyncClient
import config

# Initialize the async client globally
# We use create_async_client to allow non-blocking network calls
supabase: AsyncClient = create_async_client(config.SUPABASE_URL, config.SUPABASE_KEY)

async def insert_dish(row_data: dict):
    """Inserts a dish using the global async supabase connection."""
    return await supabase.table("menu_items").insert(row_data).execute()

async def delete_all_dishes(restaurant_id: str):
    """Deletes all dishes for a specific restaurant asynchronously."""
    return await supabase.table("menu_items").delete().eq("restaurant_id", restaurant_id).execute()
