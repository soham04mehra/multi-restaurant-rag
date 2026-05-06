#🍽️ Restaurant AI Chatbot
An AI-powered restaurant chatbot backend that lets customers explore the menu and get dish recommendations through natural conversation.
Built on a RAG (Retrieval-Augmented Generation) pipeline — the AI never guesses. It retrieves real dishes from the menu database and generates answers grounded in actual data.

How It Works
Customer sends → "suggest me a spicy veg roll under 300 rupees"

Intent Detection — extracts veg/non-veg preference, price range, allergens, and spice level from the query
Query Expansion — enriches the query with food synonyms so "roll" also matches doner, shawarma, wrap, pita
Vector Search — finds semantically relevant dishes from the menu with all filters applied at database level
Answer Generation — LLM generates a natural, friendly recommendation from the retrieved dishes


Features

🔍 Semantic search — finds dishes by meaning, not keyword matching
🥗 Veg / Non-veg filtering — enforced at database level before LLM sees results
💰 Price filtering — supports queries like "under 300 rupees" or "above 200"
🚫 Allergen safety — detects allergen mentions and excludes unsafe dishes automatically
🧠 Conversation memory — remembers context within a session for follow-up questions
🏪 Multi-tenant — one backend serves multiple restaurants with fully isolated menus
🌶️ Spice-aware — re-ranks results by spice level when customer asks for spicy food
🗣️ Hinglish friendly — LLM handles mixed Hindi-English queries naturally


Tech Stack
LayerTechnologyAPIFastAPILLMGemini 2.0 FlashEmbeddingsFastEmbed BAAI/bge-small-en-v1.5Vector DatabaseSupabase pgvectorOrchestrationLangChain

Project Structure
├── api.py          # FastAPI endpoints (/chat, /menu, /health)
├── chatbot.py      # Core RAG pipeline — search, filter, LLM call
├── embeddings.py   # Embedding model + dish alias system
├── ingest.py       # Menu ingestion pipeline
├── database.py     # Supabase client
├── config.py       # Environment config
└── supabase.sql    # Vector search function + table schema

Getting Started
bash# Install dependencies
pip install -r requirements.txt

# Add your keys to .env
GROQ_API_KEY=your_key
GOOGLE_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key

# Ingest your menu
python ingest.py

# Start the server
python api.py

API Usage
bashPOST /chat
{
  "message": "suggest me a spicy veg roll under 300 rupees",
  "restaurant_id": "rest_delhi_01",
  "session_id": "optional-for-first-message"
}
