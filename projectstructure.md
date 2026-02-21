# Project Structure

localization-api/
├── api/
│   ├── main.py                 # FastAPI application and endpoints
│   └── translation_service.py  # Core translation logic
│
├── data/
│   └── translation_memory.csv  # Mock historical translations (Apple-style)
│
├── database/
│   └── vector_store.py         # ChromaDB initialization and context retrieval
│
├── frontend/
│   └── demo.html               # Interactive demo interface
│
├── .env.example                # API key template
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

## Next Steps
1. Install dependencies: pip install -r requirements.txt
2. Set up API key in .env file
3. Run vector_store.py to initialize database
4. Start the API server
5. Open demo interface