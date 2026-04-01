import os
from dotenv import load_dotenv
from uuid import uuid4

# Load once when this module is imported
load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333"),
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = "products"
    
    # Global Cache for Categories
    VALID_CATEGORIES = []

    def get_agent_config(self, thread_id: str = None, user_id: str = None) -> dict:
        """Generates a standardized LangChain/LangGraph config object."""
        return {
            "configurable": {
                "thread_id": thread_id or str(uuid4())
            },
            "metadata": {
                "user_id": user_id or "anonymous",
                "env": os.getenv("APP_ENV", "development"),
                "app_version": "v1.0.2" # Centralized versioning
            },
            "tags": ["beta-v1", "shopping-agent"]
    }

settings = Settings()