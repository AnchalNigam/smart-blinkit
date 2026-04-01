import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

def test_cloud_connection():
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        # Simple ping
        collections = client.get_collections()
        print("✅ Successfully connected to Qdrant Cloud!")
        print(f"Current collections: {collections}")
        print(f"Current url: {os.getenv("QDRANT_URL")}")
    except Exception as e:


        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_cloud_connection()