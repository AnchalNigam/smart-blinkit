from qdrant_client import QdrantClient
from .config import settings

# def get_qdrant_client():
#     # This creates the client instance
#     client = QdrantClient(url=settings.QDRANT_URL)
#     client.set_model("BAAI/bge-small-en-v1.5")
#     client.set_sparse_model("Qdrant/bm25")
#     return client
qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
qdrant_client.set_model("BAAI/bge-small-en-v1.5")
qdrant_client.set_sparse_model("Qdrant/bm25")

def fetch_unique_categories():
    """Optimized category fetcher using Scroll API."""
    categories = set()
    offset = None
    
    while True:
        # Optimization: ONLY fetch the 'category' field, skip vectors
        response, next_offset = qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=500, # Fetch in larger batches
            offset=offset,
            with_payload=["category"], 
            with_vectors=False
        )
        
        for point in response:
            cat = point.payload.get("category")
            if cat:
                categories.add(cat)
        
        offset = next_offset
        if offset is None:
            break
            
    return list(categories)