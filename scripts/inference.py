from qdrant_client import QdrantClient, models
from fastembed.rerank.cross_encoder import TextCrossEncoder

# 1. Setup Client and Models (Must match Ingestion)
client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)
client.set_model("BAAI/bge-small-en-v1.5")
client.set_sparse_model("Qdrant/bm25")
reranker = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")

COLLECTION = "products"
DENSE_VECTOR_NAME = client.get_vector_field_name()
SPARSE_VECTOR_NAME = client.get_sparse_vector_field_name()

def smart_search(query_text, brand_filter=None, max_price=None):
    print(f"\n--- 🛒 Query: '{query_text}' | Filter: Brand={brand_filter}, Price<={max_price} ---")
    
    # Define Metadata Filters if provided
    must_filters = []
    if brand_filter:
        must_filters.append(models.FieldCondition(key="brand", match=models.MatchValue(value=brand_filter)))
    if max_price:
        must_filters.append(models.FieldCondition(key="sale_price", range=models.Range(lte=max_price)))

    # The Hybrid Search Logic
    # results = client.query_points(
    #     collection_name=COLLECTION,
    #     prefetch=[
    #         # Layer 1: Semantic (The Vibe)
    #         models.Prefetch(query=query_text, using="default", limit=10),
    #         # Layer 2: Keyword (The Specifics)
    #         models.Prefetch(query=query_text, using="sparse", limit=10),
    #     ],
    #     query=models.FusionQuery(fusion=models.Fusion.RRF), # Merges Dense + Sparse
    #     query_filter=models.Filter(must=must_filters) if must_filters else None,
    #     limit=5
    # )


    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            # Layer 1: Semantic (The Vibe)
            models.Prefetch(
                query=models.Document(
                    text=query_text, 
                    model="BAAI/bge-small-en-v1.5" # Explicitly tell it which model to use
                ),
                using=DENSE_VECTOR_NAME, 
                limit=10
            ),
            # Layer 2: Keyword (The Specifics)
            models.Prefetch(
                query=models.Document(
                    text=query_text, 
                    model="Qdrant/bm25" # Explicitly tell it which model to use
                ),
                using=SPARSE_VECTOR_NAME, 
                limit=10
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=models.Filter(must=must_filters) if must_filters else None,
        limit=5
    )

    if not results.points:
        print("❌ No results found.")
        return
    
    documents = [
        f"Product: {p.payload['product']} | Category: {p.payload['category']} | "
        f"Type: {p.payload['type']} | Description: {p.payload['description']}" 
        for p in results.points
    ]

    scores = list(reranker.rerank(query_text, documents))
    reranked_results = sorted(
            zip(scores, results.points), 
            key=lambda x: x[0], 
            reverse=True
        )
    
    # Reranker

    print(f"\n--- 🎯 RERANKED Results for: '{query_text}' ---")
    for score, point in reranked_results[:5]:
        p = point.payload
        print(f"[{score:.4f}] {p['product']} | Brand: {p['brand']} | Price: ₹{p['sale_price']}")
    # for i, res in enumerate(results.points):
    #     p = res.payload
    #     print(f"{i+1}. [{res.score:.4f}] {p['product']} | Brand: {p['brand']} | Price: ₹{p['sale_price']}")

# --- 🚀 RUNNING FEATURE TESTS ---

# Test 1: Discovery (Vibe Search)
smart_search("something healthy for light dinner")

# Test 2: Comparison (Specific Keywords)
# This will test if 'Gold' vs 'Super' works via the Sparse Vector/BM25
smart_search("Dawat Rozana Gold")
smart_search("Dawat Rozana Super")

# Test 3: Exact Metadata Filtering (The "Sri Sri" Fix)
# This tests the .strip() and KEYWORD payload index
smart_search("Garlic Oil", brand_filter="Sri Sri Ayurveda")

# Test 4: Price-Constrained Discovery
smart_search("Organic Rice", max_price=500)