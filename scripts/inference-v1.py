from qdrant_client import QdrantClient, models

# Initialize your client (same as your upsert script)
client = QdrantClient("http://localhost:6333") 
collection_name = "products"
model_name = "BAAI/bge-small-en-v1.5"
client.set_model(model_name)
def run_test(query_text, filter_brand=None):
    print(f"\n🚀 Testing Query: '{query_text}'" + (f" [Filter: {filter_brand}]" if filter_brand else ""))
    
    # Define Filter if brand is provided
    query_filter = None
    if filter_brand:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="brand", match=models.MatchValue(value=filter_brand))]
        )

    # Run the Search
    results = client.query(
        collection_name=collection_name,
        query_text=query_text,
        query_filter=query_filter,
        limit=3
    )

    if not results:
        print("❌ No results found.")
        return

    for i, res in enumerate(results):
        # Accessing metadata from the result
        meta = res.metadata
        print(f"{i+1}. [{res.score:.4f}] {meta.get('product')} | Brand: {meta.get('brand')} | Price: {meta.get('sale_price')}")

# --- TEST SUITE ---

# Test 1: Pure Semantic (Concept over Keywords)
run_test("Something for immunity and heart")

# Test 2: Specific Jargon (Checking if embedding picked up specific details)
run_test("Vegetarian capsule garlic")

# Test 3: Metadata + Semantic (The Hybrid approach you have now)
run_test("Garlic Oil", filter_brand="Sri Sri Ayurveda")

# Test 4: The "Failure" Check (Searching for a partial word in a long title)
# If this returns 0 or irrelevant items, we NEED the Text Index/Keyword search.
run_test("Organic Chips")