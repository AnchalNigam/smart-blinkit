import pandas as pd
from qdrant_client import QdrantClient, models
import os
from dotenv import load_dotenv
load_dotenv()

def run_hybrid_ingestion(csv_path):
    # 1. Initialize Client with gRPC for speed
    # client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    print(os.getenv("QDRANT_URL"), 'url--->')
    # 2. Set TWO models: Dense (for vibes) and Sparse (for keywords)
    # Dense: BGE handles the "meaning"
    # Sparse: BM25 handles the "exact text"
    client.set_model("BAAI/bge-small-en-v1.5")
    client.set_sparse_model("Qdrant/bm25") # This is the "Keyword" magic

    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # 3. Aggressive Data Cleaning (The PM Fix)
    # Stripping whitespace from brands/products to fix your "Sri Sri" issue
    initial_len = len(df)
    df = df.dropna(subset=['product'])
    df['product'] = df['product'].astype(str).str.strip()
    df['brand'] = df['brand'].fillna("Generic").astype(str).str.strip()
    df['category'] = df['category'].fillna("General").astype(str).str.strip()
    df['sub_category'] = df['sub_category'].fillna(df['category']).astype(str).str.strip()
    df['type'] = df['type'].fillna(df['sub_category']).astype(str).str.strip()
    df['description'] = df['description'].fillna(
        df['product'].astype(str) + " in " + df['category'].astype(str)
    ).astype(str).str.strip()

    df['sale_price'] = pd.to_numeric(df['sale_price'].fillna(df['market_price']), errors='coerce').fillna(0.0)
    df['market_price'] = pd.to_numeric(df['market_price'].fillna(df['sale_price']), errors='coerce').fillna(0.0)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)

    df = df.dropna(subset=['sale_price'])
    print(f"🗑️ Dropped {initial_len - len(df)} rows due to missing critical data (Name/Price).")

    # 4. Contextual Concatenation
    # We create a rich text field for the vectorizer
    df['text_for_vector'] = (
        "Product: " + df['product'].astype(str) + 
        " | Brand: " + df['brand'].astype(str) + 
        " | Category: " + df['category'].astype(str) + " > " + df['type'].astype(str) +
        " | Description: " + df['description'].astype(str)
    ).str.strip()

    # 5. Hybrid Collection Setup
    collection_name = "products"
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    print(f"🏗️ Creating Collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        # Setup Dense Vectors
        vectors_config=client.get_fastembed_vector_params(),
        # Setup Sparse Vectors (Keyword Search)
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0)
    )

    # --- PAYLOAD INDEXING (The Decision Engine) ---
    print("⚡ Creating Payload & Full-Text Indexes...")
    # 1. Exact Match Index (For Comparisons/Filtering)
    client.create_payload_index(collection_name, "brand", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection_name, "category", models.PayloadSchemaType.KEYWORD)

    # 2. Full-Text Search Index (For keyword-only queries like "Capsule")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="product",
        field_schema=models.TextIndexParams(
            type="text",
            tokenizer=models.TokenizerType.WORD,
            lowercase=True
        )
    )

    # 3. Numeric Index (For "Under ₹500" filtering)
    client.create_payload_index(collection_name, "sale_price", models.PayloadSchemaType.FLOAT)

    # 7. Hybrid Batch Upload
    # print(f"🚀 Ingesting {len(df)} products with Hybrid Vectors...")
    # payloads = df.drop(columns=['text_for_vector']).to_dict('records')
    
    # # client.add now automatically handles BOTH dense and sparse encoding 
    # # because we set both models above.
    # client.add(
    #     collection_name=collection_name,
    #     documents=df['text_for_vector'].tolist(),
    #     metadata=payloads,
    #     ids=list(range(len(df))),
    #     batch_size=128,
    #     parallel=4
    # )

    # client.update_collection(
    #     collection_name=collection_name,
    #     optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    # )
    # print(f"✅ Success! Hybrid Search is live at http://localhost:6333/dashboard")

    print(f"🚀 Starting Cloud Sync (Total: {len(df)} products)...")
    
    batch_size = 128
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i : i + batch_size]
        
        # This is your EXACT original logic, just split into smaller pieces
        client.add(
            collection_name=collection_name,
            documents=batch_df['text_for_vector'].tolist(),
            metadata=batch_df.drop(columns=['text_for_vector']).to_dict('records'),
            ids=list(range(i, i + len(batch_df))),
            batch_size=batch_size,
            parallel=4
        )
        print(f"📦 Progress: {i + len(batch_df)} / {len(df)} synced...")

    # Final step: Tell Qdrant to build the search index now that data is in
    print("⚡ Finalizing Indexing...")
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )
    print(f"✅ Success! Hybrid Search is live on Qdrant Cloud.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(PROJECT_ROOT, "data/BigBasket_Products.csv")
    run_hybrid_ingestion(csv_path)