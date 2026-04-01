import pandas as pd
from qdrant_client import QdrantClient, models
import os

def run_ingestion(csv_path):
    # 1. Initialize Client (Points to your Docker container)
    # For bulk uploads, gRPC (6334) is significantly faster.
    # prefer_grpc=True: Uses the high-speed binary protocol instead of slow JSON/HTTP.
    client = QdrantClient(url="http://localhost:6333", grpc_port=6334, prefer_grpc=True)


    # 2. Choose and Set Embedding Model (Local CPU-based embedding)
    # This model is small, fast, and great for Indian grocery context
    model_name = "BAAI/bge-small-en-v1.5"
    client.set_model(model_name)

    # 3. Load Dataset
    print(f"📂 Loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Data Cleaning of critical data 
    initial_len = len(df)
    df = df.dropna(subset=['product'])
    df['sale_price'] = df['sale_price'].fillna(df['market_price'])
    df['market_price'] = df['market_price'].fillna(df['sale_price'])

    df = df.dropna(subset=['sale_price'])
    print(f"🗑️ Dropped {initial_len - len(df)} rows due to missing critical data (Name/Price).")

    # SMART FILLERS FOR NON-CRITICAL DATA

    df['brand'] = df['brand'].fillna("Generic")
    df['rating'] = df['rating'].fillna(0.0)
    # Fill empty descriptions to avoid errors
    df['description'] = df['description'].fillna(
        df['product'].astype(str) + " in " + df['category'].astype(str)
    )

    df['sub_category'] = df['sub_category'].fillna(df['category'])
    df['type'] = df['type'].fillna(df['sub_category'])

    # 4. Create the 'Context' for the AI
    # This is what the vector will 'learn'
    df['text_for_vector'] = (
        "Product: " + df['product'].astype(str) + 
        " | Brand: " + df['brand'].astype(str) + 
        " | Category: " + df['category'].astype(str) + " > " + df['type'].astype(str) +
        " | Price: ₹" + df['sale_price'].astype(str) +
        " | Description: " + df['description'].astype(str)
    ).str.strip()


   # 5. Collection Setup (Fresh Start)
    collection_name = "products"
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
   
    print(f"🏗️ Creating collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=client.get_fastembed_vector_params(),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0)
    )

    # 6. High-Performance Batch Upload
    print(f"🚀 Ingesting {len(df)} products into Qdrant...")
    
    # We pass 'metadata' as a list of dicts (all columns except the vector text)
    payloads = df.drop(columns=['text_for_vector']).to_dict('records')
    
    # client.add(
    #     collection_name=collection_name,
    #     documents=df['text_for_vector'].tolist(),
    #     metadata=payloads,
    #     batch_size=64,
    #     parallel=2
    # )
    client.add(
        collection_name=collection_name,
        documents=df['text_for_vector'].tolist(),
        metadata=payloads,
        ids=list(range(len(df))),
        batch_size=128, # Increased batch size
        parallel=4 # Mac M1/M2/M3 can easily handle 4 cores
    )

    
    client.update_collection(
        collection_name=collection_name,
        optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
    )
    print(f"✅ Success! Go to http://localhost:6333/dashboard to see your data.")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(PROJECT_ROOT, "data/BigBasket_Products.csv")
    run_ingestion(csv_path)