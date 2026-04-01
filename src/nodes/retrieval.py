from qdrant_client import models
from ..state import AgentState
from ..config import settings
from src.database import qdrant_client

def retrieval_node(state: AgentState) -> dict:
    print(qdrant_client, 'client-=-=>')
    DENSE_VECTOR_NAME = qdrant_client.get_vector_field_name()
    SPARSE_VECTOR_NAME = qdrant_client.get_sparse_vector_field_name()
    print(DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, 'SPARSE_VECTOR_NAME====>')
    # 1. Inputs from the "Brain" (Intent Node)
    refined_query = state.get("refined_query", state["query"])
    expanded_terms = state.get("expanded_terms", [])
    target_brands = state.get("target_brands", [])
    max_price = state.get("max_price")
    exclude_categories = state.get("exclude_categories", [])

    # 2. Build Smart Filters
    must_conditions = []
    must_not_conditions = []

    if max_price:
        must_conditions.append(
            models.FieldCondition(key="sale_price", range=models.Range(lte=max_price))
        )
    
    if exclude_categories:
        must_not_conditions.append(
            models.FieldCondition(key="category", match=models.MatchAny(any=exclude_categories))
        )


    # We use 'should' logic here so that brands attract results but don't block them
    should_conditions = []
    # for brand in target_brands:
    #     # MatchText is safer than MatchValue for messy CSV data (handles "Sri Sri" vs "Sri Sri Tattva")
    #     should_conditions.append(
    #         models.FieldCondition(key="brand", match=models.MatchText(text=brand))
    #     )
    #     # Also boost if the brand is mentioned in the product name
    #     should_conditions.append(
    #         models.FieldCondition(key="product", match=models.MatchText(text=brand))
    #     )
    
    qdrant_filter = models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
        should=should_conditions if should_conditions else None
    )

    full_search_text = f"{' '.join(target_brands)} {refined_query} {' '.join(expanded_terms)}"

    # 3. Single-Shot Hybrid Search (The "No-Toy" Method)
    # We use RRF to fuse the results based on your intent's semantic weight.
    print(f"📡 Retrieval Node | Query: '{refined_query}'")
    
    result = qdrant_client.query_points(
        collection_name=settings.COLLECTION_NAME,
        prefetch=[
            # Layer 1: Dense (Conceptual Meaning)
            models.Prefetch(
                query=models.Document(text=full_search_text, model="BAAI/bge-small-en-v1.5"),
                using=DENSE_VECTOR_NAME, 
                limit=40
                # filter=qdrant_filter
            ),
            # Layer 2: Sparse (Exact Keyword Match)
            models.Prefetch(
                query=models.Document(text=full_search_text, model="Qdrant/bm25"),
                using=SPARSE_VECTOR_NAME, 
                limit=40
                # filter=qdrant_filter
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=qdrant_filter,
        limit=15
    )

    # 4. Final Processing & Score Normalization
    hits = []
    for hit in result.points:
        hits.append({
            "product": hit.payload.get("product"),
            "brand": hit.payload.get("brand"),
            "price": hit.payload.get("sale_price"),
            "category": hit.payload.get("category"),
            "description": hit.payload.get("description"),
            "score": hit.score,
            "id": hit.payload.get("index")
        })
    # print(result.points, 'hits=======> retreval node')
    return {
        "raw_hits": hits,
        "loop_count": state.get("loop_count", 0)
    }

if __name__ == "__main__":
    pass
    # import os
    # from dotenv import load_dotenv
    # from qdrant_client import QdrantClient
    
    # load_dotenv()

    # # 1. Direct Client Setup (Atomic)
    # client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)
    # # Ensure models are set so client.query_points works
    # client.set_model("BAAI/bge-small-en-v1.5")
    # client.set_sparse_model("Qdrant/bm25")

    # # 2. Mock the Request object simply
    # class SimpleMock: pass
    # mock_req = SimpleMock()
    # mock_req.app = SimpleMock()
    # mock_req.app.state = SimpleMock()
    # mock_req.app.state.qdrant_client = client

    # 3. Manually Mock the Intent Node Output (The "Brain" results)
    # This lets us test ONLY the retrieval logic
    # test_cases = [
        # {
        #     "query": "which item can i use for cleaning fridge?",
        #     "intent": "discovery",
        #     "refined_query": "refrigerator cleaning liquid spray",
        #     "semantic_weight": 0.8,
        #     "must_filters": [],
        #     "exclude_filters": ['Fruits & Vegetables', 'Bakery, Cakes & Dairy', 'Beverages'],
        # },
        # {
        #     "query": "Dawat Rozan Super vs Dawat Rozan Gold",
        #     "intent": "comparison",
        #     "refined_query": "Dawat Rozan Super Basmati Rice Dawat Rozan Gold Basmati Rice",
        #     "semantic_weight": 0.3,
        #     "must_filters": ['Dawat', 'Rozan Super', 'Rozan Gold'],
        #     "exclude_filters": ['Cleaning & Household', 'Beauty & Hygiene'],
        # },
        # {
        #     "query": "I want something healthy for dinner",
        #     "intent": "discovery",
        #     "refined_query": "healthy dinner ingredients and nutritious meals",
        #     "semantic_weight": 0.85,
        #     "must_filters": [],
        #     "exclude_filters":  ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # },
        # {
        #     "query": "Dawat Rozana Gold Rice 5kg",
        #     "intent": "navigational",
        #     "refined_query": "Dawat Rozana Gold Rice 5kg",
        #     "semantic_weight": 0.2,
        #     "must_filters":  ['Dawat', '5kg'],
        #     "exclude_filters":   ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # }
        # {
        #     "query": "Sri Sri Garlic Oil",
        #     "intent": "navigational",
        #     "refined_query": "Sri Sri Garlic Oil",
        #     "semantic_weight": 0.2,
        #     "must_filters": ['Sri Sri'],
        #     "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
        # },
        # {
        #     "query": "Healthy snacks under 400",
        #     "intent": "discovery",
        #     "refined_query": "healthy snacks",
        #     "semantic_weight": 0.8,
        #     "must_filters": ['Healthy'],
        #     "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # },
        # {
        #     "query": "Palak Paneer ingredients",
        #     "intent": "recipe",
        #     "refined_query": "Palak Paneer ingredients spinach cottage cheese spices",
        #     "semantic_weight": 0.9,
        #     "must_filters": [],
        #     "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # },
        # {
        #     "query": "Maggi vs Sunfeast Yippee",
        #     "intent": "comparison",
        #     "refined_query": "Maggi and Sunfeast Yippee noodles",
        #     "semantic_weight": 0.3,
        #     "must_filters": ['Maggi', 'Sunfeast Yippee'],
        #     "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # },  
        # {
        #     "query": "What are diff rices available",
        #     "intent": "discovery",
        #     "refined_query": "variety of rice grains",
        #     "semantic_weight": 0.8,
        #     "must_filters": [],
        #     "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # },
        # second version 
    # ]
    # test_cases = [
    #     {
    #         "query": "Sri Sri Garlic Oil",
    #         "intent": "navigational",
    #         "refined_query": "garlic oil capsules heart health immunity supplement",
    #         "target_brands": ['Sri Sri'],
    #         "max_price": None,
    #         "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Baby Care', 'Bakery, Cakes & Dairy', 'Beverages', 'Snacks & Branded Foods'],
    #         "expanded_terms": ['garlic pearls', 'allicin supplement', 'cholesterol management', 'herbal oil']
    #     },
    #     {
    #         "query": "Healthy snacks under 400",
    #         "intent": "discovery",
    #         "refined_query": "roasted makhana protein bars trail mix baked chips low calorie nuts seeds",
    #         "target_brands": [],
    #         "max_price": 400.0,
    #         "exclude_categories":  ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Baby Care'],
    #         "expanded_terms": ['roasted snacks', 'low calorie', 'high protein', 'guilt free munching', 'baked not fried']
    #     },
    #     {
    #         "query": "Palak Paneer ingredients",
    #         "intent": "recipe",
    #         "refined_query": "fresh spinach palak cottage cheese paneer cubes ginger garlic paste green chili garam masala",
    #         "target_brands": [],
    #         "max_price": None,
    #         "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Baby Care'],
    #         "expanded_terms": ['indian curry base', 'cooking essentials', 'fresh greens', 'dairy products', 'spices']
    #     },
    #     {
    #         "query": "Maggi vs Sunfeast Yippe",
    #         "intent": "comparison",
    #         "refined_query": "instant masala noodles wheat noodles",
    #         "target_brands": ['Maggi', 'Sunfeast Yippee'],
    #         "max_price": None,
    #         "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
    #         "expanded_terms": ['quick snacks', '2-minute noodles', 'masala tastemaker', 'long slurpy noodles']
    #     },
    #     {
    #         "query": "What are diff rices available",
    #         "intent": "discovery",
    #         "refined_query": "basmati jasmine brown sona masoori parboiled organic rice",
    #         "target_brands": [],
    #         "max_price": None,
    #         "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Beverages', 'Baby Care'],
    #         "expanded_terms": ['long grain', 'aromatic rice', 'staple grains', 'whole grain', 'low gi']
    #     }
    # ]

    # print("\n🚀 STARTING ATOMIC RETRIEVAL TEST\n" + "="*40)

    # for mock_state in test_cases:
    #     print(f"\n🔍 Testing Retrieval for: '{mock_state['query']}'")
        
    #     # Execute the node
    #     result = retrieval_node(mock_state, mock_req)
        
    #     hits = result.get("raw_hits", [])
        
    #     if not hits:
    #         print("❌ No products found. Check filters or collection name.")
    #     else:
    #         print(f"✅ Found {len(hits)} products:")
    #         for i, hit in enumerate(hits[:5], 1):
    #             print(f"  {i}. [{hit['brand']}] {hit['product']} - ₹{hit['price']}")

    # print("\n" + "="*40 + "\n✅ Test Finished")


# answers
# I want something healthy for dinner
# . [As Chefs Cook] Harissa Stir-Fry Sauce - ₹216.0
#   2. [Lion] Dates/Kharjura - Deseeded - ₹66.0
#   3. [NUTRIWISH] Oat Flakes - ₹400.0
#   4. [As Chefs Cook] Szechuan Stir-Fry Sauce - ₹216.0
#   5. [As Chefs Cook] Barbeque Stir-Fry Sauce - ₹216.0
# this is test cases result
# 🔍 Testing Retrieval for: 'Sri Sri Garlic Oil'
# 📡 Retrieval Node | Query: 'Sri Sri Garlic Oil' | Weight: 0.2
# ✅ Found 4 products:
#   1. [Sri Sri Tattva] Cows Ghee/Tuppa - ₹295.0
#   2. [OrientAsian] Sri Lankan Curry - ₹250.0
#   3. [Keya] Powder - Cinnamon/Chakke, Sri Lankan - ₹135.15
#   4. [Sri Sri Tattva] Amla Candy - Delicious, Healthy and Digestive - ₹170.0

# 🔍 Testing Retrieval for: 'Healthy snacks under 400'
# 📡 Retrieval Node | Query: 'healthy snacks' | Weight: 0.8
# ✅ Found 15 products:
#   1. [Snack Amor] Healthy Bites - Mini Health Bars - ₹240.0
#   2. [Mojo Bar] Healthy Snacks - Yoghurt Berry + Anti-Oxidants - ₹90.0
#   3. [Mojo Bar] Healthy Snacks - Yoghurt Berry + Anti-Oxidants - ₹90.0
#   4. [The Healthy Cravings Co.] Roasted Jowar Puffs - Herbs & Sundried Tomatoes - ₹60.0
#   5. [Open Secret] Nutty Chips-Choco Almond Butter,Baked Supergrain Chips,Protein Packed,Not Fried - ₹102.0

# 🔍 Testing Retrieval for: 'Palak Paneer ingredients'
# 📡 Retrieval Node | Query: 'Palak Paneer ingredients spinach cottage cheese spices' | Weight: 0.9
# ✅ Found 15 products:
#   1. [Fazlani Foods] Ready To Eat - Palak Paneer - ₹99.0
#   2. [MTR] Ready To Eat - Palak Paneer - ₹125.0
#   3. [Gits] Ready Meals - Palak Paneer - ₹120.0
#   4. [Gits] Ready Meals - Paneer Tikka Masala - ₹120.0
#   5. [Mom's InstaCook] Frozen Instant Indian Curry Mix - Palak Masala - ₹100.0

# 🔍 Testing Retrieval for: 'Maggi vs Sunfeast Yippee'
# 📡 Retrieval Node | Query: 'Maggi and Sunfeast Yippee noodles' | Weight: 0.3
# ❌ No products found. Check filters or collection name.

# 🔍 Testing Retrieval for: 'What are diff rices available'
# 📡 Retrieval Node | Query: 'variety of rice grains' | Weight: 0.8
# ✅ Found 15 products:
#   1. [bb Royal] Rice/Akki - Jeera/Jeeraga Samba - ₹540.0
#   2. [Nirmal] Matta Rice - Short Grain - ₹458.0
#   3. [bb Popular] Rice/Akki - Jeera/Jeeraga Samba - ₹515.0
#   4. [Aaha Oho] Organic Dosa Mix - Multi Grain - ₹75.0
#   5. [Nirmal] Matta Rice - Long Grain - ₹270.0

# ========================================
# ✅ Test Finished

# this is test case
#  {
#             "query": "Sri Sri Garlic Oil",
#             "intent": "navigational",
#             "refined_query": "Sri Sri Garlic Oil",
#             "semantic_weight": 0.2,
#             "must_filters": ['Sri Sri'],
#             "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#         },
#         {
#             "query": "Healthy snacks under 400",
#             "intent": "discovery",
#             "refined_query": "healthy snacks",
#             "semantic_weight": 0.8,
#             "must_filters": ['Healthy'],
#             "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
#         },
#         {
#             "query": "Palak Paneer ingredients",
#             "intent": "recipe",
#             "refined_query": "Palak Paneer ingredients spinach cottage cheese spices",
#             "semantic_weight": 0.9,
#             "must_filters": [],
#             "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
#         },
#         {
#             "query": "Maggi vs Sunfeast Yippee",
#             "intent": "comparison",
#             "refined_query": "Maggi and Sunfeast Yippee noodles",
#             "semantic_weight": 0.3,
#             "must_filters": ['Maggi', 'Sunfeast Yippee'],
#             "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
#         },  
#         {
#             "query": "What are diff rices available",
#             "intent": "discovery",
#             "refined_query": "variety of rice grains",
#             "semantic_weight": 0.8,
#             "must_filters": [],
#             "exclude_filters": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
#         },

# new test results

# 🔍 Testing Retrieval for: 'Sri Sri Garlic Oil'
# 📡 Retrieval Node | Query: 'Garlic Oil'
# ✅ Found 15 products:
#   1. [Colavita] Garlicolio - Garlic Seasoning Made With Extra Virgin Olive Oil - ₹795.0
#   2. [Alaska] Dietary Supplement - Deep Sea Salmon Fish Oil, Premium - ₹1500.0
#   3. [Valley Spice] Select - Garlic Powder - ₹75.0
#   4. [Graminway] Garlic Peanut Chutney Powder - ₹129.0
#   5. [Catch] Garlic Paste - ₹24.0

# 🔍 Testing Retrieval for: 'Healthy snacks under 400'
# 📡 Retrieval Node | Query: 'roasted nuts seeds low sugar makhana protein bars'
# ✅ Found 15 products:
#   1. [Cocosutra] Cereal - Granola, Chia Soya Chocolate Chunk, Breakfast Cereal With Oats, Nuts, Seeds, Dry Fruits - ₹300.0
#   2. [Mix Box] Roasted Makhana - Indian Masala - ₹60.0
#   3. [FabBox] Cheese Makhana - ₹79.0
#   4. [Organic Origins] Makhana - Mint - ₹135.0
#   5. [Cocosutra] Cereal - Granola, Millet Mocha Crunch, Breakfast Cereal With Oats, Nuts, Seeds & Dry Fruits - ₹300.0

# 🔍 Testing Retrieval for: 'Palak Paneer ingredients'
# 📡 Retrieval Node | Query: 'cottage cheese spinach indian spices garam masala ginger garlic green chili'
# ✅ Found 15 products:
#   1. [MTR] Ready To Eat - Palak Paneer - ₹125.0
#   2. [Mom's InstaCook] Frozen Instant Indian Curry Mix - Palak Masala - ₹100.0
#   3. [Fazlani Foods] Ready To Eat - Palak Paneer - ₹99.0
#   4. [Gits] Ready Meals - Palak Paneer - ₹120.0
#   5. [Gits] Ready Meals - Paneer Tikka Masala - ₹120.0

# 🔍 Testing Retrieval for: 'Maggi vs Sunfeast Yippe'
# 📡 Retrieval Node | Query: 'instant noodles masala noodles'
# ✅ Found 3 products:
#   1. [MAGGI] Maggi Sauce Hot & Sweet 500g + Nutri Licious Pazzta Cheese Macaroni 70g - ₹133.0
#   2. [Sunfeast] Sunfeast Yippee Tricolor Pasta - Masala, 2x65 g - ₹45.0
#   3. [MAGGI] Maggi Sauce Hot & Sweet 500g + Nutri Licious Pazzta Cheese Macaroni 70g - ₹133.0

# 🔍 Testing Retrieval for: 'What are diff rices available'
# 📡 Retrieval Node | Query: 'basmati brown jasmine sona masoori parboiled rice'
# ✅ Found 15 products:
#   1. [bb Royal] Organic - Sona Masoori Raw Rice 5 kg + Brown Sugar 1 kg + Tur/Toor Dal 1 kg - ₹539.0
#   2. [Safe Harvest] Sona Masuri Raw Rice/Akki - 12 Months, Pesticide Free - ₹353.0
#   3. [bb Royal] Organic Sona Masoori White Rice/Akki - ₹619.0
#   4. [Organic Tattva] Organic Sona Masuri - Brown - ₹104.5
#   5. [24 Mantra] Organic Rice - Brown Basmati/Basmati Akki - ₹250.0

# ========================================
# ✅ Test Finished

#  test_cases = [
#         {
#             "query": "Sri Sri Garlic Oil",
#             "intent": "navigational",
#             "refined_query": "Garlic Oil",
#             "target_brands": [],
#             "max_price": None,
#             "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#             "expanded_terms": ['Garlic capsules', 'Garlic supplement', 'Garlic extract oil']
#         },
#         {
#             "query": "Healthy snacks under 400",
#             "intent": "discovery",
#             "refined_query": "roasted nuts seeds low sugar makhana protein bars",
#             "target_brands": [],
#             "max_price": 400.0,
#             "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#             "expanded_terms": ['makhana', 'granola', 'dry fruits', 'roasted chana', 'diet snacks']
#         },
#         {
#             "query": "Palak Paneer ingredients",
#             "intent": "recipe",
#             "refined_query": "cottage cheese spinach indian spices garam masala ginger garlic green chili",
#             "target_brands": [],
#             "max_price": None,
#             "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#             "expanded_terms": ['paneer', 'spinach', 'palak', 'indian spices', 'cooking oil', 'cream']
#         },
#         {
#             "query": "Maggi vs Sunfeast Yippe",
#             "intent": "comparison",
#             "refined_query": "instant noodles masala noodles",
#             "target_brands": ['Maggi', 'Sunfeast Yippee'],
#             "max_price": None,
#             "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#             "expanded_terms": ['noodles', 'ramen', 'masala noodles', 'quick snacks']
#         },
#         {
#             "query": "What are diff rices available",
#             "intent": "discovery",
#             "refined_query": "basmati brown jasmine sona masoori parboiled rice",
#             "target_brands": [],
#             "max_price": None,
#             "exclude_categories": ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care'],
#             "expanded_terms": ['grains', 'long grain', 'short grain', 'organic rice', 'paddy']
#         }
#     ]

# third test cases
