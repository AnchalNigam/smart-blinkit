from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState
from ..config import settings

# 1. Define the 'Form' Gemini must fill out
# class SearchPlan(BaseModel):
#     intent: str = Field(
#         description="discovery, navigational, comparison, recipe, review, or suggestion"
#     )
#     refined_query: str = Field(
#         description="A single, optimized search string that combines user intent with necessary context. E.g., 'liquid cleaner for refrigerator' instead of 'what to use for fridge'."
#     )
#     expanded_terms: List[str] = Field(
#         description="3-5 search terms to improve retrieval. E.g. 'Oats' -> ['instant oats', 'healthy breakfast']"
#     )
#     target_brands: List[str] = Field(
#         description="Extract any brand names mentioned (e.g., ['Maggi', 'Dawat']). Use empty list if none."
#     )
#     max_price: Optional[float] = Field(
#         description="Extract numeric price limit if mentioned (e.g., 'under 500' -> 500.0)."
#     )
#     exclude_categories: List[str] = Field(
#         description="Categories to ignore (e.g., if user wants food, exclude 'Cleaning & Household')."
#     )
#     strategy_steps: List[str] = Field(
#         description="3-5 short bullet points explaining the search strategy in simple user-facing language."
#     )
class SearchPlan(BaseModel):
    intent: Literal["discovery", "navigational", "comparison", "recipe", "review", "suggestion"] = Field(
        description="Categorize the user's primary goal."
    )
    refined_query: str = Field(
        description="A semantic search string. Transform 'what to use for fridge' -> 'refrigerator deep cleaner spray'.Do not enrich or complete brand names"
    )
    expanded_terms: List[str] = Field(
        description="3-5 broad keywords to increase recall. STRICTLY PROHIBITED: Do not include brand names or company suffixes here. Focus only on product category and use-casesE.g. 'Oats' -> ['instant oats', 'healthy breakfast', 'porridge']."
    )
    target_brands: List[str] = Field(
        description="List only specific brand names mentioned. Empty list if none. Extract the literal brand keywords as written by the user."
    )
    max_price: Optional[float] = Field(
        description="The numeric upper price limit (e.g., 500.0). Null if not specified."
    )
    exclude_categories: List[str] = Field(
        description="Only exclude categories if they are completely irrelevant to the user's intent. When in doubt, leave the exclusion list empty to avoid blocking valid products."
    )
    strategy_steps: List[str] = Field(
        description="3-5 short, simple bullet points for the user explaining how we are finding their items."
    )
# 2. Initialize LLM (Ensure GOOGLE_API_KEY is in .env) gemini-flash-latest
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
structured_llm = llm.with_structured_output(SearchPlan)

async def intent_node(state: AgentState) -> dict:
    """The 'Brain' node: Analyzes query and plans the search strategy."""
    
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, 'content') else last_msg.get('parts', [{}])[0].get('text', '')
    # query = state['query']
    print(query, 'query intent node====>')
    correction = state.get("correction_notes", "")
    categories_list = settings.VALID_CATEGORIES

    # If the Evaluator sent us back for a retry, include the feedback
    system_prompt = f"""You are a Search Architect for a Grocery App. 
    AVAILABLE CATEGORIES: {categories_list}. 
    Follow the provided schema exactly. Be a minimalist with exclusions."""

    user_content = query
    if correction:
        user_content += f"\n\nCRITICAL: Previous attempt failed. Evaluator says: {correction}.Adjust refined_query and filters accordingly."

    # Call Gemini
    # print(f"📡 Sending request to Gemini for: {query[:20]}...")
    plan = await structured_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content)
    ])
    # print("✅ Gemini responded!")
    # print(plan, 'plannn=== intent node')

    return {
        "intent": plan.intent,
        "refined_query": plan.refined_query,
        "exclude_categories": plan.exclude_categories,
        "expanded_terms": plan.expanded_terms,
        "target_brands": plan.target_brands,
        "max_price": plan.max_price,
        "strategy_steps": plan.strategy_steps,
        "loop_count": state.get("loop_count", 0) + 1
    }
    # return {
    #     "intent": "discovery",
    #     "refined_query": "dairy milk plant-based milk almond soy oat cow buffalo",
    #     "exclude_categories": ['Kitchen, Garden & Pets', 'Cleaning & Household', 'Baby Care', 'Fruits & Vegetables', 'Foodgrains, Oil & Masala', 'Beauty & Hygiene', 'Eggs, Meat & Fish', 'Snacks & Branded Foods'],
    #     "expanded_terms": ['fresh milk', 'milk products', 'dairy alternatives', 'vegan milk', 'milk carton'],
    #     "target_brands": [],
    #     "max_price": None,
    #     "strategy_steps": [
    #         "Detected broad milk search",
    #         "Expanded to include dairy and plant-based options",
    #         "Filtered non-consumable categories",
    #         "Ranked results by relevance"
    #     ],
    #     "loop_count": 1
    # }

if __name__ == "__main__":
    pass
    # import os
    # from dotenv import load_dotenv
    # load_dotenv()

    # 1. Manually setup the mock data (since we aren't running the full FastAPI server)
    # settings.VALID_CATEGORIES = [
    #     "Fruits & Vegetables",
    #     "Cleaning & Household",
    #     "Kitchen, Garden & Pets",
    #     "Eggs, Meat & Fish",
    #     "Beauty & Hygiene",
    #     "Bakery, Cakes & Dairy",
    #     "Beverages",
    #     "Baby Care",
    #     "Snacks & Branded Foods",
    #     "Gourmet & World Food",
    #     "Foodgrains, Oil & Masala"
    # ]

    # test_queries = [
    #     # "Dawat Rozan Super vs Dawat Rozan Gold",
    #     # "which item can i use for cleaning fridge?",
    #     # "I want something healthy for dinner",
    #     # "Dawat Rozana Gold Rice 5kg",
    #     "Sri Sri Garlic Oil",
    #     "Healthy snacks under 400",
    #     "Palak Paneer ingredients",
    #     "Maggi vs Sunfeast Yippee",
    #     "What are diff rices available",
    # ]
    # print("\n🚀 STARTING ATOMIC NODE TEST\n" + "="*30)

    # for q in test_queries:
    #     mock_state = {
    #         "query": q,
    #         "messages": [],
    #         "loop_count": 0,
    #         "correction_notes": ""
    #     }
        
    #     print(f"\n🔍 Testing Query: '{q}'")
    #     result = intent_node(mock_state)

    #     # Verify results
    #     print(f"Detected Intent: {result['intent']}")
    #     print(f"target_brands: {result['target_brands']}")
    #     print(f"max_price: {result['max_price']}")
    #     print(f"refined_query: {result['refined_query']}")
    #     print(f"Excluding Categories: {result['exclude_categories']}")
    #     print(f"Expanded Terms: {result['expanded_terms']}")
    #     print(f"Reasonistrategy_stepsng: {result['strategy_steps']}")

        # Results
        
        # 🔍 Testing Query: 'Dawat Rozan Super vs Dawat Rozan Gold'
        # Detected Intent: comparison
        # Weight: 0.3
        # Must_filters: ['Dawat', 'Rozan Super', 'Rozan Gold']
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['Dawat Rozan Super Basmati Rice', 'Dawat Rozan Gold Basmati Rice', 'Dawat Rozan rice comparison', 'premium basmati rice Dawat']
        # Reasoning: The user is comparing two specific product variants from the brand Dawat. A low semantic weight is used to prioritize exact keyword matches for these specific rice types, and non-food categories are excluded as the items are edible grains.

        # 🔍 Testing Query: 'which item can i use for cleaning fridge?'
        # Detected Intent: discovery
        # Weight: 0.8
        # Must_filters: []
        # Excluding Categories: ['Fruits & Vegetables', 'Eggs, Meat & Fish', 'Beauty & Hygiene', 'Bakery, Cakes & Dairy', 'Beverages', 'Baby Care', 'Snacks & Branded Foods', 'Gourmet & World Food', 'Foodgrains, Oil & Masala']
        # Expanded Search Terms: ['refrigerator cleaner spray', 'fridge disinfectant', 'multipurpose surface cleaner', 'cleaning sponges and wipes']
        # Reasoning: The user is asking for product recommendations for a specific household task. Discovery intent is used to find suitable cleaning agents, and all food-related categories are excluded to focus on household supplies.

        # Testing Query: 'I want something healthy for dinner'
        # Detected Intent: discovery
        # Weight: 0.9
        # Must_filters: ['Healthy']
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['healthy dinner ideas', 'low calorie meals', 'nutritious food', 'organic vegetables and grains']
        # Reasoning: The user query is broad and focuses on a health-conscious meal, indicating a discovery intent. A high semantic weight is applied to find items conceptually related to 'healthy' across all edible categories while filtering out non-food items.

        # 🔍 Testing Query: 'Dawat Rozana Gold Rice 5kg'
        # Detected Intent: navigational
        # Weight: 0.2
        # Must_filters: ['Dawat', 'Rozana Gold', '5kg']
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['Dawat Basmati Rice', 'Rozana Gold 5kg', 'Long grain rice Dawat', 'Dawat Rozana Gold Basmati']
        # Reasoning: The query specifies a precise brand, product line, and weight, indicating a navigational intent where the user knows exactly what they want.

        # 🔍 Testing Query: 'Sri Sri Garlic Oil'
        # Detected Intent: navigational
        # Weight: 0.2
        # Must_filters: ['Sri Sri']
        # refined_query: Sri Sri Garlic Oil
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['Sri Sri Tattva Garlic Oil', 'Garlic oil capsules', 'Herbal garlic oil', 'Garlic supplement']
        # Reasoning: The user is searching for a specific brand (Sri Sri) and product (Garlic Oil), which classifies as a navigational intent. A low semantic weight is used to prioritize exact keyword matches for the brand and product name.

        # 🔍 Testing Query: 'Healthy snacks under 400'
        # Detected Intent: discovery
        # Weight: 0.8
        # Must_filters: ['Healthy']
        # refined_query: healthy snacks
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['low calorie snacks', 'roasted makhana', 'protein bars', 'nut mixes']
        # Reasoning: The user is looking for healthy food options, which indicates a discovery intent. A high semantic weight is used to capture the meaning of 'healthy' across various snack types, while non-edible categories are excluded to refine the results.

        # 🔍 Testing Query: 'Palak Paneer ingredients'
        # Detected Intent: recipe
        # Weight: 0.9
        # Must_filters: []
        # refined_query: Palak Paneer ingredients spinach cottage cheese spices
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['fresh spinach palak', 'paneer cubes cottage cheese', 'garam masala and spices', 'ginger garlic paste']
        # Reasoning: The user is looking for ingredients for a specific recipe. A high semantic weight is used to capture the various components of the dish, and non-food categories are excluded to refine the search results.

        # 🔍 Testing Query: 'Maggi vs Sunfeast Yippee'
        # Detected Intent: comparison
        # Weight: 0.3
        # Must_filters: ['Maggi', 'Sunfeast Yippee']
        # refined_query: Maggi and Sunfeast Yippee noodles
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['instant noodles', 'masala noodles', 'Maggi 2-minute noodles', 'Yippee magic masala']
        # Reasoning: The user is comparing two specific brands of instant noodles. A low semantic weight ensures brand precision, and must_filters capture the specific brands requested while excluding non-food categories.

        # 🔍 Testing Query: 'What are diff rices available'
        # Detected Intent: discovery
        # Weight: 0.8
        # Must_filters: []
        # refined_query: variety of rice grains
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Search Terms: ['Basmati rice', 'Sona Masoori rice', 'Brown rice', 'Organic rice']
        # Reasoning: The user is exploring different types of rice available in the store, indicating a discovery intent. A high semantic weight is used to capture various grain types and brands.


        # second phase
        # 🔍 Testing Query: 'Sri Sri Garlic Oil'
        # Detected Intent: navigational
        # target_brands: ['Sri Sri']
        # max_price: None
        # refined_query: Garlic Oil
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Baby Care']
        # Expanded Terms: ['Garlic capsules', 'Garlic supplement', 'Garlic extract oil']
        # Reasoning: The user is searching for a specific product from a specific brand, indicating a navigational intent. The brand name is extracted to target_brands to ensure precise filtering.

        # 🔍 Testing Query: 'Healthy snacks under 400'
        # Detected Intent: discovery
        # target_brands: []
        # max_price: 400.0
        # refined_query: roasted nuts seeds low sugar makhana protein bars
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Terms: ['makhana', 'granola', 'dry fruits', 'roasted chana', 'diet snacks']
        # Reasoning: The user is exploring nutritious food options with a price constraint, so I expanded the query to include specific healthy snack types and excluded non-food categories.

        # 🔍 Testing Query: 'Palak Paneer ingredients'
        # Detected Intent: recipe
        # target_brands: []
        # max_price: None
        # refined_query: cottage cheese spinach indian spices garam masala ginger garlic green chili
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Terms: ['paneer', 'spinach', 'palak', 'indian spices', 'cooking oil', 'cream']
        # Reasoning: The user is looking for ingredients for a specific recipe (Palak Paneer). I expanded the query to include the primary ingredients and excluded non-food categories to ensure relevant grocery results.

        # 🔍 Testing Query: 'Maggi vs Sunfeast Yippee'
        # Detected Intent: comparison
        # target_brands: ['Maggi', 'Sunfeast Yippee']
        # max_price: None
        # refined_query: instant noodles masala noodles
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Terms: ['noodles', 'ramen', 'masala noodles', 'quick snacks']
        # Reasoning: The user is comparing two specific brands of instant noodles. I extracted the brands for filtering and refined the query to the product category while excluding non-food categories.

        # 🔍 Testing Query: 'What are diff rices available'
        # Detected Intent: discovery
        # target_brands: []
        # max_price: None
        # refined_query: basmati brown jasmine sona masoori parboiled rice
        # Excluding Categories: ['Cleaning & Household', 'Kitchen, Garden & Pets', 'Beauty & Hygiene', 'Baby Care']
        # Expanded Terms: ['grains', 'long grain', 'short grain', 'organic rice', 'paddy']
        # Reasoning: The user is exploring the rice category without specific brand or type preferences. I expanded the query to include common rice varieties and excluded non-food categories to ensure relevant results.

        # third phase
