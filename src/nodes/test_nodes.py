import os
from dotenv import load_dotenv
load_dotenv()

# NOT USING IT AS OF NOW

# 1. Manually setup the mock data (since we aren't running the full FastAPI server)
from src.config import settings
settings.VALID_CATEGORIES = [
    "Fruits & Vegetables",
    "Cleaning & Household",
    "Kitchen, Garden & Pets",
    "Eggs, Meat & Fish",
    "Beauty & Hygiene",
    "Bakery, Cakes & Dairy",
    "Beverages",
    "Baby Care",
    "Snacks & Branded Foods",
    "Gourmet & World Food",
    "Foodgrains, Oil & Masala"
]

# 2. Import the node
from src.nodes.intent import intent_node

def test_brain():
    # 3. Create a fake 'Clipboard' (State)
    mock_state = {
        "query": "I want something healthy for dinner",
        "loop_count": 0,
        "correction_notes": ""
    }
    
    print(f"--- Testing Query: {mock_state['query']} ---")
    
    # 4. Run the node
    result = intent_node(mock_state)
    
    # 5. Verify results
    print(f"Detected Intent: {result['intent']}")
    print(f"Weight: {result['semantic_weight']}")
    print(f"Weight: {result['must_filters']}")
    print(f"Excluding Categories: {result['exclude_filters']}")
    print(f"Expanded Search Terms: {result['expanded_queries']}")
    print(f"Weight: {result['reasoning']}")

if __name__ == "__main__":
    test_brain()