from typing import Annotated, List, TypedDict, Union, Optional
from pydantic import BaseModel, Field

class AgentState(TypedDict):
    # Inputs
    query: str
    messages: List[dict]
    
    # Brain Output
    intent: str                      # e.g., 'discovery', 'navigational'
    refined_query: str               # Optimized search string
    expanded_terms: List[str]        # 3-5 extra terms for retrieval
    target_brands: List[str]         # Extracted brand names
    max_price: Optional[float]       # Price limit
    exclude_categories: List[str]    # Categories to filter out
    strategy_steps: List[str]

    # Search Output
    raw_hits: List[dict]
    final_response: str
    loop_count: int
