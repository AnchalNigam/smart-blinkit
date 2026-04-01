# src/nodes/response.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from ..state import AgentState

async def response_node(state: AgentState) -> dict:
    """The 'Voice' node: Summarizes search results into a friendly chat message."""
    
    # 1. Grab data from the State
    query = state.get("query", "")
    hits = state.get("raw_hits", [])
    strategy = state.get("strategy_steps", [])
    
    # 2. Format a tiny bit of context (Only top 3 names to save tokens)
   
    # 2. Format a tiny bit of context for the LLM
    # We only send the top 3-5 products so we don't waste tokens
    top_products = ", ".join([h.get('product', '') for h in hits[:3]])

    system_prompt = f"""
    You are a professional, helpful grocery assistant. 
    STRATEGY USED: {strategy}
    TOP MATCHES: {top_products}

    INSTRUCTIONS:
    - Acknowledge the search results in a calm, helpful tone.
    - CRITICAL: Use the BRAND NAMES exactly as they appear in the TOP MATCHES. Do not add suffixes, corrections
    - Mention 1 specific detail from the strategy (e.g., "focusing on healthy options under 400").
    - Keep it strictly to 2 sentences. 
    - Do NOT list products (the UI handles that).
    """

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.5, streaming=True)
    final_text = "I've curated a few options based on your request; you can find them in the product list."
    try:
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Explain results for: {query}")
        ])
        final_text = response.content[0]['text'] if isinstance(response.content, list) else response.content
    
    except Exception as e:
        print(e)
        

    return {"final_response": final_text}

    # return {
    #     "final_response": "Here is listed products check plz"
    # }