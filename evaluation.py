import asyncio
from langsmith import aevaluate
from src.graph import build_graph
from src.config import settings
from src.database import fetch_unique_categories
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# 1. Setup Environment
settings.VALID_CATEGORIES = fetch_unique_categories()
graph = build_graph()
judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)

async def target(inputs: dict):
    print(settings.VALID_CATEGORIES, 'validd---->')
    await asyncio.sleep(6)
    user_query = inputs.get("query") or inputs.get("Query")
    state_input = {
        "query": user_query,
        "messages": [
            {
                "role": "user", 
                "parts": [{"text": user_query}]
            }
        ]
    }
    return await graph.ainvoke(state_input)

async def hybrid_retrieval_evaluator(run, example):
    # Standardize inputs from CSV/Run
    intent = (example.outputs.get("expected_intent") or "").lower().strip()
    actual_hits = run.outputs.get("raw_hits", [])
    filters = (example.outputs.get("target_filters") or "").lower()
    
    # --- MODE 1: NAVIGATIONAL (Strict ID Intersection) ---
    if intent == "navigational":
        actual_ids = [str(hit.get('id')) for hit in actual_hits]
        target_ids = [i.strip() for i in str(example.outputs.get("ground_truth_ids", "")).split(",") if i.strip()]
        
        matches = set(actual_ids) & set(target_ids)
        precision = len(matches) / len(actual_ids) if actual_ids else 0
        recall = len(matches) / len(target_ids) if target_ids else 0
        mrr = next((1/(i+1) for i, idx in enumerate(actual_ids) if idx in target_ids), 0)
        
        return {"results": [{"key": "precision", "score": precision}, {"key": "recall", "score": recall}, {"key": "mrr", "score": mrr}]}

    # --- MODE 2: COMPARISON (Brand/Filter Coverage) ---
    elif intent == "comparison":
        # Parse targets: e.g., "brand: [Maggi, Sunfeast]" -> ["maggi", "sunfeast"]
        target_brands = filters.replace("brand:", "").replace("[", "").replace("]", "").split(",")
        target_brands = [b.strip() for b in target_brands if len(b.strip()) > 1]
        
        found_brands = set()
        hits_count = 0
        
        for hit in actual_hits:
            hit_text = f"{hit.get('brand', '')} {hit.get('product', '')}".lower()
            is_relevant = False
            for b in target_brands:
                if b in hit_text:
                    found_brands.add(b)
                    is_relevant = True
            if is_relevant:
                hits_count += 1

        precision = hits_count / len(actual_hits) if actual_hits else 0
        recall = len(found_brands) / len(target_brands) if target_brands else 0
        mrr = next((1/(i+1) for i, hit in enumerate(actual_hits) if any(b in str(hit).lower() for b in target_brands)), 0)
        
        return {"results": [{"key": "precision", "score": precision}, {"key": "recall", "score": recall}, {"key": "mrr", "score": mrr}]}

    # --- MODE 3: DISCOVERY / RECIPE (LLM Judge @ Fixed K=5) ---
    else:
        is_match_list = []
        # Check only the top 5 results for Discovery
        top_hits = actual_hits[:5]
        products_list = "\n".join([f"{i+1}. {h.get('product')}" for i, h in enumerate(top_hits)])
        # ONE call for ALL products
        prompt = f"""Query: {example.inputs.get('query')}
        Products: {products_list}
        For each product, decide if it is a good match. 
        Return exactly {len(top_hits)} values, separated by commas, either 'YES' or 'NO'."""
        
        resp = await judge_llm.ainvoke([HumanMessage(content=prompt)])
        is_match_list = ["YES" in x.upper() for x in resp.content.split(",")]
        hits_count = sum(is_match_list)


        hits_count = sum(is_match_list)
        precision = hits_count / len(top_hits) if top_hits else 0
        recall = hits_count / 5  # Fixed target for discovery
        mrr = next((1/(i+1) for i, is_correct in enumerate(is_match_list) if is_correct), 0)

        return {"results": [{"key": "precision", "score": precision}, {"key": "recall", "score": recall}, {"key": "mrr", "score": mrr}]}
    

def intent_evaluator(run, example):
    actual = (run.outputs.get("intent") or "").lower().strip()
    expected = (example.outputs.get("expected_intent") or "").lower().strip()
    return {"key": "intent_accuracy", "score": 1 if actual == expected else 0}

async def run_eval():
    await aevaluate(
        target,
        data="smartBlinkit-v2", 
        evaluators=[hybrid_retrieval_evaluator, intent_evaluator],
        experiment_prefix="Evaluation-v3",
        max_concurrency=1
    )

if __name__ == "__main__":
    print("🚀 Starting Evaluation on 'SmartBlinkit' dataset...") 
    asyncio.run(run_eval())
    print("✅ Evaluation complete. Check your LangSmith Dashboard!")