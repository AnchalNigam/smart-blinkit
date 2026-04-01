from langsmith import aevaluate
from src.main import graph
from src.graph import build_graph
import asyncio
from src.config import settings
from src.database import fetch_unique_categories

graph = build_graph()
settings.VALID_CATEGORIES = fetch_unique_categories()
async def target(inputs: dict):
    # LangSmith sends {"query": "..."} based on your dataset input field
    # We await the full completion to get the final state dictionary
    user_query = inputs.get("Query")
    state_input = {
        "query": user_query,
        "messages": [
            {
                "role": "user", 
                "parts": [{"text": user_query}]
            }
        ]
    }
    print(state_input, 'state_inpit====>')
    result = await graph.ainvoke(state_input)
    return result

def retrieval_evaluator(run, example):
    # What your Qdrant search found
    actual_hits = run.outputs.get("raw_hits", [])
    actual_ids = [str(hit.get('id')) for hit in actual_hits]
    
    # What the Golden Set expected (from your CSV)
    target_str = example.outputs.get("ground_truth_ids", "")
    target_ids = [i.strip() for i in target_str.split(",") if i.strip()]

    matches = set(actual_ids) & set(target_ids)
    
    # Precision: How many of the results we showed were actually correct?
    precision = len(matches) / len(actual_ids) if actual_ids else 0
    # Recall: How many of the correct results did we actually find?
    recall = len(matches) / len(target_ids) if target_ids else 0
    
    # MRR: Did the "best" result appear at the top?
    mrr = 0
    for i, idx in enumerate(actual_ids):
        if idx in target_ids:
            mrr = 1 / (i + 1)
            break

    return {
        "results": [
            {"key": "precision", "score": precision},
            {"key": "recall", "score": recall},
            {"key": "mrr", "score": mrr}
        ]
    }


def intent_evaluator(run, example):
    # Your LLM's classification
    actual = run.outputs.get("intent", "").lower().strip()
    # The classification in your CSV
    expected = example.outputs.get("Expected Intent", "").lower().strip()
    
    return {"key": "intent_accuracy", "score": 1 if actual == expected else 0}
async def run_eval():
    await aevaluate(
        target,
        data="SmartBlinkit",
        evaluators=[retrieval_evaluator, intent_evaluator],
        experiment_prefix="SmartBlinkit-Eval-v1"
    )

if __name__ == "__main__":
    print("🚀 Starting Evaluation on 'SmartBlinkit' dataset...") 
    asyncio.run(run_eval())
    print("✅ Evaluation complete. Check your LangSmith Dashboard!")