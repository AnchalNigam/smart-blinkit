from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes.intent import intent_node
from .nodes.retrieval import retrieval_node
from .nodes.response import response_node

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze_intent", intent_node)
    workflow.add_node("execute_search", retrieval_node)
    workflow.add_node("generate_response", response_node) # NEW NODE

    workflow.add_edge(START, "analyze_intent")
    workflow.add_edge("analyze_intent", "execute_search")
    workflow.add_edge("execute_search", "generate_response")
    workflow.add_edge("generate_response", END)

    return workflow.compile()