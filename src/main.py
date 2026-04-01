from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from .database import fetch_unique_categories
from .config import settings
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from src.graph import build_graph
from typing import Optional
from .rate_limiter import limiter
from fastapi import HTTPException

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 App starting... Initializing Singleton Client")
    
    # 1. Initialize Singleton Qdrant Client
    # app.state.qdrant_client = get_qdrant_client()
    
    # 2. Fetch and Cache Categories
    try:
        cats = fetch_unique_categories()
        settings.VALID_CATEGORIES = cats
        print(f"✅ Categories Cached: {len(cats)} found.")
    except Exception as e:
        print(f"❌ Error loading categories: {e}")
        settings.VALID_CATEGORIES = [] # Fallback

    yield
    # --- SHUTDOWN ---
    print("🛑 App shutting down...")

app = FastAPI(lifespan=lifespan)
graph = build_graph()


@app.get("/health")
async def health():
    return {"status": "running", "categories_loaded": settings.VALID_CATEGORIES}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5176", "https://smart-blinkit-frontend.vercel.app"], # Your React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    messages: list
    thread_id: Optional[str] = Field(None, alias="threadId") 
    user_id: Optional[str] = Field("anon_user", alias="userId")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    allowed, wait_time = limiter.check_limit()
    if not allowed:
        raise HTTPException(
            status_code=429, 
            detail={
                "code": 429,
                "status": "error",
                "message": f"Wait {wait_time}s",
                "retry_after": wait_time
            }
    )
    async def event_generator():
        
        last_msg_dict = request.messages[-1]
        user_query = last_msg_dict['parts'][0]['text']
        msg_id = f"blink-msg-{abs(hash(user_query))}"

        text_started = False 
        stream_sent = False
        async for event in graph.astream_events(
            {"query": user_query, "messages": request.messages},
            config=settings.get_agent_config(request.thread_id, request.user_id),
            version="v2"
        ):
            kind = event["event"]
            # print(kind, 'kind', event['name'])
            # print('event==>', event)
            # =========================
            # 2️⃣ INTENT REASONING
            # =========================
            if kind == "on_chain_end" and event["name"] == "analyze_intent":
                intent_data = event.get("data", {})
                intent_output = intent_data.get("output", {})
                strategy_steps = intent_output.get("strategy_steps", [])

                if strategy_steps:
                    for step in strategy_steps:
                        yield f'data: {json.dumps({
                            "type": "data-reasoning",
                            "data": {
                                "step": step
                            }
                        })}\n\n'

            # =========================
            # 3️⃣ PRODUCTS (Tool Output)
            # =========================
            if kind == "on_chain_end" and event["name"] == "execute_search":
                search_data = event.get("data", {})
                search_output = search_data.get("output", {})
                products = search_output.get("raw_hits", [])

                if products:
                 yield f'data: {json.dumps({
                    "type": "data-products",
                    "data": {
                        "products": products
                    }
                })}\n\n'

            # =========================
            # 4️⃣ STREAM LLM TEXT
            # =========================
            # if kind == "on_chat_model_stream" and event["name"] == "generate_response":
            #     chunk = event["data"]["chunk"]
            #     llm_token = chunk.content
            #     # llm_response = chunk.get("final_response")
            #     if llm_token:
            #         if not text_started:
            #             # 1️⃣ Start assistant message
            #             yield f'data: {json.dumps({
            #                 "type": "text-start",
            #                 "id": msg_id
            #             })}\n\n'

            #             text_started = True
                    
            #         yield f'data: {json.dumps({
            #             "type": "text-delta",
            #             "id": msg_id,
            #             "delta": llm_token
            #         })}\n\n'
        
            print(f"Kind: {kind} | Name: {event['name']} | Node: {event.get('metadata', {}).get('langgraph_node')}")
            # 1. HANDLE THE START (Generic for both AI and Hardcoded)
            if kind == "on_chain_start" and event["name"] == "generate_response":
                if not text_started:
                    yield f'data: {json.dumps({"type": "text-start", "id": msg_id})}\n\n'
                    text_started = True

            # 2. HANDLE REAL AI STREAMING (If llm.ainvoke/astream is called)
            elif kind == "on_chat_model_stream":
                stream_sent = True
                node_name = event.get("metadata", {}).get("langgraph_node")
                if node_name == "generate_response":
                    content = event["data"]["chunk"].content
                    token = ""
                    if isinstance(content, list) and len(content) > 0:
                        first_item = content[0]
                        # Check if it's a dict like {'text': '...'} or just a raw string
                        if isinstance(first_item, dict):
                            token = first_item.get("text", "")
                        else:
                            token = str(first_item)
                    elif isinstance(content, str):
                        token = content
                    if token:
                        yield f'data: {json.dumps({"type": "text-delta", "id": msg_id, "delta": token})}\n\n'

            # 3. HANDLE THE HARDCODED FALLBACK (When the node finishes instantly)
            elif not stream_sent and kind == "on_chain_end" and event["name"] == "generate_response":
                # Grab the hardcoded string from the node output to avoid cost 
                hardcoded_res = event["data"]["output"].get("final_response", "")
                
                for word in hardcoded_res.split():
                    yield f'data: {json.dumps({
                        "type": "text-delta",
                        "id": msg_id,
                        "delta": word + " "
                    })}\n\n'
                    await asyncio.sleep(0.05)
                    # yield f'data: {json.dumps({"type": "text-start", "id": msg_id})}\n\n'
                    # yield f'data: {json.dumps({"type": "text-delta", "id": msg_id, "delta": hardcoded_res})}\n\n'
                
        #     if kind == "on_chat_model_stream":
        #         # 2. Check the metadata to see WHICH node is running this model
        #         node_name = event.get("metadata", {}).get("langgraph_node")
                
        #         if node_name == "generate_response":
        #             chunk = event["data"]["chunk"]
        #             content = chunk.content
        #             llm_token = ""
        #             if isinstance(content, list) and len(content) > 0:
        #                 llm_token = content[0].get("text", "")
        #             elif isinstance(content, str):
        #                 llm_token = content
        #             if llm_token:
        #                 if not text_started:
        #                     yield f'data: {json.dumps({
        #                         "type": "text-start",
        #                         "id": msg_id
        #                     })}\n\n'
        #                     text_started = True
                        
        #                 yield f'data: {json.dumps({
        #                     "type": "text-delta",
        #                     "id": msg_id,
        #                     "delta": llm_token
        #                 })}\n\n'
        # # =========================
        # 5️⃣ END MESSAGE
        # =========================
        if text_started:
            yield f'data: {json.dumps({
                "type": "text-end",
                "id": msg_id
            })}\n\n'

        yield 'data: {"type":"finish"}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # "x-vercel-ai-ui-message-stream": "v1",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# Helper for nodes to get the client later
def get_db(request):
    return request.app.state.qdrant_client