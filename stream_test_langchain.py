# -*- coding: utf-8 -*-
import os
import json
import asyncio
import sys
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

# ====================== WINDOWS FIX ======================
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ====================== CONFIG ======================
MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

SYSTEM_PROMPT = """
Bạn là trợ lý AI thân thiện, thông minh và luôn trả lời bằng tiếng Việt.
Bạn có công cụ Tavily Search để lấy thông tin thời gian thực từ web.
Hãy trả lời chi tiết, vui vẻ, hữu ích. Nếu cần thông tin mới nhất, hãy dùng tool trước.
"""

DB_URI = os.environ["DB_URI"]


# Hàm khởi tạo lịch sử từ DB, dựa trên session_id
def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        connection_string=DB_URI,
        session_id=session_id,
        table_name="langchain_chat_history"  # Bảng cho LangChain Executor
    )


agent_with_history = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_with_history

    # --- 1. Model & Tools ---
    model = ChatOpenAI(
        model=MODEL,
        temperature=0.7,
        streaming=True
    )
    # create_tool_calling_agent usually expects tools, we use TavilySearchResults
    tools = [TavilySearchResults(max_results=4)]

    # --- 2. Prompt chứa biến chat_history và agent_scratchpad ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),  # memory placeholder
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # Cần thiết cho Tools
    ])

    # --- 3. Agent & Executor ---
    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    # --- 4. Bao bọc Executor với RunnableWithMessageHistory ---
    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        history_factory_config=[
            {"id": "session_id", "annotation": str, "name": "Session ID"}
        ]
    )
    try:
        get_session_history("init_db_test")
    except Exception as e:
        print("Exception:", e)
    yield


app = FastAPI(title="LangChain Streaming + Memory + Tavily", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


@app.post("/chat_x02/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get("question", "xin chào")
    # Sử dụng session_id giống stream_test
    session_id = data.get("session_id", "demo_test")

    start_time = datetime.now()
    # Configuration dict mapping parameter session_id to RunnableWithMessageHistory
    config = {"configurable": {"session_id": session_id}}

    async def event_generator():
        full_answer = ""
        try:
            # Langchain v0.2 astream_events sử dụng cho Async Streaming
            async for event in agent_with_history.astream_events(
                    {"input": question},
                    config=config,
                    version="v2"
            ):
                kind = event["event"]

                # Streaming token content
                if kind == "on_chat_model_stream":
                    chunk_content = event["data"].get("chunk", {}).content or ""
                    if chunk_content:
                        full_answer += chunk_content
                        # yield event giống logic của file HTML
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content})}\n\n"

                # Báo hiệu Tool bắt đầu
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "tavily_search_results_json")
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': 'Tavily Search'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        finally:
            duration = (datetime.now() - start_time).total_seconds()
            yield f"data: {json.dumps({'type': 'done', 'full_answer': full_answer, 'duration': f'{duration:.2f}s'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("stream_test_langchain:app", host="0.0.0.0", port=8000, reload=True)
