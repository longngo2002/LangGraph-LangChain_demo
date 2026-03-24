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
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

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

# ====================== FASTAPI LIFESPAN ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, abot
    DB_URI = "postgresql://postgres:postgres@localhost:5433/langgraph_demo"

    cm = AsyncPostgresSaver.from_conn_string(DB_URI)
    checkpointer = await cm.__aenter__()
    await checkpointer.setup()

    model = ChatOpenAI(
        model=MODEL,
        temperature=0.7,
        streaming=True,
        api_key=OPENAI_API_KEY
    )
    tool = TavilySearch(max_results=4)

    abot = create_react_agent(
        model=model,
        tools=[tool],
        checkpointer=checkpointer
    )

    yield
    await cm.__aexit__(None, None, None)


app = FastAPI(title="X02 Streaming + Memory + Tavily", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)


@app.post("/chat_x02/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get("question", "xin chào")
    session_id = data.get("session_id", "demo_test")

    start_time = datetime.now()
    config = {"configurable": {"thread_id": session_id}}

    input_data = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]
    }

    async def event_generator():
        full_answer = ""
        try:
            async for event in abot.astream_events(
                input_data, config=config, version="v2"
            ):
                # Token streaming
                if event["event"] == "on_chat_model_stream":
                    chunk_content = event["data"].get("chunk", {}).content or ""
                    if chunk_content:
                        full_answer += chunk_content
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_content})}\n\n"

                # Tool start (hiển thị trên HTML)
                elif event["event"] == "on_tool_start":
                    tool_name = event.get("name", "Tavily Search")
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name})}\n\n"

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
    uvicorn.run("stream_test:app", host="0.0.0.0", port=8000, reload=True)