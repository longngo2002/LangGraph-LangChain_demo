# -*- coding: utf-8 -*-
import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# === Hướng dẫn khởi chạy PostgreSQL (chạy 1 lần) ===
# 1. Chạy lệnh Docker sau (nếu chưa có Postgres):
# docker run -d --name postgres-langgraph \
#   -e POSTGRES_PASSWORD=postgres \
#   -e POSTGRES_USER=postgres \
#   -e POSTGRES_DB=langgraph_demo \
#   -p 5433:5432 postgres:latest
#
# 2. Chạy file này: python setup_postgres.py

# === FIX LỖI WINDOWS ===
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


async def setup_db():
    DB_URI = os.environ["DB_URI"]

    print("🔧 Đang kết nối PostgreSQL và tạo bảng checkpoint...")
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
    print("✅ PostgreSQL checkpoint tables đã được tạo thành công!")


if __name__ == "__main__":
    asyncio.run(setup_db())