import asyncpg
import os
from dotenv import load_dotenv

HARDCODED_DB_URL = "postgresql://postgres.ptscvzaofehydyayeetx:kanishk_1604@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
load_dotenv()
EFFECTIVE_DB_URL = HARDCODED_DB_URL

_pool: asyncpg.Pool | None = None


async def connect_db():
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=EFFECTIVE_DB_URL,
        min_size=2,
        max_size=10,
        ssl="require",
        statement_cache_size=0,
    )
    print("✅ Database pool created")


async def disconnect_db():
    global _pool
    if _pool:
        await _pool.close()
        print("🔌 Database pool closed")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialised. Call connect_db() first.")
    return _pool
