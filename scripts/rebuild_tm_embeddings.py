import argparse
import asyncio
import sys
from pathlib import Path

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from db import EFFECTIVE_DB_URL
from nodes.rag_tm import EMBEDDING_VECTOR_DIM, embed


async def ensure_embedding_new_column(conn: asyncpg.Connection) -> None:
    await conn.execute(
        f"""
        ALTER TABLE translation_memory
        ADD COLUMN IF NOT EXISTS embedding_new vector({EMBEDDING_VECTOR_DIM})
        """
    )


async def fetch_pending_rows(
    conn: asyncpg.Connection,
    batch_size: int,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT id, source_text
        FROM translation_memory
        WHERE embedding_new IS NULL
        ORDER BY created_at ASC, id ASC
        LIMIT $1
        """,
        batch_size,
    )


async def backfill_embeddings(conn: asyncpg.Connection, batch_size: int) -> int:
    total = 0

    while True:
        rows = await fetch_pending_rows(conn, batch_size)
        if not rows:
            break

        texts = [row["source_text"] for row in rows]
        embeddings = await asyncio.to_thread(embed, texts)

        await conn.executemany(
            """
            UPDATE translation_memory
            SET embedding_new = $1::vector
            WHERE id = $2
            """,
            [
                (str(embedding), row["id"])
                for row, embedding in zip(rows, embeddings)
            ],
        )

        total += len(rows)
        print(f"Backfilled {total} translation_memory row(s)")

    return total


async def drop_old_embedding_indexes(conn: asyncpg.Connection) -> None:
    indexes = await conn.fetch(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = 'translation_memory'
          AND indexdef ILIKE '%embedding%'
        """
    )

    for row in indexes:
        index_name = row["indexname"]
        print(f"Dropping index {index_name}")
        await conn.execute(f'DROP INDEX IF EXISTS "{index_name}"')


async def swap_embedding_columns(conn: asyncpg.Connection) -> None:
    await drop_old_embedding_indexes(conn)

    await conn.execute(
        """
        ALTER TABLE translation_memory
        DROP COLUMN IF EXISTS embedding
        """
    )
    await conn.execute(
        """
        ALTER TABLE translation_memory
        RENAME COLUMN embedding_new TO embedding
        """
    )
    await conn.execute(
        """
        CREATE INDEX IF NOT EXISTS translation_memory_embedding_idx
        ON translation_memory
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


async def count_rows(conn: asyncpg.Connection) -> int:
    return await conn.fetchval("SELECT COUNT(*) FROM translation_memory")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild translation_memory embeddings for multilingual-e5-small.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of TM rows to re-embed per batch.",
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="After backfill, replace the old embedding column with the new one.",
    )
    args = parser.parse_args()

    conn = await asyncpg.connect(
        dsn=EFFECTIVE_DB_URL,
        ssl="require",
        statement_cache_size=0,
    )
    try:
        total_before = await count_rows(conn)
        print(
            f"Preparing to rebuild embeddings for {total_before} translation_memory row(s) "
            f"to vector({EMBEDDING_VECTOR_DIM})"
        )
        await ensure_embedding_new_column(conn)
        rebuilt = await backfill_embeddings(conn, args.batch_size)
        print(f"Completed backfill for {rebuilt} row(s)")

        if args.swap:
            await swap_embedding_columns(conn)
            print("Swapped embedding_new into embedding and recreated the ivfflat index")
        else:
            print(
                "Backfill complete. Run this script again with --swap once you are ready "
                "to replace the old embedding column."
            )
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
