import asyncio
import json
import os
from datetime import datetime, timezone

from nodes.base import BaseNode
from db import get_pool

# Persistent temp dir for translated documents
TRANSLATED_DOCS_DIR = "/tmp/translatio_outputs"
os.makedirs(TRANSLATED_DOCS_DIR, exist_ok=True)


async def seed_translation_memory(seed_payload: dict) -> None:
    segment_translations: dict = seed_payload.get("segment_translations", {})
    target_language: str = seed_payload.get("target_language", "hi")

    if not segment_translations:
        return

    try:
        pool = get_pool()
        segments = list(segment_translations.keys())
        existing_rows = await pool.fetch(
            """
            SELECT source_text
            FROM translation_memory
            WHERE target_language = $1
              AND source_text = ANY($2::text[])
            """,
            target_language,
            segments,
        )
        existing_segments = {row["source_text"] for row in existing_rows}
        missing_segments = [
            segment for segment in segments
            if segment not in existing_segments
        ]

        if not missing_segments:
            print("TM seeding skipped: all segments already exist")
            return

        from nodes.rag_tm import embed

        embeddings = await asyncio.to_thread(embed, missing_segments)
        rows = [
            (
                segment,
                segment_translations[segment],
                target_language,
                str(embedding),
            )
            for segment, embedding in zip(missing_segments, embeddings)
        ]

        await pool.executemany(
            """
            INSERT INTO translation_memory
              (source_text, target_text, source_language, target_language, embedding)
            VALUES ($1, $2, 'en', $3, $4::vector)
            """,
            rows,
        )
        print(f"TM seeding complete: inserted {len(rows)} segments")
    except Exception as e:
        print(f"TM seeding failed: {e}")


class OutputNode(BaseNode):

    async def run(self, context: dict) -> dict:
        include_audit: bool = self.config.get("include_audit", True)

        translated_text: str = context.get("translated_text", "")
        rag_stats: dict = context.get("rag_stats", {})
        logs: list = context.get("_logs", [])
        execution_id: str = context.get("execution_id", "")
        workflow_id: str = context.get("workflow_id", "")

        output_doc_bytes: bytes | None = context.get("output_document_bytes")
        output_doc_format: str = context.get("output_document_format", "text")

        doc_file_path: str | None = None
        if output_doc_bytes and execution_id:
            ext = output_doc_format if output_doc_format in ("docx", "pdf") else "bin"
            doc_file_path = os.path.join(TRANSLATED_DOCS_DIR, f"{execution_id}.{ext}")
            with open(doc_file_path, "wb") as f:
                f.write(output_doc_bytes)

        output_payload = {
            "translated_text": translated_text,
            "source_language": "en",
            "target_language": context.get("target_language", "hi"),
            "segment_count": context.get("segment_count", 0),
            "rag_stats": rag_stats,
            "model": context.get("llm_model", ""),
            "tm_hit": context.get("tm_hit", False),
            "token_usage": {
                "input": context.get("input_tokens", 0),
                "output": context.get("output_tokens", 0),
            },
            "document_path": doc_file_path,
            "document_format": output_doc_format if output_doc_bytes else None,
        }

        if include_audit:
            output_payload["audit"] = {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "node_logs": logs,
            }

        pool = get_pool()

        if execution_id:
            await pool.execute(
                """
                UPDATE executions
                SET status = 'success',
                    output = $1,
                    logs   = $2,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = $3
                """,
                json.dumps(output_payload),
                json.dumps(logs),
                execution_id,
            )

        tm_seed_payload = None
        if translated_text and not context.get("tm_hit"):
            segment_translations: dict = context.get("segment_translations", {})
            if segment_translations:
                tm_seed_payload = {
                    "target_language": context.get("target_language", "hi"),
                    "segment_translations": segment_translations,
                }

        return {
            **context,
            "final_output": output_payload,
            "output_document_bytes": output_doc_bytes,
            "tm_seed_payload": tm_seed_payload,
        }
