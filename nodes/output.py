import asyncio
import json
import os
from datetime import datetime, timezone

from nodes.base import BaseNode
from db import get_pool

# Persistent temp dir for translated documents
TRANSLATED_DOCS_DIR = "/tmp/translatio_outputs"
os.makedirs(TRANSLATED_DOCS_DIR, exist_ok=True)


def is_vector_dimension_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "different vector dimensions" in message
        or "expected 1024 dimensions, not 384" in message
        or ("vector dimensions" in message and "expected" in message)
    )


def build_segment_payload(context: dict) -> list[dict]:
    override_segments = context.get("output_segments_override")
    if isinstance(override_segments, list):
        return [dict(segment) for segment in override_segments]

    original_segments: list[str] = context.get("original_segments", []) or []
    segment_translations: dict = context.get("segment_translations", {}) or {}

    if not original_segments:
        translated_text = context.get("translated_text", "")
        raw_text = context.get("original_raw_text") or context.get("raw_text", "")
        if not raw_text and not translated_text:
            return []
        return [{
            "segment_index": 1,
            "source_text": raw_text,
            "translated_text": translated_text or raw_text,
            "edited": False,
        }]

    return [
        {
            "segment_index": index,
            "source_text": source_segment,
            "translated_text": segment_translations.get(source_segment, source_segment),
            "edited": False,
        }
        for index, source_segment in enumerate(original_segments, start=1)
    ]


async def seed_translation_memory(seed_payload: dict) -> None:
    segment_translations: dict = seed_payload.get("segment_translations", {})
    target_language: str = seed_payload.get("target_language", "hi")
    source_language: str = seed_payload.get("source_language", "en")

    if not segment_translations:
        return

    try:
        pool = get_pool()
        from nodes.rag_tm import clear_runtime_caches

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

        existing_segments_to_update = [
            segment for segment in segments
            if segment in existing_segments
        ]

        if existing_segments_to_update:
            await pool.executemany(
                """
                UPDATE translation_memory
                SET target_text = $1
                WHERE target_language = $2
                  AND source_text = $3
                """,
                [
                    (
                        segment_translations[segment],
                        target_language,
                        segment,
                    )
                    for segment in existing_segments_to_update
                ],
            )

        if not missing_segments:
            clear_runtime_caches()
            print(
                "TM seeding complete: updated "
                f"{len(existing_segments_to_update)} existing segments, inserted 0 new segments"
            )
            return

        from nodes.rag_tm import embed

        embeddings = await asyncio.to_thread(embed, missing_segments)
        rows = [
            (
                segment,
                segment_translations[segment],
                source_language,
                target_language,
                str(embedding),
            )
            for segment, embedding in zip(missing_segments, embeddings)
        ]

        await pool.executemany(
            """
            INSERT INTO translation_memory
              (source_text, target_text, source_language, target_language, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            """,
            rows,
        )
        print(
            "TM seeding complete: updated "
            f"{len(existing_segments_to_update)} existing segments, inserted {len(rows)} new segments"
        )
        clear_runtime_caches()
    except Exception as e:
        if 'rows' in locals() and is_vector_dimension_mismatch_error(e):
            try:
                await pool.executemany(
                    """
                    INSERT INTO translation_memory
                      (source_text, target_text, source_language, target_language, embedding)
                    VALUES ($1, $2, $3, $4, NULL)
                    """,
                    [
                        (
                            segment,
                            segment_translations[segment],
                            source_language,
                            target_language,
                        )
                        for segment in missing_segments
                    ],
                )
                clear_runtime_caches()
                print(
                    "TM seeding fallback: inserted "
                    f"{len(missing_segments)} segment(s) without embeddings because the DB still expects the old vector dimension. "
                    "Exact-match TM will work, fuzzy/vector TM needs a DB embedding migration."
                )
                return
            except Exception as fallback_exc:
                print(f"TM seeding fallback failed: {fallback_exc}")
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
            "segments": build_segment_payload(context),
            "rag_stats": rag_stats,
            "model": context.get("llm_model", ""),
            "tm_hit": context.get("tm_hit", False),
            "token_usage": {
                "input": context.get("input_tokens", 0),
                "output": context.get("output_tokens", 0),
            },
            "document_path": doc_file_path,
            "document_format": output_doc_format if output_doc_bytes else None,
            "compliance_status": context.get("compliance_status"),
            "compliance_errors": context.get("compliance_errors", []),
            "compliance_suggestions": context.get("compliance_suggestions", []),
            "compliance_report": context.get("compliance_report"),
            "ocr_provider": context.get("ocr_provider"),
            "ocr_confidence": context.get("ocr_confidence"),
            "ocr_status": context.get("ocr_status"),
            "ocr_warnings": context.get("ocr_warnings", []),
            "review_required": context.get("review_required", False),
            "review_reason": context.get("review_reason"),
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
                    "source_language": context.get("source_language", "en"),
                    "target_language": context.get("target_language", "hi"),
                    "segment_translations": segment_translations,
                }

        return {
            **context,
            "final_output": output_payload,
            "output_document_bytes": output_doc_bytes,
            "tm_seed_payload": tm_seed_payload,
        }
