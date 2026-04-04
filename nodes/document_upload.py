import re
from nodes.base import BaseNode


# Basic sentence splitter — good enough for demo
# Swap for spaCy sentence segmentation in production
def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class DocumentUploadNode(BaseNode):
    """
    Entry point of the pipeline.
    Accepts raw text from the request (PDF parsing is a future improvement).
    Splits text into segments and stores them in context.
    """

    async def run(self, context: dict) -> dict:
        raw_text = context.get("raw_text", "")

        if not raw_text:
            raise ValueError("DocumentUploadNode: no text provided in context['raw_text']")

        segments = split_sentences(raw_text)

        return {
            **context,          # ← spread first — preserves file_bytes, source_filename, etc.
            "raw_text": raw_text,
            "segments": segments,
            "segment_count": len(segments),
        }