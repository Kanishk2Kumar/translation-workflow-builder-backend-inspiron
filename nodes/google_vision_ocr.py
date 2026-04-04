import base64
import os

import httpx

from config import settings
from nodes.base import BaseNode

_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_vision_client: httpx.AsyncClient | None = None


def get_vision_client() -> httpx.AsyncClient:
    global _vision_client
    if _vision_client is None:
        timeout = float(settings.GOOGLE_VISION_TIMEOUT_SECONDS)
        _vision_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0))
    return _vision_client


def is_supported_image(filename: str, content_type: str = "") -> bool:
    if content_type.startswith("image/"):
        return True
    _, ext = os.path.splitext(filename.lower())
    return ext in _SUPPORTED_IMAGE_EXTENSIONS


def build_word_text(word: dict) -> str:
    return "".join(symbol.get("text", "") for symbol in word.get("symbols", []))


def get_word_break_type(word: dict) -> str | None:
    symbols = word.get("symbols", [])
    if not symbols:
        return None
    return (
        symbols[-1]
        .get("property", {})
        .get("detectedBreak", {})
        .get("type")
    )


def get_vertices(item: dict) -> list[dict[str, int]]:
    vertices = item.get("boundingBox", {}).get("vertices", [])
    normalized = []
    for vertex in vertices:
        normalized.append({
            "x": int(vertex.get("x", 0)),
            "y": int(vertex.get("y", 0)),
        })
    return normalized


def merge_vertices(existing: list[dict[str, int]], new_vertices: list[dict[str, int]]) -> list[dict[str, int]]:
    vertices = existing + new_vertices
    if not vertices:
        return []

    xs = [vertex["x"] for vertex in vertices]
    ys = [vertex["y"] for vertex in vertices]
    return [
        {"x": min(xs), "y": min(ys)},
        {"x": max(xs), "y": min(ys)},
        {"x": max(xs), "y": max(ys)},
        {"x": min(xs), "y": max(ys)},
    ]


def build_line_blocks(full_text_annotation: dict) -> list[dict]:
    pages = full_text_annotation.get("pages", [])
    blocks: list[dict] = []

    for page_index, page in enumerate(pages, start=1):
        for block in page.get("blocks", []):
            for paragraph in block.get("paragraphs", []):
                current_words: list[str] = []
                current_vertices: list[dict[str, int]] = []
                current_confidences: list[float] = []

                def flush_line() -> None:
                    if not current_words:
                        return

                    confidence = None
                    if current_confidences:
                        confidence = round(sum(current_confidences) / len(current_confidences), 4)

                    blocks.append({
                        "page": page_index,
                        "text": " ".join(current_words).strip(),
                        "confidence": confidence,
                        "vertices": current_vertices.copy(),
                    })
                    current_words.clear()
                    current_vertices.clear()
                    current_confidences.clear()

                for word in paragraph.get("words", []):
                    word_text = build_word_text(word).strip()
                    if not word_text:
                        continue

                    current_words.append(word_text)
                    current_vertices[:] = merge_vertices(current_vertices, get_vertices(word))

                    confidence = word.get("confidence")
                    if isinstance(confidence, (int, float)):
                        current_confidences.append(float(confidence))

                    break_type = get_word_break_type(word)
                    if break_type in {"LINE_BREAK", "EOL_SURE_SPACE"}:
                        flush_line()

                flush_line()

    return [block for block in blocks if block["text"]]


class GoogleVisionOCRNode(BaseNode):

    async def run(self, context: dict) -> dict:
        file_bytes: bytes | None = context.get("file_bytes")
        filename: str = context.get("source_filename", "")
        content_type: str = context.get("source_content_type", "")

        if not file_bytes:
            raise ValueError("GoogleVisionOCRNode: file_bytes is required in context")
        if not is_supported_image(filename, content_type):
            raise ValueError(
                "GoogleVisionOCRNode supports image uploads only (.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff)"
            )
        if not settings.GOOGLE_VISION_API_KEY:
            raise ValueError("GoogleVisionOCRNode: GOOGLE_VISION_API_KEY is not configured")

        language_hints = self.config.get("language_hints", [])
        if isinstance(language_hints, str):
            language_hints = [hint.strip() for hint in language_hints.split(",") if hint.strip()]

        client = get_vision_client()
        response = await client.post(
            settings.GOOGLE_VISION_ENDPOINT.rstrip("/"),
            params={"key": settings.GOOGLE_VISION_API_KEY},
            json={
                "requests": [{
                    "image": {
                        "content": base64.b64encode(file_bytes).decode("utf-8"),
                    },
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {
                        "languageHints": language_hints,
                    },
                }],
            },
        )
        response.raise_for_status()

        payload = response.json()
        responses = payload.get("responses", [])
        if not responses:
            raise ValueError("GoogleVisionOCRNode: empty Google Vision response")

        annotation = responses[0]
        if annotation.get("error"):
            message = annotation["error"].get("message", "Google Vision OCR request failed")
            raise ValueError(f"GoogleVisionOCRNode: {message}")

        full_text_annotation = annotation.get("fullTextAnnotation", {})
        ocr_text = full_text_annotation.get("text", "").strip()
        ocr_blocks = build_line_blocks(full_text_annotation)

        if not ocr_blocks and ocr_text:
            ocr_blocks = [
                {"page": 1, "text": line.strip(), "confidence": None, "vertices": []}
                for line in ocr_text.splitlines()
                if line.strip()
            ]

        confidences = [
            block["confidence"]
            for block in ocr_blocks
            if isinstance(block.get("confidence"), (int, float))
        ]
        ocr_confidence = None
        if confidences:
            ocr_confidence = round(sum(confidences) / len(confidences), 4)

        warnings: list[str] = []
        if not ocr_text:
            warnings.append("No text was detected in the uploaded image.")
        if ocr_confidence is None:
            warnings.append("Google Vision did not return usable confidence values for this image.")
        if ocr_text and len(ocr_text) < int(self.config.get("min_text_chars_warning", 8)):
            warnings.append("OCR output was very short and may be incomplete.")

        return {
            **context,
            "ocr_text": ocr_text,
            "ocr_blocks": ocr_blocks,
            "ocr_confidence": ocr_confidence,
            "ocr_confidence_available": ocr_confidence is not None,
            "ocr_provider": "google_vision",
            "ocr_pages": len(full_text_annotation.get("pages", [])) or (1 if ocr_text else 0),
            "ocr_warnings": warnings,
        }
