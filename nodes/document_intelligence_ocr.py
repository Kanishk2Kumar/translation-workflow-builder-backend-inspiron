import asyncio
import os
import time

import httpx

from config import settings
from nodes.base import BaseNode

_SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".pdf",
}
_document_intelligence_client: httpx.AsyncClient | None = None


def get_document_intelligence_client() -> httpx.AsyncClient:
    global _document_intelligence_client
    if _document_intelligence_client is None:
        timeout = float(settings.AZURE_DOCUMENT_INTELLIGENCE_TIMEOUT_SECONDS)
        _document_intelligence_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0)
        )
    return _document_intelligence_client


def is_supported_document(filename: str, content_type: str = "") -> bool:
    if content_type.startswith("image/") or content_type == "application/pdf":
        return True
    _, ext = os.path.splitext(filename.lower())
    return ext in _SUPPORTED_EXTENSIONS


def guess_content_type(filename: str, content_type: str = "") -> str:
    if content_type:
        return content_type

    _, ext = os.path.splitext(filename.lower())
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def normalize_polygon(polygon: list[float] | None) -> list[dict[str, float]]:
    if not polygon:
        return []

    points: list[dict[str, float]] = []
    for index in range(0, len(polygon), 2):
        x = polygon[index]
        y = polygon[index + 1] if index + 1 < len(polygon) else 0
        points.append({"x": float(x), "y": float(y)})
    return points


def spans_overlap(left: dict, right: dict) -> bool:
    left_start = int(left.get("offset", 0))
    left_end = left_start + int(left.get("length", 0))
    right_start = int(right.get("offset", 0))
    right_end = right_start + int(right.get("length", 0))
    return left_start < right_end and right_start < left_end


def collect_line_confidence(line: dict, words: list[dict]) -> float | None:
    line_spans = line.get("spans", [])
    if not line_spans:
        return None

    confidences: list[float] = []
    for word in words:
        word_span = word.get("span")
        confidence = word.get("confidence")
        if not word_span or not isinstance(confidence, (int, float)):
            continue
        if any(spans_overlap(line_span, word_span) for line_span in line_spans):
            confidences.append(float(confidence))

    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 4)


def build_line_blocks(analyze_result: dict) -> list[dict]:
    blocks: list[dict] = []

    for page in analyze_result.get("pages", []):
        page_number = int(page.get("pageNumber", 1))
        words = page.get("words", [])
        for line in page.get("lines", []):
            text = (line.get("content") or "").strip()
            if not text:
                continue

            blocks.append({
                "page": page_number,
                "text": text,
                "confidence": collect_line_confidence(line, words),
                "vertices": normalize_polygon(line.get("polygon")),
            })

    return blocks


def get_analyze_url(model_id: str) -> str:
    if not settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT:
        raise ValueError("DocumentIntelligenceOCRNode: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT is not configured")

    base = settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT.rstrip("/")
    api_version = settings.AZURE_DOCUMENT_INTELLIGENCE_API_VERSION
    return f"{base}/documentintelligence/documentModels/{model_id}:analyze?api-version={api_version}"


def sanitize_http_error(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code
    try:
        payload = exc.response.json()
    except ValueError:
        payload = {}

    details = payload.get("error", {})
    message = details.get("message") or exc.response.text or "Document Intelligence request failed"
    return f"DocumentIntelligenceOCRNode: HTTP {status_code}: {message}"


class DocumentIntelligenceOCRNode(BaseNode):

    async def run(self, context: dict) -> dict:
        file_bytes: bytes | None = context.get("file_bytes")
        filename: str = context.get("source_filename", "")
        content_type: str = context.get("source_content_type", "")

        if not file_bytes:
            raise ValueError("DocumentIntelligenceOCRNode: file_bytes is required in context")
        if not settings.AZURE_DOCUMENT_INTELLIGENCE_KEY:
            raise ValueError("DocumentIntelligenceOCRNode: AZURE_DOCUMENT_INTELLIGENCE_KEY is not configured")
        if not is_supported_document(filename, content_type):
            raise ValueError(
                "DocumentIntelligenceOCRNode supports image and PDF uploads only "
                "(.png, .jpg, .jpeg, .webp, .bmp, .tif, .tiff, .pdf)"
            )

        model_id = self.config.get("model_id", settings.AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID)
        poll_interval_ms = int(
            self.config.get(
                "poll_interval_ms",
                settings.AZURE_DOCUMENT_INTELLIGENCE_POLL_INTERVAL_MS,
            )
        )
        language = self.config.get("locale")
        features = self.config.get("features", [])

        params = {}
        if language:
            params["locale"] = language
        if features:
            params["features"] = ",".join(features)

        client = get_document_intelligence_client()
        headers = {
            "Ocp-Apim-Subscription-Key": settings.AZURE_DOCUMENT_INTELLIGENCE_KEY,
            "Content-Type": guess_content_type(filename, content_type),
        }

        try:
            response = await client.post(
                get_analyze_url(model_id),
                headers=headers,
                params=params,
                content=file_bytes,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(sanitize_http_error(exc)) from exc

        operation_location = response.headers.get("Operation-Location")
        if not operation_location:
            raise ValueError("DocumentIntelligenceOCRNode: missing Operation-Location header from analyze response")

        deadline = time.monotonic() + float(settings.AZURE_DOCUMENT_INTELLIGENCE_TIMEOUT_SECONDS)
        while True:
            try:
                result_response = await client.get(
                    operation_location,
                    headers={"Ocp-Apim-Subscription-Key": settings.AZURE_DOCUMENT_INTELLIGENCE_KEY},
                )
                result_response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise ValueError(sanitize_http_error(exc)) from exc

            payload = result_response.json()
            status = (payload.get("status") or "").lower()

            if status == "succeeded":
                analyze_result = payload.get("analyzeResult", {})
                break
            if status == "failed":
                error = payload.get("error", {})
                message = error.get("message", "Document Intelligence analysis failed")
                raise ValueError(f"DocumentIntelligenceOCRNode: {message}")
            if time.monotonic() >= deadline:
                raise ValueError("DocumentIntelligenceOCRNode: OCR analysis timed out while waiting for completion")

            await asyncio.sleep(max(poll_interval_ms, 200) / 1000)

        ocr_text = (analyze_result.get("content") or "").strip()
        ocr_blocks = build_line_blocks(analyze_result)

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
            warnings.append("No text was detected in the uploaded document.")
        if ocr_confidence is None:
            warnings.append("Document Intelligence did not return usable confidence values for this file.")
        if ocr_text and len(ocr_text) < int(self.config.get("min_text_chars_warning", 8)):
            warnings.append("OCR output was very short and may be incomplete.")

        return {
            **context,
            "ocr_text": ocr_text,
            "ocr_blocks": ocr_blocks,
            "ocr_confidence": ocr_confidence,
            "ocr_confidence_available": ocr_confidence is not None,
            "ocr_provider": "azure_document_intelligence",
            "ocr_pages": len(analyze_result.get("pages", [])) or (1 if ocr_text else 0),
            "ocr_warnings": warnings,
        }
