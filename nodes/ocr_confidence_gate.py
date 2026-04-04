import re

from nodes.base import BaseNode

_MEDICAL_LINE_HINTS = re.compile(
    r"(\brx\b|\btab\b|\bcap\b|\bsyr\b|\binj\b|\bmg\b|\bmcg\b|\bml\b|\bg\b|"
    r"\bod\b|\bbd\b|\btid\b|\bqid\b|\bhs\b|\bprn\b|\bpo\b|\biv\b|\bim\b|"
    r"\bstat\b|\bdays?\b|\bdoctor\b|\bdr\.\b|\bdiagnosis\b|\bpatient\b)",
    flags=re.IGNORECASE,
)


def split_ocr_segments(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines or ([text.strip()] if text.strip() else [])


def is_medically_important_line(text: str) -> bool:
    if not text.strip():
        return False
    if _MEDICAL_LINE_HINTS.search(text):
        return True
    return bool(re.search(r"\d+\s*(mg|mcg|ml|g|units?)\b", text, flags=re.IGNORECASE))


class OCRConfidenceGateNode(BaseNode):

    async def run(self, context: dict) -> dict:
        ocr_text: str = context.get("ocr_text", "").strip()
        ocr_blocks: list[dict] = context.get("ocr_blocks", [])
        ocr_confidence = context.get("ocr_confidence")
        confidence_available = context.get(
            "ocr_confidence_available",
            isinstance(ocr_confidence, (int, float)),
        )

        pass_threshold = float(self.config.get("pass_threshold", 0.85))
        warn_threshold = float(self.config.get("warn_threshold", 0.65))
        critical_line_threshold = float(self.config.get("critical_line_threshold", 0.60))
        max_low_confidence_lines = int(self.config.get("max_low_confidence_lines", 2))
        min_text_chars = int(self.config.get("min_text_chars", 8))

        low_confidence_blocks = [
            block for block in ocr_blocks
            if isinstance(block.get("confidence"), (int, float))
            and float(block["confidence"]) < critical_line_threshold
        ]
        critical_low_confidence_blocks = [
            block for block in low_confidence_blocks
            if is_medically_important_line(block.get("text", ""))
        ]

        review_required = False
        ocr_status = "passed"
        reasons: list[str] = []

        if not ocr_text:
            review_required = True
            ocr_status = "failed"
            reasons.append("OCR did not extract any text.")
        elif len(ocr_text) < min_text_chars:
            review_required = True
            ocr_status = "failed"
            reasons.append("OCR extracted too little text to translate safely.")

        if not confidence_available:
            review_required = True
            if ocr_status != "failed":
                ocr_status = "review"
            reasons.append("OCR confidence metadata was unavailable.")
        elif float(ocr_confidence) < warn_threshold:
            review_required = True
            ocr_status = "failed"
            reasons.append(
                f"Overall OCR confidence {float(ocr_confidence):.2f} is below the review threshold {warn_threshold:.2f}."
            )
        elif float(ocr_confidence) < pass_threshold:
            review_required = True
            if ocr_status != "failed":
                ocr_status = "review"
            reasons.append(
                f"Overall OCR confidence {float(ocr_confidence):.2f} is below the pass threshold {pass_threshold:.2f}."
            )

        if critical_low_confidence_blocks:
            review_required = True
            if ocr_status != "failed":
                ocr_status = "review"
            reasons.append(
                f"{len(critical_low_confidence_blocks)} medically important line(s) fell below {critical_line_threshold:.2f} confidence."
            )
        elif len(low_confidence_blocks) > max_low_confidence_lines:
            review_required = True
            if ocr_status != "failed":
                ocr_status = "review"
            reasons.append(
                f"{len(low_confidence_blocks)} OCR lines fell below {critical_line_threshold:.2f} confidence."
            )

        segments = split_ocr_segments(ocr_text)
        ocr_passed = not review_required and bool(segments)

        next_context = {
            **context,
            "ocr_passed": ocr_passed,
            "ocr_status": ocr_status,
            "review_required": review_required,
            "review_reason": " ".join(reasons) if reasons else None,
            "ocr_low_confidence_blocks": low_confidence_blocks,
        }

        if ocr_passed:
            promoted_text = "\n".join(segments)
            next_context.update({
                "raw_text": promoted_text,
                "original_raw_text": promoted_text,
                "segments": segments,
                "original_segments": segments.copy(),
                "segment_count": len(segments),
            })
            return next_context

        next_context.update({
            "_stop_workflow": True,
            "translated_text": "",
            "segment_translations": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "tm_hit": False,
        })
        return next_context
