# nodes/phi_detector.py

import re

from db import get_pool
from nodes.base import BaseNode

PHI_PATTERNS = [
    # Order matters: most specific first to avoid partial overlaps.
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("NPI", r"(?<=\bXX\*)\d{10}(?=~|\*)"),
    ("NPI", r"\bNPI[-:\s]*\d{10}\b"),
    ("MRN", r"\bMRN[-:\s]*\d{6,10}\b"),
    ("PASSPORT", r"\b[A-Z]{1,2}\d{6,8}\b"),
    ("EMAIL", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("PHONE", r"\b(\+?\d[\d\s\-().]{7,}\d)\b"),
    ("DOB", r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
]


def mask_phi(text: str, type_counters: dict[str, int]) -> tuple[str, list[dict]]:
    """
    Single-pass PHI masking using a combined regex.
    Placeholders are globally unique per PHI type across the whole document.
    """
    combined = "|".join(f"(?P<TYPE_{i}>{pat})" for i, (_, pat) in enumerate(PHI_PATTERNS))
    combined_re = re.compile(combined, flags=re.IGNORECASE)

    detections: list[dict] = []

    def replace_match(match: re.Match) -> str:
        original = match.group(0)

        phi_type = "UNKNOWN"
        for index, (name, _) in enumerate(PHI_PATTERNS):
            if match.group(f"TYPE_{index}") is not None:
                phi_type = name
                break

        occurrence = type_counters.get(phi_type, 0)
        type_counters[phi_type] = occurrence + 1
        placeholder = f"PHIMASK_{phi_type}_{occurrence}"

        detections.append({
            "placeholder": placeholder,
            "original_value": original,
            "phi_type": phi_type,
        })
        return placeholder

    masked = combined_re.sub(replace_match, text)
    return masked, detections


class PHIDetectorNode(BaseNode):

    async def run(self, context: dict) -> dict:
        segments: list[str] = context.get("segments", [])
        execution_id: str = context.get("execution_id", "")
        user_id: str = context.get("user_id", "")
        document_blocks: list = context.get("document_blocks", [])

        phi_map: dict[str, str] = {}
        masked_segments: list[str] = []
        all_detections: list[dict] = []
        type_counters: dict[str, int] = {}

        for seg_idx, segment in enumerate(segments):
            masked, detections = mask_phi(segment, type_counters)
            masked_segments.append(masked)
            for det in detections:
                phi_map[det["placeholder"]] = det["original_value"]
                all_detections.append({**det, "segment_idx": seg_idx})

        if document_blocks:
            for block, masked_seg in zip(document_blocks, masked_segments):
                block.source_text = masked_seg

        if all_detections and execution_id:
            try:
                pool = get_pool()
                rows = [
                    (
                        execution_id,
                        user_id or None,
                        det["phi_type"],
                        det["placeholder"],
                        det["original_value"],
                        det["segment_idx"],
                    )
                    for det in all_detections
                ]
                await pool.executemany(
                    """
                    INSERT INTO pii_audit
                      (execution_id, user_id, phi_type, placeholder, original_value, segment_idx)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    rows,
                )
            except Exception as exc:
                print(f"PII audit write failed: {exc}")

        print(f"PHIDetectorNode: masked {len(phi_map)} PHI instances")
        print(f"DEBUG phi_map: {phi_map}")

        return {
            **context,
            "segments": masked_segments,
            "raw_text": "\n".join(masked_segments),
            "phi_map": phi_map,
            "phi_count": len(phi_map),
            "document_blocks": document_blocks,
        }
