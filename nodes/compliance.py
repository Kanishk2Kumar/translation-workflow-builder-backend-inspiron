import re

from nodes.base import BaseNode
from nodes.compliance_common import (
    ICD10_RE,
    DATE_RE,
    X12_CRITICAL_TAGS,
    clip_text,
    extract_amount_tokens,
    extract_cpt_tokens,
    extract_npi_tokens,
    extract_x12_segments,
    is_valid_ccyymmdd,
    is_valid_npi,
    should_validate_amounts,
    should_validate_dates,
)


class ComplianceNode(BaseNode):

    async def run(self, context: dict) -> dict:
        original_segments: list[str] = context.get("original_segments", [])
        current_segments: list[str] = context.get("segments", [])
        translated_text: str = context.get("translated_text", "")
        segment_translations: dict[str, str] = context.get("segment_translations", {})
        original_raw_text: str = context.get("original_raw_text", context.get("raw_text", ""))

        if not translated_text and not segment_translations:
            return {
                **context,
                "compliance_status": "warn",
                "compliance_errors": [],
                "compliance_suggestions": [],
                "compliance_report": {
                    "summary": "Compliance check skipped because no translated output was available.",
                    "critical_count": 0,
                    "warning_count": 0,
                },
            }

        paired_segments: list[tuple[int, str, str]] = []
        if original_segments and current_segments and segment_translations:
            for index, (source_segment, masked_segment) in enumerate(zip(original_segments, current_segments), start=1):
                translated_segment = segment_translations.get(masked_segment, "")
                if translated_segment:
                    paired_segments.append((index, source_segment, translated_segment))

        if not paired_segments and original_raw_text and translated_text:
            paired_segments = [(1, original_raw_text, translated_text)]

        violations: list[dict] = []
        suggestions: list[dict] = []

        for segment_index, source_segment, translated_segment in paired_segments:
            self._validate_preserved_codes(
                source_segment=source_segment,
                translated_segment=translated_segment,
                segment_index=segment_index,
                violations=violations,
                suggestions=suggestions,
            )
            self._validate_dates(
                source_segment=source_segment,
                translated_segment=translated_segment,
                segment_index=segment_index,
                violations=violations,
                suggestions=suggestions,
            )
            self._validate_amounts(
                source_segment=source_segment,
                translated_segment=translated_segment,
                segment_index=segment_index,
                violations=violations,
                suggestions=suggestions,
            )
            self._validate_x12_structure(
                source_segment=source_segment,
                translated_segment=translated_segment,
                segment_index=segment_index,
                violations=violations,
                suggestions=suggestions,
            )

        self._validate_translated_npis(
            translated_text=translated_text or "\n".join(item[2] for item in paired_segments),
            source_text=original_raw_text,
            violations=violations,
            suggestions=suggestions,
        )

        critical_count = sum(1 for item in violations if item["severity"] == "fail")
        warning_count = sum(1 for item in violations if item["severity"] == "warn")
        if critical_count:
            status = "fail"
            summary = (
                f"Compliance check found {critical_count} critical issue(s)"
                f"{f' and {warning_count} warning(s)' if warning_count else ''}. "
                "The translated response is still returned, but the EDI output needs review."
            )
        elif warning_count:
            status = "warn"
            summary = f"Compliance check found {warning_count} warning(s). Review the flagged EDI elements before export."
        else:
            status = "pass"
            summary = "Compliance check passed. Protected codes, structural hints, and core HIPAA-style formats look intact."

        return {
            **context,
            "compliance_status": status,
            "compliance_errors": violations,
            "compliance_suggestions": suggestions,
            "compliance_report": {
                "summary": summary,
                "critical_count": critical_count,
                "warning_count": warning_count,
            },
        }

    def _validate_preserved_codes(
        self,
        source_segment: str,
        translated_segment: str,
        segment_index: int,
        violations: list[dict],
        suggestions: list[dict],
    ) -> None:
        token_groups = {
            "ICD-10": list(dict.fromkeys(ICD10_RE.findall(source_segment))),
            "CPT/HCPCS": extract_cpt_tokens(source_segment),
            "NPI": extract_npi_tokens(source_segment),
        }

        for token_type, tokens in token_groups.items():
            for token in tokens:
                if token in translated_segment:
                    continue
                message = (
                    f"{token_type} value '{token}' from segment {segment_index} is missing or altered in the translated output."
                )
                violations.append(self._build_violation(
                    rule="phi_code_preservation",
                    severity="fail",
                    segment_index=segment_index,
                    source_segment=source_segment,
                    translated_segment=translated_segment,
                    expected_value=token,
                    message=message,
                ))
                suggestions.append(self._build_suggestion(
                    action="replace_token",
                    segment_index=segment_index,
                    incorrect_value=None,
                    corrected_value=token,
                    reason=f"Restore the original {token_type} token exactly as it appeared in the source.",
                ))

    def _validate_dates(
        self,
        source_segment: str,
        translated_segment: str,
        segment_index: int,
        violations: list[dict],
        suggestions: list[dict],
    ) -> None:
        source_dates = list(dict.fromkeys(DATE_RE.findall(source_segment)))
        for value in source_dates:
            if not is_valid_ccyymmdd(value):
                continue
            if value in translated_segment:
                continue
            violations.append(self._build_violation(
                rule="hipaa_date_format",
                severity="fail",
                segment_index=segment_index,
                source_segment=source_segment,
                translated_segment=translated_segment,
                expected_value=value,
                message=f"Date '{value}' should stay in CCYYMMDD format, but that exact value is missing from segment {segment_index}.",
            ))
            suggestions.append(self._build_suggestion(
                action="replace_token",
                segment_index=segment_index,
                incorrect_value=None,
                corrected_value=value,
                reason="Restore the original CCYYMMDD date string exactly.",
            ))

        if not should_validate_dates(source_segment):
            return

        for translated_value in DATE_RE.findall(translated_segment):
            if is_valid_ccyymmdd(translated_value):
                continue
            violations.append(self._build_violation(
                rule="hipaa_date_format",
                severity="warn",
                segment_index=segment_index,
                source_segment=source_segment,
                translated_segment=translated_segment,
                expected_value=None,
                actual_value=translated_value,
                message=f"Segment {segment_index} contains '{translated_value}', which is 8 digits but not a valid CCYYMMDD date.",
            ))

    def _validate_amounts(
        self,
        source_segment: str,
        translated_segment: str,
        segment_index: int,
        violations: list[dict],
        suggestions: list[dict],
    ) -> None:
        if not should_validate_amounts(source_segment):
            return

        source_amounts = extract_amount_tokens(source_segment)
        for amount in source_amounts:
            if amount in translated_segment:
                continue
            violations.append(self._build_violation(
                rule="hipaa_numeric_amount",
                severity="fail",
                segment_index=segment_index,
                source_segment=source_segment,
                translated_segment=translated_segment,
                expected_value=amount,
                message=f"Numeric amount '{amount}' from segment {segment_index} did not survive translation exactly.",
            ))
            suggestions.append(self._build_suggestion(
                action="replace_token",
                segment_index=segment_index,
                incorrect_value=None,
                corrected_value=amount,
                reason="Restore the original numeric amount so the outbound EDI stays machine-readable.",
            ))

    def _validate_x12_structure(
        self,
        source_segment: str,
        translated_segment: str,
        segment_index: int,
        violations: list[dict],
        suggestions: list[dict],
    ) -> None:
        source_x12_segments = extract_x12_segments(source_segment)
        if not source_x12_segments:
            return

        translated_x12_segments = extract_x12_segments(translated_segment)
        if not translated_x12_segments:
            tags = sorted({item["tag"] for item in source_x12_segments if item["tag"] in X12_CRITICAL_TAGS})
            violations.append(self._build_violation(
                rule="edi_structure",
                severity="fail",
                segment_index=segment_index,
                source_segment=source_segment,
                translated_segment=translated_segment,
                expected_value=", ".join(tags) if tags else "X12 segment structure",
                message=f"Segment {segment_index} looks like X12 EDI in the source, but the translated output no longer contains intact X12 segment delimiters.",
            ))
            suggestions.append(self._build_suggestion(
                action="restore_x12_structure",
                segment_index=segment_index,
                incorrect_value=None,
                corrected_value=source_segment,
                reason="Keep X12 tags, '*' element separators, and '~' terminators unchanged in the translated output.",
            ))
            return

        translated_tags = [item["tag"] for item in translated_x12_segments]
        for source_x12 in source_x12_segments:
            if source_x12["tag"] not in translated_tags:
                violations.append(self._build_violation(
                    rule="edi_structure",
                    severity="fail",
                    segment_index=segment_index,
                    source_segment=source_segment,
                    translated_segment=translated_segment,
                    expected_value=source_x12["tag"],
                    message=f"X12 tag '{source_x12['tag']}' from segment {segment_index} is missing after translation.",
                ))
                suggestions.append(self._build_suggestion(
                    action="restore_x12_tag",
                    segment_index=segment_index,
                    incorrect_value=None,
                    corrected_value=source_x12["tag"],
                    reason="Restore the original X12 segment tag and delimiters exactly.",
                ))

        translated_element_counts = {
            item["tag"]: item["element_count"]
            for item in translated_x12_segments
        }
        for source_x12 in source_x12_segments:
            translated_count = translated_element_counts.get(source_x12["tag"])
            if translated_count is None or translated_count == source_x12["element_count"]:
                continue
            violations.append(self._build_violation(
                rule="edi_structure",
                severity="warn",
                segment_index=segment_index,
                source_segment=source_segment,
                translated_segment=translated_segment,
                expected_value=str(source_x12["element_count"]),
                actual_value=str(translated_count),
                message=(
                    f"X12 tag '{source_x12['tag']}' in segment {segment_index} has "
                    f"{translated_count} element separator(s) after translation instead of {source_x12['element_count']}."
                ),
            ))

    def _validate_translated_npis(
        self,
        translated_text: str,
        source_text: str,
        violations: list[dict],
        suggestions: list[dict],
    ) -> None:
        if "npi" not in source_text.lower() and not extract_npi_tokens(source_text):
            return

        for match in re.finditer(r"\b\d{10}\b", translated_text):
            token = match.group(0)
            if is_valid_npi(token):
                continue
            violations.append({
                "rule": "hipaa_npi_format",
                "severity": "fail",
                "segment_index": None,
                "location": "translated_text",
                "expected_value": "10-digit Luhn-valid NPI",
                "actual_value": token,
                "message": f"Translated output contains '{token}', which is 10 digits but not a Luhn-valid NPI.",
                "source_excerpt": None,
                "translated_excerpt": clip_text(translated_text),
            })
            suggestions.append({
                "action": "review_npi",
                "segment_index": None,
                "incorrect_value": token,
                "corrected_value": None,
                "reason": "Replace the invalid NPI with the original 10-digit Luhn-valid value from the source document.",
            })

    def _build_violation(
        self,
        rule: str,
        severity: str,
        segment_index: int,
        source_segment: str,
        translated_segment: str,
        expected_value: str | None,
        message: str,
        actual_value: str | None = None,
    ) -> dict:
        return {
            "rule": rule,
            "severity": severity,
            "segment_index": segment_index,
            "location": f"segment_{segment_index}",
            "expected_value": expected_value,
            "actual_value": actual_value,
            "message": message,
            "source_excerpt": clip_text(source_segment),
            "translated_excerpt": clip_text(translated_segment),
        }

    def _build_suggestion(
        self,
        action: str,
        segment_index: int | None,
        incorrect_value: str | None,
        corrected_value: str | None,
        reason: str,
    ) -> dict:
        return {
            "action": action,
            "segment_index": segment_index,
            "incorrect_value": incorrect_value,
            "corrected_value": corrected_value,
            "reason": reason,
        }
