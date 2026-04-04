import re
from datetime import datetime

ICD10_RE = re.compile(r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b")
DATE_RE = re.compile(r"\b\d{8}\b")
X12_SEGMENT_RE = re.compile(r"\b([A-Z0-9]{2,3})\*[^~\r\n]*~")
SV1_CPT_RE = re.compile(r"\bSV1\*HC:([A-Z]?\d{4,5})\b", flags=re.IGNORECASE)
LABELED_CPT_RE = re.compile(
    r"\b(?:CPT|HCPCS|PROC(?:EDURE)?\s*CODE)\s*[:#-]?\s*([A-Z]?\d{4,5})\b",
    flags=re.IGNORECASE,
)
CLM_AMOUNT_RE = re.compile(r"\bCLM\*[^*\r\n~]+\*(\d+(?:\.\d{1,2})?)\b", flags=re.IGNORECASE)
SV1_AMOUNT_RE = re.compile(r"\bSV1\*[^*\r\n~]+\*(\d+(?:\.\d{1,2})?)\b", flags=re.IGNORECASE)
AMT_AMOUNT_RE = re.compile(r"\bAMT\*[^*\r\n~]+\*(\d+(?:\.\d{1,2})?)\b", flags=re.IGNORECASE)
LABELED_AMOUNT_RE = re.compile(
    r"\b(?:amount|charge|payment|paid|total)\b[^0-9]{0,8}(\d+(?:\.\d{1,2})?)",
    flags=re.IGNORECASE,
)

X12_CRITICAL_TAGS = {
    "ISA", "GS", "ST", "BHT", "NM1", "HL", "CLM", "SV1", "HI", "REF", "SE", "GE", "IEA",
}
AMOUNT_CONTEXT_HINTS = ("amount", "charge", "payment", "paid", "total", "$", "amt", "clm*", "sv1*")
DATE_CONTEXT_HINTS = ("date", "dob", "service date", "report date", "signed on", "timeline")
DATE_SEGMENT_TAGS = ("DTP*", "BHT*", "GS*")


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def clip_text(value: str, limit: int = 160) -> str:
    compact = normalize_whitespace(value)
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def is_valid_ccyymmdd(value: str) -> bool:
    if len(value) != 8 or not value.isdigit():
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def luhn_checksum(number: str) -> int:
    digits = [int(char) for char in number]
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10


def is_valid_npi(value: str) -> bool:
    return len(value) == 10 and value.isdigit() and luhn_checksum(f"80840{value}") == 0


def extract_npi_tokens(text: str) -> list[str]:
    seen: list[str] = []
    for match in re.finditer(r"\b\d{10}\b", text):
        token = match.group(0)
        if is_valid_npi(token) and token not in seen:
            seen.append(token)
    return seen


def extract_cpt_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for pattern in (SV1_CPT_RE, LABELED_CPT_RE):
        for match in pattern.finditer(text):
            token = match.group(1)
            if token not in tokens:
                tokens.append(token)
    return tokens


def should_validate_amounts(text: str) -> bool:
    lowered = text.lower()
    return any(hint in lowered for hint in AMOUNT_CONTEXT_HINTS) or bool(X12_SEGMENT_RE.search(text))


def should_validate_dates(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    return any(hint in lowered for hint in DATE_CONTEXT_HINTS) or stripped.startswith(DATE_SEGMENT_TAGS)


def extract_amount_tokens(text: str) -> list[str]:
    seen: list[str] = []
    for pattern in (CLM_AMOUNT_RE, SV1_AMOUNT_RE, AMT_AMOUNT_RE, LABELED_AMOUNT_RE):
        for match in pattern.finditer(text):
            token = match.group(1)
            if token not in seen:
                seen.append(token)
    return seen


def extract_x12_segments(text: str) -> list[dict]:
    segments: list[dict] = []
    for match in X12_SEGMENT_RE.finditer(text):
        segment = match.group(0)
        tag = match.group(1)
        segments.append({
            "raw": segment,
            "tag": tag,
            "element_count": segment.count("*"),
        })
    return segments


def is_x12_like_segment(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(X12_SEGMENT_RE.fullmatch(stripped))


def extract_date_tokens(text: str) -> list[str]:
    return [value for value in dict.fromkeys(DATE_RE.findall(text)) if is_valid_ccyymmdd(value)]


def extract_protected_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    token_groups = [
        list(dict.fromkeys(ICD10_RE.findall(text))),
        extract_cpt_tokens(text),
        extract_npi_tokens(text),
    ]
    if should_validate_dates(text):
        token_groups.append(extract_date_tokens(text))
    if should_validate_amounts(text):
        token_groups.append(extract_amount_tokens(text))

    for group in token_groups:
        for token in group:
            if token not in tokens:
                tokens.append(token)
    return tokens


def build_enforcement_plan(segments: list[str]) -> dict:
    rules = []
    total_tokens = 0
    skipped_segments = 0

    for segment in segments:
        skip_translation = is_x12_like_segment(segment)
        protected_tokens = [] if skip_translation else extract_protected_tokens(segment)
        if skip_translation:
            skipped_segments += 1
        total_tokens += len(protected_tokens)
        rules.append({
            "skip_translation": skip_translation,
            "protected_tokens": protected_tokens,
        })

    return {
        "segment_rules": rules,
        "summary": {
            "protected_segment_count": skipped_segments,
            "protected_token_count": total_tokens,
        },
    }


def ensure_enforcement_plan(context: dict) -> dict:
    if not context.get("compliance_enforcement_enabled"):
        return {"segment_rules": [], "summary": {}}

    original_segments = context.get("original_segments") or context.get("segments") or []
    plan = context.get("compliance_enforcement") or {}
    rules = plan.get("segment_rules", [])
    if len(rules) == len(original_segments):
        return plan
    return build_enforcement_plan(original_segments)


def protect_text_tokens(text: str, tokens: list[str]) -> tuple[str, dict[str, str]]:
    protected_text = text
    placeholder_map: dict[str, str] = {}

    for index, token in enumerate(sorted(set(tokens), key=len, reverse=True), start=1):
        if token not in protected_text:
            continue
        placeholder = f"COMPMASK_{index:04d}_END"
        protected_text = protected_text.replace(token, placeholder)
        placeholder_map[placeholder] = token

    return protected_text, placeholder_map


def restore_protected_text(text: str, placeholder_map: dict[str, str]) -> str:
    restored = text
    for placeholder, original in placeholder_map.items():
        restored = restored.replace(placeholder, original)
    return restored
