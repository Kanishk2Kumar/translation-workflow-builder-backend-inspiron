import httpx

from config import settings
from nodes.base import BaseNode
from nodes.compliance_common import (
    ensure_enforcement_plan,
    protect_text_tokens,
    restore_protected_text,
)
from nodes.llm_agent import restore_glossary

_DEFAULT_AZURE_TRANSLATOR_ENDPOINT = "https://api.cognitive.microsofttranslator.com"
_client: httpx.AsyncClient | None = None


def get_azure_translator_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
    return _client


def get_azure_translator_endpoint() -> str:
    endpoint = settings.AZURE_TRANSLATOR_ENDPOINT or _DEFAULT_AZURE_TRANSLATOR_ENDPOINT
    return endpoint.rstrip("/")


def build_translation_batches(texts: list[str], max_items: int, max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_chars = 0

    for text in texts:
        text_chars = len(text)
        if current_batch and (
            len(current_batch) >= max_items or current_chars + text_chars > max_chars
        ):
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(text)
        current_chars += text_chars

    if current_batch:
        batches.append(current_batch)

    return batches


def build_translation_items(
    pending_segments: list[str],
    current_segments: list[str],
    enforcement_plan: dict,
) -> list[dict]:
    items: list[dict] = []
    rules = enforcement_plan.get("segment_rules", [])

    for segment in pending_segments:
        try:
            index = current_segments.index(segment)
        except ValueError:
            index = -1

        rule = rules[index] if 0 <= index < len(rules) else {}
        items.append({
            "index": index,
            "segment": segment,
            "skip_translation": bool(rule.get("skip_translation")),
            "protected_tokens": list(rule.get("protected_tokens", [])),
        })

    return items


async def translate_batch(
    texts: list[str],
    target_language: str,
    source_language: str | None,
    text_type: str,
    category: str | None,
) -> list[str]:
    if not settings.AZURE_TRANSLATOR_KEY:
        raise ValueError("AzureTranslateNode: AZURE_TRANSLATOR_KEY is not configured")

    client = get_azure_translator_client()
    params = {
        "api-version": "3.0",
        "to": target_language,
        "textType": text_type,
    }
    if source_language:
        params["from"] = source_language
    if category:
        params["category"] = category

    headers = {
        "Ocp-Apim-Subscription-Key": settings.AZURE_TRANSLATOR_KEY,
        "Content-Type": "application/json",
    }
    if settings.AZURE_TRANSLATOR_REGION:
        headers["Ocp-Apim-Subscription-Region"] = settings.AZURE_TRANSLATOR_REGION

    body = [{"Text": text} for text in texts]
    response = await client.post(
        f"{get_azure_translator_endpoint()}/translate",
        params=params,
        headers=headers,
        json=body,
    )
    response.raise_for_status()

    payload = response.json()
    if len(payload) != len(texts):
        raise ValueError(
            "AzureTranslateNode: translation count mismatch "
            f"(expected {len(texts)}, got {len(payload)})"
        )

    translations = []
    for item in payload:
        item_translations = item.get("translations", [])
        if not item_translations:
            raise ValueError("AzureTranslateNode: missing translations in API response")
        translations.append(item_translations[0].get("text", ""))

    return translations


class AzureTranslateNode(BaseNode):

    async def run(self, context: dict) -> dict:
        raw_text: str = context.get("raw_text", "")
        segments: list[str] = context.get("segments", [])
        rag_matches: list = context.get("rag_matches", [])

        if not raw_text and not segments and not rag_matches:
            raise ValueError("AzureTranslateNode: no text available in context")

        target_language: str = context.get("target_language", self.config.get("target_language", "hi"))
        source_language: str | None = context.get("source_language", self.config.get("source_language", "en"))
        batch_size: int = int(self.config.get("batch_size", 25))
        max_batch_chars: int = int(self.config.get("max_batch_chars", 5000))
        text_type: str = self.config.get("text_type", "plain")
        category: str | None = self.config.get("category")
        glossary_map: dict[str, str] = context.get("glossary_map", {})
        current_segments: list[str] = context.get("segments", [])
        enforcement_plan = ensure_enforcement_plan(context)

        if rag_matches and all(
            match.get("match_type") == "exact" and match.get("matches")
            for match in rag_matches
        ):
            translation_items = build_translation_items(
                pending_segments=[match["segment"] for match in rag_matches],
                current_segments=current_segments,
                enforcement_plan=enforcement_plan,
            )
            segment_translations = {}
            for match, item in zip(rag_matches, translation_items):
                if item["skip_translation"]:
                    segment_translations[match["segment"]] = match["segment"]
                    continue
                segment_translations[match["segment"]] = restore_glossary(
                    match["matches"][0]["translation"],
                    glossary_map,
                )
            translated_text = "\n".join(
                segment_translations[match["segment"]] for match in rag_matches
            )
            return {
                **context,
                "translated_text": translated_text,
                "segment_translations": segment_translations,
                "llm_model": "translation_memory",
                "input_tokens": 0,
                "output_tokens": 0,
                "tm_hit": True,
            }

        segment_translations: dict[str, str] = {}

        if rag_matches:
            ordered_segments = [match["segment"] for match in rag_matches]
            pending_segments = []
            for match in rag_matches:
                segment = match["segment"]
                if match.get("match_type") == "exact" and match.get("matches"):
                    item = build_translation_items(
                        pending_segments=[segment],
                        current_segments=current_segments,
                        enforcement_plan=enforcement_plan,
                    )[0]
                    if item["skip_translation"]:
                        segment_translations[segment] = segment
                    else:
                        segment_translations[segment] = restore_glossary(
                            match["matches"][0]["translation"],
                            glossary_map,
                        )
                else:
                    pending_segments.append(segment)
        elif segments:
            ordered_segments = segments
            pending_segments = segments
        else:
            ordered_segments = [raw_text]
            pending_segments = [raw_text]

        translation_items = build_translation_items(
            pending_segments=pending_segments,
            current_segments=current_segments,
            enforcement_plan=enforcement_plan,
        )

        batchable_items = []
        for item in translation_items:
            if item["skip_translation"]:
                segment_translations[item["segment"]] = item["segment"]
            else:
                batchable_items.append(item)

        for batch in build_translation_batches(
            texts=[item["segment"] for item in batchable_items],
            max_items=batch_size,
            max_chars=max_batch_chars,
        ):
            batch_items = []
            for segment in batch:
                item = next(candidate for candidate in batchable_items if candidate["segment"] == segment)
                protected_text, placeholder_map = protect_text_tokens(segment, item["protected_tokens"])
                batch_items.append({
                    **item,
                    "source_text": protected_text,
                    "placeholder_map": placeholder_map,
                })
            translated_batch = await translate_batch(
                texts=[item["source_text"] for item in batch_items],
                target_language=target_language,
                source_language=source_language,
                text_type=text_type,
                category=category,
            )
            for item, translated in zip(batch_items, translated_batch):
                restored = restore_protected_text(translated, item["placeholder_map"])
                print(f"DEBUG azure_translation: '{translated[:100]}'")
                segment_translations[item["segment"]] = restore_glossary(restored, glossary_map)

        translated_text = "\n".join(
            segment_translations.get(segment, segment)
            for segment in ordered_segments
        )

        return {
            **context,
            "translated_text": translated_text,
            "segment_translations": segment_translations,
            "llm_model": "azure_translator",
            "input_tokens": 0,
            "output_tokens": 0,
            "tm_hit": False,
        }


GoogleTranslateNode = AzureTranslateNode
