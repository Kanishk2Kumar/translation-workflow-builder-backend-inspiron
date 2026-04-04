import re

from openai import OpenAI

from nodes.base import BaseNode
from nodes.compliance_common import (
    ensure_enforcement_plan,
    protect_text_tokens,
    restore_protected_text,
)
from config import settings

_client: OpenAI | None = None
_SEGMENT_MARKER_RE = re.compile(
    r"<<<(?P<id>SEG_\d+)>>>\s*(?P<text>.*?)(?=(?:\n<<<SEG_\d+>>>|\Z))",
    flags=re.DOTALL,
)

LANGUAGE_NAMES = {
    "hi": "Hindi", "es": "Spanish", "fr": "French", "de": "German",
    "ja": "Japanese", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
}

TONE_INSTRUCTIONS = {
    "clinical": "Use formal clinical terminology. This text is for healthcare professionals.",
    "patient_friendly": "Use simple, clear language. This text is for patients with no medical background.",
    "formal": "Use formal language appropriate for official documents.",
    "technical": "Use precise technical language. Preserve all technical terms.",
}


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def build_system_prompt(target_language, tone, system_prompt, glossary_terms=None):
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])

    glossary_block = ""
    if glossary_terms:
        term_lines = "\n".join(
            f'  - "{t["source_term"]}" must always be translated as "{t["target_term"]}"'
            for t in glossary_terms[:30]
        )
        glossary_block = (
            "\n\nMANDATORY GLOSSARY - these translations are fixed and must not be altered:\n"
            f"{term_lines}\n"
            "Do not paraphrase, synonymize, or skip any of the above terms."
        )

    return system_prompt or (
        f"You are an expert translator specialising in English to {lang_name} translation. "
        f"{tone_instruction} "
        "Never translate ICD codes, CPT codes, MRN numbers, or passport numbers - keep them exactly as-is. "
        "Return only the translated text with no explanation or preamble."
        f"{glossary_block}"
    )


def build_prompt(source_text, target_language, tone, rag_matches, system_prompt, glossary_terms=None):
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    system = build_system_prompt(
        target_language=target_language,
        tone=tone,
        system_prompt=system_prompt,
        glossary_terms=glossary_terms,
    )

    examples_block = ""
    useful_matches = [m for m in rag_matches if m.get("match_type") == "fuzzy" and m.get("matches")]
    if useful_matches:
        examples = [
            f"Source: {m['matches'][0]['source']}\nTranslation ({lang_name}): {m['matches'][0]['translation']}"
            for m in useful_matches[:3]
        ]
        examples_block = "\n\nReference translations from approved memory:\n" + "\n\n".join(examples)

    user = (
        f"Translate the following text from English to {lang_name}."
        f"{examples_block}"
        f"\n\nText to translate:\n{source_text}"
    )

    return system, user


def build_batch_prompt(batch_items, target_language, tone, system_prompt, glossary_terms=None):
    lang_name = LANGUAGE_NAMES.get(target_language, target_language)
    system = build_system_prompt(
        target_language=target_language,
        tone=tone,
        system_prompt=system_prompt,
        glossary_terms=glossary_terms,
    )

    item_blocks = []
    for item in batch_items:
        reference_block = ""
        if item["match"].get("match_type") == "fuzzy" and item["match"].get("matches"):
            best_match = item["match"]["matches"][0]
            reference_block = (
                "\nReference translation from approved memory:\n"
                f"Source: {best_match['source']}\n"
                f"Translation ({lang_name}): {best_match['translation']}\n"
            )

        item_blocks.append(
            f"<<<{item['id']}>>>\n"
            f"Source text:\n{item.get('source_for_translation', item['segment'])}\n"
            f"{reference_block}"
        )

    user = (
        f"Translate each segment below from English to {lang_name}.\n"
        "Translate each segment independently and preserve the segment IDs.\n"
        "Return your answer using exactly this format for every segment and nothing else:\n"
        "<<<SEG_0001>>>\n<translation>\n<<<SEG_0002>>>\n<translation>\n\n"
        "Segments:\n\n"
        + "\n".join(item_blocks)
    )

    return system, user


def build_translation_batches(batch_items, max_batch_segments: int, max_batch_chars: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    current_batch: list[dict] = []
    current_chars = 0

    for item in batch_items:
        reference_chars = 0
        if item["match"].get("match_type") == "fuzzy" and item["match"].get("matches"):
            best_match = item["match"]["matches"][0]
            reference_chars = len(best_match["source"]) + len(best_match["translation"])

        item_chars = len(item["segment"]) + reference_chars
        would_overflow = current_batch and (
            len(current_batch) >= max_batch_segments
            or current_chars + item_chars > max_batch_chars
        )

        if would_overflow:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append(item)
        current_chars += item_chars

    if current_batch:
        batches.append(current_batch)

    return batches


def parse_batch_response(response_text: str) -> dict[str, str]:
    translations = {}
    for match in _SEGMENT_MARKER_RE.finditer(response_text.strip()):
        translations[match.group("id")] = match.group("text").strip()
    return translations


def restore_glossary(text: str, glossary_map: dict[str, str]) -> str:
    """
    Post-translation correction pass.
    1. If the source term still appears in translated text -> replace with target term
    2. No placeholder logic needed - terms go via prompt instructions
    """
    if not glossary_map:
        return text

    for source_term, target_term in glossary_map.items():
        pattern = r"(?<!\w)" + re.escape(source_term) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            text = re.sub(pattern, target_term, text, flags=re.IGNORECASE)
            print(f"glossary correction: '{source_term}' -> '{target_term}'")

    return text


def translate_single_segment(
    client: OpenAI,
    segment: str,
    match: dict,
    target_language: str,
    tone: str,
    model: str,
    max_tokens: int,
    system_prompt: str | None,
    glossary_terms: list,
    glossary_map: dict[str, str],
) -> tuple[str, int, int]:
    system, user = build_prompt(
        source_text=segment,
        target_language=target_language,
        tone=tone,
        rag_matches=[match],
        system_prompt=system_prompt,
        glossary_terms=glossary_terms,
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw_translation = response.choices[0].message.content.strip()
    print(f"DEBUG raw_translation: '{raw_translation[:100]}'")
    return (
        restore_glossary(raw_translation, glossary_map),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def translate_batch(
    client: OpenAI,
    batch_items: list[dict],
    target_language: str,
    tone: str,
    model: str,
    max_tokens: int,
    system_prompt: str | None,
    glossary_terms: list,
    glossary_map: dict[str, str],
) -> tuple[dict[str, str], int, int]:
    system, user = build_batch_prompt(
        batch_items=batch_items,
        target_language=target_language,
        tone=tone,
        system_prompt=system_prompt,
        glossary_terms=glossary_terms,
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    response_text = response.choices[0].message.content.strip()
    parsed = parse_batch_response(response_text)
    expected_ids = {item["id"] for item in batch_items}
    if set(parsed) != expected_ids:
        missing = sorted(expected_ids - set(parsed))
        raise ValueError(f"Batch response missing segment IDs: {missing}")

    translations = {
        item["segment"]: restore_glossary(parsed[item["id"]], glossary_map)
        for item in batch_items
    }
    preview = response_text[:100].replace("\n", " ")
    print(f"DEBUG raw_translation_batch: '{preview}'")
    return translations, response.usage.prompt_tokens, response.usage.completion_tokens


class LLMAgentNode(BaseNode):

    async def run(self, context: dict) -> dict:
        raw_text: str = context.get("raw_text", "")
        if not raw_text:
            raise ValueError("LLMAgentNode: no raw_text in context")

        target_language: str = context.get("target_language", self.config.get("target_language", "hi"))
        tone: str = self.config.get("tone", "formal")
        model: str = self.config.get("model", "gpt-4o")
        max_tokens: int = self.config.get("max_tokens", 2048)
        system_prompt: str | None = self.config.get("system_prompt")
        rag_matches: list = context.get("rag_matches", [])
        batch_max_segments: int = self.config.get("batch_max_segments", 12)
        batch_max_chars: int = self.config.get("batch_max_chars", 4000)
        segments: list[str] = context.get("segments", [])
        enforcement_plan = ensure_enforcement_plan(context)

        client = get_openai_client()
        total_input_tokens = 0
        total_output_tokens = 0
        glossary_terms = context.get("glossary_terms", [])
        glossary_map: dict[str, str] = context.get("glossary_map", {})

        print(f"DEBUG llm_agent: glossary_map has {len(glossary_map)} entries: {glossary_map}")

        if rag_matches and all(
            m.get("match_type") == "exact" and m.get("matches")
            for m in rag_matches
        ):
            segment_translations = {}
            rules = enforcement_plan.get("segment_rules", [])
            for index, m in enumerate(rag_matches):
                rule = rules[index] if index < len(rules) else {}
                if rule.get("skip_translation"):
                    segment_translations[m["segment"]] = m["segment"]
                    continue
                segment_translations[m["segment"]] = restore_glossary(
                    m["matches"][0]["translation"],
                    glossary_map,
                )
            translated_text = "\n".join(
                segment_translations[m["segment"]] for m in rag_matches
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
            batch_items = []
            rules = enforcement_plan.get("segment_rules", [])
            for index, match in enumerate(rag_matches, start=1):
                segment = match["segment"]
                rule = rules[index - 1] if index - 1 < len(rules) else {}
                if match.get("match_type") == "exact" and match.get("matches"):
                    if rule.get("skip_translation"):
                        segment_translations[segment] = segment
                    else:
                        segment_translations[segment] = restore_glossary(
                            match["matches"][0]["translation"],
                            glossary_map,
                        )
                    continue

                protected_source, placeholder_map = protect_text_tokens(
                    segment,
                    rule.get("protected_tokens", []),
                )
                batch_items.append({
                    "id": f"SEG_{index:04d}",
                    "segment": segment,
                    "source_for_translation": segment if rule.get("skip_translation") else protected_source,
                    "match": match,
                    "skip_translation": bool(rule.get("skip_translation")),
                    "placeholder_map": placeholder_map,
                })

            skipped_items = [item for item in batch_items if item["skip_translation"]]
            for item in skipped_items:
                segment_translations[item["segment"]] = item["segment"]

            translatable_items = [item for item in batch_items if not item["skip_translation"]]

            batches = build_translation_batches(
                batch_items=translatable_items,
                max_batch_segments=batch_max_segments,
                max_batch_chars=batch_max_chars,
            )

            for batch in batches:
                try:
                    batch_translations, prompt_tokens, completion_tokens = translate_batch(
                        client=client,
                        batch_items=batch,
                        target_language=target_language,
                        tone=tone,
                        model=model,
                        max_tokens=max_tokens,
                        system_prompt=system_prompt,
                        glossary_terms=glossary_terms,
                        glossary_map=glossary_map,
                    )
                    restored_batch = {
                        item["segment"]: restore_protected_text(
                            batch_translations[item["segment"]],
                            item["placeholder_map"],
                        )
                        for item in batch
                    }
                    restored_batch = {
                        key: restore_glossary(value, glossary_map)
                        for key, value in restored_batch.items()
                    }
                    segment_translations.update(restored_batch)
                    total_input_tokens += prompt_tokens
                    total_output_tokens += completion_tokens
                except Exception as e:
                    print(f"LLM batch failed, falling back to single segments: {e}")
                    for item in batch:
                        try:
                            translation, prompt_tokens, completion_tokens = translate_single_segment(
                                client=client,
                                segment=item["source_for_translation"],
                                match=item["match"],
                                target_language=target_language,
                                tone=tone,
                                model=model,
                                max_tokens=max_tokens,
                                system_prompt=system_prompt,
                                glossary_terms=glossary_terms,
                                glossary_map=glossary_map,
                            )
                            restored = restore_protected_text(translation, item["placeholder_map"])
                            segment_translations[item["segment"]] = restore_glossary(restored, glossary_map)
                            total_input_tokens += prompt_tokens
                            total_output_tokens += completion_tokens
                        except Exception as inner_e:
                            print(f"LLM failed for segment: {inner_e}")
                            segment_translations[item["segment"]] = restore_glossary(item["segment"], glossary_map)

            translated_text = "\n".join(
                segment_translations.get(m["segment"], m["segment"])
                for m in rag_matches
            )

        else:
            rules = enforcement_plan.get("segment_rules", [])
            if rules and rules[0].get("skip_translation"):
                translated_text = raw_text
                segment_translations = {}
                total_input_tokens = 0
                total_output_tokens = 0
                return {
                    **context,
                    "translated_text": translated_text,
                    "segment_translations": segment_translations,
                    "llm_model": model,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                    "tm_hit": False,
                }

            protected_source, placeholder_map = protect_text_tokens(
                raw_text,
                rules[0].get("protected_tokens", []) if rules else [],
            )
            system, user = build_prompt(
                source_text=protected_source,
                target_language=target_language,
                tone=tone,
                rag_matches=[],
                system_prompt=system_prompt,
                glossary_terms=glossary_terms,
            )
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            raw_translation = response.choices[0].message.content.strip()
            print(f"DEBUG raw_translation: '{raw_translation[:100]}'")
            translated_text = restore_glossary(
                restore_protected_text(raw_translation, placeholder_map),
                glossary_map,
            )
            segment_translations = {}
            total_input_tokens = response.usage.prompt_tokens
            total_output_tokens = response.usage.completion_tokens

        print(
            f"DEBUG llm: {len(segment_translations)} segments translated, "
            f"tokens in={total_input_tokens} out={total_output_tokens}"
        )
        print(
            f"DEBUG llm: rag_matches={len(rag_matches)}, "
            f"segment_translations={len(segment_translations)}, tm_hit={False}"
        )
        return {
            **context,
            "translated_text": translated_text,
            "segment_translations": segment_translations,
            "llm_model": model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "tm_hit": False,
        }
