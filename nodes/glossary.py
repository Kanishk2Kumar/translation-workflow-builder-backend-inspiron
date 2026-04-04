from nodes.base import BaseNode
from db import get_pool


class GlossaryNode(BaseNode):

    async def run(self, context: dict) -> dict:
        user_id: str = context.get("user_id", "")
        target_language: str = context.get("target_language", "hi")
        source_lang: str = context.get("source_language", "en")

        if not user_id:
            print("GlossaryNode: no user_id - skipping")
            return {**context, "glossary_terms": [], "glossary_map": {}}

        pool = get_pool()
        rows = await pool.fetch(
            """
            SELECT source_term, target_term, case_sensitive, domain
            FROM glossary_terms
            WHERE user_id = $1
              AND source_lang = $2
              AND target_lang = $3
            ORDER BY length(source_term) DESC
            """,
            user_id,
            source_lang,
            target_language,
        )

        if not rows:
            print(f"GlossaryNode: no terms for user {user_id}")
            return {**context, "glossary_terms": [], "glossary_map": {}}

        glossary_terms = [dict(r) for r in rows]

        # glossary_map: source_term -> target_term (used for post-translation correction)
        glossary_map: dict[str, str] = {
            r["source_term"]: r["target_term"] for r in glossary_terms
        }

        print(f"GlossaryNode: {len(glossary_terms)} terms loaded")
        print(f"DEBUG glossary_map: {glossary_map}")

        # No segment mutation. Terms are passed via prompt instructions and corrected post-translation.
        return {
            **context,
            "glossary_terms": glossary_terms,
            "glossary_map": glossary_map,
        }
