# nodes/phi_restore.py — update restore() to match new PHIMASK format

import re
from nodes.base import BaseNode

# phi_restore.py — update the sentinel pattern only, nothing else changes
PLACEHOLDER_RE = re.compile(r"PHIMASK_[A-Z]+_\d+")


class PHIRestoreNode(BaseNode):

    async def run(self, context: dict) -> dict:
        phi_map: dict[str, str] = context.get("phi_map", {})

        if not phi_map:
            print("ℹ️  PHIRestoreNode: no phi_map, nothing to restore")
            return context

        translated_text: str = context.get("translated_text", "")
        segment_translations: dict = context.get("segment_translations", {})

        print(f"DEBUG phi_restore: {len(phi_map)} entries: {list(phi_map.keys())}")

        def restore(text: str) -> str:
            # PHIMASK_N_END format is LLM-safe — exact match always works
            for placeholder, original in phi_map.items():
                text = text.replace(placeholder, original)
            return text

        restored_translated_text = restore(translated_text)
        restored_segment_translations = {
            seg: restore(trans)
            for seg, trans in segment_translations.items()
        }

        remaining = sum(
            1 for trans in restored_segment_translations.values()
            if PLACEHOLDER_RE.search(trans)
        )
        if remaining:
            print(f"⚠️  PHIRestoreNode: {remaining} segments still have unrestored placeholders")
        else:
            print(f"✅ PHIRestoreNode: all {len(phi_map)} values restored")

        return {
            **context,
            "translated_text": restored_translated_text,
            "segment_translations": restored_segment_translations,
            "phi_map": phi_map,
        }