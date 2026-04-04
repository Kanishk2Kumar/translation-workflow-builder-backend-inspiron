import io
import copy
from nodes.base import BaseNode
import docx

class DocumentRebuilderNode(BaseNode):
    """
    Writes translated_text back into each DocumentBlock's live object.
    Produces a bytes payload of the translated document.
    DOCX: mutates python-docx paragraph runs in place.
    PDF:  uses reportlab to stamp translated text over the original.
    """

    async def run(self, context: dict) -> dict:
        blocks: list = context.get("document_blocks", [])
        segment_translations: dict = context.get("segment_translations", {})
        filename: str = context.get("source_filename", "")

        # Loud diagnostics — remove after confirming fix
        print(f"DEBUG rebuilder: filename={filename}")
        print(f"DEBUG rebuilder: blocks={len(blocks)}, translations={len(segment_translations)}")
        print(f"DEBUG rebuilder: file_bytes_in_context={bool(context.get('file_bytes'))}")
        if blocks:
            print(f"DEBUG rebuilder: first block has _docx_doc={bool(blocks[0].metadata.get('_docx_doc'))}")

        if not blocks:
            print("⚠️  DocumentRebuilderNode: no document_blocks in context — skipping")
            return context

        if not segment_translations:
            print("⚠️  DocumentRebuilderNode: no segment_translations in context — skipping")
            return context

        if not filename.endswith((".docx", ".pdf")):
            print(f"⚠️  DocumentRebuilderNode: unsupported filename '{filename}' — skipping")
            return context

        # Write translations back into blocks
        for block in blocks:
            translated = segment_translations.get(block.source_text)
            if translated:
                block.translated_text = translated

        if filename.endswith(".docx"):
            output_bytes = self._rebuild_docx(blocks)
            output_format = "docx"
        elif filename.endswith(".pdf"):
            output_bytes = self._rebuild_pdf(blocks, context)
            output_format = "pdf"
        else:
            return context

        print(f"✅ DocumentRebuilderNode: produced {len(output_bytes)} bytes as {output_format}")

        return {
            **context,
            "output_document_bytes": output_bytes,
            "output_document_format": output_format,
        }

    # ── DOCX rebuild ─────────────────────────────────────────────────────────

    def _rebuild_docx(self, blocks: list) -> bytes:
        from lxml import etree

        doc = blocks[0].metadata.get("_docx_doc")
        if not doc:
            raise ValueError("No _docx_doc reference found in blocks")

        # Word XML namespace
        W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

        for block in blocks:
            if not block.translated_text or not block.block_ref:
                continue

            para = block.block_ref  # python-docx Paragraph

            # Find all <w:r> run elements that contain <w:t> text nodes
            # Runs with <w:drawing> or <w:pict> are images — never touch them
            text_runs = [
                r for r in para._element.findall(f"{{{W}}}r")
                if r.find(f"{{{W}}}t") is not None
                and r.find(f"{{{W}}}drawing") is None
                and r.find(f"{{{W}}}pict") is None
            ]

            if not text_runs:
                # Paragraph has no text runs (maybe it's pure image) — skip entirely
                continue

            # Put the full translated text into the FIRST text run's <w:t>
            first_text_run = text_runs[0]
            t_elem = first_text_run.find(f"{{{W}}}t")
            if t_elem is not None:
                t_elem.text = block.translated_text
                # xml:space="preserve" prevents Word from trimming leading/trailing spaces
                t_elem.set(
                    "{http://www.w3.org/XML/1998/namespace}space",
                    "preserve"
                )

            # Clear <w:t> content from all OTHER text runs (they're now redundant)
            # but leave the run element itself so formatting/spacing isn't broken
            for run in text_runs[1:]:
                t_elem = run.find(f"{{{W}}}t")
                if t_elem is not None:
                    t_elem.text = ""

            # Drawing/image runs are never touched — they stay exactly where they are

        buffer = io.BytesIO()
        doc.save(buffer)
        return buffer.getvalue()

    # ── PDF rebuild ──────────────────────────────────────────────────────────

    def _rebuild_pdf(self, blocks: list, context: dict) -> bytes:
        """
        PDF rebuild strategy:
        1. Use pypdf to blank out original text layers per page
        2. Use reportlab to render translated text in the same positions
        3. Merge the two layers

        Note: full positional accuracy requires pdfplumber for bbox extraction.
        This implementation does a best-effort line-by-line overlay.
        For production, swap the extraction step to pdfplumber.
        """
        try:
            from reportlab.pdfgen import canvas as rl_canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import pypdf
        except ImportError:
            raise ImportError(
                "PDF rebuild requires: pip install reportlab pypdf"
            )

        # Get original PDF bytes from first block metadata
        original_bytes = next(
            (b.metadata.get("_pdf_bytes") for b in blocks if b.metadata.get("_pdf_bytes")),
            None
        )
        if not original_bytes:
            raise ValueError("Original PDF bytes not found in block metadata")

        # Group blocks by page
        pages: dict[int, list] = {}
        for block in blocks:
            page_num = block.metadata.get("page", 0)
            pages.setdefault(page_num, []).append(block)

        # Read original PDF to get page dimensions
        reader = pypdf.PdfReader(io.BytesIO(original_bytes))
        writer = pypdf.PdfWriter()

        for page_num, page in enumerate(reader.pages):
            page_width = float(page.mediabox.width)
            page_height = float(page.mediabox.height)
            page_blocks = pages.get(page_num, [])

            # Build a reportlab overlay with translated text
            overlay_buffer = io.BytesIO()
            c = rl_canvas.Canvas(overlay_buffer, pagesize=(page_width, page_height))

            # Try to register a Unicode font (Noto Sans covers most scripts)
            # Fall back to Helvetica if not available
            try:
                pdfmetrics.registerFont(TTFont("NotoSans", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"))
                font_name = "NotoSans"
            except Exception:
                font_name = "Helvetica"

            c.setFont(font_name, 10)

            # Simple layout: evenly space lines top-to-bottom
            # For production: use pdfplumber bboxes to place text at exact coordinates
            y_start = page_height - 50
            line_height = 14
            x_margin = 50

            translated_lines = [b.translated_text or b.source_text for b in page_blocks]
            for i, line in enumerate(translated_lines):
                y = y_start - (i * line_height)
                if y < 40:
                    break
                try:
                    c.drawString(x_margin, y, line)
                except Exception:
                    # Fallback: encode to latin-1 lossy if font can't handle char
                    c.drawString(x_margin, y, line.encode("latin-1", "replace").decode("latin-1"))

            c.save()

            # Merge: use original page as base, stamp overlay on top
            overlay_reader = pypdf.PdfReader(overlay_buffer)
            overlay_page = overlay_reader.pages[0]

            # Blank the original text by merging with a white rect is not possible
            # in pure pypdf — instead we merge overlay ON TOP of original.
            # For full text removal, use pikepdf:
            #   page.Resources.Font.clear()  # removes font resources → text invisible
            base_page = copy.copy(page)
            base_page.merge_page(overlay_page)
            writer.add_page(base_page)

        output_buffer = io.BytesIO()
        writer.write(output_buffer)
        return output_buffer.getvalue()