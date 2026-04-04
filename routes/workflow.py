import hashlib
import json
import uuid
from datetime import datetime
import base64
import os

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from db import get_pool
from executor import execute_workflow
from nodes.output import seed_translation_memory

router = APIRouter(prefix="/workflow", tags=["workflow"])


class RunWorkflowResponse(BaseModel):
    execution_id: str
    status: str
    output: dict
    logs: list
    cache_hit: bool = False


# ─── Text extraction ──────────────────────────────────────────────────────────

def extract_text(filename: str, contents: bytes) -> str:
    if filename.endswith(".txt"):
        return contents.decode("utf-8", errors="ignore")

    if filename.endswith(".pdf"):
        try:
            import io
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(contents))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="PDF support requires pypdf. Run: pip install pypdf",
            )

    if filename.endswith(".docx"):
        try:
            import io
            import docx
            doc = docx.Document(io.BytesIO(contents))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise HTTPException(
                status_code=400,
                detail="DOCX support requires python-docx. Run: pip install python-docx",
            )

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: {filename}. Supported: .txt, .pdf, .docx",
    )


# ─── Document hash helpers ────────────────────────────────────────────────────

def compute_hash(contents: bytes, target_language: str) -> str:
    """
    Hash = SHA256(file bytes + target language).
    Same file in different target languages = different cache entries.
    """
    h = hashlib.sha256()
    h.update(contents)
    h.update(target_language.encode())
    return h.hexdigest()


async def get_cached_execution(
    pool, document_hash: str, workflow_id: str
) -> dict | None:
    """
    Look up a previous successful execution with the same document hash
    for the same workflow. Returns the output payload or None.
    """
    row = await pool.fetchrow(
        """
        SELECT id, output
        FROM executions
        WHERE workflow_id = $1
          AND status = 'success'
          AND input::jsonb->>'document_hash' = $2
        ORDER BY started_at DESC
        LIMIT 1
        """,
        workflow_id,
        document_hash,
    )
    if not row or not row["output"]:
        return None
    return {
        "execution_id": str(row["id"]),
        "output": json.loads(row["output"]) if isinstance(row["output"], str) else row["output"],
    }


# ─── POST /workflow/{workflow_id}/run ─────────────────────────────────────────

@router.post("/{workflow_id}/run", response_model=RunWorkflowResponse)
async def run_workflow(
    workflow_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form(default="hi"),
):
    pool = get_pool()
    contents = await file.read()

    document_hash = compute_hash(contents, target_language)

    cached = await get_cached_execution(pool, document_hash, workflow_id)
    if cached:
        return RunWorkflowResponse(
            execution_id=cached["execution_id"],
            status="success",
            output=cached["output"],
            logs=[],
            cache_hit=True,
        )

    raw_text = extract_text(file.filename, contents)

    row = await pool.fetchrow(
        "SELECT id, nodes, edges FROM workflows WHERE id = $1", workflow_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")

    nodes = json.loads(row["nodes"]) if isinstance(row["nodes"], str) else row["nodes"]
    edges = json.loads(row["edges"]) if isinstance(row["edges"], str) else row["edges"]

    if not nodes:
        raise HTTPException(status_code=400, detail="Workflow has no nodes")

    execution_id = str(uuid.uuid4())
    await pool.execute(
        """
        INSERT INTO executions (id, workflow_id, status, input, started_at)
        VALUES ($1, $2, 'running', $3, $4)
        """,
        execution_id,
        workflow_id,
        json.dumps({
            "filename": file.filename,
            "target_language": target_language,
            "document_hash": document_hash,
        }),
        datetime.utcnow(),
    )

    initial_context = {
        "raw_text": raw_text,
        "file_bytes": contents,
        "target_language": target_language,
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "source_filename": file.filename,
        "document_hash": document_hash,
        "user_id": '37720c15-ff75-49eb-a538-b25fd2273d30',           # ← add this (from auth header once JWT is wired)
    }

    try:
        final_context = await execute_workflow(
            nodes=nodes,
            edges=edges,
            initial_context=initial_context,
        )
    except Exception as e:
        await pool.execute(
            """
            UPDATE executions
            SET status = 'failed', logs = $1, completed_at = $2
            WHERE id = $3
            """,
            json.dumps([{"error": str(e)}]),
            datetime.utcnow(),
            execution_id,
        )
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")

    final_output = final_context.get("final_output", {})
    logs = final_context.get("_logs", [])
    tm_seed_payload = final_context.get("tm_seed_payload")

    if tm_seed_payload:
        background_tasks.add_task(seed_translation_memory, tm_seed_payload)

    # Attach b64 directly to run response (not stored in DB)
    output_doc_bytes = final_context.get("output_document_bytes")
    if output_doc_bytes:
        final_output["document_b64"] = base64.b64encode(output_doc_bytes).decode("utf-8")

    return RunWorkflowResponse(
        execution_id=execution_id,
        status="success",
        output=final_output,
        logs=logs,
        cache_hit=False,
    )

# ─── GET /execution/{execution_id}/status ────────────────────────────────────

@router.get("/{workflow_id}/execution/{execution_id}/download")
async def download_translated_document(workflow_id: str, execution_id: str):
    import re
    from fastapi.responses import Response

    pool = get_pool()
    row = await pool.fetchrow(
        "SELECT output FROM executions WHERE id = $1 AND workflow_id = $2",
        execution_id, workflow_id,
    )
    if not row or not row["output"]:
        raise HTTPException(status_code=404, detail="Execution not found")

    output = json.loads(row["output"]) if isinstance(row["output"], str) else row["output"]

    doc_path: str | None = output.get("document_path")
    doc_format: str | None = output.get("document_format")
    phi_map: dict = output.get("phi_map", {})

    if not doc_path or not doc_format:
        raise HTTPException(
            status_code=404,
            detail="No document output for this execution — was a DOCX or PDF uploaded?"
        )

    if not os.path.exists(doc_path):
        raise HTTPException(
            status_code=410,
            detail="Translated document has expired from temporary storage"
        )

    with open(doc_path, "rb") as f:
        file_bytes = f.read()

    # ── Safety net: re-apply PHI restore directly to document bytes ──────────
    if phi_map:
        if doc_format == "docx":
            file_bytes = _restore_phi_in_docx(file_bytes, phi_map)
        # PDF restore can be added later

    content_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf":  "application/pdf",
    }

    return Response(
        content=file_bytes,
        media_type=content_types.get(doc_format, "application/octet-stream"),
        headers={
            "Content-Disposition": f"attachment; filename=translated.{doc_format}",
            "Content-Length": str(len(file_bytes)),
        },
    )


def _restore_phi_in_docx(file_bytes: bytes, phi_map: dict) -> bytes:
    import io
    import docx

    if not phi_map:
        return file_bytes

    doc = docx.Document(io.BytesIO(file_bytes))
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    def restore_text(text: str) -> str:
        for placeholder, original in phi_map.items():
            text = text.replace(placeholder, original)
        return text

    for para in doc.paragraphs:
        for run in para.runs:
            if run.text and "PHIMASK" in run.text:
                run.text = restore_text(run.text)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        if run.text and "PHIMASK" in run.text:
                            run.text = restore_text(run.text)

    # Direct XML walk catches placeholders split across <w:t> elements
    for elem in doc.element.iter(f"{{{W}}}t"):
        if elem.text and "PHIMASK" in elem.text:
            elem.text = restore_text(elem.text)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()
