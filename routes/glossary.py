# routes/glossary.py — new file
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from db import get_pool

router = APIRouter(prefix="/glossary", tags=["glossary"])


class GlossaryTermIn(BaseModel):
    source_term: str
    source_lang: str = "en"
    target_term: str
    target_lang: str
    domain: str | None = None
    case_sensitive: bool = False


@router.post("/{user_id}/terms")
async def add_term(user_id: str, term: GlossaryTermIn):
    pool = get_pool()
    try:
        row = await pool.fetchrow(
            """
            INSERT INTO glossary_terms
              (user_id, source_term, source_lang, target_term, target_lang, domain, case_sensitive)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (user_id, source_term, source_lang, target_lang)
            DO UPDATE SET
              target_term = EXCLUDED.target_term,
              domain = EXCLUDED.domain,
              updated_at = CURRENT_TIMESTAMP
            RETURNING id, source_term, target_term
            """,
            user_id, term.source_term, term.source_lang,
            term.target_term, term.target_lang, term.domain, term.case_sensitive,
        )
        return {"id": str(row["id"]), "source_term": row["source_term"], "target_term": row["target_term"]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{user_id}/terms")
async def list_terms(user_id: str):
    pool = get_pool()
    query = "SELECT * FROM glossary_terms WHERE user_id = $1"
    args = [user_id]
    rows = await pool.fetch(query, *args)
    return [dict(r) for r in rows]


@router.delete("/{user_id}/terms/{term_id}")
async def delete_term(user_id: str, term_id: str):
    pool = get_pool()
    result = await pool.execute(
        "DELETE FROM glossary_terms WHERE id = $1 AND user_id = $2",
        term_id, user_id,
    )
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Term not found")
    return {"deleted": term_id}