from nodes.base import BaseNode
from db import get_pool


# Lazy-load the embedding model so startup stays fast
_model = None


def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/multilingual-e5-large")
        print("Embedding model loaded")
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    # multilingual-e5 expects "query: " prefix for queries
    prefixed = [f"query: {t}" for t in texts]
    embeddings = model.encode(prefixed, normalize_embeddings=True)
    return embeddings.tolist()


async def fetch_rag_candidates(
    segments: list[str],
    embeddings: list[list[float]],
    target_language: str,
    top_k: int,
) -> dict[int, list[dict]]:
    pool = get_pool()
    embedding_texts = [str(embedding) for embedding in embeddings]

    rows = await pool.fetch(
        """
        WITH input_segments AS (
            SELECT segment, embedding_text, ord::int AS ord
            FROM unnest($1::text[], $2::text[]) WITH ORDINALITY
              AS t(segment, embedding_text, ord)
        )
        SELECT
            i.ord,
            i.segment,
            tm.source_text,
            tm.target_text,
            CASE
                WHEN tm.source_text IS NULL THEN NULL
                ELSE 1 - (tm.embedding <=> i.embedding_text::vector)
            END AS similarity
        FROM input_segments i
        LEFT JOIN LATERAL (
            SELECT source_text, target_text, embedding
            FROM translation_memory
            WHERE target_language = $3
            ORDER BY embedding <=> i.embedding_text::vector
            LIMIT $4
        ) tm ON TRUE
        ORDER BY i.ord, similarity DESC NULLS LAST
        """,
        segments,
        embedding_texts,
        target_language,
        top_k,
    )

    grouped: dict[int, list[dict]] = {idx: [] for idx in range(len(segments))}
    for row in rows:
        idx = int(row["ord"]) - 1
        if row["source_text"] is None:
            continue
        grouped[idx].append({
            "source": row["source_text"],
            "translation": row["target_text"],
            "score": round(float(row["similarity"]), 4),
        })

    return grouped


def build_rag_match(segment: str, matches: list[dict], exact_threshold: float, fuzzy_threshold: float) -> tuple[dict, str]:
    if not matches:
        return {
            "segment": segment,
            "match_type": "new",
            "matches": [],
        }, "new"

    best_score = float(matches[0]["score"])
    if best_score >= exact_threshold:
        match_type = "exact"
    elif best_score >= fuzzy_threshold:
        match_type = "fuzzy"
    else:
        match_type = "new"

    return {
        "segment": segment,
        "match_type": match_type,
        "best_score": round(best_score, 4),
        "matches": matches,
    }, match_type


class RAGNode(BaseNode):
    """
    Queries the translation memory (pgvector) for each segment.
    Classifies matches as:
      exact  - cosine similarity >= exact_threshold  (auto-fill)
      fuzzy  - cosine similarity >= fuzzy_threshold  (suggestion)
      new    - below fuzzy threshold (send to LLM)
    """

    async def run(self, context: dict) -> dict:
        segments: list[str] = context.get("segments", [])
        if not segments:
            return {**context, "rag_matches": [], "rag_stats": {}}

        exact_threshold: float = self.config.get("exact_threshold", 1.0)
        fuzzy_threshold: float = self.config.get("fuzzy_threshold", 0.75)
        top_k: int = self.config.get("top_k", 5)
        target_language: str = context.get("target_language", "hi")

        embeddings = embed(segments)
        candidate_map = await fetch_rag_candidates(
            segments=segments,
            embeddings=embeddings,
            target_language=target_language,
            top_k=top_k,
        )

        rag_matches = []
        stats = {"exact": 0, "fuzzy": 0, "new": 0}

        for idx, segment in enumerate(segments):
            match, match_type = build_rag_match(
                segment=segment,
                matches=candidate_map.get(idx, []),
                exact_threshold=exact_threshold,
                fuzzy_threshold=fuzzy_threshold,
            )
            stats[match_type] += 1
            rag_matches.append(match)

        return {
            **context,
            "rag_matches": rag_matches,
            "rag_stats": stats,
        }


# SQL to create the translation_memory table
#
# Run this once in your PostgreSQL instance:
#
# CREATE EXTENSION IF NOT EXISTS vector;
#
# CREATE TABLE translation_memory (
#   id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
#   source_text TEXT NOT NULL,
#   target_text TEXT NOT NULL,
#   source_language TEXT NOT NULL DEFAULT 'en',
#   target_language TEXT NOT NULL,
#   embedding vector(1024),   -- multilingual-e5-large produces 1024-dim vectors
#   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX ON translation_memory
#   USING ivfflat (embedding vector_cosine_ops)
#   WITH (lists = 100);
