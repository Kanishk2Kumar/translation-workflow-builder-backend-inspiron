import hashlib
import json
from collections import OrderedDict

from nodes.base import BaseNode
from db import get_pool
from config import settings


# Lazy-load the embedding model so startup stays fast
_model = None
_EMBEDDING_CACHE_MAX = 500
_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
_redis_client = None
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_VECTOR_DIM = 384
EMBEDDING_CACHE_KEY_PREFIX = "embedding-cache:v1:"

def get_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded")
    return _model


def get_redis_client():
    global _redis_client
    if _redis_client is None and settings.REDIS_URL:
        try:
            import redis

            _redis_client = redis.Redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
        except Exception as exc:
            print(f"Redis cache unavailable, falling back to in-process cache: {exc}")
            _redis_client = False
    return _redis_client if _redis_client is not False else None


def _make_embedding_cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{EMBEDDING_CACHE_KEY_PREFIX}{digest}"


def _get_cached_embedding(text: str) -> list[float] | None:
    cached = _embedding_cache.get(text)
    if cached is not None:
        _embedding_cache.move_to_end(text)
        return cached

    client = get_redis_client()
    if client is None:
        return None

    try:
        raw = client.get(_make_embedding_cache_key(text))
        if not raw:
            return None
        embedding = json.loads(raw)
        _embedding_cache[text] = embedding
        _embedding_cache.move_to_end(text)
        while len(_embedding_cache) > _EMBEDDING_CACHE_MAX:
            _embedding_cache.popitem(last=False)
        return embedding
    except Exception as exc:
        print(f"Redis embedding cache read failed: {exc}")
        return None


def _store_cached_embedding(text: str, embedding: list[float]) -> None:
    _embedding_cache[text] = embedding
    _embedding_cache.move_to_end(text)
    while len(_embedding_cache) > _EMBEDDING_CACHE_MAX:
        _embedding_cache.popitem(last=False)

    client = get_redis_client()
    if client is None:
        return

    try:
        client.setex(
            _make_embedding_cache_key(text),
            settings.EMBEDDING_CACHE_TTL_SECONDS,
            json.dumps(embedding),
        )
    except Exception as exc:
        print(f"Redis embedding cache write failed: {exc}")


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    cached_results: dict[str, list[float]] = {}
    uncached_texts: list[str] = []
    uncached_seen: set[str] = set()

    for text in texts:
        cached = _get_cached_embedding(text)
        if cached is not None:
            cached_results[text] = cached
            continue
        if text not in uncached_seen:
            uncached_texts.append(text)
            uncached_seen.add(text)

    if uncached_texts:
        model = get_embedding_model()
        prefixed = [f"query: {text}" for text in uncached_texts]
        generated_embeddings = model.encode(prefixed, normalize_embeddings=True).tolist()

        for text, embedding in zip(uncached_texts, generated_embeddings):
            _store_cached_embedding(text, embedding)
            cached_results[text] = embedding

    return [cached_results[text] for text in texts]


def clear_runtime_caches() -> None:
    _embedding_cache.clear()
    client = get_redis_client()
    if client is None:
        return
    try:
        keys = list(client.scan_iter(f"{EMBEDDING_CACHE_KEY_PREFIX}*"))
        if keys:
            client.delete(*keys)
    except Exception as exc:
        print(f"Redis embedding cache clear failed: {exc}")


async def fetch_exact_matches(
    segments: list[str],
    target_language: str,
) -> dict[str, str]:
    if not segments:
        return {}

    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT DISTINCT ON (source_text)
            source_text,
            target_text
        FROM translation_memory
        WHERE target_language = $1
          AND source_text = ANY($2::text[])
        ORDER BY source_text, created_at DESC
        """,
        target_language,
        segments,
    )
    return {
        row["source_text"]: row["target_text"]
        for row in rows
    }


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
              AND embedding IS NOT NULL
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


def is_vector_dimension_mismatch_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "different vector dimensions" in message
        or ("vector dimensions" in message and "different" in message)
    )


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
        exact_only: bool = bool(
            context.get("tm_exact_only", False)
            or self.config.get("exact_only", False)
        )

        exact_match_map = await fetch_exact_matches(
            segments=segments,
            target_language=target_language,
        )

        unresolved_segments = [
            segment for segment in segments
            if segment not in exact_match_map
        ]
        candidate_map: dict[int, list[dict]] = {}
        unresolved_index_map: dict[str, int] = {
            segment: index
            for index, segment in enumerate(unresolved_segments)
        }
        if unresolved_segments and not exact_only:
            try:
                embeddings = embed(unresolved_segments)
                candidate_map = await fetch_rag_candidates(
                    segments=unresolved_segments,
                    embeddings=embeddings,
                    target_language=target_language,
                    top_k=top_k,
                )
            except Exception as exc:
                if is_vector_dimension_mismatch_error(exc):
                    print(
                        "RAGNode: vector dimension mismatch detected between runtime embeddings "
                        "and translation_memory. Falling back to exact-match-only TM for this run."
                    )
                    candidate_map = {}
                    exact_only = True
                else:
                    raise

        rag_matches = []
        stats = {"exact": 0, "fuzzy": 0, "new": 0}

        for segment in segments:
            exact_translation = exact_match_map.get(segment)
            if exact_translation is not None:
                stats["exact"] += 1
                rag_matches.append({
                    "segment": segment,
                    "match_type": "exact",
                    "best_score": 1.0,
                    "matches": [{
                        "source": segment,
                        "translation": exact_translation,
                        "score": 1.0,
                    }],
                })
                continue

            idx = unresolved_index_map.get(segment)
            match, match_type = build_rag_match(
                segment=segment,
                matches=candidate_map.get(idx, []) if idx is not None else [],
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
#   embedding vector(384),    -- multilingual-e5-small produces 384-dim vectors
#   created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
# );
#
# CREATE INDEX ON translation_memory
#   USING ivfflat (embedding vector_cosine_ops)
#   WITH (lists = 100);
