CREATE EXTENSION IF NOT EXISTS vector;

DROP INDEX IF EXISTS translation_memory_embedding_idx;

TRUNCATE TABLE translation_memory;

ALTER TABLE translation_memory
DROP COLUMN IF EXISTS embedding;

ALTER TABLE translation_memory
ADD COLUMN embedding vector(384);

CREATE INDEX IF NOT EXISTS translation_memory_embedding_idx
ON translation_memory
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
