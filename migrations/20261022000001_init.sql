CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Free-form key/value store (e.g. which embedding model produced chunks.embedding)
CREATE TABLE kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One indexed text unit: an entity card, a help page, a rule group, a code symbol, a summary, a topic dossier
CREATE TABLE documents (
    id           BIGSERIAL PRIMARY KEY,
    key          TEXT NOT NULL UNIQUE,              -- stable id, e.g. 'card:item:water_#00:fr'
    kind         TEXT NOT NULL,                     -- entity_card | help | rule | code | summary | topic
    audience     TEXT NOT NULL DEFAULT 'both',      -- player | dev | both
    lang         TEXT,                              -- en | fr | de | es, NULL when language-neutral
    title        TEXT NOT NULL,
    entity_id    TEXT,
    path         TEXT,
    symbol       TEXT,
    line_start   INT,
    line_end     INT,
    git_sha      TEXT,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    meta         JSONB NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX documents_kind_idx ON documents (kind);
CREATE INDEX documents_entity_idx ON documents (entity_id);

-- Embedding dimension depends on the configured model, so the column is untyped and the
-- HNSW index is created at runtime on (embedding::vector(<dims>)) — see kb_core::store::ensure_vector_index
CREATE TABLE chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ord         INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    UNIQUE (document_id, ord)
);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_pending_idx ON chunks (id) WHERE embedding IS NULL;

-- Structured game content from the fixture export
CREATE TABLE entities (
    id   TEXT PRIMARY KEY,                          -- '<kind>:<fixture key>', e.g. 'item:water_#00'
    kind TEXT NOT NULL,
    data JSONB NOT NULL
);

CREATE TABLE entity_names (
    entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    name      TEXT NOT NULL,
    PRIMARY KEY (entity_id, lang, name)
);
CREATE INDEX entity_names_trgm_idx ON entity_names USING gin (lower(name) gin_trgm_ops);

CREATE TABLE edges (
    src    TEXT NOT NULL,
    rel    TEXT NOT NULL,
    dst    TEXT NOT NULL,
    weight DOUBLE PRECISION,
    PRIMARY KEY (src, rel, dst)
);
CREATE INDEX edges_dst_idx ON edges (dst);
