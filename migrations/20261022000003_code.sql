-- Snapshot of indexed source files: lets the bot quote exact lines without a MyHordes checkout
CREATE TABLE source_files (
    path         TEXT PRIMARY KEY,              -- relative to the MyHordes root, forward slashes
    content_hash TEXT NOT NULL,
    git_sha      TEXT,
    content      TEXT NOT NULL
);

-- Claude summaries per source file, reused while the file content is unchanged
CREATE TABLE summary_cache (
    path         TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    result       JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
