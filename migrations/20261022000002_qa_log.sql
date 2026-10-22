-- Every question answered by the bot: for rate limits, the daily budget, feedback and future eval sets
CREATE TABLE qa_log (
    id                 BIGSERIAL PRIMARY KEY,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    guild_id           TEXT,
    channel_id         TEXT,
    user_id            TEXT NOT NULL,
    mode               TEXT NOT NULL,              -- player | dev
    lang               TEXT NOT NULL,
    question           TEXT NOT NULL,
    answer             TEXT,
    sources            JSONB,
    tool_calls         JSONB,
    model              TEXT,
    outcome            TEXT,                       -- end_turn | refusal | max_tokens | max_steps | error
    input_tokens       BIGINT,
    output_tokens      BIGINT,
    cache_read_tokens  BIGINT,
    cache_write_tokens BIGINT,
    cost_usd           DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms         INT,
    error              TEXT,
    feedback           SMALLINT,                   -- 1 = 👍, -1 = 👎
    feedback_by        TEXT
);
CREATE INDEX qa_log_created_idx ON qa_log (created_at);
CREATE INDEX qa_log_user_idx ON qa_log (user_id, created_at);
