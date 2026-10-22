use std::collections::HashSet;

use anyhow::{Result, bail};
use pgvector::Vector;
use serde_json::Value;
use sqlx::{AssertSqlSafe, PgPool, Row};

use crate::db;
use crate::embed::Embedder;
use crate::text::{chunk, sha256_hex};

/// Max characters per chunk. E5 models truncate at 512 tokens; ~1800 chars of mixed
/// German/French/English stays below that.
pub const CHUNK_CHARS: usize = 1800;

#[derive(Debug, Clone, Default)]
pub struct NewDoc {
    pub key: String,
    pub kind: String,
    pub audience: String,
    pub lang: Option<String>,
    pub title: String,
    pub entity_id: Option<String>,
    pub path: Option<String>,
    pub symbol: Option<String>,
    pub line_start: Option<i32>,
    pub line_end: Option<i32>,
    pub git_sha: Option<String>,
    pub content: String,
    pub meta: Value,
}

#[derive(Debug, Default)]
pub struct UpsertStats {
    pub inserted: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub chunks: usize,
}

/// Inserts or updates documents by `key`. Documents whose title+content did not change keep their
/// chunks and embeddings; changed ones get fresh chunks with `embedding = NULL` (see [`embed_pending`]).
pub async fn upsert_docs(pool: &PgPool, docs: &[NewDoc]) -> Result<UpsertStats> {
    let mut stats = UpsertStats::default();
    let mut tx = pool.begin().await?;

    for doc in docs {
        let hash = sha256_hex(&format!("{}\n{}", doc.title, doc.content));
        let existing: Option<(i64, String)> = sqlx::query_as("SELECT id, content_hash FROM documents WHERE key = $1")
            .bind(&doc.key)
            .fetch_optional(&mut *tx)
            .await?;
        if let Some((_, old_hash)) = &existing
            && *old_hash == hash
        {
            stats.unchanged += 1;
            continue;
        }

        let id: i64 = sqlx::query_scalar(
            "INSERT INTO documents (key, kind, audience, lang, title, entity_id, path, symbol, line_start, line_end, git_sha, content, content_hash, meta)
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
             ON CONFLICT (key) DO UPDATE SET
               kind = EXCLUDED.kind, audience = EXCLUDED.audience, lang = EXCLUDED.lang, title = EXCLUDED.title,
               entity_id = EXCLUDED.entity_id, path = EXCLUDED.path, symbol = EXCLUDED.symbol,
               line_start = EXCLUDED.line_start, line_end = EXCLUDED.line_end, git_sha = EXCLUDED.git_sha,
               content = EXCLUDED.content, content_hash = EXCLUDED.content_hash, meta = EXCLUDED.meta, updated_at = now()
             RETURNING id",
        )
        .bind(&doc.key)
        .bind(&doc.kind)
        .bind(if doc.audience.is_empty() { "both" } else { &doc.audience })
        .bind(&doc.lang)
        .bind(&doc.title)
        .bind(&doc.entity_id)
        .bind(&doc.path)
        .bind(&doc.symbol)
        .bind(doc.line_start)
        .bind(doc.line_end)
        .bind(&doc.git_sha)
        .bind(&doc.content)
        .bind(&hash)
        .bind(&doc.meta)
        .fetch_one(&mut *tx)
        .await?;

        sqlx::query("DELETE FROM chunks WHERE document_id = $1").bind(id).execute(&mut *tx).await?;
        let chunks = chunk(&doc.title, &doc.content, CHUNK_CHARS);
        let ords: Vec<i32> = (0..chunks.len() as i32).collect();
        sqlx::query("INSERT INTO chunks (document_id, ord, content) SELECT $1, * FROM UNNEST($2::int[], $3::text[])")
            .bind(id)
            .bind(&ords)
            .bind(&chunks)
            .execute(&mut *tx)
            .await?;

        stats.chunks += chunks.len();
        if existing.is_some() { stats.updated += 1 } else { stats.inserted += 1 }
    }

    tx.commit().await?;
    Ok(stats)
}

/// Deletes documents whose key starts with `prefix` and that are not in `keep`
/// (content that disappeared from the source since the last ingest).
pub async fn prune(pool: &PgPool, prefix: &str, keep: &HashSet<String>) -> Result<u64> {
    let keep: Vec<&String> = keep.iter().collect();
    let res = sqlx::query("DELETE FROM documents WHERE starts_with(key, $1) AND NOT (key = ANY($2))")
        .bind(prefix)
        .bind(&keep)
        .execute(pool)
        .await?;
    Ok(res.rows_affected())
}

pub struct EntityRow {
    pub id: String,
    pub kind: String,
    pub data: Value,
}

pub struct EdgeRow {
    pub src: String,
    pub rel: String,
    pub dst: String,
    pub weight: Option<f64>,
}

/// Replaces the whole entity graph (entities, their names, edges between entities). It is derived
/// from the fixture export in one pass, so a full rebuild is simpler and cheap. Code edges are kept.
pub async fn replace_entities(pool: &PgPool, entities: &[EntityRow], names: &[(String, String, String)], edges: &[EdgeRow]) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM edges WHERE NOT starts_with(src, 'code:')").execute(&mut *tx).await?;
    sqlx::query("DELETE FROM entities").execute(&mut *tx).await?;

    for batch in entities.chunks(500) {
        let ids: Vec<&str> = batch.iter().map(|e| e.id.as_str()).collect();
        let kinds: Vec<&str> = batch.iter().map(|e| e.kind.as_str()).collect();
        let data: Vec<&Value> = batch.iter().map(|e| &e.data).collect();
        sqlx::query("INSERT INTO entities (id, kind, data) SELECT * FROM UNNEST($1::text[], $2::text[], $3::jsonb[])")
            .bind(&ids)
            .bind(&kinds)
            .bind(&data)
            .execute(&mut *tx)
            .await?;
    }

    let mut seen = HashSet::new();
    let names: Vec<_> = names.iter().filter(|n| !n.2.trim().is_empty() && seen.insert(*n)).collect();
    for batch in names.chunks(1000) {
        let ids: Vec<&str> = batch.iter().map(|n| n.0.as_str()).collect();
        let langs: Vec<&str> = batch.iter().map(|n| n.1.as_str()).collect();
        let values: Vec<&str> = batch.iter().map(|n| n.2.as_str()).collect();
        sqlx::query("INSERT INTO entity_names (entity_id, lang, name) SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[])")
            .bind(&ids)
            .bind(&langs)
            .bind(&values)
            .execute(&mut *tx)
            .await?;
    }

    insert_edges(&mut tx, edges).await?;
    tx.commit().await?;
    Ok(())
}

async fn insert_edges(tx: &mut sqlx::Transaction<'_, sqlx::Postgres>, edges: &[EdgeRow]) -> Result<()> {
    let mut seen = HashSet::new();
    let edges: Vec<_> = edges.iter().filter(|e| seen.insert((&e.src, &e.rel, &e.dst))).collect();
    for batch in edges.chunks(1000) {
        let src: Vec<&str> = batch.iter().map(|e| e.src.as_str()).collect();
        let rel: Vec<&str> = batch.iter().map(|e| e.rel.as_str()).collect();
        let dst: Vec<&str> = batch.iter().map(|e| e.dst.as_str()).collect();
        let weight: Vec<Option<f64>> = batch.iter().map(|e| e.weight).collect();
        sqlx::query("INSERT INTO edges (src, rel, dst, weight) SELECT * FROM UNNEST($1::text[], $2::text[], $3::text[], $4::float8[])")
            .bind(&src)
            .bind(&rel)
            .bind(&dst)
            .bind(&weight)
            .execute(&mut **tx)
            .await?;
    }
    Ok(())
}

/// Replaces the edges from code symbols (`code:...`) to entities and settings.
pub async fn replace_code_edges(pool: &PgPool, edges: &[EdgeRow]) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM edges WHERE starts_with(src, 'code:')").execute(&mut *tx).await?;
    insert_edges(&mut tx, edges).await?;
    tx.commit().await?;
    Ok(())
}

/// Replaces the snapshot of indexed source files: (path, content hash, content).
pub async fn replace_source_files(pool: &PgPool, files: &[(String, String, String)], git_sha: Option<&str>) -> Result<()> {
    let mut tx = pool.begin().await?;
    sqlx::query("DELETE FROM source_files").execute(&mut *tx).await?;
    for batch in files.chunks(200) {
        let paths: Vec<&str> = batch.iter().map(|f| f.0.as_str()).collect();
        let hashes: Vec<&str> = batch.iter().map(|f| f.1.as_str()).collect();
        let contents: Vec<&str> = batch.iter().map(|f| f.2.as_str()).collect();
        sqlx::query("INSERT INTO source_files (path, content_hash, content, git_sha) SELECT p, h, c, $4 FROM UNNEST($1::text[], $2::text[], $3::text[]) AS t(p, h, c)")
            .bind(&paths)
            .bind(&hashes)
            .bind(&contents)
            .bind(git_sha)
            .execute(&mut *tx)
            .await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Creates the HNSW index for the embedding dimension of the active model.
pub async fn ensure_vector_index(pool: &PgPool, dims: usize) -> Result<()> {
    sqlx::query(AssertSqlSafe(format!(
        "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_{dims} ON chunks USING hnsw ((embedding::vector({dims})) vector_cosine_ops)"
    )))
    .execute(pool)
    .await?;
    Ok(())
}

/// Checks that stored embeddings were produced by `embedder`'s model. With `reset`, drops all
/// embeddings (and stale indexes) so they are recomputed with the new model.
pub async fn check_embed_model(pool: &PgPool, embedder: &Embedder, reset: bool) -> Result<()> {
    let wanted = embedder.kind().id();
    match db::get_meta(pool, "embed_model").await? {
        Some(current) if current != wanted => {
            if !reset {
                bail!("chunks are embedded with '{current}' but EMBED_MODEL is '{wanted}'; run `mhkb embed --reset` to re-embed");
            }
            sqlx::query("UPDATE chunks SET embedding = NULL").execute(pool).await?;
            let stale: Vec<String> = sqlx::query_scalar("SELECT indexname FROM pg_indexes WHERE indexname LIKE 'chunks_embedding_hnsw_%'")
                .fetch_all(pool)
                .await?;
            for idx in stale {
                sqlx::query(AssertSqlSafe(format!("DROP INDEX IF EXISTS {idx}"))).execute(pool).await?;
            }
        }
        _ => {}
    }
    db::set_meta(pool, "embed_model", wanted).await?;
    ensure_vector_index(pool, embedder.kind().dims()).await
}

/// Embeds every chunk that has no embedding yet. Returns the number of chunks embedded.
pub async fn embed_pending(pool: &PgPool, embedder: &Embedder, batch: usize, mut progress: impl FnMut(usize, usize)) -> Result<usize> {
    let total: i64 = sqlx::query_scalar("SELECT count(*) FROM chunks WHERE embedding IS NULL").fetch_one(pool).await?;
    let mut done = 0usize;
    loop {
        let rows = sqlx::query("SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id LIMIT $1")
            .bind(batch as i64)
            .fetch_all(pool)
            .await?;
        if rows.is_empty() {
            break;
        }
        let ids: Vec<i64> = rows.iter().map(|r| r.get("id")).collect();
        let texts: Vec<String> = rows.iter().map(|r| r.get("content")).collect();
        let vectors = embedder.passages(&texts)?;

        let mut tx = pool.begin().await?;
        for (id, v) in ids.iter().zip(vectors) {
            sqlx::query("UPDATE chunks SET embedding = $1 WHERE id = $2")
                .bind(Vector::from(v))
                .bind(id)
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        done += ids.len();
        progress(done, total as usize);
    }
    Ok(done)
}
