use std::collections::HashMap;

use anyhow::Result;
use pgvector::Vector;
use serde::Serialize;
use sqlx::{AssertSqlSafe, FromRow, PgPool};

use crate::embed::Embedder;

#[derive(Debug, Clone, Default)]
pub struct SearchOptions {
    pub k: usize,
    /// Restrict to documents in this language (language-neutral documents always match)
    pub lang: Option<String>,
    /// Restrict to these document kinds
    pub kinds: Option<Vec<String>>,
    /// 'player' or 'dev'; documents with audience 'both' always match
    pub audience: Option<String>,
}

#[derive(Debug, Clone, Serialize, FromRow)]
pub struct Hit {
    pub chunk_id: i64,
    pub document_id: i64,
    pub key: String,
    pub kind: String,
    pub lang: Option<String>,
    pub title: String,
    pub entity_id: Option<String>,
    pub path: Option<String>,
    pub line_start: Option<i32>,
    pub git_sha: Option<String>,
    pub content: String,
    #[sqlx(skip)]
    pub score: f64,
    #[sqlx(skip)]
    pub matched_by: Vec<&'static str>,
}

/// Candidates fetched per retriever before fusion
const CANDIDATES: i64 = 40;
/// Reciprocal Rank Fusion constant
const RRF_K: f64 = 60.0;

const FILTERS: &str = "($2::text IS NULL OR d.lang = $2 OR d.lang IS NULL)
    AND ($3::text[] IS NULL OR d.kind = ANY($3))
    AND ($4::text IS NULL OR d.audience = 'both' OR d.audience = $4)";

/// Hybrid search: dense vectors + full-text + fuzzy entity-name matching, fused with RRF and
/// deduplicated to the best chunk per document.
pub async fn search(pool: &PgPool, embedder: &Embedder, query: &str, opts: &SearchOptions) -> Result<Vec<Hit>> {
    let k = if opts.k == 0 { 8 } else { opts.k };
    let mut fused: HashMap<i64, (f64, Vec<&'static str>)> = HashMap::new();
    let mut add = |ids: Vec<i64>, source: &'static str, weight: f64| {
        for (rank, id) in ids.into_iter().enumerate() {
            let e = fused.entry(id).or_default();
            e.0 += weight / (RRF_K + rank as f64 + 1.0);
            if !e.1.contains(&source) {
                e.1.push(source);
            }
        }
    };

    add(vector_candidates(pool, embedder, query, opts).await?, "vector", 1.0);
    add(fulltext_candidates(pool, query, opts).await?, "fulltext", 1.0);
    // A named entity in the question is a strong signal: weight its card higher
    add(entity_candidates(pool, query, opts).await?, "name", 1.5);

    let ids: Vec<i64> = fused.keys().copied().collect();
    let rows: Vec<Hit> = sqlx::query_as(
        "SELECT c.id AS chunk_id, d.id AS document_id, d.key, d.kind, d.lang, d.title, d.entity_id, d.path, d.line_start, d.git_sha, c.content
         FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ANY($1)",
    )
    .bind(&ids)
    .fetch_all(pool)
    .await?;

    let mut hits: Vec<Hit> = rows
        .into_iter()
        .map(|mut h| {
            let (score, sources) = fused.remove(&h.chunk_id).unwrap_or_default();
            h.score = score;
            h.matched_by = sources;
            h
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));

    let mut seen_docs = std::collections::HashSet::new();
    hits.retain(|h| seen_docs.insert(h.document_id));
    hits.truncate(k);
    Ok(hits)
}

async fn vector_candidates(pool: &PgPool, embedder: &Embedder, query: &str, opts: &SearchOptions) -> Result<Vec<i64>> {
    let dims = embedder.kind().dims();
    // Embedding is CPU-bound: keep it off the async worker so the bot stays responsive
    let qv = Vector::from(tokio::task::block_in_place(|| embedder.query(query))?);
    let sql = format!(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
         WHERE c.embedding IS NOT NULL AND {FILTERS}
         ORDER BY c.embedding::vector({dims}) <=> $1::vector({dims}) LIMIT $5"
    );
    Ok(sqlx::query_scalar(AssertSqlSafe(sql))
        .bind(qv)
        .bind(&opts.lang)
        .bind(&opts.kinds)
        .bind(&opts.audience)
        .bind(CANDIDATES)
        .fetch_all(pool)
        .await?)
}

async fn fulltext_candidates(pool: &PgPool, query: &str, opts: &SearchOptions) -> Result<Vec<i64>> {
    // OR of the query's words: ts_rank_cd still favours chunks matching more of them
    let terms: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 3)
        .map(|w| w.to_lowercase())
        .collect();
    if terms.is_empty() {
        return Ok(vec![]);
    }
    let tsquery = terms.join(" | ");
    let sql = format!(
        "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id, to_tsquery('simple', $1) q
         WHERE c.tsv @@ q AND {FILTERS}
         ORDER BY ts_rank_cd(c.tsv, q) DESC LIMIT $5"
    );
    Ok(sqlx::query_scalar(AssertSqlSafe(sql))
        .bind(tsquery)
        .bind(&opts.lang)
        .bind(&opts.kinds)
        .bind(&opts.audience)
        .bind(CANDIDATES)
        .fetch_all(pool)
        .await?)
}

/// First chunk of the cards of entities whose name appears (fuzzily) in the query.
async fn entity_candidates(pool: &PgPool, query: &str, opts: &SearchOptions) -> Result<Vec<i64>> {
    let card_lang = opts.lang.clone().unwrap_or_else(|| "en".to_string());
    let sql = format!(
        "WITH named AS (
             SELECT entity_id, max(word_similarity(lower(name), lower($1))) AS sim
             FROM entity_names
             WHERE char_length(name) >= 4 AND lower(name) <% lower($1)
             GROUP BY entity_id ORDER BY sim DESC LIMIT 8
         )
         SELECT c.id FROM named n
         JOIN documents d ON d.entity_id = n.entity_id AND d.kind = 'entity_card' AND d.lang = $6
         JOIN chunks c ON c.document_id = d.id AND c.ord = 0
         WHERE {FILTERS}
         ORDER BY n.sim DESC LIMIT $5"
    );
    Ok(sqlx::query_scalar(AssertSqlSafe(sql))
        .bind(query)
        .bind(&opts.lang)
        .bind(&opts.kinds)
        .bind(&opts.audience)
        .bind(CANDIDATES)
        .bind(card_lang)
        .fetch_all(pool)
        .await?)
}
