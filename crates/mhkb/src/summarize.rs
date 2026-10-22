//! Claude summaries of the game-logic source files: one structured request per file, results
//! cached by file content hash, turned into `summary:` documents that explain each mechanic in plain
//! language with its numbers and conditions.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use kb_core::claude::{Claude, Usage};
use kb_core::db;
use kb_core::store::NewDoc;
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use tracing::{info, warn};

/// Source directories whose files are summarized (event payload classes and enums are left out)
const SUMMARIZED: &[&str] = &[
    "src/Service/",
    "src/EventListener/Game/",
    "src/Controller/",
    "src/Structures/",
    "src/Entity/",
    "src/Hooks/",
    "src/Configuration/",
];

const SYSTEM: &str = r#"You document the game mechanics implemented in the source code of MyHordes, a multiplayer zombie survival web game (PHP / Symfony). Your summaries are indexed in a knowledge base that answers player and developer questions such as "what is the chance to find an item when digging?" or "do scouts improve digging?".

You receive one source file with line numbers. Describe every method that implements, computes or changes game behaviour: chances, damage, defense, action points, statuses, items, constructions, the nightly attack, zombie movement, camping, digging, votes, roles, souls, rewards, and so on.

For each such method:
- Say what it does in game terms, and when it runs (player action, nightly attack stage, page load, ...) when the code shows it.
- Give every number, probability and formula with its conditions, e.g. "base 60% (TownSetting::MapParamsDigChancesBase) + job dig bonus, -10% at night unless a torch is on the zone floor, +2.5% per zone scout level (max 3)".
- Name what it depends on: TownSetting / CitizenProperties keys, statuses ('camper', 'wound5'), jobs ('hunter'), roles, item and building ids ('pelle_#00', 'small_watchmen_#01'), town types ('panda', 'small').
- Stay factual: only what the code shows. If a value comes from configuration, name the setting instead of guessing its value.

Leave out methods without game logic (getters, setters, wiring, rendering helpers, persistence). The file summary says what the file is responsible for in 1 to 3 sentences.
Write in English. Use the exact class and method names in `symbol`, e.g. "CitizenChanceQueryListener::getDigChances"."#;

fn output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "file_summary": {"type": "string"},
            "symbols": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "line_start": {"type": "integer"},
                        "line_end": {"type": "integer"},
                        "summary": {"type": "string"}
                    },
                    "required": ["symbol", "line_start", "line_end", "summary"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["file_summary", "symbols"],
        "additionalProperties": false
    })
}

#[derive(Deserialize)]
struct FileSummary {
    file_summary: String,
    symbols: Vec<SymbolSummary>,
}

#[derive(Deserialize)]
struct SymbolSummary {
    symbol: String,
    line_start: i32,
    line_end: i32,
    summary: String,
}

pub struct Options {
    pub model: String,
    pub effort: String,
    /// Only files whose path contains one of these (pilot runs); all pending files otherwise
    pub only: Vec<String>,
    /// Synchronous requests instead of a batch (full price, immediate)
    pub sync: bool,
    /// Max files to send
    pub limit: Option<usize>,
}

fn request(model: &str, effort: &str, path: &str, content: &str) -> Value {
    let numbered: String = content.lines().enumerate().map(|(i, l)| format!("{:>5}| {l}\n", i + 1)).collect();
    json!({
        "model": model,
        "max_tokens": 32000,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": format!("<file path=\"{path}\">\n{numbered}</file>")}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": output_schema()}}
    })
}

/// Files in scope whose current content has no cached summary from `model`
async fn pending(pool: &PgPool, opts: &Options) -> Result<Vec<(String, String, String)>> {
    let rows: Vec<(String, String, String)> = sqlx::query_as(
        "SELECT f.path, f.content_hash, f.content FROM source_files f
         LEFT JOIN summary_cache s ON s.path = f.path AND s.content_hash = f.content_hash AND s.model = $1
         WHERE s.path IS NULL ORDER BY f.path",
    )
    .bind(&opts.model)
    .fetch_all(pool)
    .await?;
    let mut files: Vec<_> = rows
        .into_iter()
        .filter(|(p, _, _)| SUMMARIZED.iter().any(|d| p.starts_with(d)))
        .filter(|(p, _, _)| opts.only.is_empty() || opts.only.iter().any(|o| p.contains(o.as_str())))
        .collect();
    if let Some(limit) = opts.limit {
        files.truncate(limit);
    }
    Ok(files)
}

async fn store_result(pool: &PgPool, model: &str, path: &str, hash: &str, message: &Value) -> Result<FileSummary> {
    let text = message
        .get("content")
        .and_then(Value::as_array)
        .and_then(|blocks| blocks.iter().find(|b| b.get("type").and_then(Value::as_str) == Some("text")))
        .and_then(|b| b.get("text").and_then(Value::as_str))
        .ok_or_else(|| anyhow!("no text block (stop_reason {:?})", message.get("stop_reason")))?;
    let parsed: FileSummary = serde_json::from_str(text).context("parsing summary JSON")?;
    let result: Value = serde_json::from_str(text)?;
    sqlx::query(
        "INSERT INTO summary_cache (path, content_hash, model, result) VALUES ($1, $2, $3, $4)
         ON CONFLICT (path) DO UPDATE SET content_hash = EXCLUDED.content_hash, model = EXCLUDED.model, result = EXCLUDED.result, created_at = now()",
    )
    .bind(path)
    .bind(hash)
    .bind(model)
    .bind(&result)
    .execute(pool)
    .await?;
    Ok(parsed)
}

pub async fn run(pool: &PgPool, claude: &Claude, opts: &Options) -> Result<()> {
    // A batch submitted earlier and not collected yet takes precedence
    if let Some(pending_batch) = db::get_meta(pool, "summary_batch").await? {
        info!("collecting previously submitted batch");
        return collect(pool, claude, &pending_batch).await;
    }

    let files = pending(pool, opts).await?;
    if files.is_empty() {
        info!("all files already summarized");
        return Ok(());
    }
    let input_chars: usize = files.iter().map(|f| f.2.len()).sum();
    info!(files = files.len(), approx_input_tokens = input_chars / 3, model = %opts.model, sync = opts.sync, "summarizing");

    if opts.sync {
        let mut usage = Usage::default();
        for (path, hash, content) in &files {
            let response = claude.messages(request(&opts.model, &opts.effort, path, content)).await?;
            usage.add(response.usage);
            match store_result(pool, &opts.model, path, hash, &response.body).await {
                Ok(s) => info!(%path, symbols = s.symbols.len(), out_tokens = response.usage.output_tokens, "summarized"),
                Err(e) => warn!(%path, error = %e, "summary failed"),
            }
        }
        info!(cost_usd = format!("{:.3}", usage.cost_usd(&opts.model)), input = usage.input_tokens, output = usage.output_tokens, "done");
        return Ok(());
    }

    let mut ids = serde_json::Map::new();
    let requests: Vec<(String, Value)> = files
        .iter()
        .enumerate()
        .map(|(i, (path, hash, content))| {
            let id = format!("f{i:04}");
            ids.insert(id.clone(), json!([path, hash]));
            (id, request(&opts.model, &opts.effort, path, content))
        })
        .collect();
    let batch_id = claude.create_batch(&requests).await?;
    let state = json!({"id": batch_id, "model": opts.model, "files": ids});
    db::set_meta(pool, "summary_batch", &state.to_string()).await?;
    info!(%batch_id, requests = requests.len(), "batch submitted; waiting for results (re-run `mhkb summarize` to resume if interrupted)");
    collect(pool, claude, &state.to_string()).await
}

async fn collect(pool: &PgPool, claude: &Claude, state: &str) -> Result<()> {
    let state: Value = serde_json::from_str(state)?;
    let batch_id = state["id"].as_str().context("batch id")?;
    let model = state["model"].as_str().unwrap_or_default();
    let files: HashMap<String, (String, String)> = state["files"]
        .as_object()
        .map(|m| m.iter().map(|(k, v)| (k.clone(), (v[0].as_str().unwrap_or("").into(), v[1].as_str().unwrap_or("").into()))).collect())
        .unwrap_or_default();

    let batch = loop {
        let batch = claude.get_batch(batch_id).await?;
        let status = batch["processing_status"].as_str().unwrap_or("");
        let c = &batch["request_counts"];
        info!(status, processing = %c["processing"], succeeded = %c["succeeded"], errored = %c["errored"], "batch progress");
        if status == "ended" {
            break batch;
        }
        tokio::time::sleep(Duration::from_secs(30)).await;
    };

    let url = batch["results_url"].as_str().context("ended batch without results_url")?;
    let mut usage = Usage::default();
    let (mut ok, mut failed) = (0, 0);
    for (id, result) in claude.batch_results(url).await? {
        let Some((path, hash)) = files.get(&id) else { continue };
        if result["type"] != "succeeded" {
            warn!(%path, result = %result, "batch request did not succeed");
            failed += 1;
            continue;
        }
        let message = &result["message"];
        usage.add(Usage::from_response(message));
        match store_result(pool, model, path, hash, message).await {
            Ok(_) => ok += 1,
            Err(e) => {
                warn!(%path, error = %e, "summary failed");
                failed += 1;
            }
        }
    }
    // Batch prices are half the standard ones
    info!(ok, failed, cost_usd = format!("{:.3}", usage.cost_usd(model) / 2.0), "batch collected");
    sqlx::query("DELETE FROM kb_meta WHERE key = 'summary_batch'").execute(pool).await?;
    if failed > 0 {
        warn!("{failed} files failed; re-run `mhkb summarize` to retry them");
    }
    Ok(())
}

/// `summary:` documents from every cached summary of a current source file
pub async fn documents(pool: &PgPool) -> Result<Vec<NewDoc>> {
    let rows: Vec<(String, Value, Option<String>)> = sqlx::query_as(
        "SELECT s.path, s.result, f.git_sha FROM summary_cache s JOIN source_files f ON f.path = s.path AND f.content_hash = s.content_hash ORDER BY s.path",
    )
    .fetch_all(pool)
    .await?;
    if rows.is_empty() {
        bail!("no summaries yet; run `mhkb summarize` first");
    }
    let mut docs = Vec::new();
    for (path, result, git_sha) in rows {
        let Ok(summary) = serde_json::from_value::<FileSummary>(result) else {
            warn!(%path, "unreadable cached summary");
            continue;
        };
        let file_name = path.rsplit('/').next().unwrap_or(&path).to_string();
        let mut overview = format!("{}\n\nMechanics in this file:\n", summary.file_summary);
        for s in &summary.symbols {
            overview.push_str(&format!("- {} (lines {}–{})\n", s.symbol, s.line_start, s.line_end));
        }
        docs.push(NewDoc {
            key: format!("summary:{path}"),
            kind: "summary".into(),
            audience: "both".into(),
            title: format!("What {file_name} does"),
            path: Some(path.clone()),
            git_sha: git_sha.clone(),
            content: overview,
            meta: json!({}),
            ..Default::default()
        });
        for s in summary.symbols {
            docs.push(NewDoc {
                key: format!("summary:{path}:{}", s.symbol),
                kind: "summary".into(),
                audience: "both".into(),
                title: format!("How it works: {} ({file_name})", s.symbol),
                path: Some(path.clone()),
                symbol: Some(s.symbol.clone()),
                line_start: Some(s.line_start),
                line_end: Some(s.line_end),
                git_sha: git_sha.clone(),
                content: format!("Source: {path}, lines {}–{}\n\n{}", s.line_start, s.line_end, s.summary),
                meta: json!({}),
                ..Default::default()
            });
        }
    }
    Ok(docs)
}
