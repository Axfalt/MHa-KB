mod bot;
mod dossier;
mod eval;
mod ingest;
mod summarize;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use kb_core::answer::{self, AnswerConfig, Mode};
use kb_core::claude::Claude;
use kb_core::embed::{Embedder, ModelKind};
use kb_core::search::{self, SearchOptions};
use kb_core::{db, store};
use sqlx::PgPool;
use tracing::info;

#[derive(Parser)]
#[command(name = "mhkb", about = "MyHordes knowledge base: ingest, embed and search game mechanics")]
struct Cli {
    #[arg(long, env = "DATABASE_URL", default_value = "postgres://mhkb:mhkb@127.0.0.1:5433/mhkb", global = true)]
    database_url: String,

    /// Embedding model: e5-small (384d), e5-base (768d) or bge-m3 (1024d)
    #[arg(long, env = "EMBED_MODEL", default_value = "e5-base", global = true)]
    embed_model: String,

    /// Where embedding models are downloaded
    #[arg(long, env = "MODEL_CACHE", default_value = ".cache/models", global = true)]
    model_cache: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Apply database migrations
    Migrate,
    /// Build documents from the MyHordes export and sources, then embed new chunks
    Ingest {
        #[arg(value_enum, default_value = "all")]
        source: ingest::Source,
        /// Output of php/export.php
        #[arg(long, env = "EXPORT_DIR", default_value = "export")]
        export: PathBuf,
        /// MyHordes checkout (for translations, templates and code)
        #[arg(long, env = "MYHORDES_DIR", default_value = "../myhordes")]
        myhordes: PathBuf,
        /// Curated knowledge (aliases, topics)
        #[arg(long, default_value = "knowledge")]
        knowledge: PathBuf,
        /// Player glossary to download ({lang} is replaced); empty to skip
        #[arg(long, env = "GLOSSARY_URL", default_value = ingest::glossary::DEFAULT_URL)]
        glossary_url: String,
        #[arg(long, value_delimiter = ',', default_value = "en,fr,de,es")]
        langs: Vec<String>,
        /// Skip embedding the new chunks
        #[arg(long)]
        no_embed: bool,
    },
    /// Embed all chunks that have no embedding yet
    Embed {
        /// Drop existing embeddings (required after changing EMBED_MODEL)
        #[arg(long)]
        reset: bool,
        #[arg(long, default_value_t = 64)]
        batch: usize,
    },
    /// Hybrid search, for debugging retrieval
    Search {
        query: String,
        #[arg(short, default_value_t = 8)]
        k: usize,
        #[arg(long)]
        lang: Option<String>,
        /// Comma-separated document kinds (entity_card, help, rule, ...)
        #[arg(long, value_delimiter = ',')]
        kind: Option<Vec<String>>,
        /// player or dev
        #[arg(long)]
        audience: Option<String>,
        /// Print full chunk contents
        #[arg(long)]
        full: bool,
    },
    /// Measure retrieval quality on a question set, or answer quality with --answers
    Eval {
        #[arg(long, default_value = "eval/questions.yaml")]
        questions: PathBuf,
        #[arg(short, default_value_t = 5)]
        k: usize,
        /// Write per-question results as JSON
        #[arg(long)]
        out: Option<PathBuf>,
        /// Show expected vs. retrieved documents (or the judge's comment) for misses
        #[arg(short, long)]
        verbose: bool,
        /// Grade bot answers against the questions' reference facts (calls Claude, costs money)
        #[arg(long)]
        answers: bool,
        /// Model answering the questions (with --answers)
        #[arg(long, env = "ANSWER_MODEL", default_value = "claude-opus-5-5")]
        model: String,
        /// Model grading the answers (with --answers)
        #[arg(long, default_value = "claude-opus-5-5")]
        judge_model: String,
        /// Only questions containing this text
        #[arg(long)]
        only: Option<String>,
    },
    /// Answer a question with Claude, as the Discord bot does
    Ask {
        question: String,
        /// Preferred document language (en, fr, de, es)
        #[arg(long, default_value = "en")]
        lang: String,
        /// Answer for developers instead of players
        #[arg(long)]
        dev: bool,
        #[arg(long, env = "ANSWER_MODEL", default_value = "claude-opus-5-5")]
        model: String,
        /// Override the effort level (default: low for players, medium for dev)
        #[arg(long)]
        effort: Option<String>,
        /// Show tool calls and token usage
        #[arg(short, long)]
        verbose: bool,
    },
    /// Run the Discord bot (DISCORD_TOKEN, DISCORD_GUILD_ID, DAILY_BUDGET_USD, USER_LIMIT_PER_HOUR, ALLOWED_CHANNEL_IDS, DEV_ROLE_ID)
    Bot {
        #[arg(long, env = "ANSWER_MODEL", default_value = "claude-opus-5-5")]
        model: String,
        #[arg(long, env = "PLAYER_EFFORT", default_value = "low")]
        player_effort: String,
        #[arg(long, env = "DEV_EFFORT", default_value = "medium")]
        dev_effort: String,
    },
    /// Summarize game-logic source files with Claude (Batch API by default) and index the summaries
    Summarize {
        #[arg(long, default_value = "claude-opus-5-5")]
        model: String,
        #[arg(long, default_value = "medium")]
        effort: String,
        /// Only files whose path contains this (repeatable), e.g. for a pilot
        #[arg(long)]
        only: Vec<String>,
        /// Send requests one by one at full price instead of a half-price batch
        #[arg(long)]
        sync: bool,
        /// Maximum number of files to send
        #[arg(long)]
        limit: Option<usize>,
        /// Don't rebuild the summary documents afterwards
        #[arg(long)]
        no_index: bool,
    },
    /// Write topic dossiers (knowledge/topics/*.md) with Claude, then index them
    Dossier {
        /// Only these topic ids (repeatable)
        #[arg(long)]
        only: Vec<String>,
        /// Regenerate existing dossiers
        #[arg(long)]
        force: bool,
        #[arg(long, env = "ANSWER_MODEL", default_value = "claude-opus-5-5")]
        model: String,
        #[arg(long, default_value = "medium")]
        effort: String,
        #[arg(long, default_value_t = 3)]
        parallel: usize,
        #[arg(long, default_value = "knowledge")]
        knowledge: PathBuf,
    },
    /// List recent bot answers that got a 👎
    Feedback {
        #[arg(short, default_value_t = 20)]
        n: i64,
    },
    /// Document and chunk counts
    Stats,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    let pool = db::connect(&cli.database_url).await?;
    db::migrate(&pool).await?;
    let load_embedder = || -> Result<Embedder> { Embedder::load(ModelKind::parse(&cli.embed_model)?, &cli.model_cache) };

    match cli.cmd {
        Cmd::Migrate => info!("migrations applied"),
        Cmd::Ingest { source, export, myhordes, knowledge, glossary_url, langs, no_embed } => {
            let glossary_url = Some(glossary_url).filter(|u| !u.is_empty());
            ingest::run(&pool, source, &export, &myhordes, &knowledge, glossary_url.as_deref(), &langs).await?;
            if !no_embed {
                embed(&pool, &load_embedder()?, false, 64).await?;
            }
        }
        Cmd::Embed { reset, batch } => embed(&pool, &load_embedder()?, reset, batch).await?,
        Cmd::Search { query, k, lang, kind, audience, full } => {
            let embedder = load_embedder()?;
            let hits = search::search(&pool, &embedder, &query, &SearchOptions { k, lang, kinds: kind, audience }).await?;
            for (i, h) in hits.iter().enumerate() {
                println!(
                    "{:>2}. [{:.4}] {} · {} · {}  ({})",
                    i + 1,
                    h.score,
                    h.kind,
                    h.lang.as_deref().unwrap_or("-"),
                    h.title,
                    h.matched_by.join("+")
                );
                let body = h.content.split_once('\n').map(|(_, b)| b).unwrap_or(&h.content);
                if full {
                    println!("{body}\n");
                } else {
                    let preview: String = body.chars().take(220).collect::<String>().replace('\n', " ");
                    println!("    {preview}…");
                }
            }
        }
        Cmd::Eval { questions, k, out, verbose, answers, model, judge_model, only } => {
            if answers {
                let cfg = AnswerConfig { model, ..Default::default() };
                eval::run_answers(&pool, &load_embedder()?, &Claude::from_env()?, &questions, only.as_deref(), &cfg, &judge_model, out.as_deref(), verbose).await?;
            } else {
                eval::run(&pool, &load_embedder()?, &questions, only.as_deref(), k, out.as_deref(), verbose).await?;
            }
        }
        Cmd::Ask { question, lang, dev, model, effort, verbose } => {
            let embedder = load_embedder()?;
            let claude = Claude::from_env()?;
            let mut cfg = AnswerConfig { model, ..Default::default() };
            if let Some(e) = effort {
                cfg.effort = (e.clone(), e);
            }
            let mode = if dev { Mode::Dev } else { Mode::Player };
            let started = std::time::Instant::now();
            let a = answer::answer(&pool, &embedder, &claude, &cfg, mode, &lang, &question).await?;
            println!("{}\n", a.text);
            for (i, s) in a.sources.iter().enumerate() {
                println!("[{}] {} ({})", i + 1, s.title, s.key);
            }
            if !a.unknown_citations.is_empty() {
                println!("\n⚠ cited unknown documents: {}", a.unknown_citations.join(", "));
            }
            if verbose {
                println!("\ntool calls:");
                for c in &a.tool_calls {
                    println!("  {c}");
                }
            }
            println!(
                "\n{} · {} · {:.1}s · in {} (cache read {}, write {}) out {} · ~${:.4}",
                a.model,
                a.outcome,
                started.elapsed().as_secs_f32(),
                a.usage.input_tokens,
                a.usage.cache_read_input_tokens,
                a.usage.cache_creation_input_tokens,
                a.usage.output_tokens,
                a.cost_usd
            );
        }
        Cmd::Bot { model, player_effort, dev_effort } => {
            let cfg = AnswerConfig { model, effort: (player_effort, dev_effort), ..Default::default() };
            bot::run(pool, load_embedder()?, Claude::from_env()?, bot::BotConfig::from_env(cfg)?).await?;
        }
        Cmd::Summarize { model, effort, only, sync, limit, no_index } => {
            let opts = summarize::Options { model, effort, only, sync, limit };
            summarize::run(&pool, &Claude::from_env()?, &opts).await?;
            if !no_index {
                ingest::save(&pool, "summary:", summarize::documents(&pool).await?).await?;
                embed(&pool, &load_embedder()?, false, 64).await?;
            }
        }
        Cmd::Dossier { only, force, model, effort, parallel, knowledge } => {
            let embedder = load_embedder()?;
            let cfg = AnswerConfig { model, max_steps: 30, max_tokens: 24_000, ..Default::default() };
            let opts = dossier::Options { only, force, effort, parallel };
            dossier::run(&pool, &embedder, &Claude::from_env()?, &cfg, &knowledge, &opts).await?;
            ingest::save(&pool, "topic:", dossier::documents(&knowledge)?).await?;
            embed(&pool, &embedder, false, 64).await?;
        }
        Cmd::Feedback { n } => {
            let rows: Vec<(i64, chrono::DateTime<chrono::Utc>, String, String, Option<String>, Option<serde_json::Value>)> = sqlx::query_as(
                "SELECT id, created_at, lang, question, answer, sources FROM qa_log WHERE feedback < 0 ORDER BY id DESC LIMIT $1",
            )
            .bind(n)
            .fetch_all(&pool)
            .await?;
            for (id, at, lang, question, answer, sources) in rows {
                let keys: Vec<String> = sources.as_ref().and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|s| s["key"].as_str().map(str::to_string)).collect()).unwrap_or_default();
                let answer: String = answer.unwrap_or_default().replace('\n', " ").chars().take(300).collect();
                println!("#{id} {} [{lang}] {question}\n  sources: {}\n  {answer}\n", at.format("%Y-%m-%d %H:%M"), keys.join(", "));
            }
        }
        Cmd::Stats => stats(&pool).await?,
    }
    Ok(())
}

async fn embed(pool: &PgPool, embedder: &Embedder, reset: bool, batch: usize) -> Result<()> {
    store::check_embed_model(pool, embedder, reset).await?;
    let started = std::time::Instant::now();
    let n = store::embed_pending(pool, embedder, batch, |done, total| {
        if done % (batch * 10) < batch || done == total {
            info!("embedded {done}/{total} chunks");
        }
    })
    .await?;
    info!(chunks = n, secs = started.elapsed().as_secs(), model = embedder.kind().id(), "embedding done");
    Ok(())
}

async fn stats(pool: &PgPool) -> Result<()> {
    let rows: Vec<(String, Option<String>, i64, i64, i64)> = sqlx::query_as(
        "SELECT d.kind, d.lang, count(DISTINCT d.id), count(c.id), count(c.embedding)
         FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
         GROUP BY 1, 2 ORDER BY 1, 2",
    )
    .fetch_all(pool)
    .await?;
    println!("{:<14} {:<5} {:>9} {:>8} {:>9}", "kind", "lang", "documents", "chunks", "embedded");
    for (kind, lang, docs, chunks, embedded) in rows {
        println!("{:<14} {:<5} {:>9} {:>8} {:>9}", kind, lang.unwrap_or_else(|| "-".into()), docs, chunks, embedded);
    }
    let entities: i64 = sqlx::query_scalar("SELECT count(*) FROM entities").fetch_one(pool).await?;
    let edges: i64 = sqlx::query_scalar("SELECT count(*) FROM edges").fetch_one(pool).await?;
    let model = db::get_meta(pool, "embed_model").await?.unwrap_or_else(|| "-".into());
    println!("\nentities: {entities} · edges: {edges} · embed model: {model}");
    Ok(())
}
