pub mod entities;
pub mod help;
pub mod rules;

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::i18n::Translations;
use kb_core::store::{self, NewDoc};
use sqlx::PgPool;
use tracing::info;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Source {
    All,
    Entities,
    Help,
    Rules,
}

/// Git commit of the MyHordes checkout the export was made from
fn export_git_sha(export: &Path) -> Option<String> {
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(export.join("manifest.json")).ok()?).ok()?;
    manifest.get("git_sha")?.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

pub async fn run(pool: &PgPool, source: Source, export: &Path, myhordes: &Path, langs: &[String]) -> Result<()> {
    let git_sha = export_git_sha(export);
    let needs_translations = matches!(source, Source::All | Source::Entities | Source::Help);
    let tr = if needs_translations {
        Translations::load(&myhordes.join("translations")).context("loading MyHordes translations")?
    } else {
        Translations::default()
    };

    if matches!(source, Source::All | Source::Entities) {
        let out = entities::build(export, &tr, langs, git_sha.as_deref())?;
        store::replace_entities(pool, &out.entities, &out.names, &out.edges).await?;
        info!(entities = out.entities.len(), names = out.names.len(), edges = out.edges.len(), "entity graph replaced");
        save(pool, "card:", out.docs).await?;
    }
    if matches!(source, Source::All | Source::Help) {
        save(pool, "help:", help::build(myhordes, &tr, langs, git_sha.as_deref())?).await?;
    }
    if matches!(source, Source::All | Source::Rules) {
        save(pool, "rule:", rules::build(export, git_sha.as_deref())?).await?;
    }
    Ok(())
}

async fn save(pool: &PgPool, prefix: &str, docs: Vec<NewDoc>) -> Result<()> {
    let keep: HashSet<String> = docs.iter().map(|d| d.key.clone()).collect();
    let stats = store::upsert_docs(pool, &docs).await?;
    let pruned = store::prune(pool, prefix, &keep).await?;
    info!(
        source = prefix.trim_end_matches(':'),
        documents = docs.len(),
        inserted = stats.inserted,
        updated = stats.updated,
        unchanged = stats.unchanged,
        chunks = stats.chunks,
        pruned,
        "documents saved"
    );
    Ok(())
}
