pub mod code;
pub mod entities;
pub mod glossary;
pub mod help;
pub mod rules;

use std::collections::{BTreeMap, HashSet};
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
    Code,
    Topics,
}

/// Git commit of the MyHordes checkout the export was made from
fn export_git_sha(export: &Path) -> Option<String> {
    let manifest: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(export.join("manifest.json")).ok()?).ok()?;
    manifest.get("git_sha")?.as_str().filter(|s| !s.is_empty()).map(str::to_string)
}

/// Current commit of the MyHordes checkout
fn checkout_git_sha(myhordes: &Path) -> Option<String> {
    let out = std::process::Command::new("git").arg("-C").arg(myhordes).args(["rev-parse", "HEAD"]).output().ok()?;
    out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Curated community nicknames from `knowledge/aliases.yaml`
fn load_aliases(path: &Path) -> Result<BTreeMap<String, Vec<String>>> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    serde_yaml::from_str(&std::fs::read_to_string(path)?).with_context(|| format!("parsing {}", path.display()))
}

/// Adds nicknames to the entity names (for name search) and to the entity's cards
fn apply_aliases(out: &mut entities::Output, aliases: &BTreeMap<String, Vec<String>>) {
    for (entity_id, nicknames) in aliases {
        for nick in nicknames {
            out.names.push((entity_id.clone(), "alias".into(), nick.clone()));
        }
        for doc in out.docs.iter_mut().filter(|d| d.entity_id.as_deref() == Some(entity_id)) {
            doc.content.push_str(&format!("\nNicknames used by players: {}\n", nicknames.join(", ")));
        }
    }
}

pub async fn run(
    pool: &PgPool,
    source: Source,
    export: &Path,
    myhordes: &Path,
    knowledge: &Path,
    glossary_url: Option<&str>,
    langs: &[String],
) -> Result<()> {
    let git_sha = export_git_sha(export);
    let needs_translations = matches!(source, Source::All | Source::Entities | Source::Help | Source::Code);
    let tr = if needs_translations {
        Translations::load(&myhordes.join("translations")).context("loading MyHordes translations")?
    } else {
        Translations::default()
    };

    if matches!(source, Source::All | Source::Entities) {
        let mut out = entities::build(export, &tr, langs, git_sha.as_deref())?;
        let mut aliases = load_aliases(&knowledge.join("aliases.yaml"))?;
        let glossary = match glossary_url {
            Some(url) => glossary::build(&glossary::fetch(url, langs).await, &out.entities, &out.names),
            None => glossary::Output { aliases: BTreeMap::new(), docs: vec![] },
        };
        for (id, nicks) in glossary.aliases {
            let list = aliases.entry(id).or_default();
            for n in nicks {
                if !list.contains(&n) {
                    list.push(n);
                }
            }
        }
        apply_aliases(&mut out, &aliases);
        store::replace_entities(pool, &out.entities, &out.names, &out.edges).await?;
        info!(entities = out.entities.len(), names = out.names.len(), edges = out.edges.len(), aliases = aliases.len(), "entity graph replaced");
        save(pool, "card:", out.docs).await?;
        if glossary_url.is_some() {
            save(pool, "glossary:", glossary.docs).await?;
        }
    }
    if matches!(source, Source::All | Source::Help) {
        save(pool, "help:", help::build(myhordes, &tr, langs, git_sha.as_deref())?).await?;
    }
    if matches!(source, Source::All | Source::Rules) {
        save(pool, "rule:", rules::build(export, git_sha.as_deref())?).await?;
    }
    if matches!(source, Source::All | Source::Code) {
        let sha = checkout_git_sha(myhordes);
        let known: HashSet<String> = sqlx::query_scalar("SELECT id FROM entities").fetch_all(pool).await?.into_iter().collect();
        let out = code::build(myhordes, &known, sha.as_deref())?;
        let files: Vec<(String, String, String)> = out.files.into_iter().map(|f| (f.path, f.hash, f.content)).collect();
        store::replace_source_files(pool, &files, sha.as_deref()).await?;
        store::replace_code_edges(pool, &out.edges).await?;
        info!(files = files.len(), edges = out.edges.len(), "source files and code edges replaced");
        let translations: Vec<_> = tr.entries().collect();
        store::replace_translations(pool, &translations).await?;
        info!(translations = translations.len(), "translations replaced");
        save(pool, "code:", out.docs).await?;
    }
    if matches!(source, Source::All | Source::Topics) {
        save(pool, "topic:", crate::dossier::documents(knowledge)?).await?;
    }
    Ok(())
}

pub async fn save(pool: &PgPool, prefix: &str, docs: Vec<NewDoc>) -> Result<()> {
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
