//! Player abbreviations and nicknames from L'Argordien (https://argordien.dev.ctruillet.eu,
//! GPL-3.0, gitlab.com/ctruillet/argordien). Downloaded at ingest time and kept only in the
//! database: entries naming a game entity become nicknames on its cards, and every entry goes into
//! a per-language glossary document.

use std::collections::{BTreeMap, HashMap};

use anyhow::{Context, Result};
use kb_core::store::{EntityRow, NewDoc};
use serde::Deserialize;
use serde_json::json;
use tracing::warn;

pub const DEFAULT_URL: &str = "https://argordien.dev.ctruillet.eu/data_{lang}.json";
const SOURCE: &str = "L'Argordien (argordien.dev.ctruillet.eu)";

#[derive(Deserialize, Clone)]
pub struct Entry {
    word: String,
    meaning: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default)]
    icon: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct DataFile {
    data: Vec<Entry>,
}

/// Downloads the glossary for each language; languages that fail are skipped with a warning.
pub async fn fetch(url_template: &str, langs: &[String]) -> Vec<(String, Vec<Entry>)> {
    let client = reqwest::Client::new();
    let mut out = Vec::new();
    for lang in langs {
        let url = url_template.replace("{lang}", lang);
        let result: Result<DataFile> = async {
            let resp = client.get(&url).send().await?.error_for_status()?;
            let text = resp.text().await?;
            serde_json::from_str(&text).with_context(|| format!("parsing {url}"))
        }
        .await;
        match result {
            Ok(file) => out.push((lang.clone(), file.data)),
            Err(e) => warn!(%url, error = %e, "glossary not available"),
        }
    }
    out
}

/// Item categories of the glossary; other categories name constructions or non-entities
fn is_item_category(category: &str) -> bool {
    ["Objet", "Arme", "Nourriture", "Drogue", "Alcool"].iter().any(|c| category.contains(c))
}

/// "Tour De Guet (TDG)" → "tour de guet"
fn meaning_name(meaning: &str) -> String {
    let name = match meaning.rfind(" (") {
        Some(i) if meaning.ends_with(')') => &meaning[..i],
        _ => meaning,
    };
    name.trim().to_lowercase().replace('’', "'")
}

pub struct Output {
    /// entity id → nicknames
    pub aliases: BTreeMap<String, Vec<String>>,
    pub docs: Vec<NewDoc>,
}

pub fn build(glossaries: &[(String, Vec<Entry>)], entities: &[EntityRow], names: &[(String, String, String)]) -> Output {
    let mut by_name: HashMap<String, Vec<&str>> = HashMap::new();
    for (id, _, name) in names {
        let v = by_name.entry(name.trim().to_lowercase().replace('’', "'")).or_default();
        if !v.contains(&id.as_str()) {
            v.push(id);
        }
    }
    let mut item_by_icon: HashMap<&str, Vec<&str>> = HashMap::new();
    for e in entities.iter().filter(|e| e.kind == "item") {
        if let Some(icon) = e.data.get("icon").and_then(|v| v.as_str()) {
            item_by_icon.entry(icon).or_default().push(&e.id);
        }
    }

    let mut aliases: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut docs = Vec::new();
    for (lang, entries) in glossaries {
        let mut by_category: BTreeMap<String, String> = BTreeMap::new();
        for entry in entries {
            let category = entry.category.clone().unwrap_or_else(|| "Other".into());

            // Name first (works for constructions and items), then the item icon for item categories
            let wanted_kind = if is_item_category(&category) { "item:" } else if category.contains("Chantier") || category.contains("Batiment") { "building:" } else { "" };
            let mut entity = by_name
                .get(&meaning_name(&entry.meaning))
                .and_then(|ids| ids.iter().find(|id| id.starts_with(wanted_kind)).or(ids.first()))
                .map(|id| id.to_string());
            if entity.is_none() && is_item_category(&category) {
                let icon = entry.icon.as_deref().and_then(|i| i.strip_prefix("item/")).map(|i| i.trim_end_matches(".gif"));
                entity = icon.and_then(|i| item_by_icon.get(i)).filter(|ids| ids.len() == 1).map(|ids| ids[0].to_string());
            }
            if let Some(id) = &entity {
                let list = aliases.entry(id.clone()).or_default();
                if !list.contains(&entry.word) {
                    list.push(entry.word.clone());
                }
            }

            let line = by_category.entry(category).or_default();
            line.push_str(&format!(
                "- **{}** — {}{}{}\n",
                entry.word,
                entry.meaning,
                entity.as_deref().map(|id| format!(" (`{id}`)")).unwrap_or_default(),
                entry.description.as_deref().filter(|d| !d.is_empty()).map(|d| format!(": {}", d.replace('\n', " "))).unwrap_or_default()
            ));
        }
        for (category, body) in by_category {
            let slug: String = category.to_lowercase().chars().map(|c| if c.is_alphanumeric() { c } else { '-' }).collect();
            docs.push(NewDoc {
                key: format!("glossary:{lang}:{slug}"),
                kind: "glossary".into(),
                audience: "both".into(),
                lang: Some(lang.clone()),
                title: format!("Player glossary: {category}"),
                content: format!("Abbreviations and nicknames used by MyHordes players, category \"{category}\". Source: {SOURCE}.\n\n{body}"),
                meta: json!({ "source": SOURCE, "category": category }),
                ..Default::default()
            });
        }
    }
    Output { aliases, docs }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_abbreviation_from_meaning() {
        assert_eq!(meaning_name("Tour De Guet (TDG)"), "tour de guet");
        assert_eq!(meaning_name("Abri Anti-Atomique (AAA)"), "abri anti-atomique");
        assert_eq!(meaning_name("Outre-Monde"), "outre-monde");
    }
}
