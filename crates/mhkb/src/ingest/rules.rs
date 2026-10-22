//! Effective town rules per town type (`export/rules/resolved/*.json`), annotated with the
//! `TownSetting` enum case that reads each value.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, Result};
use kb_core::store::NewDoc;
use serde_json::{Value, json};

fn flatten(prefix: &str, v: &Value, out: &mut BTreeMap<String, Value>) {
    match v {
        // Nested objects become dotted paths, matching TownSetting::key(); arrays stay whole
        Value::Object(m) if !m.is_empty() => {
            for (k, child) in m {
                let path = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
                flatten(&path, child, out);
            }
        }
        other => {
            out.insert(prefix.to_string(), other.clone());
        }
    }
}

fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_str(&std::fs::read_to_string(path)?).with_context(|| format!("parsing {}", path.display()))
}

pub fn build(export: &Path, git_sha: Option<&str>) -> Result<Vec<NewDoc>> {
    let settings = read_json(&export.join("settings/TownSetting.json"))?;
    let setting_by_key: HashMap<String, String> = settings
        .as_object()
        .map(|m| m.iter().filter_map(|(name, s)| Some((s.get("key")?.as_str()?.to_string(), name.clone()))).collect())
        .unwrap_or_default();

    let dir = export.join("rules/resolved");
    let default = {
        let mut flat = BTreeMap::new();
        flatten("", &read_json(&dir.join("default.json"))?, &mut flat);
        flat
    };

    let mut docs = Vec::new();
    let mut types: Vec<String> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".json") && !n.ends_with(".private.json"))
        .map(|n| n.trim_end_matches(".json").to_string())
        .collect();
    types.sort();

    // Each value is written with a readable form of its key first ("map params › dig chances › depleted"):
    // dotted keys are single tokens for full-text search and opaque to embeddings
    let line = |path: &str, value: &Value| -> String {
        let readable = path.replace('.', " › ").replace('_', " ");
        match setting_by_key.get(path) {
            Some(name) => format!("- {readable}: {value} — `{path}`, TownSetting::{name}\n"),
            None => format!("- {readable}: {value} — `{path}`\n"),
        }
    };

    for town_type in types {
        let mut flat = BTreeMap::new();
        flatten("", &read_json(&dir.join(format!("{town_type}.json")))?, &mut flat);

        // Full values only for the default rules, one document per top-level group (map, modifiers,
        // features, ...). Other town types only get their differences: full copies per type would be
        // near-duplicates crowding each other out of search results.
        if town_type == "default" {
            let mut groups: BTreeMap<String, String> = BTreeMap::new();
            for (path, value) in &flat {
                let group = path.split('.').next().unwrap_or("").to_string();
                groups.entry(group).or_default().push_str(&line(path, value));
            }
            for (group, body) in groups {
                docs.push(NewDoc {
                    key: format!("rule:default:{group}"),
                    kind: "rule".into(),
                    audience: "both".into(),
                    title: format!("Default town rules: {}", group.replace('_', " ")),
                    path: Some("config/app/rules.yml".into()),
                    git_sha: git_sha.map(str::to_string),
                    content: format!(
                        "Configuration values in group `{group}` of the default town rules (config/app/rules.yml). \
                         They apply to every town type unless that type's \"differs from the default rules\" document overrides them.\n\n{body}"
                    ),
                    meta: json!({ "town_type": "default", "group": group }),
                    ..Default::default()
                });
            }
        } else {
            let mut diff = String::new();
            for (path, value) in &flat {
                if default.get(path) != Some(value) {
                    let before = default.get(path).map(Value::to_string).unwrap_or_else(|| "(unset)".into());
                    let _ = write!(diff, "{}", line(path, value).trim_end());
                    let _ = writeln!(diff, " — default: {before}");
                }
            }
            for path in default.keys().filter(|p| !flat.contains_key(*p)) {
                let _ = writeln!(diff, "- {} removed — `{path}` (default: {})", path.replace('.', " › ").replace('_', " "), default[path]);
            }
            docs.push(NewDoc {
                key: format!("rule:{town_type}:diff"),
                kind: "rule".into(),
                audience: "both".into(),
                title: format!("How town type '{town_type}' differs from the default rules"),
                path: Some("config/app/rules.yml".into()),
                git_sha: git_sha.map(str::to_string),
                content: if diff.is_empty() { "No differences from the default rules.".into() } else { diff },
                meta: json!({ "town_type": town_type, "group": "diff" }),
                ..Default::default()
            });
        }
    }
    Ok(docs)
}
