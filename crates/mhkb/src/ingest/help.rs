//! In-game help pages (`templates/ajax/help/*.html.twig`), rendered to Markdown per language.

use std::path::Path;
use std::sync::LazyLock;

use anyhow::Result;
use kb_core::i18n::Translations;
use kb_core::store::NewDoc;
use kb_core::text::html_to_markdown;
use regex::{Captures, Regex};
use serde_json::json;

/// `{{ 'German text'|trans({...}, 'domain')|raw }}`, single or double quoted
static TRANS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?s)\{\{\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")\s*\|\s*trans\b\s*(?:\(([^)]*)\))?.*?\}\}"#).unwrap()
});
static DOMAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"['"](\w+)['"]\s*$"#).unwrap());
static CONTENT_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)\{%\s*block\s+helpContent\s*%\}(.*?)\{%\s*endblock\s*%\}").unwrap());
static H1: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^# (.+)$").unwrap());

fn unescape(s: &str) -> String {
    s.replace("\\'", "'").replace("\\\"", "\"").replace("\\\\", "\\")
}

pub fn build(myhordes: &Path, tr: &Translations, langs: &[String], git_sha: Option<&str>) -> Result<Vec<NewDoc>> {
    let dir = myhordes.join("templates/ajax/help");
    let mut files: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".html.twig") && !p.ends_with("shell.html.twig"))
        .collect();
    files.sort();

    let mut docs = Vec::new();
    for path in files {
        let page = path.file_name().unwrap().to_string_lossy().trim_end_matches(".html.twig").to_string();
        let raw = std::fs::read_to_string(&path)?;
        let body = CONTENT_BLOCK.captures(&raw).map(|c| c[1].to_string()).unwrap_or(raw);

        for lang in langs {
            let translated = TRANS.replace_all(&body, |c: &Captures| {
                let text = unescape(c.get(1).or_else(|| c.get(2)).map(|m| m.as_str()).unwrap_or(""));
                let domain = c.get(3).and_then(|args| DOMAIN.captures(args.as_str())).map(|d| d[1].to_string());
                let domains: Vec<&str> = domain.as_deref().into_iter().collect();
                tr.tr(lang, &domains, &text)
            });
            let markdown = html_to_markdown(&translated);
            if markdown.trim().is_empty() {
                continue;
            }
            let title = H1.captures(&markdown).map(|c| c[1].trim().to_string()).unwrap_or_else(|| page.replace('_', " "));
            // The H1 becomes the document title; chunks carry it already
            let content = H1.replace(&markdown, "").trim().to_string();
            docs.push(NewDoc {
                key: format!("help:{page}:{lang}"),
                kind: "help".into(),
                audience: "player".into(),
                lang: Some(lang.clone()),
                title: format!("Help: {title}"),
                path: Some(format!("templates/ajax/help/{page}.html.twig")),
                git_sha: git_sha.map(str::to_string),
                content,
                meta: json!({ "page": page }),
                ..Default::default()
            });
        }
    }
    Ok(docs)
}
