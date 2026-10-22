//! PHP source code: one document per method (tree-sitter), a per-file outline, a snapshot of each
//! file, and edges from code to the game entities and settings it references.

use std::collections::HashSet;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result};
use kb_core::store::{EdgeRow, NewDoc};
use kb_core::text::sha256_hex;
use regex::Regex;
use serde_json::json;
use tree_sitter::{Node, Parser};
use walkdir::WalkDir;

/// Game-logic code, relative to the MyHordes root
pub const SCOPE: &[&str] = &[
    "src/Service",
    "src/EventListener/Game",
    "src/Event/Game",
    "src/Controller/BeyondController.php",
    "src/Controller/ExplorationController.php",
    "src/Controller/InventoryAwareController.php",
    "src/Controller/Town",
    "src/Controller/REST/Game",
    "src/Structures",
    "src/Enum",
    "src/Entity",
    "src/Hooks",
    "src/Configuration",
];

pub struct SourceFile {
    pub path: String,
    pub content: String,
    pub hash: String,
}

pub struct Method {
    pub class: String,
    pub name: String,
    pub signature: String,
    pub start: usize,
    pub end: usize,
    /// Doc comment + full method source
    pub text: String,
}

impl Method {
    pub fn symbol(&self) -> String {
        format!("{}::{}", self.class, self.name)
    }

    /// Accessors and other one-liners carry no mechanics
    pub fn is_trivial(&self) -> bool {
        let lines = self.end - self.start + 1;
        let accessor = ["get", "set", "is", "has", "add", "remove"].iter().any(|p| self.name.starts_with(p));
        (accessor && lines <= 5) || lines <= 2
    }
}

pub struct ParsedFile {
    pub namespace: String,
    /// (kind, name, start line, end line)
    pub types: Vec<(String, String, usize, usize)>,
    pub methods: Vec<Method>,
}

pub fn collect_files(myhordes: &Path) -> Result<Vec<SourceFile>> {
    let mut files = Vec::new();
    for scope in SCOPE {
        for entry in WalkDir::new(myhordes.join(scope)).sort_by_file_name() {
            let entry = entry?;
            if !entry.file_type().is_file() || entry.path().extension().is_none_or(|e| e != "php") {
                continue;
            }
            let path = entry.path().strip_prefix(myhordes)?.to_string_lossy().replace('\\', "/");
            let content = std::fs::read_to_string(entry.path()).with_context(|| format!("reading {path}"))?;
            files.push(SourceFile { hash: sha256_hex(&content), path, content });
        }
    }
    Ok(files)
}

fn text<'a>(node: Node, src: &'a str) -> &'a str {
    &src[node.byte_range()]
}

pub fn parse(src: &str) -> Result<ParsedFile> {
    let mut parser = Parser::new();
    parser.set_language(&tree_sitter_php::LANGUAGE_PHP.into())?;
    let tree = parser.parse(src, None).context("tree-sitter returned no tree")?;
    let mut out = ParsedFile { namespace: String::new(), types: vec![], methods: vec![] };
    walk(tree.root_node(), src, None, &mut out);
    Ok(out)
}

fn walk(node: Node, src: &str, class: Option<&str>, out: &mut ParsedFile) {
    match node.kind() {
        "namespace_definition" => {
            if let Some(name) = node.child_by_field_name("name") {
                out.namespace = text(name, src).to_string();
            }
        }
        "class_declaration" | "interface_declaration" | "trait_declaration" | "enum_declaration" => {
            let name = node.child_by_field_name("name").map(|n| text(n, src).to_string()).unwrap_or_default();
            out.types.push((node.kind().trim_end_matches("_declaration").to_string(), name.clone(), node.start_position().row + 1, node.end_position().row + 1));
            let mut cursor = node.walk();
            for child in node.children(&mut cursor) {
                walk(child, src, Some(&name), out);
            }
            return;
        }
        "method_declaration" | "function_definition" => {
            let name = node.child_by_field_name("name").map(|n| text(n, src).to_string()).unwrap_or_default();
            let sig_end = node.child_by_field_name("body").map(|b| b.start_byte()).unwrap_or(node.end_byte());
            let signature = src[node.start_byte()..sig_end].split_whitespace().collect::<Vec<_>>().join(" ");
            // A doc comment directly above belongs to the method
            let (doc, start) = match node.prev_sibling() {
                Some(prev) if prev.kind() == "comment" && text(prev, src).starts_with("/**") && node.start_position().row - prev.end_position().row <= 1 => {
                    (format!("{}\n", text(prev, src)), prev.start_position().row + 1)
                }
                _ => (String::new(), node.start_position().row + 1),
            };
            // Keep the method's original indentation for the first line
            let line_start = src[..node.start_byte()].rfind('\n').map(|i| i + 1).unwrap_or(0);
            out.methods.push(Method {
                class: class.unwrap_or("(global)").to_string(),
                name,
                signature,
                start,
                end: node.end_position().row + 1,
                text: format!("{doc}{}", &src[line_start..node.end_byte()]),
            });
            return;
        }
        _ => {}
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, src, class, out);
    }
}

static ITEM_ID: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"'([a-z0-9_]+_#\d{2})'").unwrap());
static STATUS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:hasStatus|addStatus|removeStatus|inflictStatus|healStatus)\w*\(\s*(?:\$\w+\s*,\s*)?'(\w+)'").unwrap());
static PROFESSION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"isProfession\(\s*'(\w+)'").unwrap());
static ROLE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"hasRole\(\s*'(\w+)'").unwrap());
static BUILDING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"getBuilding\(\s*[^,]+,\s*'([a-z0-9_]+_#\d{2})'").unwrap());
static SETTING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(TownSetting|CitizenProperties)::(\w+)").unwrap());

/// Edges from a code symbol to the entities and settings its source mentions.
fn references(src_key: &str, code: &str, known: &HashSet<String>) -> Vec<EdgeRow> {
    let mut edges = Vec::new();
    let mut push = |rel: &str, dst: String| {
        if !edges.iter().any(|e: &EdgeRow| e.rel == rel && e.dst == dst) {
            edges.push(EdgeRow { src: src_key.to_string(), rel: rel.to_string(), dst, weight: None });
        }
    };
    for c in ITEM_ID.captures_iter(code) {
        for kind in ["item", "building"] {
            let id = format!("{kind}:{}", &c[1]);
            if known.contains(&id) {
                push("mentions", id);
            }
        }
    }
    for c in BUILDING.captures_iter(code) {
        push("mentions", format!("building:{}", &c[1]));
    }
    for (re, kind) in [(&*STATUS, "status"), (&*PROFESSION, "profession"), (&*ROLE, "role")] {
        for c in re.captures_iter(code) {
            let id = format!("{kind}:{}", &c[1]);
            if known.contains(&id) {
                push("mentions", id);
            }
        }
    }
    for c in SETTING.captures_iter(code) {
        let kind = if &c[1] == "TownSetting" { "setting" } else { "citizenprop" };
        push("reads", format!("{kind}:{}", &c[2]));
    }
    edges
}

pub struct Output {
    pub files: Vec<SourceFile>,
    pub docs: Vec<NewDoc>,
    pub edges: Vec<EdgeRow>,
}

pub fn build(myhordes: &Path, known_entities: &HashSet<String>, git_sha: Option<&str>) -> Result<Output> {
    let files = collect_files(myhordes)?;
    let mut docs = Vec::new();
    let mut edges = Vec::new();

    for file in &files {
        let parsed = parse(&file.content).with_context(|| format!("parsing {}", file.path))?;

        // File outline: what is where, for navigation
        let mut outline = format!("File: {}\nNamespace: {}\n\n", file.path, parsed.namespace);
        for (kind, name, start, end) in &parsed.types {
            outline.push_str(&format!("{kind} {name} (lines {start}–{end})\n"));
        }
        outline.push('\n');
        for m in &parsed.methods {
            outline.push_str(&format!("- {} — line {}: `{}`\n", m.symbol(), m.start, m.signature));
        }
        docs.push(NewDoc {
            key: format!("code:{}", file.path),
            kind: "code".into(),
            audience: "dev".into(),
            title: format!("Code outline: {}", file.path),
            path: Some(file.path.clone()),
            git_sha: git_sha.map(str::to_string),
            content: outline,
            meta: json!({ "outline": true }),
            ..Default::default()
        });

        for m in parsed.methods.iter().filter(|m| !m.is_trivial()) {
            let key = format!("code:{}:{}", file.path, m.symbol());
            edges.extend(references(&key, &m.text, known_entities));
            docs.push(NewDoc {
                key: key.clone(),
                kind: "code".into(),
                audience: "dev".into(),
                title: format!("Code: {} ({})", m.symbol(), file.path),
                path: Some(file.path.clone()),
                symbol: Some(m.symbol()),
                line_start: Some(m.start as i32),
                line_end: Some(m.end as i32),
                git_sha: git_sha.map(str::to_string),
                content: format!(
                    "File: {} (lines {}–{})\nClass: {}\\{}\n\n```php\n{}\n```",
                    file.path, m.start, m.end, parsed.namespace, m.class, m.text
                ),
                meta: json!({}),
                ..Default::default()
            });
        }
    }
    Ok(Output { files, docs, edges })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = r#"<?php
namespace App\Entity;

class Zone {
    private int $x;

    public function getX(): int { return $this->x; }

    /**
     * Scout level, 0 to 3
     */
    public function getScoutLevel(): int
    {
        if ($this->isTownZone()) return 0;
        return min(3, max(0, floor($this->count()/5)));
    }
}
"#;

    #[test]
    fn parses_methods_with_doc_comments() {
        let p = parse(SRC).unwrap();
        assert_eq!(p.namespace, "App\\Entity");
        assert_eq!(p.types[0].1, "Zone");
        let names: Vec<_> = p.methods.iter().map(|m| m.symbol()).collect();
        assert_eq!(names, vec!["Zone::getX", "Zone::getScoutLevel"]);
        let scout = &p.methods[1];
        assert!(scout.text.starts_with("/**"));
        assert_eq!(scout.start, 9);
        assert!(p.methods[0].is_trivial());
        assert!(!scout.is_trivial());
    }

    #[test]
    fn finds_entity_and_setting_references() {
        let known: HashSet<String> = ["item:pelle_#00".to_string(), "status:camper".to_string(), "profession:hunter".to_string()].into();
        let code = "if ($c->hasStatus('camper') && $c->isProfession('hunter')) $x = 'pelle_#00'; $conf->get(TownSetting::MapParamsDigChancesBase);";
        let edges = references("code:x", code, &known);
        let dsts: Vec<_> = edges.iter().map(|e| e.dst.as_str()).collect();
        assert!(dsts.contains(&"item:pelle_#00"));
        assert!(dsts.contains(&"status:camper"));
        assert!(dsts.contains(&"profession:hunter"));
        assert!(dsts.contains(&"setting:MapParamsDigChancesBase"));
    }
}
