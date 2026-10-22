use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde_yaml::Value;

use crate::text::normalize_ws;

/// Languages the knowledge base renders documents in. German is the source language:
/// every MyHordes translation key is the German text itself.
pub const LANGS: [&str; 4] = ["en", "fr", "de", "es"];

/// MyHordes translations, loaded from `translations/<domain>+intl-icu.<lang>.yml`.
#[derive(Default)]
pub struct Translations {
    /// (domain, lang) -> German source -> translation
    domains: HashMap<(String, String), HashMap<String, String>>,
    /// lang -> German source -> translation, across all domains
    global: HashMap<String, HashMap<String, String>>,
    /// lang -> whitespace-normalized German source -> translation (fallback for multi-line Twig literals)
    normalized: HashMap<String, HashMap<String, String>>,
}

impl Translations {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut tr = Self::default();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "yml"))
            .collect();
        files.sort();

        for path in files {
            let name = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
            // "<domain>+intl-icu.<lang>" or "<domain>.<lang>"
            let Some((domain, lang)) = name.rsplit_once('.') else { continue };
            let domain = domain.trim_end_matches("+intl-icu").to_string();
            if !LANGS.contains(&lang) {
                continue;
            }
            let raw = std::fs::read_to_string(&path)?;
            let value: Value = serde_yaml::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
            let mut map = HashMap::new();
            flatten(&value, "", &mut map);

            let global = tr.global.entry(lang.to_string()).or_default();
            let normalized = tr.normalized.entry(lang.to_string()).or_default();
            for (k, v) in &map {
                global.entry(k.clone()).or_insert_with(|| v.clone());
                normalized.entry(normalize_ws(k)).or_insert_with(|| v.clone());
            }
            tr.domains.insert((domain, lang.to_string()), map);
        }
        Ok(tr)
    }

    /// Translates German source text `de` into `lang`, trying `domains` first, then any domain.
    /// Falls back to the German text when no translation exists.
    pub fn tr(&self, lang: &str, domains: &[&str], de: &str) -> String {
        simplify_icu(&self.lookup(lang, domains, de).unwrap_or_else(|| de.to_string()))
    }

    pub fn lookup(&self, lang: &str, domains: &[&str], de: &str) -> Option<String> {
        if de.is_empty() {
            return None;
        }
        for d in domains {
            if let Some(v) = self.domains.get(&(d.to_string(), lang.to_string())).and_then(|m| m.get(de)) {
                return Some(v.clone());
            }
        }
        if let Some(v) = self.global.get(lang).and_then(|m| m.get(de)) {
            return Some(v.clone());
        }
        if let Some(v) = self.normalized.get(lang).and_then(|m| m.get(&normalize_ws(de))) {
            return Some(v.clone());
        }
        // German "translations" may be missing when the key already is the final German text
        (lang == "de").then(|| de.to_string())
    }
}

/// Resolves ICU `select` / `plural` messages to their `other` branch (or first branch), e.g.
/// `{ref__gender, select, female {Droguée} other {Drogué}}` → `Drogué`. Plain `{placeholders}` are kept.
pub fn simplify_icu(s: &str) -> String {
    if !s.contains(", select,") && !s.contains(", plural,") && !s.contains(", selectordinal,") {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '{'
            && let Some((end, branch)) = parse_icu_choice(&chars, i)
        {
            out.push_str(&simplify_icu(&branch));
            i = end;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out.trim().to_string()
}

/// Index just past the `}` matching the `{` at `open`
fn matching_brace(chars: &[char], open: usize) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in chars.iter().enumerate().skip(open) {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parses `{var, select|plural, key {msg} key {msg}}` starting at `open`; returns (end, chosen msg)
fn parse_icu_choice(chars: &[char], open: usize) -> Option<(usize, String)> {
    let end = matching_brace(chars, open)?;
    let inner: String = chars[open + 1..end - 1].iter().collect();
    let mut parts = inner.splitn(3, ',');
    let (_var, kind, rest) = (parts.next()?, parts.next()?.trim(), parts.next()?);
    if !matches!(kind, "select" | "plural" | "selectordinal") {
        return None;
    }
    let rest: Vec<char> = rest.chars().collect();
    let mut options: Vec<(String, String)> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        while i < rest.len() && rest[i].is_whitespace() {
            i += 1;
        }
        let key_start = i;
        while i < rest.len() && !rest[i].is_whitespace() && rest[i] != '{' {
            i += 1;
        }
        let key: String = rest[key_start..i].iter().collect();
        while i < rest.len() && rest[i].is_whitespace() {
            i += 1;
        }
        if i >= rest.len() || rest[i] != '{' {
            break;
        }
        let close = matching_brace(&rest, i)?;
        options.push((key, rest[i + 1..close - 1].iter().collect()));
        i = close;
    }
    let chosen = options.iter().find(|(k, _)| k == "other").or(options.first())?;
    Some((end, chosen.1.clone()))
}

fn flatten(value: &Value, prefix: &str, out: &mut HashMap<String, String>) {
    match value {
        Value::Mapping(map) => {
            for (k, v) in map {
                let key = match k {
                    Value::String(s) => s.clone(),
                    other => serde_yaml::to_string(other).unwrap_or_default().trim().to_string(),
                };
                let full = if prefix.is_empty() { key } else { format!("{prefix}.{key}") };
                flatten(v, &full, out);
            }
        }
        Value::String(s) => {
            out.insert(prefix.to_string(), s.clone());
        }
        Value::Null => {}
        other => {
            out.insert(prefix.to_string(), serde_yaml::to_string(other).unwrap_or_default().trim().to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::simplify_icu;

    #[test]
    fn resolves_nested_select_to_other_branch() {
        let s = "{ref__icu, select,\non {\n{ref__gender, select,\nfemale {Droguée}\nother {Drogué}\n}\n}\nother {Drogué(e)}\n}";
        assert_eq!(simplify_icu(s), "Drogué(e)");
    }

    #[test]
    fn keeps_plain_placeholders() {
        assert_eq!(simplify_icu("You have {count} AP"), "You have {count} AP");
        assert_eq!(simplify_icu("Lost {n, plural, one {# point} other {# points}} today"), "Lost # points today");
    }
}
