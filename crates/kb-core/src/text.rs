use std::sync::LazyLock;

use regex::Regex;
use sha2::{Digest, Sha256};

pub fn sha256_hex(s: &str) -> String {
    hex::encode(Sha256::digest(s.as_bytes()))
}

/// Collapses all whitespace runs into single spaces.
pub fn normalize_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

macro_rules! re {
    ($name:ident, $pat:expr) => {
        static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($pat).unwrap());
    };
}

re!(TWIG_COMMENT, r"(?s)\{#.*?#\}");
re!(TWIG_TAG, r"(?s)\{%.*?%\}");
re!(TWIG_EXPR, r"(?s)\{\{.*?\}\}");
re!(HEADING_OPEN, r"(?i)<h([1-6])[^>]*>");
re!(HEADING_CLOSE, r"(?i)</h[1-6]\s*>");
re!(LI_OPEN, r"(?i)<li[^>]*>");
re!(BR, r"(?i)<br\s*/?>|<hr\s*/?>");
re!(BLOCK, r"(?i)</?(p|div|ul|ol|table|tr|section|article|blockquote)[^>]*>");
re!(CELL, r"(?i)</?(td|th)[^>]*>");
re!(BOLD, r"(?i)</?(strong|b)\s*>");
re!(ITALIC, r"(?i)</?(em|i)\s*>");
re!(ANY_TAG, r"(?s)<[^>]+>");
re!(IMG_PLACEHOLDER, r"\{img_(\w+)\}");
re!(SPACES, r"[ \t]+");
re!(BLANK_LINES, r"\n{3,}");
re!(DANGLING_BULLET, r"(?m)^-\n+");

/// Converts the small HTML subset used in MyHordes templates and translations to Markdown.
pub fn html_to_markdown(html: &str) -> String {
    let s = TWIG_COMMENT.replace_all(html, "");
    let s = TWIG_TAG.replace_all(&s, "");
    let s = TWIG_EXPR.replace_all(&s, "");
    let s = HEADING_OPEN.replace_all(&s, |c: &regex::Captures| format!("\n\n{} ", "#".repeat(c[1].parse().unwrap_or(2))));
    let s = HEADING_CLOSE.replace_all(&s, "\n\n");
    let s = LI_OPEN.replace_all(&s, "\n- ");
    let s = BR.replace_all(&s, "\n");
    let s = BLOCK.replace_all(&s, "\n\n");
    let s = CELL.replace_all(&s, " | ");
    let s = BOLD.replace_all(&s, "**");
    let s = ITALIC.replace_all(&s, "_");
    let s = ANY_TAG.replace_all(&s, "");
    let s = IMG_PLACEHOLDER.replace_all(&s, |c: &regex::Captures| c[1].to_uppercase());
    let s = decode_entities(&s);

    let lines: Vec<String> = s
        .lines()
        .map(|l| SPACES.replace_all(l.trim(), " ").to_string())
        .collect();
    let s = lines.join("\n");
    // "<li>\n text" leaves the bullet alone on its line
    let s = DANGLING_BULLET.replace_all(&s, "- ");
    BLANK_LINES.replace_all(s.trim(), "\n\n").to_string()
}

fn decode_entities(s: &str) -> String {
    s.replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&rsquo;", "'")
        .replace("&hellip;", "…")
        .replace("&amp;", "&")
}

/// Splits a document body into chunks of at most ~`max_chars`, cutting at headings first,
/// then at blank lines, then at line breaks. Every chunk is prefixed with the document title
/// so it stays meaningful on its own (for embeddings and for the LLM).
pub fn chunk(title: &str, body: &str, max_chars: usize) -> Vec<String> {
    let header = format!("# {title}\n");
    let budget = max_chars.saturating_sub(header.len()).max(200);

    // Sections start at markdown headings (##, ###, ...)
    let mut sections: Vec<String> = Vec::new();
    for line in body.lines() {
        if line.starts_with("## ") || line.starts_with("### ") || sections.is_empty() {
            sections.push(String::new());
        }
        let cur = sections.last_mut().unwrap();
        cur.push_str(line);
        cur.push('\n');
    }

    let mut pieces: Vec<String> = Vec::new();
    for section in sections {
        if section.trim().is_empty() {
            continue;
        }
        if section.len() <= budget {
            pieces.push(section);
        } else {
            pieces.extend(split_to_budget(&section, budget));
        }
    }

    // Greedily merge small consecutive pieces
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if !cur.is_empty() && cur.len() + p.len() > budget {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&p);
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    if chunks.is_empty() {
        chunks.push(String::new());
    }
    chunks.into_iter().map(|c| format!("{header}{}", c.trim_end())).collect()
}

fn split_to_budget(text: &str, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let units: Vec<&str> = if text.contains("\n\n") { text.split_inclusive("\n\n").collect() } else { text.split_inclusive('\n').collect() };
    for unit in units {
        if unit.len() > budget {
            // A single oversized paragraph: hard-split at char boundaries
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            let chars: Vec<char> = unit.chars().collect();
            for part in chars.chunks(budget) {
                out.push(part.iter().collect());
            }
            continue;
        }
        if !cur.is_empty() && cur.len() + unit.len() > budget {
            out.push(std::mem::take(&mut cur));
        }
        cur.push_str(unit);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_help_html() {
        let md = html_to_markdown("<h2>Title</h2>\n<p>\n<strong>Bold</strong> text {img_ap}</p><ul><li>one</li><li>two</li></ul>");
        assert!(md.starts_with("## Title"));
        assert!(md.contains("**Bold** text AP"));
        assert!(md.contains("- one\n- two"));
    }

    #[test]
    fn chunks_respect_budget_and_carry_title() {
        let body = (0..50).map(|i| format!("Paragraph {i} with some words in it.\n\n")).collect::<String>();
        let chunks = chunk("Doc", &body, 300);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|c| c.starts_with("# Doc\n") && c.len() <= 300 + 10));
    }
}
