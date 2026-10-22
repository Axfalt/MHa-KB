//! Question answering: hybrid retrieval + a Claude tool loop over the knowledge base.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::LazyLock;

use anyhow::Result;
use regex::Regex;
use serde::Serialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use tracing::debug;

use crate::claude::{Claude, Usage};
use crate::embed::Embedder;
use crate::i18n::LANGS;
use crate::search::{self, Hit, SearchOptions};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Plain-language answers for players
    Player,
    /// Technical answers with internal ids, config keys and file paths
    Dev,
}

impl Mode {
    fn audience(self) -> &'static str {
        match self {
            Mode::Player => "player",
            Mode::Dev => "dev",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnswerConfig {
    pub model: String,
    /// Effort per mode: (player, dev)
    pub effort: (String, String),
    pub max_steps: usize,
    /// Documents injected before the first model call
    pub initial_k: usize,
    pub max_tokens: u32,
}

impl Default for AnswerConfig {
    fn default() -> Self {
        Self {
            model: "claude-opus-5-5".into(),
            effort: ("low".into(), "medium".into()),
            max_steps: 6,
            initial_k: 6,
            max_tokens: 16_000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Source {
    pub key: String,
    pub title: String,
    pub kind: String,
    pub path: Option<String>,
    pub git_sha: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    /// Answer text, with citations rewritten to [1], [2], ...
    pub text: String,
    /// Cited documents, in citation order
    pub sources: Vec<Source>,
    /// Citations to keys that were never shown to the model
    pub unknown_citations: Vec<String>,
    pub tool_calls: Vec<String>,
    pub usage: Usage,
    pub cost_usd: f64,
    pub model: String,
    /// "end_turn", "refusal", "max_tokens", "max_steps"
    pub outcome: String,
}

const SYSTEM_BASE: &str = r#"You answer questions about the game MyHordes (the community remake of Hordes / Die2Nite / Zombinoia) for players on a Discord server.

Your knowledge comes from a knowledge base generated from the MyHordes source code:
- entity cards: items, constructions (buildings), ruins, jobs, statuses, roles, hero skills, town types. Cards list actions with their internal requirement and effect ids, workshop recipes, drop tables, and building costs.
- help: the in-game help pages.
- rule: the default town configuration values, plus, for each other town type (small, remote, panda = Pandemonium, custom), the list of values that differ from the default.

How to answer:
- Base your answer on knowledge base documents. Search again with different words when the first results don't contain the answer: try the official in-game name, English, or a broader topic. Players use nicknames and slang.
- MyHordes differs from the original Hordes in many details. If the knowledge base does not contain the answer, say so plainly instead of guessing. If you add something from general knowledge, mark it as unverified.
- Not finding a rule is not proof that it doesn't exist: the knowledge base does not cover every formula of the game code, and many bonuses and modifiers are hidden. Before concluding that an effect does not exist, run at least one more search aimed at it from another angle (constructions, items, statuses or hero skills related to the subject). If you still find nothing, don't open with "no": say what your documents do show, that they don't mention such an effect, and point out any related hint you found (for example a construction or item description that alludes to it).
- Reply in the language of the question, using the in-game names of that language (they are on the cards).
- Internal ids in cards (such as `drink_tl0` or `StatusEffect(pointType=1)`) are implementation details: translate them into what they mean for the player. Drop-table percentages are the share of finds in that table, not the chance per search.
- Mention it when a rule depends on the town type, a job, or a status.
- Cite the documents you used: write their key in square brackets right after the statement they support, e.g. [card:item:water_#00:fr]. Only cite keys that appear in the documents you were given."#;

const SYSTEM_PLAYER: &str = r#"Audience: players. Keep the answer short enough for a Discord message (a few sentences or a short list, under about 200 words). Give concrete numbers and conditions. No source code, file paths or configuration keys."#;

const SYSTEM_DEV: &str = r#"Audience: MyHordes developers and contributors. Be precise and technical: include internal ids, configuration keys with their TownSetting names, and file paths where the documents give them. Up to about 400 words."#;

fn tools() -> Value {
    let langs: Vec<&str> = LANGS.to_vec();
    json!([
        {
            "name": "search",
            "description": "Hybrid (semantic + keyword + name) search over the MyHordes knowledge base. Returns the best matching document excerpts with their keys.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, in natural language or with in-game names"},
                    "lang": {"type": "string", "enum": langs, "description": "Language of the documents to search (language-neutral rule documents are always included)"},
                    "kinds": {"type": "array", "items": {"type": "string", "enum": ["entity_card", "help", "rule"]}, "description": "Restrict to these document kinds"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Number of results (default 6)"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "get_document",
            "description": "Read a whole document by key (for example a full help page or a long entity card). Keys look like card:item:water_#00:fr, help:extreme_camping:en, rule:small:diff.",
            "input_schema": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"]
            }
        },
        {
            "name": "find_entity",
            "description": "Find items, buildings, ruins, jobs, statuses, roles or hero skills by (partial or approximate) name in any language. Returns entity ids and the keys of their cards.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "lang": {"type": "string", "enum": langs, "description": "Language of the cards to return"}
                },
                "required": ["name"]
            }
        }
    ])
}

/// Keys of the documents shown to the model, with their metadata
type Seen = HashMap<String, Source>;

fn render_hits(hits: &[Hit], seen: &mut Seen) -> String {
    if hits.is_empty() {
        return "No results.".into();
    }
    let mut out = String::new();
    for h in hits {
        seen.entry(h.key.clone()).or_insert_with(|| Source { key: h.key.clone(), title: h.title.clone(), kind: h.kind.clone(), path: h.path.clone(), git_sha: h.git_sha.clone() });
        let _ = writeln!(out, "<document key=\"{}\" kind=\"{}\">\n{}\n</document>", h.key, h.kind, h.content);
    }
    out
}

async fn run_tool(pool: &PgPool, embedder: &Embedder, mode: Mode, default_lang: &str, name: &str, input: &Value, seen: &mut Seen) -> Result<String> {
    let str_arg = |k: &str| input.get(k).and_then(Value::as_str).map(str::to_string);
    match name {
        "search" => {
            let query = str_arg("query").unwrap_or_default();
            let kinds = input.get("kinds").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect());
            let k = input.get("k").and_then(Value::as_u64).unwrap_or(6).clamp(1, 10) as usize;
            let opts = SearchOptions {
                k,
                lang: Some(str_arg("lang").unwrap_or_else(|| default_lang.to_string())),
                kinds,
                audience: Some(mode.audience().to_string()),
            };
            let hits = search::search(pool, embedder, &query, &opts).await?;
            Ok(render_hits(&hits, seen))
        }
        "get_document" => {
            let key = str_arg("key").unwrap_or_default();
            let row: Option<(String, String, String, Option<String>, Option<String>)> =
                sqlx::query_as("SELECT title, kind, content, path, git_sha FROM documents WHERE key = $1").bind(&key).fetch_optional(pool).await?;
            Ok(match row {
                Some((title, kind, content, path, git_sha)) => {
                    seen.entry(key.clone()).or_insert_with(|| Source { key: key.clone(), title: title.clone(), kind: kind.clone(), path, git_sha });
                    let content: String = content.chars().take(12_000).collect();
                    format!("<document key=\"{key}\" kind=\"{kind}\">\n# {title}\n{content}\n</document>")
                }
                None => format!("No document with key '{key}'. Use search or find_entity to get valid keys."),
            })
        }
        "find_entity" => {
            let name = str_arg("name").unwrap_or_default();
            let lang = str_arg("lang").unwrap_or_else(|| default_lang.to_string());
            let rows: Vec<(String, String, f32)> = sqlx::query_as(
                "SELECT n.entity_id, (SELECT name FROM entity_names WHERE entity_id = n.entity_id AND lang = $2 LIMIT 1),
                        max(greatest(similarity(lower(n.name), lower($1)), word_similarity(lower($1), lower(n.name))))
                 FROM entity_names n
                 WHERE lower(n.name) % lower($1) OR lower($1) <% lower(n.name) OR lower(n.name) LIKE '%' || lower($1) || '%'
                 GROUP BY n.entity_id ORDER BY 3 DESC LIMIT 10",
            )
            .bind(&name)
            .bind(&lang)
            .fetch_all(pool)
            .await?;
            if rows.is_empty() {
                return Ok(format!("No entity matches '{name}'."));
            }
            let mut out = String::new();
            for (id, local_name, score) in rows {
                let _ = writeln!(out, "- {local_name} — entity {id}, card key card:{id}:{lang} (match {score:.2})");
            }
            Ok(out)
        }
        other => Ok(format!("Unknown tool '{other}'.")),
    }
}

static CITATION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[((?:card|help|rule|code|summary|topic):[^\]\s]+)\]").unwrap());

/// Rewrites `[doc-key]` citations to `[n]` and returns the cited sources in order.
fn resolve_citations(text: &str, seen: &Seen) -> (String, Vec<Source>, Vec<String>) {
    let mut order: Vec<String> = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let rewritten = CITATION.replace_all(text, |c: &regex::Captures| {
        let key = c[1].to_string();
        if !seen.contains_key(&key) {
            if !unknown.contains(&key) {
                unknown.push(key);
            }
            return String::new();
        }
        let n = match order.iter().position(|k| *k == key) {
            Some(i) => i + 1,
            None => {
                order.push(key);
                order.len()
            }
        };
        format!("[{n}]")
    });
    // "[1][2]" → "[1, 2]" reads better; also drop spaces left before removed citations
    let rewritten = rewritten.replace("][", ", ").replace(" .", ".").replace(" ,", ",");
    let sources = order.iter().filter_map(|k| seen.get(k).cloned()).collect();
    (rewritten.trim().to_string(), sources, unknown)
}

/// Answers `question`. `lang` is the preferred document language (e.g. from the Discord locale);
/// the model replies in the language of the question regardless.
pub async fn answer(pool: &PgPool, embedder: &Embedder, claude: &Claude, cfg: &AnswerConfig, mode: Mode, lang: &str, question: &str) -> Result<Answer> {
    let lang = if LANGS.contains(&lang) { lang } else { "en" };
    let mut seen: Seen = HashMap::new();

    let initial = search::search(
        pool,
        embedder,
        question,
        &SearchOptions { k: cfg.initial_k, lang: Some(lang.to_string()), kinds: None, audience: Some(mode.audience().to_string()) },
    )
    .await?;

    let mut messages = vec![json!({
        "role": "user",
        "content": format!(
            "<search_results>\n{}</search_results>\n\n<question>{}</question>",
            render_hits(&initial, &mut seen),
            question
        )
    })];
    let system = json!([
        {"type": "text", "text": SYSTEM_BASE},
        {"type": "text", "text": match mode { Mode::Player => SYSTEM_PLAYER, Mode::Dev => SYSTEM_DEV }, "cache_control": {"type": "ephemeral"}}
    ]);
    let effort = match mode {
        Mode::Player => &cfg.effort.0,
        Mode::Dev => &cfg.effort.1,
    };

    let mut usage = Usage::default();
    let mut tool_calls = Vec::new();
    let mut model_used = cfg.model.clone();
    let mut final_text = String::new();
    let mut outcome = "max_steps".to_string();

    for step in 0..cfg.max_steps {
        // Last step: no tools, so the model must answer with what it has
        let last = step + 1 == cfg.max_steps;
        let mut body = json!({
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"effort": effort},
            // Caches the growing conversation prefix between tool-loop steps
            "cache_control": {"type": "ephemeral"},
        });
        body["tools"] = tools();
        if last {
            body["tool_choice"] = json!({"type": "none"});
        }

        let response = claude.messages(body).await?;
        usage.add(response.usage);
        model_used = response.model().to_string();
        debug!(step, stop = response.stop_reason(), "model step");

        match response.stop_reason() {
            "tool_use" => {
                // Append the assistant turn verbatim (thinking blocks included), then all results in one user turn
                messages.push(json!({"role": "assistant", "content": response.content().clone()}));
                let mut results = Vec::new();
                for (id, name, input) in response.tool_uses() {
                    tool_calls.push(format!("{name}({input})"));
                    let (content, is_error) = match run_tool(pool, embedder, mode, lang, &name, &input, &mut seen).await {
                        Ok(c) => (c, false),
                        Err(e) => (format!("Tool failed: {e}"), true),
                    };
                    results.push(json!({"type": "tool_result", "tool_use_id": id, "content": content, "is_error": is_error}));
                }
                messages.push(json!({"role": "user", "content": results}));
            }
            "refusal" => {
                outcome = "refusal".into();
                final_text = "Sorry, I can't answer that one.".into();
                break;
            }
            other => {
                outcome = other.to_string();
                final_text = response.text();
                break;
            }
        }
    }

    let (text, sources, unknown_citations) = resolve_citations(&final_text, &seen);
    Ok(Answer {
        text,
        sources,
        unknown_citations,
        tool_calls,
        cost_usd: usage.cost_usd(&model_used),
        usage,
        model: model_used,
        outcome,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn citations_are_numbered_and_unknown_ones_dropped() {
        let mut seen = Seen::new();
        for k in ["card:item:water_#00:fr", "help:ghouls:fr"] {
            seen.insert(k.into(), Source { key: k.into(), title: k.into(), kind: "x".into(), path: None, git_sha: None });
        }
        let (text, sources, unknown) = resolve_citations(
            "Oui [card:item:water_#00:fr][help:ghouls:fr], mais [card:item:fake:fr]. Encore [card:item:water_#00:fr].",
            &seen,
        );
        assert_eq!(text, "Oui [1, 2], mais. Encore [1].");
        assert_eq!(sources.len(), 2);
        assert_eq!(unknown, vec!["card:item:fake:fr".to_string()]);
    }
}
