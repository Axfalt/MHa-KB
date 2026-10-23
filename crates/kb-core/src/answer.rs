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
use crate::store;

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
- summary: plain-language explanations (in English) of the game's source code, method by method, with the exact numbers, chances, formulas and conditions. They are the only place where hidden mechanics are documented: use them for any question about chances, formulas, modifiers, or "does X affect Y". Search them with kinds ["summary"].
- glossary: abbreviations and nicknames used by players (e.g. AAA, OM, TDG).
- topic: dossiers (in English) that connect all the pieces of one mechanic (digging, camping, night watch, scouts, ...) with their numbers and sources. Start with them for "how does X work" questions, then open their sources for details.

Summaries are written automatically and can contain mistakes. When a number or rule in your answer comes only from a summary, check it in the code with get_source (the summary gives the file and lines) before stating it, and cite the summary.

Texts shown to players are written in German in the code; get_source follows them with their official in-game translation. When you quote a game text, quote that translation in the language of the answer, not your own.

How to answer:
- Base your answer on knowledge base documents. Search again with different words when the first results don't contain the answer: try the official in-game name, English, or a broader topic. Players use nicknames and slang.
- MyHordes differs from the original Hordes in many details. If the knowledge base does not contain the answer, say so plainly instead of guessing. If you add something from general knowledge, mark it as unverified.
- Not finding a rule is not proof that it doesn't exist: the knowledge base does not cover every formula of the game code, and many bonuses and modifiers are hidden. Before concluding that an effect does not exist, run at least one more search aimed at it from another angle (constructions, items, statuses or hero skills related to the subject). If you still find nothing, don't open with "no": say what your documents do show, that they don't mention such an effect, and point out any related hint you found (for example a construction or item description that alludes to it).
- Reply in the language of the question, using the in-game names of that language (they are on the cards).
- Internal ids in cards (such as `drink_tl0` or `StatusEffect(pointType=1)`) are implementation details: translate them into what they mean for the player. Drop-table percentages are the share of finds in that table, not the chance per search.
- The asker cannot reply to you: this is a single answer, not a conversation. Do the lookups now instead of offering to do them later, and don't end with questions.
- For questions that compare or rank ("the best", "the most", "which ... gives the highest ..."), use list_entities to see every candidate instead of judging from a few search results.
- Mention it when a rule depends on the town type, a job, or a status.
- Cite the documents you used: write their key in square brackets right after the statement they support, e.g. [card:item:water_#00:fr]. Only cite keys that appear in the documents you were given."#;

const SYSTEM_PLAYER: &str = r#"Audience: players. Keep the answer short enough for a Discord message (a few sentences or a short list, under about 200 words). Give concrete numbers and conditions. Keep the limits that change what a player can do: how often something works (per day, per town) and how the numbers differ by town type (small, default, Pandemonium) when the documents say so; cut background before cutting these. No source code, file paths or configuration keys."#;

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
                    "kinds": {"type": "array", "items": {"type": "string", "enum": ["entity_card", "help", "rule", "glossary", "summary", "topic", "code"]}, "description": "Restrict to these document kinds"},
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
        },
        {
            "name": "get_source",
            "description": "Read lines of a MyHordes source file, e.g. to verify a number or condition given by a summary document, or to find the exact text the game shows (German in the code, followed by its official translation). Paths look like src/EventListener/Game/Citizen/CitizenChanceQueryListener.php. At most 150 lines per call.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": ["path", "start_line", "end_line"]
            }
        },
        {
            "name": "list_entities",
            "description": "List every entity of a kind with its key numbers, optionally sorted (highest first). Use it to compare or rank: the best camping ruin, the cheapest or strongest construction, all heavy items... Fields: ruin: camping (camping bonus), min_dist, max_dist (km), capacity, empty (chance to be empty), chance (spawn weight); building: defense, ap, health, isTemporary, blueprintLevel, parentBuilding; item: category, heavy, deco, watchpoint (night watch defense); profession: dig_bonus, hero; heroskill: unlockAt (hero days).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["ruin", "building", "item", "profession", "heroskill", "status", "role", "town"]},
                    "sort_by": {"type": "string", "description": "Field to sort by, highest first"},
                    "limit": {"type": "integer", "description": "Maximum number of rows (default 100)"},
                    "lang": {"type": "string", "enum": langs, "description": "Language of the names"}
                },
                "required": ["kind"]
            }
        }
    ])
}

/// Fields shown by `list_entities`, per entity kind
fn list_fields(kind: &str) -> &'static [&'static str] {
    match kind {
        "ruin" => &["camping", "min_dist", "max_dist", "capacity", "empty", "chance"],
        "building" => &["defense", "ap", "health", "isTemporary", "blueprintLevel", "parentBuilding"],
        "item" => &["category", "heavy", "deco", "watchpoint"],
        "profession" => &["dig_bonus", "hero"],
        "heroskill" => &["unlockAt"],
        "status" => &["volatile"],
        "role" => &["vote"],
        "town" => &["ranked"],
        _ => &[],
    }
}

fn as_number(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Keys of the documents shown to the model, with their metadata
type Seen = HashMap<String, Source>;

/// Entity cards up to this size are shown whole: search returns one chunk per document, and a
/// card's later chunks hold what players ask about (a job's abilities, an item's actions)
const WHOLE_CARD_CHARS: i32 = 5000;

/// Replaces the matched chunk of short entity cards with the whole card, and puts the opening of
/// topic dossiers (their key facts) before the matched chunk: a later chunk alone can mislead, e.g.
/// "Scout digging bonus +0%" without the zone scout levels that do raise digging chances
async fn whole_cards(pool: &PgPool, hits: &mut [Hit]) -> Result<()> {
    let topics: Vec<i64> = hits.iter().filter(|h| h.kind == "topic").map(|h| h.document_id).collect();
    if !topics.is_empty() {
        let rows: Vec<(i64, i64, String)> = sqlx::query_as("SELECT document_id, id, content FROM chunks WHERE document_id = ANY($1) AND ord = 0")
            .bind(&topics)
            .fetch_all(pool)
            .await?;
        for (document_id, chunk_id, intro) in rows {
            if let Some(h) = hits.iter_mut().find(|h| h.document_id == document_id && h.chunk_id != chunk_id) {
                h.content = format!("{intro}
[…]
{}", h.content);
            }
        }
    }
    let ids: Vec<i64> = hits.iter().filter(|h| h.kind == "entity_card").map(|h| h.document_id).collect();
    if ids.is_empty() {
        return Ok(());
    }
    let rows: Vec<(i64, String)> = sqlx::query_as("SELECT id, content FROM documents WHERE id = ANY($1) AND length(content) <= $2")
        .bind(&ids)
        .bind(WHOLE_CARD_CHARS)
        .fetch_all(pool)
        .await?;
    let whole: HashMap<i64, String> = rows.into_iter().collect();
    for h in hits.iter_mut() {
        if let Some(content) = whole.get(&h.document_id) {
            h.content = format!("# {}
{}", h.title, content);
        }
    }
    Ok(())
}

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
            let mut hits = search::search(pool, embedder, &query, &opts).await?;
            whole_cards(pool, &mut hits).await?;
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
        "get_source" => {
            let path = str_arg("path").unwrap_or_default().trim_start_matches('/').replace('\\', "/");
            let start = input.get("start_line").and_then(Value::as_u64).unwrap_or(1).max(1) as usize;
            let end = input.get("end_line").and_then(Value::as_u64).unwrap_or(start as u64) as usize;
            let end = end.clamp(start, start + 149);
            let row: Option<(String, Option<String>)> =
                sqlx::query_as("SELECT content, git_sha FROM source_files WHERE path = $1").bind(&path).fetch_optional(pool).await?;
            let Some((content, git_sha)) = row else {
                return Ok(format!("No indexed source file '{path}'."));
            };
            let key = format!("code:{path}");
            seen.entry(key.clone()).or_insert_with(|| Source { key: key.clone(), title: format!("Source code: {path}"), kind: "code".into(), path: Some(path.clone()), git_sha });
            let lines: String = content.lines().enumerate().skip(start - 1).take(end + 1 - start).map(|(i, l)| format!("{:>5}| {l}\n", i + 1)).collect();
            let mut out = format!("<source key=\"{key}\" lines=\"{start}-{end}\">\n{lines}</source>");
            out.push_str(&game_texts(pool, default_lang, &lines).await?);
            Ok(out)
        }
        "list_entities" => {
            let kind = str_arg("kind").unwrap_or_default();
            let lang = str_arg("lang").unwrap_or_else(|| default_lang.to_string());
            let fields = list_fields(&kind);
            if fields.is_empty() {
                return Ok(format!("Unknown kind '{kind}'."));
            }
            let limit = input.get("limit").and_then(Value::as_u64).unwrap_or(100).clamp(1, 400) as usize;
            let mut rows: Vec<(String, Option<String>, Value)> = sqlx::query_as(
                "SELECT e.id, (SELECT name FROM entity_names n WHERE n.entity_id = e.id AND n.lang = $2 LIMIT 1), e.data
                 FROM entities e WHERE e.kind = $1 ORDER BY e.id",
            )
            .bind(&kind)
            .bind(&lang)
            .fetch_all(pool)
            .await?;
            let sort_by = str_arg("sort_by").filter(|f| fields.contains(&f.as_str()));
            if let Some(field) = &sort_by {
                let key = |d: &Value| d.get(field).and_then(as_number).unwrap_or(f64::NEG_INFINITY);
                rows.sort_by(|a, b| key(&b.2).total_cmp(&key(&a.2)));
            }
            let mut out = format!(
                "{} entities of kind {kind}{}:\n",
                rows.len(),
                sort_by.as_ref().map(|f| format!(", sorted by {f} (highest first)")).unwrap_or_default()
            );
            for (id, name, data) in rows.iter().take(limit) {
                let card = format!("card:{id}:{lang}");
                let name = name.clone().unwrap_or_else(|| id.clone());
                seen.entry(card.clone()).or_insert_with(|| Source { key: card.clone(), title: name.clone(), kind: "entity_card".into(), path: None, git_sha: None });
                let values: Vec<String> = fields
                    .iter()
                    .filter_map(|f| data.get(*f).filter(|v| !v.is_null()).map(|v| format!("{f}={}", v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))))
                    .collect();
                let _ = writeln!(out, "- {name} [{card}]: {}", values.join(", "));
            }
            if rows.len() > limit {
                let _ = writeln!(out, "… {} more not shown (raise limit)", rows.len() - limit);
            }
            Ok(out)
        }
        other => Ok(format!("Unknown tool '{other}'.")),
    }
}

/// German string literals passed to the translator: `$trans->trans('…', [], 'game')`, `T::__("…", 'game')`
static TRANS_LITERAL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?:trans|__)\(\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")"#).unwrap());

/// Official `lang` translations of the game texts in a piece of source code, or "" when there are none
async fn game_texts(pool: &PgPool, lang: &str, code: &str) -> Result<String> {
    let mut texts: Vec<String> = Vec::new();
    for c in TRANS_LITERAL.captures_iter(code) {
        let text = c.get(1).or_else(|| c.get(2)).map_or("", |m| m.as_str()).replace("\\'", "'").replace("\\\"", "\"").replace("\\\\", "\\");
        if !text.is_empty() && !texts.contains(&text) {
            texts.push(text);
        }
    }
    if texts.is_empty() || lang == "de" {
        return Ok(String::new());
    }
    let found = store::translate(pool, lang, &texts).await?;
    if found.is_empty() {
        return Ok(String::new());
    }
    let mut out = format!("\n<game_texts lang=\"{lang}\" note=\"official in-game wording of the German texts above\">\n");
    for (de, text) in found {
        let _ = writeln!(out, "- \"{de}\" → \"{text}\"");
    }
    out.push_str("</game_texts>");
    Ok(out)
}

static CITATION: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[((?:card|help|rule|code|summary|topic|glossary):[^\]\s]+)\]").unwrap());

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

    let mut initial = search::search(
        pool,
        embedder,
        question,
        &SearchOptions { k: cfg.initial_k, lang: Some(lang.to_string()), kinds: None, audience: Some(mode.audience().to_string()) },
    )
    .await?;
    whole_cards(pool, &mut initial).await?;

    let messages = vec![json!({
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
    tool_loop(pool, embedder, claude, cfg, effort, mode, lang, system, messages, seen).await
}

/// Runs the model with the knowledge-base tools until it answers (or `cfg.max_steps` is reached),
/// then resolves its citations against the documents it was shown.
#[allow(clippy::too_many_arguments)]
async fn tool_loop(
    pool: &PgPool,
    embedder: &Embedder,
    claude: &Claude,
    cfg: &AnswerConfig,
    effort: &str,
    mode: Mode,
    lang: &str,
    system: Value,
    mut messages: Vec<Value>,
    mut seen: Seen,
) -> Result<Answer> {
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
        // The model may write its answer next to a last tool call and end with an empty turn
        let text = response.text();
        if !text.trim().is_empty() {
            final_text = text;
        }

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
                // The next request has tools disabled: say so, or the model may keep "researching" in
                // its thinking and end without writing anything
                if step + 2 == cfg.max_steps {
                    results.push(json!({"type": "text", "text": "That was your last tool call: no more tools are available. Write your final answer now with what you have."}));
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
                if final_text.trim().is_empty() {
                    let types: Vec<&str> = response.content().as_array().map(|b| b.iter().filter_map(|b| b["type"].as_str()).collect()).unwrap_or_default();
                    tracing::warn!(stop = other, ?types, "model ended without text");
                }
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

const SYSTEM_RESEARCH: &str = r#"You write reference dossiers about the game mechanics of MyHordes (the community remake of Hordes / Die2Nite / Zombinoia) for a knowledge base that answers player and developer questions.

The knowledge base contains entity cards, in-game help pages, town rules per town type, a player glossary, and "summary" documents that explain the game's source code method by method. Summaries are written automatically and may contain mistakes: verify every number and condition you take from a summary in the code with get_source before writing it down.

Research the topic thoroughly with the tools (searches from several angles, kinds ["summary"] for formulas, get_source to verify, list_entities to compare), then write the dossier in English Markdown:

# <Topic title>
## In short
2 to 4 sentences a player can act on.
## How it works
The mechanic step by step: when it happens, every number, chance and formula with its conditions.
## What changes it
Jobs, statuses, items, constructions, hero skills, roles and settings that modify it, each with its exact effect.
## Town types
Differences between default, small, remote, panda (Pandemonium) and custom towns, if any.
## Related
Connected mechanics.
## Not verified
What you could not confirm in the code or documents (omit the section if nothing).

Cite every factual statement with the key of the document it comes from in square brackets, e.g. [summary:src/Entity/Zone.php:Zone::getScoutLevel] or [code:src/Entity/Zone.php] for code you read with get_source. Only state what the documents and code show. Prefer precision over length, but leave out no number that matters to players."#;

/// Researches a topic with the knowledge-base tools and writes a cited Markdown dossier.
pub async fn research(pool: &PgPool, embedder: &Embedder, claude: &Claude, cfg: &AnswerConfig, effort: &str, topic: &str, scope: &str) -> Result<Answer> {
    let system = json!([{"type": "text", "text": SYSTEM_RESEARCH, "cache_control": {"type": "ephemeral"}}]);
    let messages = vec![json!({
        "role": "user",
        "content": format!("<topic>{topic}</topic>\n<scope>{scope}</scope>\n\nWrite the dossier.")
    })];
    tool_loop(pool, embedder, claude, cfg, effort, Mode::Dev, "en", system, messages, Seen::new()).await
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
