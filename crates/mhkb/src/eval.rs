//! Evaluation against `eval/questions.yaml`:
//! - retrieval: hit@1, hit@k and MRR, overall and per level
//! - answers (`--answers`): the bot answers questions that have reference `facts`, and Claude grades them

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use kb_core::answer::{self, AnswerConfig, Mode};
use kb_core::claude::{Claude, Usage};
use kb_core::embed::Embedder;
use kb_core::search::{self, SearchOptions};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::PgPool;

#[derive(Deserialize)]
struct Question {
    q: String,
    lang: Option<String>,
    level: String,
    /// Empty for questions the KB should not answer (off-topic, nonexistent things)
    #[serde(default)]
    expect: Vec<String>,
    #[serde(default)]
    facts: Vec<String>,
    note: Option<String>,
}

fn load(path: &Path, only: Option<&str>) -> Result<Vec<Question>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let questions: Vec<Question> = serde_yaml::from_str(&raw)?;
    Ok(questions.into_iter().filter(|q| only.is_none_or(|o| q.q.to_lowercase().contains(&o.to_lowercase()))).collect())
}

#[derive(Serialize)]
struct Outcome {
    q: String,
    level: String,
    /// 1-based rank of the first expected document, if retrieved
    rank: Option<usize>,
    top: Vec<String>,
}

/// `expected` is a full key or a key without its trailing `:<lang>`
fn matches(key: &str, expected: &str) -> bool {
    key == expected || key.strip_prefix(expected).is_some_and(|rest| rest.starts_with(':'))
}

pub async fn run(pool: &PgPool, embedder: &Embedder, questions: &Path, only: Option<&str>, k: usize, out: Option<&Path>, verbose: bool) -> Result<()> {
    let (questions, unanswerable): (Vec<Question>, Vec<Question>) = load(questions, only)?.into_iter().partition(|q| !q.expect.is_empty());
    check_expectations(pool, &questions).await?;

    let mut outcomes = Vec::new();
    for question in &questions {
        let opts = SearchOptions { k, lang: question.lang.clone(), ..Default::default() };
        let hits = search::search(pool, embedder, &question.q, &opts).await?;
        let rank = hits.iter().position(|h| question.expect.iter().any(|e| matches(&h.key, e))).map(|p| p + 1);

        let mark = match rank {
            Some(1) => "✔ 1".to_string(),
            Some(r) => format!("~ {r}"),
            None => "✘ -".to_string(),
        };
        let note = question.note.as_deref().map(|n| format!("  ({n})")).unwrap_or_default();
        println!("{mark:<4} [{:<6}] {}{note}", question.level, question.q);
        if verbose && rank != Some(1) {
            println!("         expected: {}", question.expect.join(", "));
            for h in hits.iter().take(3) {
                println!("         got: {} ({})", h.key, h.matched_by.join("+"));
            }
        }
        outcomes.push(Outcome {
            q: question.q.clone(),
            level: question.level.clone(),
            rank,
            top: hits.iter().map(|h| h.key.clone()).collect(),
        });
    }

    let summarize = |set: &[&Outcome]| -> String {
        let n = set.len().max(1) as f64;
        let hit1 = set.iter().filter(|o| o.rank == Some(1)).count();
        let hitk = set.iter().filter(|o| o.rank.is_some_and(|r| r <= k)).count();
        let mrr: f64 = set.iter().filter_map(|o| o.rank).map(|r| 1.0 / r as f64).sum::<f64>() / n;
        format!("hit@1 {:>3.0}%  hit@{k} {:>3.0}%  MRR {:.2}  (n={})", hit1 as f64 / n * 100.0, hitk as f64 / n * 100.0, mrr, set.len())
    };

    println!();
    let mut by_level: BTreeMap<&str, Vec<&Outcome>> = BTreeMap::new();
    for o in &outcomes {
        by_level.entry(o.level.as_str()).or_default().push(o);
    }
    for (level, set) in &by_level {
        println!("{level:<8} {}", summarize(set));
    }
    println!("{:<8} {}", "all", summarize(&outcomes.iter().collect::<Vec<_>>()));
    if !unanswerable.is_empty() {
        println!("({} questions without expected documents skipped)", unanswerable.len());
    }

    if let Some(out) = out {
        std::fs::write(out, serde_json::to_string_pretty(&outcomes)?)?;
        println!("\nresults written to {}", out.display());
    }
    Ok(())
}

/// Warns about expected keys that match no document: a typo, or a document that was renamed or
/// pruned, would otherwise silently count as a miss
async fn check_expectations(pool: &PgPool, questions: &[Question]) -> Result<()> {
    let keys: Vec<String> = sqlx::query_scalar("SELECT key FROM documents").fetch_all(pool).await?;
    let mut unknown: Vec<&str> = questions
        .iter()
        .flat_map(|q| &q.expect)
        .filter(|e| !keys.iter().any(|k| matches(k, e)))
        .map(String::as_str)
        .collect();
    unknown.sort_unstable();
    unknown.dedup();
    for e in &unknown {
        println!("⚠ expected document not in the KB: {e}");
    }
    if !unknown.is_empty() {
        println!();
    }
    Ok(())
}

/// Prints YAML stubs for bot questions (from `qa_log`) that aren't in the test set yet, with the
/// documents the bot cited as candidate expectations. Expectations and facts must be checked by
/// hand before the stub goes into `eval/questions.yaml`.
pub async fn stubs(pool: &PgPool, questions: &Path) -> Result<()> {
    let known: Vec<String> = load(questions, None)?.into_iter().map(|q| q.q.trim().to_lowercase()).collect();
    let rows: Vec<(i64, String, String, Option<i16>, Option<Value>, Option<String>)> =
        sqlx::query_as("SELECT id, lang, question, feedback, sources, feedback_comment FROM qa_log ORDER BY id").fetch_all(pool).await?;
    let mut seen = known;
    for (id, lang, question, feedback, sources, comment) in rows {
        let norm = question.trim().to_lowercase();
        if seen.contains(&norm) {
            continue;
        }
        seen.push(norm);
        let mut keys: Vec<String> = sources
            .as_ref()
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|s| s["key"].as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        keys.dedup();
        let feedback = match feedback {
            Some(f) if f < 0 => ", 👎",
            Some(f) if f > 0 => ", 👍",
            _ => "",
        };
        println!("- q: {}", serde_json::to_string(question.trim())?);
        println!("  lang: {lang}");
        println!("  level: casual");
        println!("  expect: [{}]", keys.join(", "));
        match comment {
            // The remark usually names the wrong or missing fact: a starting point for `facts`
            Some(c) => println!("  facts: []  # remark: {}", c.replace('\n', " ")),
            None => println!("  facts: []"),
        }
        println!("  note: \"qa_log #{id}{feedback}\"\n");
    }
    Ok(())
}

// ---- Answer grading ------------------------------------------------------------------------

const JUDGE_SYSTEM: &str = r#"You grade answers given by a Discord bot to questions about the game MyHordes.
You get the question, reference facts that were verified against the game's source code and data, and the bot's answer (in any language).

For each reference fact, decide whether the answer states it ("covered"), does not state it ("missing"), or states something incompatible with it ("contradicted"). Equivalent wording, translations and rounding count as covered.

Then give one verdict:
- "correct": every fact is covered and nothing in the answer contradicts a fact.
- "partial": at least one fact is covered, none is contradicted.
- "abstained": the answer says it doesn't know or can't find the information, and contradicts no fact.
- "wrong": the answer contradicts at least one fact, or covers none of them while claiming to answer.

Set "overconfident" to true when the answer states something incompatible with the facts as certain, without hedging."#;

fn judge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "status": {"type": "string", "enum": ["covered", "missing", "contradicted"]}
                    },
                    "required": ["fact", "status"],
                    "additionalProperties": false
                }
            },
            "verdict": {"type": "string", "enum": ["correct", "partial", "abstained", "wrong"]},
            "overconfident": {"type": "boolean"},
            "comment": {"type": "string"}
        },
        "required": ["facts", "verdict", "overconfident", "comment"],
        "additionalProperties": false
    })
}

#[derive(Serialize, Deserialize)]
struct Judgement {
    facts: Vec<Value>,
    verdict: String,
    overconfident: bool,
    comment: String,
}

#[derive(Serialize)]
struct Graded {
    q: String,
    level: String,
    answer: String,
    sources: Vec<String>,
    tool_calls: Vec<String>,
    judgement: Judgement,
    answer_cost_usd: f64,
    latency_s: f32,
}

fn score(verdict: &str) -> f64 {
    match verdict {
        "correct" => 1.0,
        "partial" => 0.5,
        "abstained" => 0.25,
        _ => 0.0,
    }
}

async fn judge(claude: &Claude, model: &str, q: &Question, answer: &str) -> Result<(Judgement, Usage)> {
    let facts = q.facts.iter().map(|f| format!("- {f}")).collect::<Vec<_>>().join("\n");
    let body = json!({
        "model": model,
        "max_tokens": 4000,
        "system": JUDGE_SYSTEM,
        "messages": [{
            "role": "user",
            "content": format!("<question>{}</question>\n\n<reference_facts>\n{facts}\n</reference_facts>\n\n<answer>\n{answer}\n</answer>", q.q)
        }],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": judge_schema()}}
    });
    let response = claude.messages(body).await?;
    if response.stop_reason() == "refusal" {
        return Err(anyhow!("judge refused"));
    }
    let judgement: Judgement = serde_json::from_str(&response.text()).context("parsing judge output")?;
    Ok((judgement, response.usage))
}

pub async fn run_answers(
    pool: &PgPool,
    embedder: &Embedder,
    claude: &Claude,
    questions: &Path,
    only: Option<&str>,
    cfg: &AnswerConfig,
    judge_model: &str,
    out: Option<&Path>,
    verbose: bool,
) -> Result<()> {
    let questions: Vec<Question> = load(questions, only)?.into_iter().filter(|q| !q.facts.is_empty()).collect();
    println!("Grading {} answers from {} (judge: {judge_model})\n", questions.len(), cfg.model);

    let mut graded = Vec::new();
    let mut judge_usage = Usage::default();
    for q in &questions {
        let started = Instant::now();
        let lang = q.lang.as_deref().unwrap_or("en");
        let a = answer::answer(pool, embedder, claude, cfg, Mode::Player, lang, &q.q).await?;
        let latency_s = started.elapsed().as_secs_f32();
        let (judgement, usage) = judge(claude, judge_model, q, &a.text).await?;
        judge_usage.add(usage);

        let mark = match judgement.verdict.as_str() {
            "correct" => "✔",
            "partial" => "~",
            "abstained" => "?",
            _ => "✘",
        };
        let flag = if judgement.overconfident { " ⚠ overconfident" } else { "" };
        println!("{mark} {:<9} [{:<6}] {}{flag}", judgement.verdict, q.level, q.q);
        if verbose && judgement.verdict != "correct" {
            println!("    judge: {}", judgement.comment);
        }
        graded.push(Graded {
            q: q.q.clone(),
            level: q.level.clone(),
            answer: a.text,
            sources: a.sources.iter().map(|s| s.key.clone()).collect(),
            tool_calls: a.tool_calls,
            judgement,
            answer_cost_usd: a.cost_usd,
            latency_s,
        });
    }

    let n = graded.len().max(1) as f64;
    let count = |v: &str| graded.iter().filter(|g| g.judgement.verdict == v).count();
    let mean: f64 = graded.iter().map(|g| score(&g.judgement.verdict)).sum::<f64>() / n;
    let overconfident = graded.iter().filter(|g| g.judgement.overconfident).count();
    let answer_cost: f64 = graded.iter().map(|g| g.answer_cost_usd).sum();
    let latency: f32 = graded.iter().map(|g| g.latency_s).sum::<f32>() / n as f32;

    println!(
        "\nscore {:.2}  ·  correct {}  partial {}  abstained {}  wrong {}  ·  overconfident {}",
        mean,
        count("correct"),
        count("partial"),
        count("abstained"),
        count("wrong"),
        overconfident
    );
    let mut by_level: BTreeMap<&str, Vec<&Graded>> = BTreeMap::new();
    for g in &graded {
        by_level.entry(g.level.as_str()).or_default().push(g);
    }
    for (level, set) in &by_level {
        let s: f64 = set.iter().map(|g| score(&g.judgement.verdict)).sum::<f64>() / set.len() as f64;
        println!("  {level:<8} {s:.2} (n={})", set.len());
    }
    println!(
        "answers: ${answer_cost:.3} total, ${:.4}/answer, {latency:.1}s avg  ·  judge: ${:.3}",
        answer_cost / n,
        judge_usage.cost_usd(judge_model)
    );

    if let Some(out) = out {
        std::fs::write(out, serde_json::to_string_pretty(&graded)?)?;
        println!("results written to {}", out.display());
    }
    Ok(())
}
