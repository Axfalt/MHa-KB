//! Review of 👎 answers: re-answers the question with a stronger model on the same knowledge base,
//! then has Claude compare both answers with the voter's remark to tell where the problem lies:
//! - "model": the stronger model gets it right, so the data was there and the answering model reasoned poorly
//! - "knowledge": the stronger model misses it too, so the knowledge base (or what the bot is shown) lacks it
//! - "knowledge_fixed": the re-answer gets it right thanks to documents updated after the original answer
//! - "original_fine": the remark isn't supported, the original answer was acceptable

use anyhow::{Context, Result, anyhow};
use kb_core::answer::{self, AnswerConfig, Mode};
use kb_core::claude::Claude;
use kb_core::embed::Embedder;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::PgPool;
use tracing::info;

const JUDGE_SYSTEM: &str = r#"You review answers given by a Discord bot to questions about the game MyHordes. A user voted 👎 on an answer.
You get the question, the user's remark (may be empty), the original answer, and a second answer to the same question from a stronger model using the knowledge base as it is now. Documents updated since the original answer are listed: the second answer may be better only because of them.

Neither you nor the answers have access to the game itself. The remark comes from a player who usually knows the game well, but it can be wrong. Do not invent game facts.

Decide:
- original_problems: what is wrong, missing or unclear in the original answer, in light of the remark and the second answer. Empty if nothing.
- diagnosis:
  - "model": the second answer fixes the problems without relying on updated documents, so the information was available and the original model reasoned or searched poorly.
  - "knowledge_fixed": the second answer fixes the problems thanks to documents updated since the original answer.
  - "knowledge": the second answer has the same problems (it misses or contradicts what the remark says), so the knowledge base or what the bot is shown probably lacks it.
  - "original_fine": the original answer is acceptable and the remark is not supported by either answer.
  - "unclear": you cannot tell, for example the answers contradict each other and the remark gives no hint.
- missing_facts: facts the remark (or the second answer) says are needed and that the original answer lacks; mark each with whether the second answer has it.
- summary: one to three sentences for the maintainer.

Write every field in English, whatever the language of the question and answers."#;

fn judge_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "original_problems": {"type": "array", "items": {"type": "string"}},
            "diagnosis": {"type": "string", "enum": ["model", "knowledge", "knowledge_fixed", "original_fine", "unclear"]},
            "missing_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fact": {"type": "string"},
                        "in_second_answer": {"type": "boolean"}
                    },
                    "required": ["fact", "in_second_answer"],
                    "additionalProperties": false
                }
            },
            "summary": {"type": "string"}
        },
        "required": ["original_problems", "diagnosis", "missing_facts", "summary"],
        "additionalProperties": false
    })
}

#[derive(Serialize, Deserialize)]
struct Verdict {
    original_problems: Vec<String>,
    diagnosis: String,
    missing_facts: Vec<Value>,
    summary: String,
}

pub struct Options {
    pub id: Option<i64>,
    pub model: String,
    pub judge_model: String,
    /// Review again answers that already have a review
    pub force: bool,
}

type Row = (i64, chrono::DateTime<chrono::Utc>, String, String, String, Option<String>, Option<String>, Option<String>);

pub async fn run(pool: &PgPool, embedder: &Embedder, claude: &Claude, opts: &Options) -> Result<()> {
    let rows: Vec<Row> = match opts.id {
        Some(id) => sqlx::query_as("SELECT id, created_at, mode, lang, question, answer, model, feedback_comment FROM qa_log WHERE id = $1")
            .bind(id)
            .fetch_all(pool)
            .await?,
        None => sqlx::query_as(
            "SELECT id, created_at, mode, lang, question, answer, model, feedback_comment FROM qa_log
             WHERE feedback < 0 AND answer IS NOT NULL AND ($1 OR review IS NULL) ORDER BY id",
        )
        .bind(opts.force)
        .fetch_all(pool)
        .await?,
    };
    if rows.is_empty() {
        println!("Nothing to review{}.", if opts.id.is_some() { "" } else { " (no 👎 answer without a review; --force reviews them again)" });
        return Ok(());
    }

    let mut total = 0.0;
    for (id, asked_at, mode, lang, question, original, original_model, remark) in rows {
        let original = original.ok_or_else(|| anyhow!("#{id} has no answer (it failed)"))?;
        let mode = if mode == "dev" { Mode::Dev } else { Mode::Player };
        let cfg = AnswerConfig { model: opts.model.clone(), ..Default::default() };
        let second = answer::answer(pool, embedder, claude, &cfg, mode, &lang, &question).await?;

        let titles: Vec<&str> = second.sources.iter().map(|s| s.title.as_str()).collect();
        let keys: Vec<&str> = second.sources.iter().map(|s| s.key.as_str()).collect();
        let updated: Vec<String> = sqlx::query_scalar("SELECT title FROM documents WHERE key = ANY($1) AND updated_at > $2")
            .bind(&keys)
            .bind(asked_at)
            .fetch_all(pool)
            .await?;
        let body = json!({
            "model": opts.judge_model,
            "max_tokens": 4000,
            "system": JUDGE_SYSTEM,
            "messages": [{
                "role": "user",
                "content": format!(
                    "<question>{question}</question>\n\n<remark>{}</remark>\n\n<original_answer model=\"{}\">\n{original}\n</original_answer>\n\n<second_answer model=\"{}\" documents=\"{}\" updated_since_original=\"{}\">\n{}\n</second_answer>",
                    remark.as_deref().unwrap_or(""),
                    original_model.as_deref().unwrap_or("?"),
                    second.model,
                    titles.join("; "),
                    updated.join("; "),
                    second.text
                )
            }],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": judge_schema()}}
        });
        let response = claude.messages(body).await?;
        if response.stop_reason() == "refusal" {
            return Err(anyhow!("judge refused #{id}"));
        }
        let verdict: Verdict = serde_json::from_str(&response.text()).context("parsing review verdict")?;
        let cost = second.cost_usd + response.usage.cost_usd(&opts.judge_model);
        total += cost;

        sqlx::query("UPDATE qa_log SET review = $1 WHERE id = $2")
            .bind(json!({
                "at": chrono::Utc::now(),
                "model": second.model,
                "answer": second.text,
                "sources": second.sources.iter().map(|s| &s.key).collect::<Vec<_>>(),
                "updated_documents": updated,
                "verdict": verdict,
                "cost_usd": cost,
            }))
            .bind(id)
            .execute(pool)
            .await?;
        info!(qa_id = id, diagnosis = verdict.diagnosis, "reviewed");

        println!("#{id} [{lang}] {question}");
        if let Some(remark) = &remark {
            println!("  remark: {remark}");
        }
        println!("\n  ── original ({}) ──\n{}\n", original_model.as_deref().unwrap_or("?"), indent(&original));
        println!("  ── re-answer ({}) ──\n{}\n", second.model, indent(&second.text));
        if !updated.is_empty() {
            println!("  documents updated since the original answer: {}", updated.join("; "));
        }
        println!("  → diagnosis: {} — {}", verdict.diagnosis, verdict.summary);
        for p in &verdict.original_problems {
            println!("    - problem: {p}");
        }
        for f in &verdict.missing_facts {
            let fixed = if f["in_second_answer"].as_bool() == Some(true) { "re-answer has it" } else { "still missing" };
            println!("    - missing: {} ({fixed})", f["fact"].as_str().unwrap_or_default());
        }
        println!("  cost ${cost:.3}\n");
    }
    println!("total ${total:.3}");
    Ok(())
}

fn indent(text: &str) -> String {
    text.lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n")
}
