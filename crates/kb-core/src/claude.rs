//! Minimal Claude Messages API client (raw HTTP: there is no official Rust SDK).

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::warn;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const BATCHES_URL: &str = "https://api.anthropic.com/v1/messages/batches";
const API_VERSION: &str = "2023-06-01";
/// Server-side refusal fallback, `fallbacks: "default"` form
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

pub struct Claude {
    http: reqwest::Client,
    api_key: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub cache_creation_input_tokens: u64,
}

impl Usage {
    /// Usage of a Messages API response body
    pub fn from_response(v: &Value) -> Self {
        let n = |k: &str| v.get("usage").and_then(|u| u.get(k)).and_then(Value::as_u64).unwrap_or(0);
        Self {
            input_tokens: n("input_tokens"),
            output_tokens: n("output_tokens"),
            cache_read_input_tokens: n("cache_read_input_tokens"),
            cache_creation_input_tokens: n("cache_creation_input_tokens"),
        }
    }

    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
    }

    /// Approximate cost in USD at list prices (5-minute cache writes).
    pub fn cost_usd(&self, model: &str) -> f64 {
        // (input, output) $ per million tokens
        let (input, output) = match model {
            m if m.starts_with("claude-opus-5-5") => (4.0, 20.0),
            m if m.starts_with("claude-sonnet-5") => (2.0, 10.0),
            m if m.starts_with("claude-haiku-5-5") => (0.10, 0.50),
            m if m.starts_with("claude-fable") => (10.0, 50.0),
            _ => (5.0, 25.0),
        };
        (self.input_tokens as f64 * input
            + self.cache_creation_input_tokens as f64 * input * 1.25
            + self.cache_read_input_tokens as f64 * input * 0.05
            + self.output_tokens as f64 * output)
            / 1_000_000.0
    }
}

pub struct Response {
    pub body: Value,
    pub usage: Usage,
}

impl Response {
    pub fn stop_reason(&self) -> &str {
        self.body.get("stop_reason").and_then(Value::as_str).unwrap_or("")
    }

    pub fn content(&self) -> &Value {
        self.body.get("content").unwrap_or(&Value::Null)
    }

    pub fn model(&self) -> &str {
        self.body.get("model").and_then(Value::as_str).unwrap_or("")
    }

    pub fn text(&self) -> String {
        self.content()
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|b| b.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default()
    }

    /// (id, name, input) of every tool_use block
    pub fn tool_uses(&self) -> Vec<(String, String, Value)> {
        self.content()
            .as_array()
            .map(|blocks| {
                blocks
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                    .map(|b| {
                        (
                            b.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                            b.get("name").and_then(Value::as_str).unwrap_or("").to_string(),
                            b.get("input").cloned().unwrap_or(json!({})),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Models that accept `fallbacks: "default"` on the Claude API
fn supports_default_fallback(model: &str) -> bool {
    ["claude-opus-5", "claude-sonnet-5-5", "claude-fable-5"].iter().any(|p| model.starts_with(p))
}

impl Claude {
    pub fn new(api_key: String) -> Result<Self> {
        if api_key.trim().is_empty() {
            bail!("ANTHROPIC_API_KEY is not set");
        }
        let http = reqwest::Client::builder().timeout(Duration::from_secs(300)).build()?;
        Ok(Self { http, api_key })
    }

    pub fn from_env() -> Result<Self> {
        Self::new(std::env::var("ANTHROPIC_API_KEY").unwrap_or_default())
    }

    fn get(&self, url: &str) -> reqwest::RequestBuilder {
        self.http.get(url).header("x-api-key", &self.api_key).header("anthropic-version", API_VERSION)
    }

    /// Submits a Message Batch (50% price, results within 24h). `requests` are (custom_id, params);
    /// params must not contain `fallbacks`, which the Batches API rejects. Returns the batch id.
    pub async fn create_batch(&self, requests: &[(String, Value)]) -> Result<String> {
        let body = json!({
            "requests": requests.iter().map(|(id, params)| json!({"custom_id": id, "params": params})).collect::<Vec<_>>()
        });
        let resp = self
            .http
            .post(BATCHES_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("creating batch failed ({status}): {text}");
        }
        let v: Value = serde_json::from_str(&text)?;
        v.get("id").and_then(Value::as_str).map(str::to_string).context("batch response without id")
    }

    /// Current state of a batch (`processing_status`, `request_counts`, `results_url`, ...)
    pub async fn get_batch(&self, id: &str) -> Result<Value> {
        let resp = self.get(&format!("{BATCHES_URL}/{id}")).send().await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("reading batch {id} failed ({status}): {text}");
        }
        Ok(serde_json::from_str(&text)?)
    }

    /// Results of an ended batch: (custom_id, result object with `type` succeeded/errored/...)
    pub async fn batch_results(&self, results_url: &str) -> Result<Vec<(String, Value)>> {
        let text = self.get(results_url).send().await?.error_for_status()?.text().await?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: Value = serde_json::from_str(l)?;
                let id = v.get("custom_id").and_then(Value::as_str).unwrap_or("").to_string();
                Ok((id, v.get("result").cloned().unwrap_or(Value::Null)))
            })
            .collect()
    }

    /// POST /v1/messages. `body` must contain `model`; adds the refusal fallback where supported.
    /// Retries rate limits, overload and server errors with backoff.
    pub async fn messages(&self, mut body: Value) -> Result<Response> {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("").to_string();
        let mut betas = Vec::new();
        if supports_default_fallback(&model) {
            body["fallbacks"] = json!("default");
            betas.push(FALLBACK_BETA);
        }

        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut req = self
                .http
                .post(API_URL)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", API_VERSION)
                .json(&body);
            if !betas.is_empty() {
                req = req.header("anthropic-beta", betas.join(","));
            }
            let result = req.send().await;

            let retry_after = match result {
                Ok(resp) => {
                    let status = resp.status();
                    let retry_after = resp
                        .headers()
                        .get("retry-after")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.parse::<u64>().ok());
                    let text = resp.text().await.context("reading Claude API response")?;
                    if status.is_success() {
                        let body: Value = serde_json::from_str(&text).context("parsing Claude API response")?;
                        let usage = Usage::from_response(&body);
                        return Ok(Response { body, usage });
                    }
                    let retryable = status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error();
                    if !retryable || attempt >= 4 {
                        bail!("Claude API error {status}: {text}");
                    }
                    warn!(%status, attempt, "Claude API call failed, retrying");
                    retry_after
                }
                Err(e) => {
                    if attempt >= 4 {
                        return Err(e).context("calling Claude API");
                    }
                    warn!(error = %e, attempt, "Claude API request failed, retrying");
                    None
                }
            };
            let wait = retry_after.unwrap_or(2u64.pow(attempt)).min(60);
            tokio::time::sleep(Duration::from_secs(wait)).await;
        }
    }
}
