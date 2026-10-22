//! Discord bot: `/ask` for players, `/dev` for developers, with feedback buttons,
//! per-user rate limits and a daily spending cap.

use std::sync::{Arc, LazyLock};
use std::time::Instant;

use anyhow::Result;
use kb_core::answer::{self, Answer, AnswerConfig, Mode};
use kb_core::claude::Claude;
use kb_core::embed::Embedder;
use poise::serenity_prelude as serenity;
use regex::Regex;
use serde_json::json;
use sqlx::PgPool;
use tracing::{error, info};

type Error = Box<dyn std::error::Error + Send + Sync>;
type Context<'a> = poise::Context<'a, Data, Error>;

const GITLAB_BLOB: &str = "https://gitlab.com/eternaltwin/myhordes/myhordes/-/blob";
const EMBED_COLOR: u32 = 0x8b1a1a;

/// Citation markers like " [1]" or " [2] [3]" in an answer
static MARKERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*\[\d+\](?:\s*\[\d+\])*").unwrap());

pub struct BotConfig {
    pub token: String,
    /// Register commands on this server only (instant updates while testing); global otherwise
    pub guild_id: Option<u64>,
    pub answer: AnswerConfig,
    pub daily_budget_usd: f64,
    pub user_limit_per_hour: i64,
    /// When set, the bot only answers in these channels
    pub allowed_channels: Vec<u64>,
    /// When set, `/dev` requires this role
    pub dev_role: Option<u64>,
}

impl BotConfig {
    pub fn from_env(answer: AnswerConfig) -> Result<Self> {
        let var = |k: &str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let ids = |k: &str| -> Vec<u64> { var(k).map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect()).unwrap_or_default() };
        Ok(Self {
            token: var("DISCORD_TOKEN").ok_or_else(|| anyhow::anyhow!("DISCORD_TOKEN is not set"))?,
            guild_id: var("DISCORD_GUILD_ID").and_then(|v| v.parse().ok()),
            answer,
            daily_budget_usd: var("DAILY_BUDGET_USD").and_then(|v| v.parse().ok()).unwrap_or(5.0),
            user_limit_per_hour: var("USER_LIMIT_PER_HOUR").and_then(|v| v.parse().ok()).unwrap_or(10),
            allowed_channels: ids("ALLOWED_CHANNEL_IDS"),
            dev_role: var("DEV_ROLE_ID").and_then(|v| v.parse().ok()),
        })
    }
}

pub struct Data {
    pool: PgPool,
    embedder: Arc<Embedder>,
    claude: Claude,
    cfg: BotConfig,
}

/// Knowledge-base language for a Discord locale ("fr", "es-ES", "en-US", ...)
fn locale_lang(locale: Option<&str>) -> &'static str {
    match locale.unwrap_or("en").split('-').next().unwrap_or("en") {
        "fr" => "fr",
        "de" => "de",
        "es" => "es",
        _ => "en",
    }
}

/// The few bot messages that are not written by the model
fn t(lang: &str, key: &str) -> &'static str {
    match (key, lang) {
        ("budget", "fr") => "Le budget du jour est épuisé, réessaie demain.",
        ("budget", "de") => "Das Tagesbudget ist aufgebraucht, versuch es morgen wieder.",
        ("budget", "es") => "El presupuesto de hoy se ha agotado, inténtalo mañana.",
        ("budget", _) => "Today's question budget is used up, please try again tomorrow.",
        ("rate", "fr") => "Tu as posé beaucoup de questions cette dernière heure, réessaie un peu plus tard.",
        ("rate", "de") => "Du hast in der letzten Stunde viele Fragen gestellt, versuch es etwas später.",
        ("rate", "es") => "Has hecho muchas preguntas en la última hora, inténtalo un poco más tarde.",
        ("rate", _) => "You have asked a lot of questions in the last hour, please try again a bit later.",
        ("channel", "fr") => "Je ne réponds pas dans ce salon.",
        ("channel", "de") => "In diesem Kanal antworte ich nicht.",
        ("channel", "es") => "No respondo en este canal.",
        ("channel", _) => "I don't answer in this channel.",
        ("role", "fr") => "La commande /dev est réservée aux développeurs.",
        ("role", "de") => "Der Befehl /dev ist Entwicklern vorbehalten.",
        ("role", "es") => "El comando /dev está reservado a los desarrolladores.",
        ("role", _) => "The /dev command is reserved for developers.",
        ("error", "fr") => "Désolé, quelque chose s'est mal passé. Réessaie plus tard.",
        ("error", "de") => "Entschuldigung, da ist etwas schiefgelaufen. Versuch es später noch einmal.",
        ("error", "es") => "Lo siento, algo salió mal. Inténtalo más tarde.",
        ("error", _) => "Sorry, something went wrong. Please try again later.",
        ("thanks", "fr") => "Merci pour ton retour !",
        ("thanks", "de") => "Danke für dein Feedback!",
        ("thanks", "es") => "¡Gracias por tu opinión!",
        ("thanks", _) => "Thanks for your feedback!",
        _ => "",
    }
}

/// Why this request may not be answered, if it may not
async fn refusal_reason(ctx: Context<'_>, mode: Mode) -> Result<Option<&'static str>, Error> {
    let data = ctx.data();
    let cfg = &data.cfg;
    if !cfg.allowed_channels.is_empty() && !cfg.allowed_channels.contains(&ctx.channel_id().get()) {
        return Ok(Some("channel"));
    }
    if mode == Mode::Dev
        && let Some(role) = cfg.dev_role
    {
        let has_role = match ctx.author_member().await {
            Some(member) => member.roles.iter().any(|r| r.get() == role),
            None => false,
        };
        if !has_role {
            return Ok(Some("role"));
        }
    }
    let spent: f64 = sqlx::query_scalar("SELECT COALESCE(sum(cost_usd), 0) FROM qa_log WHERE created_at >= date_trunc('day', now())")
        .fetch_one(&data.pool)
        .await?;
    if spent >= cfg.daily_budget_usd {
        return Ok(Some("budget"));
    }
    let recent: i64 = sqlx::query_scalar("SELECT count(*) FROM qa_log WHERE user_id = $1 AND created_at > now() - interval '1 hour'")
        .bind(ctx.author().id.to_string())
        .fetch_one(&data.pool)
        .await?;
    if recent >= cfg.user_limit_per_hour {
        return Ok(Some("rate"));
    }
    Ok(None)
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn render(question: &str, a: &Answer, mode: Mode, qa_id: i64, secs: f32) -> poise::CreateReply {
    // Players can't open the cited documents, so their answers carry no citations (they stay in qa_log)
    let (text, cited) = match mode {
        Mode::Player => (MARKERS.replace_all(&a.text, "").into_owned(), &[][..]),
        Mode::Dev => (a.text.clone(), &a.sources[..]),
    };
    let mut sources = String::new();
    for (i, s) in cited.iter().enumerate() {
        let link = match (mode, &s.path, &s.git_sha) {
            (Mode::Dev, Some(path), Some(sha)) => Some(format!("{GITLAB_BLOB}/{sha}/{path}")),
            _ => None,
        };
        let title = match s.kind.as_str() {
            // Machine-written: say so, so readers weigh it accordingly
            "summary" => format!("{} — automatic summary of the game code", s.title.trim_start_matches("How it works: ")),
            "code" => s.title.clone(),
            _ => s.title.trim_start_matches("Help: ").to_string(),
        };
        let line = match link {
            Some(url) => format!("[{}] [{}]({url})\n", i + 1, title),
            None => format!("[{}] {}\n", i + 1, title),
        };
        if sources.len() + line.len() > 1000 {
            sources.push('…');
            break;
        }
        sources.push_str(&line);
    }

    let mut footer = format!("{} · {:.0}s", a.model, secs);
    if mode == Mode::Dev {
        footer.push_str(&format!(" · ${:.3}", a.cost_usd));
    }
    let mut embed = serenity::CreateEmbed::new()
        .title(truncate(question, 250))
        .description(truncate(&text, 4000))
        .color(EMBED_COLOR)
        .footer(serenity::CreateEmbedFooter::new(footer));
    if !sources.is_empty() {
        embed = embed.field("Sources", sources, false);
    }

    let buttons = serenity::CreateActionRow::Buttons(vec![
        serenity::CreateButton::new(format!("fb:{qa_id}:1")).emoji('👍').style(serenity::ButtonStyle::Secondary),
        serenity::CreateButton::new(format!("fb:{qa_id}:-1")).emoji('👎').style(serenity::ButtonStyle::Secondary),
    ]);
    poise::CreateReply::default().embed(embed).components(vec![buttons])
}

async fn handle(ctx: Context<'_>, mode: Mode, question: String) -> Result<(), Error> {
    let lang = locale_lang(ctx.locale());
    if let Some(reason) = refusal_reason(ctx, mode).await? {
        ctx.send(poise::CreateReply::default().content(t(lang, reason)).ephemeral(true)).await?;
        return Ok(());
    }
    // Answers take several seconds; Discord needs an acknowledgement within 3
    ctx.defer().await?;

    let data = ctx.data();
    let started = Instant::now();
    let result = answer::answer(&data.pool, &data.embedder, &data.claude, &data.cfg.answer, mode, lang, &question).await;
    let latency_ms = started.elapsed().as_millis() as i32;
    let mode_name = if mode == Mode::Dev { "dev" } else { "player" };

    let insert = sqlx::query_scalar::<_, i64>(
        "INSERT INTO qa_log (guild_id, channel_id, user_id, mode, lang, question, answer, sources, tool_calls, model, outcome,
                             input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, error)
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18) RETURNING id",
    )
    .bind(ctx.guild_id().map(|g| g.to_string()))
    .bind(ctx.channel_id().to_string())
    .bind(ctx.author().id.to_string())
    .bind(mode_name)
    .bind(lang);

    match result {
        Ok(a) => {
            let qa_id = insert
                .bind(&question)
                .bind(&a.text)
                .bind(json!(a.sources))
                .bind(json!(a.tool_calls))
                .bind(&a.model)
                .bind(&a.outcome)
                .bind(a.usage.input_tokens as i64)
                .bind(a.usage.output_tokens as i64)
                .bind(a.usage.cache_read_input_tokens as i64)
                .bind(a.usage.cache_creation_input_tokens as i64)
                .bind(a.cost_usd)
                .bind(latency_ms)
                .bind(None::<String>)
                .fetch_one(&data.pool)
                .await?;
            info!(qa_id, mode = mode_name, lang, cost = a.cost_usd, latency_ms, outcome = %a.outcome, "answered");
            ctx.send(render(&question, &a, mode, qa_id, latency_ms as f32 / 1000.0)).await?;
        }
        Err(e) => {
            error!(error = %e, "answer failed");
            insert
                .bind(&question)
                .bind(None::<String>)
                .bind(None::<serde_json::Value>)
                .bind(None::<serde_json::Value>)
                .bind(&data.cfg.answer.model)
                .bind("error")
                .bind(None::<i64>)
                .bind(None::<i64>)
                .bind(None::<i64>)
                .bind(None::<i64>)
                .bind(0.0)
                .bind(latency_ms)
                .bind(e.to_string())
                .fetch_one(&data.pool)
                .await?;
            ctx.say(t(lang, "error")).await?;
        }
    }
    Ok(())
}

/// Ask a question about MyHordes game mechanics
#[poise::command(
    slash_command,
    name_localized("fr", "question"),
    description_localized("fr", "Pose une question sur les mécaniques de MyHordes"),
    description_localized("de", "Stelle eine Frage zu den Spielmechaniken von MyHordes"),
    description_localized("es-ES", "Haz una pregunta sobre las mecánicas de MyHordes"),
    description_localized("es-419", "Haz una pregunta sobre las mecánicas de MyHordes")
)]
async fn ask(
    ctx: Context<'_>,
    #[description = "Your question"]
    #[description_localized("fr", "Ta question")]
    #[description_localized("de", "Deine Frage")]
    #[description_localized("es-ES", "Tu pregunta")]
    #[description_localized("es-419", "Tu pregunta")]
    #[max_length = 500]
    question: String,
) -> Result<(), Error> {
    handle(ctx, Mode::Player, question).await
}

/// Technical question about MyHordes internals (ids, config keys, source files)
#[poise::command(slash_command)]
async fn dev(
    ctx: Context<'_>,
    #[description = "Your question"]
    #[max_length = 500]
    question: String,
) -> Result<(), Error> {
    handle(ctx, Mode::Dev, question).await
}

async fn on_event(framework: poise::FrameworkContext<'_, Data, Error>, event: &serenity::FullEvent) -> Result<(), Error> {
    let ctx = framework.serenity_context;
    match event {
        serenity::FullEvent::Ready { data_about_bot, .. } => info!("logged in as {}", data_about_bot.user.name),
        serenity::FullEvent::InteractionCreate { interaction } => {
            let Some(component) = interaction.as_message_component() else { return Ok(()) };
            let mut parts = component.data.custom_id.split(':');
            let (Some("fb"), Some(id), Some(vote)) = (parts.next(), parts.next(), parts.next()) else { return Ok(()) };
            let (Ok(id), Ok(vote)) = (id.parse::<i64>(), vote.parse::<i16>()) else { return Ok(()) };
            sqlx::query("UPDATE qa_log SET feedback = $1, feedback_by = $2 WHERE id = $3")
                .bind(vote)
                .bind(component.user.id.to_string())
                .bind(id)
                .execute(&framework.user_data.pool)
                .await?;
            info!(qa_id = id, vote, "feedback");
            let lang = locale_lang(Some(component.locale.as_str()));
            component
                .create_response(
                    ctx,
                    serenity::CreateInteractionResponse::Message(serenity::CreateInteractionResponseMessage::new().content(t(lang, "thanks")).ephemeral(true)),
                )
                .await?;
        }
        _ => {}
    }
    Ok(())
}

pub async fn run(pool: PgPool, embedder: Embedder, claude: Claude, cfg: BotConfig) -> Result<()> {
    let token = cfg.token.clone();
    let guild_id = cfg.guild_id;
    let data = Data { pool, embedder: Arc::new(embedder), claude, cfg };

    let framework = poise::Framework::builder()
        .options(poise::FrameworkOptions {
            commands: vec![ask(), dev()],
            event_handler: |framework, event| Box::pin(on_event(framework, event)),
            on_error: |err| {
                Box::pin(async move {
                    // A failed setup (e.g. command registration rejected by Discord) leaves a bot that can't
                    // answer anything: exit instead of staying online silently
                    if let poise::FrameworkError::Setup { error, .. } = &err {
                        error!("bot setup failed: {error}");
                        std::process::exit(1);
                    }
                    error!("command error: {err}");
                    if let Err(e) = poise::builtins::on_error(err).await {
                        error!("error while handling error: {e}");
                    }
                })
            },
            ..Default::default()
        })
        .setup(move |ctx, _ready, framework| {
            Box::pin(async move {
                match guild_id {
                    Some(g) => {
                        poise::builtins::register_in_guild(ctx, &framework.options().commands, serenity::GuildId::new(g)).await?;
                        info!(guild = g, "commands registered on test server");
                    }
                    None => {
                        poise::builtins::register_globally(ctx, &framework.options().commands).await?;
                        info!("commands registered globally");
                    }
                }
                Ok(data)
            })
        })
        .build();

    let mut client = serenity::ClientBuilder::new(token, serenity::GatewayIntents::non_privileged()).framework(framework).await?;
    client.start().await?;
    Ok(())
}
