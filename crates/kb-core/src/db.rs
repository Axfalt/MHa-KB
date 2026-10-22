use anyhow::{Context, Result};
use sqlx::PgPool;
use sqlx::postgres::PgPoolOptions;

pub async fn connect(url: &str) -> Result<PgPool> {
    PgPoolOptions::new()
        .max_connections(8)
        .connect(url)
        .await
        .with_context(|| format!("connecting to {url}"))
}

pub async fn migrate(pool: &PgPool) -> Result<()> {
    sqlx::migrate!("../../migrations").run(pool).await?;
    Ok(())
}

pub async fn get_meta(pool: &PgPool, key: &str) -> Result<Option<String>> {
    Ok(sqlx::query_scalar("SELECT value FROM kb_meta WHERE key = $1")
        .bind(key)
        .fetch_optional(pool)
        .await?)
}

pub async fn set_meta(pool: &PgPool, key: &str, value: &str) -> Result<()> {
    sqlx::query("INSERT INTO kb_meta (key, value) VALUES ($1, $2) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value")
        .bind(key)
        .bind(value)
        .execute(pool)
        .await?;
    Ok(())
}
