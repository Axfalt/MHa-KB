# myhordes-kb

Knowledge base of MyHordes game mechanics: Postgres + pgvector, fed from the MyHordes source code,
queried by a Discord bot and by an MCP server for development.

## Layout

| Path | What |
|---|---|
| `php/` | Docker image + `export.php`, which boots the MyHordes kernel and dumps fixtures, town rules and setting schemas as JSON |
| `crates/kb-core` | Storage, embeddings (fastembed, local ONNX), translations, text processing, hybrid search |
| `crates/mhkb` | CLI: `ingest`, `embed`, `search`, `stats` |
| `migrations/` | SQL schema (applied automatically by every `mhkb` command) |
| `export/` | Output of the PHP export (git-ignored) |

## Setup

Requirements: Docker Desktop, Rust (stable, MSVC on Windows), a MyHordes checkout next to this repo
(`../myhordes`, override with `MYHORDES_DIR`).

```sh
# 1. Postgres + pgvector on 127.0.0.1:5433
docker compose up -d db

# 2. Export game data from MyHordes (first run installs its Composer deps into a Docker volume;
#    the MyHordes checkout itself is not modified)
docker compose --profile export run --rm php composer install --no-interaction --no-scripts
docker compose --profile export run --rm php php /kb/export.php /app /export

# 3. Build documents and embed them (first run downloads the embedding model into .cache/models)
cargo run --release -- ingest all

# 4. Try it
cargo run --release -- search "où trouver des piles" --lang fr
cargo run --release -- stats
```

Re-run steps 2–3 after pulling MyHordes: unchanged documents keep their embeddings, removed ones are pruned.

## Answering questions

```sh
cargo run --release -- ask "can a ghoul drink water?" -v          # one answer in the terminal
cargo run --release -- ask "quel TownSetting règle les fouilles ?" --lang fr --dev
cargo run --release -- bot                                       # Discord bot: /ask and /dev
```

Answers come from Claude (`ANSWER_MODEL`, default `claude-opus-5-5`) with tools over the knowledge base
(`search`, `get_document`, `find_entity`). Every bot answer is logged in `qa_log` with its cost and the
👍/👎 feedback.

## Evaluation

```sh
cargo run --release -- eval -v                    # retrieval: hit@1, hit@5, MRR (free)
cargo run --release -- eval --answers -v \
  --out eval/results/answers-<label>.json         # answer quality graded by Claude (~$1–2 per run)
```

`eval/questions.yaml` holds the questions, the documents that should be retrieved and, for answer grading,
reference facts verified against the game code. Add questions that got a 👎 on Discord.

## Configuration

| Env var | Default | |
|---|---|---|
| `DATABASE_URL` | `postgres://mhkb:mhkb@127.0.0.1:5433/mhkb` | |
| `EMBED_MODEL` | `e5-base` | `e5-small` (384d, for 1 GB hosts), `e5-base` (768d), `bge-m3` (1024d). Changing it requires `mhkb embed --reset` |
| `MODEL_CACHE` | `.cache/models` | |
| `EXPORT_DIR` | `export` | |
| `MYHORDES_DIR` | `../myhordes` | |
| `ANTHROPIC_API_KEY` | | Required by `ask`, `bot`, `eval --answers` |
| `ANSWER_MODEL` | `claude-opus-5-5` | |
| `PLAYER_EFFORT` / `DEV_EFFORT` | `low` / `medium` | Claude effort for `/ask` and `/dev` |
| `DISCORD_TOKEN` | | Required by `bot` |
| `DISCORD_GUILD_ID` | | Register commands on one server only (instant, for testing); global otherwise |
| `DAILY_BUDGET_USD` | `5` | Bot stops answering for the day once reached |
| `USER_LIMIT_PER_HOUR` | `10` | Questions per user per hour |
| `ALLOWED_CHANNEL_IDS` | | Comma-separated; the bot only answers there when set |
| `DEV_ROLE_ID` | | Role required for `/dev` when set |

Put secrets in `.env` (git-ignored); see `.env.example`.

## Documents

| Kind | Source | Key |
|---|---|---|
| `entity_card` | Items (actions, recipes, drop rates, buildings using them), buildings, ruins, jobs, statuses, roles, hero skills, town types, pictos, home levels — one card per language | `card:<kind>:<id>:<lang>` |
| `help` | In-game help pages, rendered to Markdown per language | `help:<page>:<lang>` |
| `rule` | Effective `rules.yml` values per town type and group, annotated with `TownSetting` names, plus a diff against the default rules | `rule:<type>:<group>` |

Search fuses vector similarity, Postgres full-text search and fuzzy entity-name matching (pg_trgm) with
Reciprocal Rank Fusion.
