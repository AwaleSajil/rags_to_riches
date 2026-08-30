---
title: Rags2Riches
emoji: 💰
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: apache-2.0
short_description: Where did my money go? Chat with your bank statements
app_port: 7860
---
# Rags2Riches - Personal Finance Transaction Analysis

AI-powered financial transaction analysis using RAG (Retrieval-Augmented Generation) with Model Context Protocol (MCP) integration. Upload your bank statements and chat with your financial data.

## Features

- **Smart CSV Ingestion**: Automatically maps any CSV format to standardized transaction schema using LLM
- **Multi-Provider Support**: Works with Google Gemini and OpenAI models
- **Merchant Enrichment**: Automatically enriches transactions with web-searched merchant information
- **Semantic + Structured Search**: Supabase Postgres with pgvector for semantic search + structured queries (one database)
- **MCP Integration**: Leverages Model Context Protocol for tool-based agent interactions
- **Mobile-First UI**: Expo (React Native) frontend with Android support
- **Auth**: Supabase authentication with JWT validation
- **Streaming Chat**: Server-Sent Events (SSE) for real-time AI responses

## Architecture

  <img width="900" height="594" alt="architecture" src="https://github.com/user-attachments/assets/f432c094-dc59-4046-aee6-607e278d3917" />

- **Frontend**: Expo (React Native Web) - serves as static build in production
- **Backend**: FastAPI wrapping the RAG engine
- **RAG Engine**: LangChain + LangGraph with MCP tool server
- **Auth**: Supabase (client-side JS + server-side JWT validation)
- **Vector DB**: Supabase Postgres + pgvector for semantic search (multi-tenant via user_id).
- **Database**: Supabase PostgreSQL for structured transaction queries.



## Environment Variables

Set these as **Repository secrets** in HF Space settings:

### Required

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL (auth) |
| `SUPABASE_KEY` | Supabase anon/service key (auth) |
| `APP_ENCRYPTION_KEY` | Fernet key encrypting stored LLM API keys. **The API refuses to start without it.** |

#### `APP_ENCRYPTION_KEY`

Users' LLM API keys are encrypted before they reach the database, so a dump or
backup no longer exposes credentials that can spend their money. Generate one:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Set it as a repository/Space secret in deployment, or in `.env` locally. **Keep
it safe** — losing it makes every stored API key unreadable and users must
re-enter theirs. Startup fails loudly when it's missing rather than silently
falling back to plaintext.

Keys written before this existed still work and are re-encrypted whenever a user
saves their config. To convert the backlog in one go:

```bash
python scripts/encrypt_existing_api_keys.py          # dry run
python scripts/encrypt_existing_api_keys.py --apply
```

### Account deletion (required by both app stores)

| Variable | Description |
|---|---|
| `SUPABASE_SERVICE_KEY` | Supabase **service_role** key. Required for `DELETE /api/v1/auth/account`. |
| `SUPPORT_EMAIL` | Contact address shown on `/account-deletion`, `/privacy` and `/terms`. |
| `PUBLISHER_NAME` | The party named as responsible for user data in the privacy policy. |

### Policy pages

Three public pages are served from the API, so their URLs exist wherever the
backend does and cannot drift out of step with what the app actually does:

| Path | Required by |
|---|---|
| `/privacy` | Apple and Play both refuse a listing without a resolving privacy policy URL |
| `/terms` | Apple expects an EULA; Play expects it in the listing |
| `/account-deletion` | Play requires this to be readable **without** installing the app |

They are plain HTML in `backend/pages/`, editable without touching code.
`PUBLISHER_NAME` and `SUPPORT_EMAIL` are substituted at request time; anything
unset renders as a visible `[not configured]` marker, and a test asserts no
`{{placeholder}}` can survive to a served page.

**Before publishing**, set both variables and fill in the governing-law
jurisdiction in `backend/pages/terms.html` — it is deliberately left as a loud
red marker rather than defaulted, because it is a real legal choice.

`tests/test_policy_pages.py` also holds the policy to the code: it fails if
`deep_enrichment` stops defaulting to off, if an analytics or crash-reporting
SDK appears in `frontend/package.json`, or if the capture route grows a
latitude/longitude parameter — each of which would make a specific sentence on
the privacy page untrue.

Apple guideline 5.1.1(v) and Google Play's deletion policy both require an app
that offers account creation to offer account *deletion* from inside the app —
clearing your data, or writing to support, does not satisfy either. Play
additionally requires a deletion page reachable **without** installing the app;
this backend serves one at `/account-deletion`, so the URL to give the Play
Console is `https://<your-backend>/account-deletion`.

Deleting the `auth.users` row is what makes it a real deletion: `"User".id`
references it `ON DELETE CASCADE` and every user-scoped table cascades from
`"User"`, so one delete takes all of it. Object storage does not cascade and is
cleared explicitly first — if that fails the account survives, so the user can
retry, rather than losing their files with nothing left pointing at them.

The anon key cannot delete a user, so this needs the service key. **It bypasses
row-level security**, so it is deliberately kept apart from `SUPABASE_KEY`: it
is reachable only through `dependencies.admin_client()` (grep that name to find
every place RLS is skipped), and `tests/test_production_guards.py` asserts it
never reaches `/public-config`.

Without it the app still boots and logs a warning at startup; "Delete my
account" answers 503 with a message telling the user to contact support. Set it
before submitting a build.

### Database

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (Supabase) |

### Vector DB

Semantic search runs on **pgvector inside your Supabase Postgres** — no separate
service or keys. It reuses `DATABASE_URL` above. Enable the extension by running
`supabase/migrations/000_base_schema.sql` (it's pre-installed on Supabase).

## Deployment

### Two-Space Deployment (HF Spaces)

The application is deployed as two separate Hugging Face Spaces:

- **Backend** ([rags2riches-backend](https://huggingface.co/spaces/ksmu/rags2riches-backend)): FastAPI server with RAG engine
- **Frontend** ([rags2riches-frontend](https://huggingface.co/spaces/ksmu/rags2riches-frontend)): Expo web static build

The frontend Space requires one build secret:

| Variable | Description |
|---|---|
| `EXPO_PUBLIC_API_URL` | Backend Space URL, e.g. `https://ksmu-rags2riches-backend.hf.space/api/v1` |

### Android app (Play Store)

Build profiles live in `frontend/eas.json`. `eas.json` is strict JSON and takes
no comments, so the reasoning is here.

```bash
cd frontend
npm i -g eas-cli && eas login
eas init          # first time only: writes extra.eas.projectId into app.config.js
eas build --platform android --profile production
```

| Profile | Output | For |
|---|---|---|
| `development` | APK, dev client | Emulator. Sets `EXPO_ALLOW_CLEARTEXT=1` so it can reach the backend over plain HTTP. |
| `preview` | APK | Sideloading a release-configured build onto a real phone. |
| `production` | AAB | Play upload. |

**Why `EXPO_PUBLIC_API_URL` is pinned in the two release profiles:** without it,
`src/lib/apiUrl.ts` falls back to `http://10.0.2.2:8000/api/v1` — the Android
*emulator's* address for the host machine. A release build also has
`usesCleartextTraffic: false`, so that request is blocked outright. The app would
install, open, and fail every call with nothing in the UI to explain why. The
pinned HTTPS URL is what stops a shipped build pointing at a dev machine.

No Supabase keys are needed at build time: when `EXPO_PUBLIC_SUPABASE_*` are
absent, `src/lib/supabase.ts` fetches them from the backend's
`/api/v1/public-config`. That is why this file contains no secrets and can be
committed.

`appVersionSource: "remote"` with `autoIncrement` lets EAS own the Android
`versionCode`, so two builds can never collide on an upload Play would reject.
The user-facing `version` stays in `app.config.js`.

### Single-Container Deployment

```bash
docker build -t r2r .
docker run -p 7860:7860 --env-file .env r2r
```
Open http://localhost:7860

### Docker Compose (Separate Containers)

```bash
docker compose build
docker compose up -d
```
- Backend: http://localhost:7860
- Frontend: http://localhost:8081

## Local Development

### Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npx expo start
```

## Usage

1. Register/login with your email
2. Configure your LLM provider and API key in Settings
3. Upload CSV transaction files via the Ingest tab
4. Chat with your financial data

### Example Questions

- "How much did I spend on restaurants last month?"
- "What are my top 5 spending categories?"
- "Show me all transactions over $100"
- "Analyze my spending patterns"

## Supported CSV Formats

MoneyRAG automatically handles different CSV formats:
- Chase Bank, Discover, and custom formats
- LLM-based column mapping (works with any column names)
- Required: Date, Merchant/Description, Amount

## Technologies

- **LangChain & LangGraph**: Agent orchestration
- **Google Gemini / OpenAI GPT**: LLM providers
- **Supabase**: Auth + PostgreSQL database for structured queries
- **pgvector (Supabase Postgres)**: Vector search for semantic retrieval
- **FastMCP**: Model Context Protocol server
- **Expo (React Native)**: Cross-platform frontend
- **FastAPI**: Backend API framework

## Contributors

- **Sajil Awale** - [GitHub](https://github.com/AwaleSajil)
- **Simran KC** - [GitHub](https://github.com/iamsims)

## License

MIT
