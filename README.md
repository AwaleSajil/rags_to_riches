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
- **Semantic + Structured Search**: Qdrant vector DB for semantic search + PostgreSQL for structured queries
- **MCP Integration**: Leverages Model Context Protocol for tool-based agent interactions
- **Mobile-First UI**: Expo (React Native) frontend with Android support
- **Auth**: Supabase authentication with JWT validation
- **Streaming Chat**: Server-Sent Events (SSE) for real-time AI responses

## Architecture

- **Frontend**: Expo (React Native Web) - serves as static build in production
- **Backend**: FastAPI wrapping the RAG engine
- **RAG Engine**: LangChain + LangGraph with MCP tool server
- **Auth**: Supabase (client-side JS + server-side JWT validation)
- **Vector DB**: Qdrant Cloud (multi-tenant via user_id)
- **Database**: Supabase PostgreSQL with RLS

## Environment Variables

Set these as **Repository secrets** in HF Space settings:

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Supabase anon/service key |
| `QDRANT_URL` | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `DATABASE_URL` | PostgreSQL connection string |

## Local Development

### Backend
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npx expo start
```

### Docker (matches HF Spaces deployment)
```bash
docker build -t r2r .
docker run -p 7860:7860 --env-file .env r2r
```
Open http://localhost:7860

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
- **Qdrant Cloud**: Vector database for semantic search
- **Supabase**: Auth + PostgreSQL database
- **FastMCP**: Model Context Protocol server
- **Expo (React Native)**: Cross-platform frontend
- **FastAPI**: Backend API framework

## Contributors

- **Sajil Awale** - [GitHub](https://github.com/AwaleSajil)
- **Simran KC** - [GitHub](https://github.com/iamsims)

## License

MIT
