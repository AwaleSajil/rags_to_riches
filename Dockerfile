# ===========================================================
# Dockerfile for Hugging Face Spaces (Docker SDK)
# Builds Expo web frontend + serves it from FastAPI backend
# HF Spaces expects port 7860
# ===========================================================

# ---- Stage 1: Build Expo web frontend ----
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend

# Install dependencies first (cache layer)
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy frontend source and build for web
COPY frontend/ ./

# Remove any .env that may have been copied — in production the API URL
# must resolve to a relative path (/api/v1), not a hardcoded http:// URL.
RUN rm -f .env .env.local .env.*

# Export static web build
RUN npx expo export --platform web


# ---- Stage 2: Python backend ----
FROM python:3.11-slim

# System deps for psycopg2-binary, grpcio (actiancortex), and general build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cache layer)
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r backend-requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY money_rag.py mcp_server.py ./

# Copy built frontend static files from Stage 1
COPY --from=frontend-build /app/frontend/dist ./static/

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Run with uvicorn
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
