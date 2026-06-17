#!/bin/bash
# Start the MoneyRAG backend server
# Binds to 0.0.0.0 so mobile devices on the same network can connect
cd "$(dirname "$0")"
.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
