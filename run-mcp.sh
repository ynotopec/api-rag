#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="api-rag"
VENV_DIR="${VENV_DIR:-${HOME}/venv/${PROJECT_NAME}}"
PYTHON_BIN="${VENV_DIR}/bin/python"

cd "${PROJECT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  "${PROJECT_DIR}/install.sh"
fi

# Activate venv so mcp and uvicorn are on PATH
export PATH="${VENV_DIR}/bin:${PATH}"

# Load environment (API auth token, upstream config)
if [ -f "${PROJECT_DIR}/.env" ]; then
  set -a
  source "${PROJECT_DIR}/.env"
  set +a
fi

# MCP defaults
export MCP_HOST="${MCP_HOST:-0.0.0.0}"
export MCP_PORT="${MCP_PORT:-8085}"
export RAG_API_URL="${RAG_API_URL:-http://localhost:8080}"
export RAG_AUTH_TOKEN="${RAG_AUTH_TOKEN:-${API_AUTH_TOKEN:-changeme}}"

exec "${PYTHON_BIN}" -m uvicorn \
  mcp_server.main:app \
  --host "${MCP_HOST}" \
  --port "${MCP_PORT}" \
  --log-level "${UVICORN_LOG_LEVEL:-info}"
