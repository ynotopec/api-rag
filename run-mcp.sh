#!/usr/bin/env bash
# Run the api-rag MCP server — one command.
#   ./run-mcp.sh              → stdio (Hermes config.yaml)
#   ./run-mcp.sh --sse        → HTTP/SSE (OpenWebUI / any client)
#
# Required env:  RAG_API_URL   (default http://localhost:8080)
#                RAG_AUTH_TOKEN (default changeme)
# Optional env:  MCP_PORT       (default 8085)

set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV="${VENV_DIR:-${HOME}/venv/api-rag}/bin/python"
if [ ! -x "$VENV" ]; then
  ./install.sh >/dev/null 2>&1 || true
fi

# Load .env if present
[ -f .env ] && set -a && source .env && set +a

# Defaults
export RAG_API_URL="${RAG_API_URL:-http://localhost:8080}"
export RAG_AUTH_TOKEN="${RAG_AUTH_TOKEN:-changeme}"
export MCP_PORT="${MCP_PORT:-8085}"

# Detect mode
if [[ "${1:-}" == "--sse" ]]; then
  exec "$VENV" -m uvicorn "mcp_wrapper.main:_create_app" \
    --factory --host 0.0.0.0 --port "${MCP_PORT}" --log-level info
else
  exec "$VENV" -m mcp_wrapper.main
fi
