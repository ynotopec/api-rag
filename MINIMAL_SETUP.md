# Minimal setup for api-rag MCP server

## 1-server side — one command

### Prerequisites

1. **RAG API running** on port 8080 (the main api-rag service).
2. **mcp package installed** in the same venv: `pip install mcp>=2.0.0`
3. **Environment** — at minimum:

```bash
export RAG_API_URL=http://localhost:8080   # where the RAG API lives
export RAG_AUTH_TOKEN=your-api-token       # Bearer token for /v1/chat/completions
```

### Start

```bash
python -m mcp_wrapper.main              # stdio mode  (Hermes config.yaml)
python -m mcp_wrapper.main --sse        # HTTP/SSE mode (OpenWebUI / any client)
```

That's it — **1 command + 2 env vars**. The server discovers its own tools automatically.

---

## 2-client side — minimal integration

### A) Hermes Agent (stdio)

In `~/.hermes/config.yaml` under `mcp_servers`:

```yaml
mcp_servers:
  api-rag:
    command: "bash"
    args: ["-c", "cd /path/to/api-rag && RAG_API_URL=http://localhost:8080 RAG_AUTH_TOKEN=xyz ./run-mcp.sh"]
```

Tools appear as `mcp_api_rag_rag_chat`, `mcp_api_rag_rag_health`, …

### B) OpenWebUI (SSE)

1. Start the server: `python -m mcp_wrapper.main --sse` (listens on `http://0.0.0.0:8085`)
2. In OpenWebUI: **Admin → Integrations → MCP Servers** → add URL `http://<host>:8085/sse`

### C) Any MCP client (SSE)

Same URL as OpenWebUI: `http://<host>:8085/sse`

Clients that support the MCP Streamable HTTP or SSE transport can connect directly.

---

## Quick health check

After starting the server in SSE mode, verify:

```bash
# List tools
curl -s http://localhost:8085/sse

# Health check (through MCP)
# Just call the rag_health tool — it should return {"status": "ok", "vectors": <N>}
```
