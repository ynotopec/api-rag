"""MCP server wrapping the api-rag RAG service — minimal version.

Usage
-----
  python -m mcp_wrapper.main                  # stdio (Hermes config.yaml)
  python -m mcp_wrapper.main --sse            # HTTP/SSE (OpenWebUI / any client)

Required env vars
-----------------
  RAG_API_URL      base URL of the RAG service       (default http://localhost:8080)
  RAG_AUTH_TOKEN    Bearer token for the RAG service  (default changeme)
  MCP_PORT          SSE listen port                   (default 8085)

Tools exposed
-------------
  rag_chat       – query the RAG knowledge base
  rag_health     – health check
  rag_list_models – list available models
  rag_get_extract – retrieve a source extract by ID
"""

from __future__ import annotations

import os
import json
import httpx
import sys
import asyncio

from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool, ListToolsResult
from mcp_types._types import ListToolsRequest, CallToolRequest

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

# ── Configuration ──────────────────────────────────────────────────

RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8080")
RAG_AUTH_TOKEN: str = os.getenv("RAG_AUTH_TOKEN", "changeme")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8085"))


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {RAG_AUTH_TOKEN}", "Content-Type": "application/json"}

# ── Tool definitions ───────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="rag_chat",
        title="RAG Chat",
        description="Query the RAG knowledge base (chat/completions).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "User question."},
                "stream": {"type": "boolean", "description": "Stream the response.", "default": False},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_health",
        title="RAG Health",
        description="Check RAG service health.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_list_models",
        title="RAG Models",
        description="List available RAG models.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_get_extract",
        title="RAG Get Extract",
        description="Retrieve a stored source extract by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "extract_id": {"type": "string", "description": "Extract ID from a prior rag_chat response."},
            },
            "required": ["extract_id"],
        },
    ),
]

# ── Handlers ───────────────────────────────────────────────────────

async def _call_rag_chat(args: dict) -> str:
    """Call POST /v1/chat/completions."""
    query: str = args.get("query", "")
    stream: bool = args.get("stream", False)
    if not query:
        return "Error: 'query' is required."

    payload = {
        "model": "ai-rag",
        "messages": [{"role": "user", "content": query}],
        "stream": stream,
    }
    url = f"{RAG_API_URL}/v1/chat/completions"

    if stream:
        lines: list[str] = []
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=_auth_headers()) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            lines.append(line[6:])
        except Exception as exc:
            return f"Stream error: {exc}"
        return "\n\n".join(lines) if lines else "No data."

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=_auth_headers())
            resp.raise_for_status()
            data = resp.json()
        if data.get("choices"):
            return data["choices"][0].get("message", {}).get("content", "")
        return json.dumps(data, ensure_ascii=False)
    except Exception as exc:
        return f"Error: {exc}"


async def _call_rag_health() -> str:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/healthz")
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except Exception as exc:
        return f"Error: {exc}"


async def _call_rag_list_models() -> str:
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/v1/models")
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except Exception as exc:
        return f"Error: {exc}"


async def _call_rag_get_extract(args: dict) -> str:
    eid = args.get("extract_id", "")
    if not eid:
        return "Error: 'extract_id' required."
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/extract/{eid}")
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"Not found: extract '{eid}'"
        return f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
    except Exception as exc:
        return f"Error: {exc}"

_HANDLERS: dict[str, callable] = {
    "rag_chat": _call_rag_chat,
    "rag_health": _call_rag_health,
    "rag_list_models": _call_rag_list_models,
    "rag_get_extract": _call_rag_get_extract,
}

# ── MCP v2 callbacks ──────────────────────────────────────────────

async def _on_list_tools(_ctx: ServerRequestContext, _params: ListToolsRequest | None) -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)

async def _on_call_tool(_ctx: ServerRequestContext, params: CallToolRequest) -> CallToolResult:
    name = params.name
    args = params.arguments or {}
    handler = _HANDLERS.get(name)
    if not handler:
        return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")])
    try:
        return CallToolResult(content=[TextContent(type="text", text=await handler(args))])
    except Exception as exc:
        return CallToolResult(content=[TextContent(type="text", text=f"Error: {exc}")], isError=True)

# ── App ───────────────────────────────────────────────────────────

app = Server(name="api-rag", description="Minimal MCP wrapper for api-rag RAG service.")
app.add_request_handler("tools/list", ListToolsRequest, _on_list_tools)
app.add_request_handler("tools/call", CallToolRequest, _on_call_tool)

async def _run_stdio():
    async with stdio_server() as (rs, ws):
        await app.run(rs, ws, app.create_initialization_options())

# ── Starlette app — exposed as uvicorn factory for HTTP/SSE mode ──

_sse = SseServerTransport("/messages/")


async def _handle_sse(request: Request) -> Response:
    async with _sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())
    return Response()


_starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=_handle_sse, methods=["GET"]),
        Mount("/messages/", app=_sse.handle_post_message),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )
    ],
)


def _create_app():
    """Uvicorn factory: returns the Starlette app instance."""
    return _starlette_app


if __name__ == "__main__":
    if "--sse" in sys.argv:
        import uvicorn
        uvicorn.run(
            "mcp_wrapper.main:_create_app",
            host="0.0.0.0",
            port=MCP_PORT,
            factory=True,
            log_level="info",
        )
    else:
        asyncio.run(_run_stdio())
