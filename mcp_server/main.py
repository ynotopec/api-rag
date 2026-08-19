"""MCP server wrapping the api-rag RAG service.

Exposes 4 tools over stdio or SSE transport:
  - rag_chat       – query the RAG knowledge base
  - rag_health     – health check (status + vector count)
  - rag_list_models – list available models
  - rag_get_extract – fetch a source extract by ID

Usage
-----
  # Stdio (default for Hermes config.yaml)
  python -m mcp_server.main

  # HTTP/SSE (for OpenWebUI or remote clients)
  python -m mcp_server.main --sse

Environment
-----------
  RAG_API_URL    – base URL of the RAG service (default http://localhost:8080)
  RAG_AUTH_TOKEN  – Bearer token for the RAG service (falls back to API_AUTH_TOKEN)
  MCP_HOST        – bind address for SSE mode (default 0.0.0.0)
  MCP_PORT        – port for SSE mode (default 8085)
"""

from __future__ import annotations

import os
import json
import httpx

from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
    ListToolsResult,
)

# MCP v2.0.0 uses mcp_types for request / result type annotations.
# ListToolsRequest and CallToolRequest carry the method literal so
# add_request_handler can match them at dispatch time.
from mcp_types._types import (
    ListToolsRequest,
    CallToolRequest,
)

from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import Response
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


# ── Configuration ──────────────────────────────────────────────

RAG_API_URL: str = os.getenv("RAG_API_URL", "http://localhost:8080")
RAG_AUTH_TOKEN: str = os.getenv(
    "RAG_AUTH_TOKEN", os.getenv("API_AUTH_TOKEN", "changeme")
)
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT: int = int(os.getenv("MCP_PORT", "8085"))


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {RAG_AUTH_TOKEN}", "Content-Type": "application/json"}


# ── Tool definitions ───────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="rag_chat",
        title="RAG Chat",
        description=(
            "Query the RAG knowledge base. Send a user question and get an answer "
            "synthesised from the indexed documents. Supports non-streaming and "
            "streaming responses, plus optional chunk / citation / source extras."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The user question or message to send to the RAG service.",
                },
                "model": {
                    "type": "string",
                    "description": "Model name (defaults to 'ai-rag').",
                    "default": "ai-rag",
                },
                "temperature": {
                    "type": "number",
                    "description": "Sampling temperature 0.0–2.0 (default 0.2).",
                    "default": 0.2,
                },
                "top_p": {
                    "type": "number",
                    "description": "Nucleus sampling threshold (default 1.0).",
                    "default": 1.0,
                },
                "max_tokens": {
                    "type": ["integer", "null"],
                    "description": "Maximum tokens in the response.",
                    "default": None,
                },
                "include_chunks": {
                    "type": "boolean",
                    "description": "Append the list of source chunks used for the answer.",
                    "default": False,
                },
                "include_citations": {
                    "type": "boolean",
                    "description": "Include structured citation objects with source excerpts and scores.",
                    "default": False,
                },
                "include_sources": {
                    "type": "boolean",
                    "description": "Append a plain-text 'Sources: …' footer.",
                    "default": False,
                },
                "stream": {
                    "type": "boolean",
                    "description": "If True, returns streaming SSE lines. If False, returns the final answer.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="rag_health",
        title="RAG Health Check",
        description="Check the RAG service health. Returns status and vector count.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_list_models",
        title="RAG List Models",
        description="List available RAG models.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="rag_get_extract",
        title="RAG Get Extract",
        description=(
            "Retrieve a previously stored source extract by ID. The extract ID is "
            "returned in a prior rag_chat response when include_citations=True."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "extract_id": {
                    "type": "string",
                    "description": "The extract ID returned in a prior rag_chat response.",
                },
            },
            "required": ["extract_id"],
        },
    ),
]

# ── Tool implementations ───────────────────────────────────────


async def _handle_rag_chat(args: dict) -> str:
    """Call POST /v1/chat/completions on the RAG service."""
    model: str = args.get("model", "ai-rag")
    temperature: float = args.get("temperature", 0.2)
    top_p: float = args.get("top_p", 1.0)
    max_tokens: int | None = args.get("max_tokens")
    include_chunks: bool = args.get("include_chunks", False)
    include_citations: bool = args.get("include_citations", False)
    include_sources: bool = args.get("include_sources", False)
    stream: bool = args.get("stream", False)
    query: str = args.get("query", "")

    if not query:
        return "Error: 'query' is required and must be a non-empty string."

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "temperature": temperature,
        "top_p": top_p,
        "include_chunks": include_chunks,
        "include_citations": include_citations,
        "include_sources": include_sources,
        "stream": stream,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    url: str = f"{RAG_API_URL}/v1/chat/completions"

    # ── Streaming mode ───────────────────────────────────────────
    if stream:
        data_lines: list[str] = []
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream("POST", url, json=payload, headers=_headers()) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data_lines.append(line[6:])
        except Exception as exc:
            return f"Streaming error: {exc}"
        return "\n\n".join(data_lines) if data_lines else "No data received from stream."

    # ── Non-streaming mode ───────────────────────────────────────
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=_headers())
            resp.raise_for_status()
            data = resp.json()

        if "choices" in data and data["choices"]:
            content: str = data["choices"][0].get("message", {}).get("content", "")
            extras: list[str] = []
            if include_chunks and "chunks" in data:
                extras.append(f"## Chunks\n{json.dumps(data['chunks'], ensure_ascii=False, indent=2)}")
            if include_citations and "citations" in data:
                extras.append(f"## Citations\n{json.dumps(data['citations'], ensure_ascii=False, indent=2)}")
            if include_sources and "extra" in data and "sources" in data["extra"]:
                src_names = [s.get("filename", "?") for s in data["extra"]["sources"]]
                extras.append(f"## Sources\n{', '.join(src_names)}")
            if extras:
                content += "\n\n" + "\n\n".join(extras)
            return content
        return json.dumps(data, ensure_ascii=False, indent=2)

    except httpx.HTTPStatusError as exc:
        return f"HTTP error {exc.response.status_code}: {exc.response.text[:500]}"
    except Exception as exc:
        return f"Error: {exc}"


async def _handle_rag_health() -> str:
    """GET /healthz — status and vector count."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/healthz")
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


async def _handle_rag_list_models() -> str:
    """GET /v1/models — list available models."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/v1/models")
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"Error: {exc}"


async def _handle_rag_get_extract(args: dict) -> str:
    """GET /extract/<id> — retrieve a stored source extract."""
    extract_id: str = args.get("extract_id", "")
    if not extract_id:
        return "Error: 'extract_id' is required."
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{RAG_API_URL}/extract/{extract_id}")
            resp.raise_for_status()
            return json.dumps(resp.json(), ensure_ascii=False, indent=2)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return f"Not found: extract '{extract_id}' does not exist."
        return f"HTTP error {exc.response.status_code}: {exc.response.text[:500]}"
    except Exception as exc:
        return f"Error: {exc}"


_TOOL_HANDLERS: dict[str, callable] = {
    "rag_chat": _handle_rag_chat,
    "rag_health": _handle_rag_health,
    "rag_list_models": _handle_rag_list_models,
    "rag_get_extract": _handle_rag_get_extract,
}


# ── MCP callbacks ──────────────────────────────────────────────


async def _on_list_tools(
    _ctx: ServerRequestContext,
    _params: ListToolsRequest | None,
) -> ListToolsResult:
    """Return the full tool list.  Registered for the 'tools/list' method."""
    return ListToolsResult(tools=TOOLS)


async def _on_call_tool(
    _ctx: ServerRequestContext,
    params: CallToolRequest,
) -> CallToolResult:
    """Route a tool call to the appropriate handler."""
    name: str = params.name
    arguments: dict = params.arguments or {}
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Unknown tool: {name}. Available: {', '.join(_TOOL_HANDLERS.keys())}",
                )
            ],
        )
    try:
        result_text: str = await handler(arguments)
        return CallToolResult(content=[TextContent(type="text", text=result_text)])
    except Exception as exc:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool error: {exc}")],
            isError=True,
        )


# ── App entry-point ────────────────────────────────────────────

app = Server(
    name="api-rag",
    description="MCP server wrapping the api-rag RAG service.",
)

# Register the two MCP request handlers using the v2 add_request_handler() pattern.
app.add_request_handler("tools/list", ListToolsRequest, _on_list_tools)
app.add_request_handler("tools/call", CallToolRequest, _on_call_tool)


async def _run_stdio():
    """Run the MCP server over stdio (default for Hermes config.yaml)."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


async def _run_sse():
    """Run the MCP server over SSE (for HTTP transport / OpenWebUI)."""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> Response:
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await app.run(
                streams[0], streams[1], app.create_initialization_options()
            )
        return Response()

    routes = [
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages/", app=sse.handle_post_message),
    ]

    starlette_app = Starlette(
        routes=routes,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["*"],
            )
        ],
    )

    uvicorn.run(starlette_app, host=MCP_HOST, port=MCP_PORT)


if __name__ == "__main__":
    import sys
    import asyncio

    if "--sse" in sys.argv or "--transport" in sys.argv:
        asyncio.run(_run_sse())
    else:
        asyncio.run(_run_stdio())
