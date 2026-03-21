from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import SETTINGS
from .retrieval import RetrievalEngine
from .schemas import ChatCompletionRequest, ModelCard

engine = RetrievalEngine()
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global http_client
    engine.load_or_build()
    http_client = httpx.AsyncClient(timeout=60.0)
    yield
    if http_client:
        await http_client.aclose()


app = FastAPI(title="Lean RAG API", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def check_auth(authorization: str = Header(default="")) -> None:
    if not SETTINGS.api_auth_token:
        return
    token = authorization.split(" ", 1)[-1].strip() if authorization else ""
    if token != SETTINGS.api_auth_token:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
def models() -> dict:
    return {"data": [ModelCard(id=SETTINGS.model_rag).model_dump()]}


async def _call_upstream(messages: list[dict], temperature: float, stream: bool) -> dict | AsyncGenerator[str, None]:
    if http_client is None:
        raise HTTPException(status_code=503, detail="HTTP client unavailable")

    payload = {
        "model": SETTINGS.upstream_model_rag,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    headers = {"Authorization": f"Bearer {SETTINGS.openai_api_key}"}

    if not stream:
        response = await http_client.post(SETTINGS.chat_completions_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

    async def iterator() -> AsyncGenerator[str, None]:
        async with http_client.stream("POST", SETTINGS.chat_completions_url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    yield f"{line}\n"

    return iterator()


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_completions(req: ChatCompletionRequest):
    if req.model != SETTINGS.model_rag:
        raise HTTPException(status_code=400, detail=f"Unknown model. Use {SETTINGS.model_rag}")

    question = req.messages[-1].content
    retrieval = engine.retrieve(question)

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using the provided context. If context is insufficient, say you don't know.\n\n"
                f"Context:\n{retrieval.context or 'No relevant context found.'}"
            ),
        },
        *[m.model_dump() for m in req.messages],
    ]

    if req.stream:
        stream = await _call_upstream(messages, req.temperature or 0.2, stream=True)
        assert not isinstance(stream, dict)
        return StreamingResponse(stream, media_type="text/event-stream")

    data = await _call_upstream(messages, req.temperature or 0.2, stream=False)
    assert isinstance(data, dict)
    content = data["choices"][0]["message"]["content"]
    if retrieval.sources:
        content += f"\n\nSources: {', '.join(retrieval.sources)}"

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": SETTINGS.model_rag,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }
