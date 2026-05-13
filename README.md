# api-rag

Minimal OpenAI-compatible RAG API with token auth.

## Start

```bash
./install.sh
nano .env
source run.sh 0.0.0.0 8080
```

The installer is idempotent and uses `uv` inside `~/venv/api-rag` (or `~/venv/<project-dir>` if you rename the directory).

## Required `.env`

```bash
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=changeme-upstream-token
API_AUTH_TOKEN=changeme-inbound-token
UPSTREAM_MODEL_RAG=gpt-4o-mini
MODEL_RAG=ai-rag
```

Copy `.env.example` for the important optional defaults.

## API

Popular OpenAI-compatible endpoints:

* `GET /healthz`
* `GET /v1/models`
* `POST /v1/chat/completions`

Authentication for chat accepts either:

* `Authorization: Bearer <API_AUTH_TOKEN>`
* `X-API-Key: <API_AUTH_TOKEN>`

Example:

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer changeme-inbound-token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-rag",
    "messages": [{"role": "user", "content": "Answer using the knowledge base."}]
  }'
```

## Documents

By default the API reads `wiki.txt` and writes the FAISS index to `vectorstore_db/`.
For multiple files or directories, set:

```bash
INGESTION_TEXT_PATHS=docs,README.md
RAG_FORCE_REBUILD=false
```

## H100 / DGX Spark notes

For accelerator servers, keep embedding and reranking behind your OpenAI-compatible backend (for example vLLM) when possible:

```bash
EMBEDDING_BACKEND=external
EMBEDDINGS_API_BASE=http://localhost:8000/v1
RERANKING_BACKEND=external
RERANKING_API_BASE=http://localhost:8000/v1
```

If this API must load local models, use CUDA:

```bash
EMBEDDING_BACKEND=local
EMBEDDING_DEVICE=cuda
RERANKING_BACKEND=local
RERANKER_DEVICE=cuda
```

`run.sh` sets safe default CUDA/PyTorch environment variables and can be used directly by systemd:

```ini
[Service]
WorkingDirectory=/workspace/api-rag
ExecStart=/workspace/api-rag/run.sh 0.0.0.0 8080
Restart=always
```
