# Lean OpenAI-Compatible RAG API

A stripped-down, production-friendly FastAPI service exposing `/v1/chat/completions` with retrieval-augmented generation over local text files.

## What changed

- Single responsibility package layout under `src/api_rag/`.
- Removed dead/duplicate docs and brittle shell bootstrapping.
- Centralized config in one dataclass (`Settings`).
- Deterministic ingestion and index lifecycle (`load_or_build`).
- Repeatable automation through `Makefile` (`setup`, `run`, `test`, `clean`).

## Project structure

```text
.
├── app.py
├── evaluate_rag.py
├── Makefile
├── requirements.txt
└── src/api_rag
    ├── __init__.py
    ├── config.py
    ├── ingest.py
    ├── retrieval.py
    ├── schemas.py
    └── server.py
```

## Quick start

```bash
cp .env.example .env
make setup
set -a && source .env && set +a
make run
```

## Required env vars

- `OPENAI_API_BASE` (example: `https://api.openai.com/v1`)
- `OPENAI_API_KEY`

Optional: `API_AUTH_TOKEN`, `INGESTION_TEXT_PATHS`, `VECTORSTORE_DIR`, `RAG_TOP_K`, `RAG_FORCE_REBUILD`.

## API

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

Payload stays OpenAI-compatible, with one exposed logical model (`MODEL_RAG`, default `ai-rag`).

## Evaluation

```bash
python evaluate_rag.py --dataset eval_dataset.sample.jsonl --token "$API_AUTH_TOKEN"
```
