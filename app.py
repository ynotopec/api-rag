import os
import time
import json
import uuid
import hashlib
import asyncio
import pickle
import logging
import mailbox
import email
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator, Set, Iterable

import numpy as np
import httpx
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# LangChain / RAG Imports
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# --- Optional Dependencies ---
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

try:
    from langchain_experimental.text_splitter import SemanticChunker
    SEMANTIC_CHUNKER_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKER_AVAILABLE = False

# Try to import RapidFuzz for fast string matching, fallback to standard if missing
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# ===============================
# Configuration & Environment
# ===============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "changeme")
OPENAI_CHAT_COMPLETIONS_URL = OPENAI_API_BASE.rstrip("/") + "/chat/completions"

AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "changeme")

# Models
UPSTREAM_MODEL_RAG = os.getenv("UPSTREAM_MODEL_RAG", "gpt-4o-mini")
UPSTREAM_MODEL_REWRITE = os.getenv("UPSTREAM_MODEL_REWRITE", UPSTREAM_MODEL_RAG)
MODEL_RAG_NAME = os.getenv("MODEL_RAG", "ai-rag")

# Vector Store
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore_db")
WIKI_PATH = os.getenv("WIKI_TXT", "wiki.txt")
RAG_FORCE_REBUILD = os.getenv("RAG_FORCE_REBUILD", "true").lower() in {"1", "true", "yes"}
INGESTION_SOURCES = os.getenv("INGESTION_SOURCES", "text")
INGESTION_TEXT_PATHS = os.getenv("INGESTION_TEXT_PATHS", "").strip()
THUNDERBIRD_PROFILE_DIR = os.getenv("THUNDERBIRD_PROFILE_DIR", "").strip()
THUNDERBIRD_MAX_MESSAGES = int(os.getenv("THUNDERBIRD_MAX_MESSAGES", "10000"))
INGESTION_REFRESH_INTERVAL = int(os.getenv("INGESTION_REFRESH_INTERVAL", "0"))

# RAG Params
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
# Strategies: "simple", "rewrite", "hyde", "rewrite+hyde"
RAG_QUERY_STRATEGY = os.getenv("RAG_QUERY_STRATEGY", "rewrite+hyde")
HISTORY_WINDOW = int(os.getenv("RAG_HISTORY_WINDOW", "6"))

# Optimization Flags
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_QUERY_CLASSIFICATION = os.getenv("ENABLE_QUERY_CLASSIFICATION", "true").lower() == "true"
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Settings
MMR_K = int(os.getenv("MMR_K", "8"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "16"))
BM25_K = int(os.getenv("BM25_K", "4"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
#cross-encoder/ms-marco-MiniLM-L-6-v2")

# System Resources
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CACHE_MAX_SIZE = 1000
PORT = int(os.getenv("PORT", "8080"))

# ===============================
# Global State
# ===============================
_vector: Optional[FAISS] = None
_bm25_retriever: Optional[BM25Retriever] = None
_all_docs: List[Document] = []
_embeddings: Optional[HuggingFaceEmbeddings] = None
_reranker: Optional[Any] = None
_http_client: Optional[httpx.AsyncClient] = None
_executor: Optional[ThreadPoolExecutor] = None
_index_lock: Optional[asyncio.Lock] = None
_refresh_task: Optional[asyncio.Task] = None
_last_ingestion_mtime: Optional[float] = None

# Simple in-memory LRU-like caches
_embedding_cache: Dict[str, List[float]] = {}
_query_rewrite_cache: Dict[str, str] = {}
_hyde_cache: Dict[str, str] = {}


# ===============================
# Lifespan Events
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _executor, _index_lock, _refresh_task
    
    # 1. Thread Pool for CPU bound tasks
    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    _index_lock = asyncio.Lock()
    
    # 2. HTTP Client for upstream LLM calls
    _http_client = httpx.AsyncClient(timeout=60.0, limits=httpx.Limits(max_keepalive_connections=20))
    
    # 3. Load Models
    _get_embeddings()
    if ENABLE_RERANKING and RERANKER_AVAILABLE:
        # Load reranker in background to not block startup entirely
        asyncio.get_event_loop().run_in_executor(_executor, _get_reranker)

    # 4. Load/Build Index
    await asyncio.get_event_loop().run_in_executor(_executor, _ensure_vectorstore)

    if INGESTION_REFRESH_INTERVAL > 0:
        _refresh_task = asyncio.create_task(_refresh_index_loop())
    
    logger.info("System Ready.")
    yield
    
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
    if _http_client:
        await _http_client.aclose()
    if _executor:
        _executor.shutdown(wait=False)


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    logger.info(f"Loading embedding model: {model_name}")
    
    # Intelligent device selection
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # Apple Silicon
    
    logger.info(f"Using device: {device}")

    encode_kwargs = {'normalize_embeddings': True, 'batch_size': 32}
    
    emb = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs=encode_kwargs
    )
    _embeddings = emb
    return emb


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    if not RERANKER_AVAILABLE:
        return None
    try:
        logger.info(f"Loading Reranker: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
        return _reranker
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        return None


def _ensure_vectorstore():
    global _vector, _bm25_retriever, _all_docs, _last_ingestion_mtime
    
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
    
    rebuild = RAG_FORCE_REBUILD
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        rebuild = True
    else:
        ingestion_mtime = _get_ingestion_latest_mtime()
        _last_ingestion_mtime = ingestion_mtime
        if ingestion_mtime and ingestion_mtime > os.path.getmtime(VECTORSTORE_DIR):
            logger.info("Ingestion sources updated, rebuilding index...")
            rebuild = True

    if rebuild:
        _build_index()
        _last_ingestion_mtime = _get_ingestion_latest_mtime()
    else:
        logger.info("Loading existing vector store...")
        try:
            _vector = FAISS.load_local(VECTORSTORE_DIR, _get_embeddings(), allow_dangerous_deserialization=True)
            with open(chunks_path, "rb") as f:
                _all_docs = pickle.load(f)
            
            if ENABLE_HYBRID_SEARCH and _all_docs:
                _bm25_retriever = BM25Retriever.from_documents(_all_docs)
                _bm25_retriever.k = BM25_K
        except Exception as e:
            logger.error(f"Error loading index: {e}. Rebuilding...")
            _build_index()


def _build_index():
    global _vector, _bm25_retriever, _all_docs, _last_ingestion_mtime
    docs = _collect_ingestion_documents()
    if not docs:
        logger.warning("No ingestion documents found. Creating empty index.")
        docs = [Document(page_content="Welcome to the RAG system.", metadata={"source": "system"})]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = splitter.split_documents(docs)

    embeddings = _get_embeddings()
    _vector = FAISS.from_documents(docs, embeddings)
    _vector.save_local(VECTORSTORE_DIR)
    
    _all_docs = docs
    with open(os.path.join(VECTORSTORE_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(docs, f)
        
    if ENABLE_HYBRID_SEARCH:
        _bm25_retriever = BM25Retriever.from_documents(docs)
        _bm25_retriever.k = BM25_K
    _last_ingestion_mtime = _get_ingestion_latest_mtime()


async def _refresh_index_loop():
    global _last_ingestion_mtime
    while True:
        await asyncio.sleep(INGESTION_REFRESH_INTERVAL)
        ingestion_mtime = _get_ingestion_latest_mtime()
        if not ingestion_mtime:
            continue
        if _last_ingestion_mtime and ingestion_mtime <= _last_ingestion_mtime:
            continue
        if not _index_lock:
            continue
        async with _index_lock:
            logger.info("Detected new ingestion data. Rebuilding index...")
            await asyncio.get_event_loop().run_in_executor(_executor, _build_index)
            _last_ingestion_mtime = ingestion_mtime


def _iter_text_paths(paths: Iterable[str]) -> List[str]:
    collected = []
    for raw_path in paths:
        path = raw_path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.lower().endswith(".txt"):
                        collected.append(os.path.join(root, filename))
        elif os.path.isfile(path):
            collected.append(path)
        else:
            logger.warning(f"Text ingestion path not found: {path}")
    return collected


def _load_text_documents() -> List[Document]:
    if INGESTION_TEXT_PATHS:
        raw_paths = [p for p in INGESTION_TEXT_PATHS.split(",") if p.strip()]
    else:
        raw_paths = [WIKI_PATH]
    text_paths = _iter_text_paths(raw_paths)
    docs: List[Document] = []
    for path in text_paths:
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
    return docs


def _is_thunderbird_mbox(filename: str) -> bool:
    lower = filename.lower()
    if lower.endswith(".msf"):
        return False
    if lower.endswith(".mbox"):
        return True
    return "." not in filename


def _extract_email_body(message: email.message.Message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            disposition = part.get_content_disposition()
            if content_type == "text/plain" and disposition != "attachment":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload is None:
                    return ""
                return payload.decode(charset, errors="replace")
        return ""
    payload = message.get_payload(decode=True)
    if payload is None:
        return ""
    charset = message.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def _load_thunderbird_documents() -> List[Document]:
    profile_dir = THUNDERBIRD_PROFILE_DIR
    if not profile_dir:
        logger.info("THUNDERBIRD_PROFILE_DIR not set; skipping Thunderbird ingestion.")
        return []
    if not os.path.isdir(profile_dir):
        logger.warning(f"Thunderbird profile directory not found: {profile_dir}")
        return []
    docs: List[Document] = []
    count = 0
    for root, _, files in os.walk(profile_dir):
        for filename in files:
            if not _is_thunderbird_mbox(filename):
                continue
            mbox_path = os.path.join(root, filename)
            try:
                mbox = mailbox.mbox(mbox_path)
            except Exception as exc:
                logger.warning(f"Failed to open mbox {mbox_path}: {exc}")
                continue
            for message in mbox:
                if THUNDERBIRD_MAX_MESSAGES and count >= THUNDERBIRD_MAX_MESSAGES:
                    return docs
                if not isinstance(message, email.message.Message):
                    continue
                subject = message.get("subject", "")
                sender = message.get("from", "")
                date = message.get("date", "")
                body = _extract_email_body(message)
                content = "\n".join(
                    [
                        f"Subject: {subject}",
                        f"From: {sender}",
                        f"Date: {date}",
                        "",
                        body.strip(),
                    ]
                ).strip()
                if not content:
                    continue
                relative_path = os.path.relpath(mbox_path, profile_dir)
                metadata = {
                    "source": f"thunderbird:{relative_path}",
                    "message_id": message.get("message-id", ""),
                }
                docs.append(Document(page_content=content, metadata=metadata))
                count += 1
    return docs


def _collect_ingestion_documents() -> List[Document]:
    sources = {s.strip().lower() for s in INGESTION_SOURCES.split(",") if s.strip()}
    logger.info(f"Ingestion sources enabled: {', '.join(sorted(sources)) or 'none'}")
    docs: List[Document] = []
    if "text" in sources:
        docs.extend(_load_text_documents())
    if "thunderbird" in sources:
        docs.extend(_load_thunderbird_documents())
    return docs


def _get_ingestion_latest_mtime() -> Optional[float]:
    mtimes: List[float] = []
    if "text" in {s.strip().lower() for s in INGESTION_SOURCES.split(",") if s.strip()}:
        if INGESTION_TEXT_PATHS:
            text_paths = _iter_text_paths(INGESTION_TEXT_PATHS.split(","))
        else:
            text_paths = _iter_text_paths([WIKI_PATH])
        for path in text_paths:
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                continue
    if "thunderbird" in {s.strip().lower() for s in INGESTION_SOURCES.split(",") if s.strip()}:
        if THUNDERBIRD_PROFILE_DIR and os.path.isdir(THUNDERBIRD_PROFILE_DIR):
            try:
                mtimes.append(os.path.getmtime(THUNDERBIRD_PROFILE_DIR))
            except OSError:
                pass
    return max(mtimes) if mtimes else None


# ===============================
# API & Auth
# ===============================
app = FastAPI(title="Optimized RAG API", version="2.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0

def check_auth(authorization: str = Header(None)):
    if AUTH_TOKEN:
        if not authorization or authorization.split(" ", 1)[-1] != AUTH_TOKEN:
            raise HTTPException(401, "Invalid Token")


# ===============================
# Logic: Helpers
# ===============================
def _get_cache_key(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

async def _call_upstream_llm(
    messages: List[Dict[str, str]], 
    model: str, 
    max_tokens: int = 1000,
    temp: float = 0.1
) -> str:
    """Helper to call OpenAI compatible endpoint."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    try:
        resp = await _http_client.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"LLM Call failed: {e}")
        return ""


# ===============================
# Logic: RAG Core
# ===============================
async def _classify_query(query: str) -> str:
    if not ENABLE_QUERY_CLASSIFICATION: 
        return "RAG"
    # Simple heuristic fallback to avoid latency
    if len(query.split()) < 3: return "CHAT"
    return "RAG" # Assume RAG by default for strictness


async def _rewrite_query(history_str: str, current_query: str) -> str:
    """Generates a search-optimized query."""
    cache_key = _get_cache_key(f"{history_str}|{current_query}")
    if ENABLE_CACHING and cache_key in _query_rewrite_cache:
        return _query_rewrite_cache[cache_key]

    system = "You are a search query optimizer. Output ONLY the improved search query, no explanation."
    prompt = f"Context: {history_str}\nUser Question: {current_query}\nOptimized Search Query:"
    
    rewritten = await _call_upstream_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=UPSTREAM_MODEL_REWRITE,
        max_tokens=64
    )
    
    result = rewritten.strip() if rewritten else current_query
    if ENABLE_CACHING: _query_rewrite_cache[cache_key] = result
    return result


async def _hyde_expand(query: str) -> str:
    """Generates a hypothetical document."""
    cache_key = _get_cache_key(f"hyde|{query}")
    if ENABLE_CACHING and cache_key in _hyde_cache:
        return _hyde_cache[cache_key]

    system = "Write a short, factual 3-sentence excerpt from a technical manual answering this question."
    hyde_doc = await _call_upstream_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        model=UPSTREAM_MODEL_REWRITE,
        max_tokens=128
    )
    
    if ENABLE_CACHING: _hyde_cache[cache_key] = hyde_doc
    return hyde_doc


def _fast_deduplicate(docs: List[Document]) -> List[Document]:
    """
    Deduplicates documents using Jaccard Similarity (or RapidFuzz) 
    instead of heavy embedding calculations.
    """
    unique_docs = []
    seen_hashes = set()
    
    for doc in docs:
        # 1. Fast Hash Check (Exact Duplicates)
        content = doc.page_content
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        
        if doc_hash in seen_hashes:
            continue
            
        # 2. Fuzzy Check (Near Duplicates)
        is_duplicate = False
        if RAPIDFUZZ_AVAILABLE and len(unique_docs) < 20: # Limit fuzzy check to small sets for speed
            for kept_doc in unique_docs:
                ratio = fuzz.ratio(content, kept_doc.page_content)
                if ratio > 90: # 90% similar
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_docs.append(doc)
            seen_hashes.add(doc_hash)
            
    return unique_docs


def _reciprocal_rank_fusion(results_list: List[List[Document]], k=60) -> List[Document]:
    """Combines results from multiple lists."""
    scores = {}
    doc_map = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            # Use hash as ID
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[id] for id in sorted_ids]


def _rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    """Reranks documents using CrossEncoder on CPU (Threaded)."""
    if not ENABLE_RERANKING or not _reranker or not docs:
        return docs
    
    # Optimization: Don't rerank if few docs
    if len(docs) < 3:
        return docs

    try:
        pairs = [(query, d.page_content) for d in docs]
        scores = _reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        return docs


# ===============================
# Logic: Main Retrieval Pipeline
# ===============================
async def _retrieve_pipeline(messages: List[ChatMessage]) -> Dict[str, Any]:
    user_query = messages[-1].content
    history = "\n".join([f"{m.role}: {m.content}" for m in messages[-4:-1]])

    # 1. Check classification
    if await _classify_query(user_query) == "CHAT":
        return {"context": "", "sources": [], "skip": True, "query": user_query}

    # 2. Parallel Execution Setup
    # We want to search for the original query IMMEDIATELY while the LLM thinks about rewrites.
    tasks = []
    
    # Task A: Search using Original Query (Immediate)
    async def search_original():
        res = []
        if _vector:
            res.extend(await asyncio.get_event_loop().run_in_executor(
                _executor, 
                lambda: _vector.similarity_search_with_score(user_query, k=MMR_K)
            ))
            # Unwrap tuple (doc, score) -> doc
            res = [r[0] for r in res]
        if ENABLE_HYBRID_SEARCH and _bm25_retriever:
             res.extend(await asyncio.get_event_loop().run_in_executor(
                _executor, _bm25_retriever.invoke, user_query
            ))
        return res

    original_search_task = asyncio.create_task(search_original())
    
    # Task B: Query Rewriting & HyDE (Parallel with search)
    rewritten_query = user_query
    
    if RAG_QUERY_STRATEGY != "simple":
        # Launch Rewrite
        rewrite_task = asyncio.create_task(_rewrite_query(history, user_query))
        
        # Launch HyDE (if enabled)
        hyde_task = None
        if "hyde" in RAG_QUERY_STRATEGY:
            hyde_task = asyncio.create_task(_hyde_expand(user_query))
        
        # Await rewrite to trigger secondary search
        rewritten_query = await rewrite_task
        
        # Trigger secondary search with rewritten query
        async def search_rewritten():
            res = []
            if _vector:
                res.extend(await asyncio.get_event_loop().run_in_executor(
                    _executor, 
                    lambda: _vector.similarity_search(rewritten_query, k=MMR_K)
                ))
            return res
        
        tasks.append(asyncio.create_task(search_rewritten()))
        
        # If HyDE finishes, search with that too
        if hyde_task:
            hyde_doc = await hyde_task
            async def search_hyde():
                if _vector and hyde_doc:
                    return await asyncio.get_event_loop().run_in_executor(
                        _executor, 
                        lambda: _vector.similarity_search(hyde_doc, k=MMR_K)
                    )
                return []
            tasks.append(asyncio.create_task(search_hyde()))

    # 3. Gather All Results
    # Include the original search which started first
    all_results_raw = [await original_search_task]
    if tasks:
        secondary_results = await asyncio.gather(*tasks)
        all_results_raw.extend(secondary_results)
    
    # 4. Fusion & Deduplication (CPU Bound -> Thread Pool)
    def process_results():
        # Flatten
        flat_results = [docs for docs in all_results_raw if docs]
        if not flat_results: return []
        
        # RRF Fusion
        fused = _reciprocal_rank_fusion(flat_results)
        
        # Fast Dedup
        deduped = _fast_deduplicate(fused)
        
        # Rerank (Heavy)
        final_docs = _rerank_documents(rewritten_query, deduped)
        return final_docs[:RAG_TOP_K]

    final_docs = await asyncio.get_event_loop().run_in_executor(_executor, process_results)

    # 5. Format Output
    if not final_docs:
        return {"context": "", "sources": [], "skip": False, "query": rewritten_query}

    # On ajoute des marqueurs clairs pour que le LLM sache que ce sont des morceaux distincts
    formatted_chunks = []
    for i, doc in enumerate(final_docs):
        # Petit nettoyage : retirer les sauts de ligne excessifs dans le chunk lui-même
        clean_content = " ".join(doc.page_content.split())
        formatted_chunks.append(f"[Excerpt {i+1}]: {clean_content}")

#    context_text = "\n\n".join([d.page_content for d in final_docs])
    context_text = "\n\n".join(formatted_chunks)
    sources = list(set([d.metadata.get("source", "unknown") for d in final_docs]))
    
    return {
        "context": context_text,
        "sources": sources,
        "skip": False,
        "query": rewritten_query
    }


# ===============================
# Endpoints
# ===============================
@app.get("/healthz")
def health():
    return {"status": "ok", "vectors": _vector.index.ntotal if _vector else 0}

@app.get("/v1/models")
def list_models():
    return {"data": [{"id": MODEL_RAG_NAME, "object": "model"}]}

@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_endpoint(req: ChatReq):
    if req.model != MODEL_RAG_NAME:
        raise HTTPException(400, f"Unknown model. Use {MODEL_RAG_NAME}")

    # 1. Retrieve Context
    rag_start = time.time()
    rag_result = await _retrieve_pipeline(req.messages)
    rag_time = time.time() - rag_start
    
    logger.info(f"RAG Retrieval took {rag_time:.2f}s | Docs: {len(rag_result.get('sources', []))}")

    # 2. Build Prompt
    final_messages = []
    
    if rag_result["skip"] or not rag_result["context"]:
        # Fallback to pure chat
        final_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if not rag_result["skip"]:
             final_messages.insert(0, {"role": "system", "content": "Knowledge base check returned no results. Answer based on general knowledge."})
    else:
        # Inject Context
        sys_prompt = (
#            "You are a helpful assistant. Use the following context to answer the user request. "
#            "If the answer is not in the context, say you don't know.\n\n"
#            f"Context:\n{rag_result['context']}"
            "You are a helpful assistant. "
            "Task: Synthesize the answer using the provided context. "
            "Guidelines:\n"
            "1. If the context contains duplicate information, mention it only once.\n"
            "2. Be concise and structured.\n"
            "3. If the answer is not in the context, say you don't know.\n"
            "4. Answer in the same language as the user's question.\n\n"
            f"Context:\n{rag_result['context']}"
        )
        final_messages = [{"role": "system", "content": sys_prompt}]
        # Append recent history
        final_messages.extend([{"role": m.role, "content": m.content} for m in req.messages[-HISTORY_WINDOW:]])

    # 3. Stream or Return
    if req.stream:
        return StreamingResponse(
            _stream_generator(final_messages, req, rag_result["sources"]),
            media_type="text/event-stream"
        )
    else:
        # Sync Call
        payload = {
            "model": UPSTREAM_MODEL_RAG,
            "messages": final_messages,
            "temperature": req.temperature,
            "presence_penalty": 0.0,   # Nouveau
            "frequency_penalty": 0.3,  # <--- AJOUTER CECI (0.3 à 0.5)
            "stream": False
        }

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        async with httpx.AsyncClient() as client:
            resp = await client.post(OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=headers)
            data = resp.json()
            
        content = data["choices"][0]["message"]["content"]
        if rag_result["sources"]:
            content += f"\n\nSources: {', '.join(rag_result['sources'])}"
            
        return {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_RAG_NAME,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]
        }

async def _stream_generator(messages, req, sources):
    """Yields SSE events."""
    payload = {
        "model": UPSTREAM_MODEL_RAG,
        "messages": messages,
        "temperature": req.temperature,
        "presence_penalty": 0.0,   # Nouveau
        "frequency_penalty": 0.3,  # <--- AJOUTER CECI
        "stream": True
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", OPENAI_CHAT_COMPLETIONS_URL, json=payload, headers=headers) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        # Clean up model name in response
                        data["model"] = MODEL_RAG_NAME
                        yield f"data: {json.dumps(data)}\n\n"
                    except: pass
    
    # Append sources at the end
    if sources:
        src_text = f"\n\nSources: {', '.join(sources)}"
        chunk = {
            "id": "sources",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_RAG_NAME,
            "choices": [{"index": 0, "delta": {"content": src_text}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
    yield "data: [DONE]\n\n"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
