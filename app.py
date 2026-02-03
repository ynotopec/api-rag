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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
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
RAG_QUERY_STRATEGY = os.getenv("RAG_QUERY_STRATEGY", "rewrite+hyde")
HISTORY_WINDOW = int(os.getenv("RAG_HISTORY_WINDOW", "6"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))

# Optimization Flags
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
ENABLE_QUERY_CLASSIFICATION = os.getenv("ENABLE_QUERY_CLASSIFICATION", "true").lower() == "true"
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
ENABLE_MMR = os.getenv("ENABLE_MMR", "true").lower() == "true"

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval Settings
MMR_K = int(os.getenv("MMR_K", "8"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "20"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))
BM25_K = int(os.getenv("BM25_K", "4"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# System Resources
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60.0"))
HTTP_MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "3"))
PORT = int(os.getenv("PORT", "8080"))


# ===============================
# LRU Cache with TTL
# ===============================
class LRUCache:
    """Thread-safe LRU Cache with optional TTL."""
    
    def __init__(self, max_size: int = 1000, ttl: Optional[int] = None):
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    def _is_expired(self, timestamp: float) -> bool:
        if self._ttl is None:
            return False
        return time.time() - timestamp > self._ttl
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        value, timestamp = self._cache[key]
        if self._is_expired(timestamp):
            del self._cache[key]
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        return value
    
    def set(self, key: str, value: Any) -> None:
        # Remove oldest entries if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (value, time.time())
    
    def clear(self) -> None:
        self._cache.clear()
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# ===============================
# Performance Timer
# ===============================
@contextmanager
def timer(name: str, log_level: str = "debug"):
    """Context manager to time operations."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[PERF] {name}: {elapsed:.3f}s"
    if log_level == "info":
        logger.info(msg)
    else:
        logger.debug(msg)


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

# LRU Caches with TTL
_embedding_cache: Optional[LRUCache] = None
_query_rewrite_cache: Optional[LRUCache] = None
_hyde_cache: Optional[LRUCache] = None
_retrieval_cache: Optional[LRUCache] = None

# Metrics
_metrics = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "retrieval_time_total": 0.0,
    "llm_calls_total": 0,
    "errors_total": 0,
}


def _init_caches():
    """Initialize all caches."""
    global _embedding_cache, _query_rewrite_cache, _hyde_cache, _retrieval_cache
    _embedding_cache = LRUCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
    _query_rewrite_cache = LRUCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
    _hyde_cache = LRUCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL_SECONDS)
    _retrieval_cache = LRUCache(max_size=CACHE_MAX_SIZE // 2, ttl=CACHE_TTL_SECONDS // 2)


# ===============================
# Lifespan Events
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client, _executor, _index_lock, _refresh_task
    
    # 1. Initialize Caches
    _init_caches()
    
    # 2. Thread Pool for CPU bound tasks
    _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    _index_lock = asyncio.Lock()
    
    # 3. HTTP Client for upstream LLM calls (REUSED for all requests)
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(HTTP_TIMEOUT, connect=10.0),
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0
        ),
        http2=True  # Enable HTTP/2 for better performance
    )
    
    # 4. Load Models
    _get_embeddings()
    if ENABLE_RERANKING and RERANKER_AVAILABLE:
        asyncio.get_event_loop().run_in_executor(_executor, _get_reranker)

    # 5. Load/Build Index
    await asyncio.get_event_loop().run_in_executor(_executor, _ensure_vectorstore)

    if INGESTION_REFRESH_INTERVAL > 0:
        _refresh_task = asyncio.create_task(_refresh_index_loop())
    
    logger.info("System Ready.")
    yield
    
    # Cleanup
    if _refresh_task:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
    if _http_client:
        await _http_client.aclose()
    if _executor:
        _executor.shutdown(wait=True, cancel_futures=False)
    
    logger.info("Shutdown complete.")


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    logger.info(f"Loading embedding model: {model_name}")
    
    import torch
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    
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
    bm25_path = os.path.join(VECTORSTORE_DIR, "bm25.pkl")
    
    rebuild = RAG_FORCE_REBUILD
    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        rebuild = True
    else:
        ingestion_mtime = _get_ingestion_latest_mtime()
        _last_ingestion_mtime = ingestion_mtime
        if ingestion_mtime:
            try:
                index_mtime = os.path.getmtime(index_path)
                if ingestion_mtime > index_mtime:
                    logger.info("Ingestion sources updated, rebuilding index...")
                    rebuild = True
            except OSError:
                rebuild = True

    if rebuild:
        _build_index()
        _last_ingestion_mtime = _get_ingestion_latest_mtime()
    else:
        logger.info("Loading existing vector store...")
        try:
            with timer("load_faiss", "info"):
                _vector = FAISS.load_local(
                    VECTORSTORE_DIR, 
                    _get_embeddings(), 
                    allow_dangerous_deserialization=True
                )
            
            with open(chunks_path, "rb") as f:
                _all_docs = pickle.load(f)
            
            # Load or rebuild BM25
            if ENABLE_HYBRID_SEARCH and _all_docs:
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, "rb") as f:
                            _bm25_retriever = pickle.load(f)
                        _bm25_retriever.k = BM25_K
                        logger.info("Loaded BM25 index from cache.")
                    except Exception as e:
                        logger.warning(f"Failed to load BM25 cache: {e}. Rebuilding...")
                        _bm25_retriever = BM25Retriever.from_documents(_all_docs)
                        _bm25_retriever.k = BM25_K
                else:
                    with timer("build_bm25", "info"):
                        _bm25_retriever = BM25Retriever.from_documents(_all_docs)
                        _bm25_retriever.k = BM25_K
                        # Save BM25 for next time
                        with open(bm25_path, "wb") as f:
                            pickle.dump(_bm25_retriever, f)
                            
            logger.info(f"Loaded {len(_all_docs)} document chunks.")
        except Exception as e:
            logger.error(f"Error loading index: {e}. Rebuilding...")
            _build_index()


def _build_index():
    global _vector, _bm25_retriever, _all_docs, _last_ingestion_mtime
    
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    with timer("collect_documents", "info"):
        docs = _collect_ingestion_documents()
    
    if not docs:
        logger.warning("No ingestion documents found. Creating empty index.")
        docs = [Document(page_content="Welcome to the RAG system.", metadata={"source": "system"})]
    else:
        with timer("chunking", "info"):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", ", ", " ", ""],
                length_function=len,
            )
            docs = splitter.split_documents(docs)
    
    logger.info(f"Created {len(docs)} chunks from ingestion sources.")

    embeddings = _get_embeddings()
    
    with timer("build_faiss", "info"):
        _vector = FAISS.from_documents(docs, embeddings)
        _vector.save_local(VECTORSTORE_DIR)
    
    _all_docs = docs
    chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(docs, f)
    
    if ENABLE_HYBRID_SEARCH:
        with timer("build_bm25", "info"):
            _bm25_retriever = BM25Retriever.from_documents(docs)
            _bm25_retriever.k = BM25_K
            # Cache BM25
            bm25_path = os.path.join(VECTORSTORE_DIR, "bm25.pkl")
            with open(bm25_path, "wb") as f:
                pickle.dump(_bm25_retriever, f)
    
    _last_ingestion_mtime = _get_ingestion_latest_mtime()
    logger.info("Index build complete.")


async def _refresh_index_loop():
    global _last_ingestion_mtime
    while True:
        await asyncio.sleep(INGESTION_REFRESH_INTERVAL)
        try:
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
                # Clear caches after rebuild
                if _retrieval_cache:
                    _retrieval_cache.clear()
        except Exception as e:
            logger.error(f"Error in refresh loop: {e}")


def _iter_text_paths(paths: Iterable[str]) -> List[str]:
    collected = []
    for raw_path in paths:
        path = raw_path.strip()
        if not path:
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.lower().endswith((".txt", ".md", ".rst")):
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
        try:
            loader = TextLoader(path, encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = path
            docs.extend(loaded)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    logger.info(f"Loaded {len(docs)} text documents.")
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
                content = "\n".join([
                    f"Subject: {subject}",
                    f"From: {sender}",
                    f"Date: {date}",
                    "",
                    body.strip(),
                ]).strip()
                if not content:
                    continue
                relative_path = os.path.relpath(mbox_path, profile_dir)
                metadata = {
                    "source": f"thunderbird:{relative_path}",
                    "message_id": message.get("message-id", ""),
                }
                docs.append(Document(page_content=content, metadata=metadata))
                count += 1
    logger.info(f"Loaded {len(docs)} Thunderbird emails.")
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
    sources = {s.strip().lower() for s in INGESTION_SOURCES.split(",") if s.strip()}
    
    if "text" in sources:
        if INGESTION_TEXT_PATHS:
            text_paths = _iter_text_paths(INGESTION_TEXT_PATHS.split(","))
        else:
            text_paths = _iter_text_paths([WIKI_PATH])
        for path in text_paths:
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                continue
    
    if "thunderbird" in sources:
        if THUNDERBIRD_PROFILE_DIR and os.path.isdir(THUNDERBIRD_PROFILE_DIR):
            try:
                mtimes.append(os.path.getmtime(THUNDERBIRD_PROFILE_DIR))
            except OSError:
                pass
    
    return max(mtimes) if mtimes else None


# ===============================
# API & Auth
# ===============================
app = FastAPI(title="Optimized RAG API", version="2.2.0", lifespan=lifespan)
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
    if AUTH_TOKEN and AUTH_TOKEN != "changeme":
        if not authorization:
            raise HTTPException(401, "Authorization header required")
        token = authorization.split(" ", 1)[-1]
        if token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid Token")


# ===============================
# Logic: Helpers
# ===============================
def _get_cache_key(*args) -> str:
    """Generate a cache key from multiple arguments."""
    combined = "|".join(str(a) for a in args)
    return hashlib.md5(combined.encode()).hexdigest()


async def _call_upstream_llm(
    messages: List[Dict[str, str]], 
    model: str, 
    max_tokens: int = 1000,
    temp: float = 0.1,
    retries: int = HTTP_MAX_RETRIES
) -> Optional[str]:
    """
    Helper to call OpenAI compatible endpoint with retry logic.
    Returns None on failure instead of empty string.
    """
    global _metrics
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
        "stream": False
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    last_error = None
    for attempt in range(retries):
        try:
            _metrics["llm_calls_total"] += 1
            resp = await _http_client.post(
                OPENAI_CHAT_COMPLETIONS_URL, 
                json=payload, 
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code >= 500:
                # Server error, retry
                wait_time = 2 ** attempt
                logger.warning(f"LLM server error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            elif e.response.status_code == 429:
                # Rate limited
                wait_time = 5 * (attempt + 1)
                logger.warning(f"Rate limited (attempt {attempt + 1}). Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            else:
                # Client error, don't retry
                logger.error(f"LLM client error: {e}")
                _metrics["errors_total"] += 1
                return None
        except httpx.TimeoutException as e:
            last_error = e
            wait_time = 2 ** attempt
            logger.warning(f"LLM timeout (attempt {attempt + 1}). Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            continue
        except Exception as e:
            last_error = e
            logger.error(f"LLM unexpected error: {e}")
            _metrics["errors_total"] += 1
            return None
    
    logger.error(f"LLM call failed after {retries} attempts: {last_error}")
    _metrics["errors_total"] += 1
    return None


# ===============================
# Logic: RAG Core
# ===============================
async def _classify_query(query: str) -> str:
    """
    Classify if query needs RAG or is simple chat.
    Uses fast heuristics to avoid latency.
    """
    if not ENABLE_QUERY_CLASSIFICATION:
        return "RAG"
    
    # Simple heuristics
    query_lower = query.lower().strip()
    
    # Very short queries are usually chat
    if len(query.split()) < 3:
        return "CHAT"
    
    # Greetings and simple phrases
    chat_patterns = [
        "hello", "hi ", "hey ", "thanks", "thank you", "bye", "goodbye",
        "how are you", "what's up", "good morning", "good evening"
    ]
    if any(query_lower.startswith(p) or query_lower == p.strip() for p in chat_patterns):
        return "CHAT"
    
    # Default to RAG for information-seeking queries
    return "RAG"


async def _rewrite_query(history_str: str, current_query: str) -> str:
    """Generates a search-optimized query based on conversation context."""
    cache_key = _get_cache_key("rewrite", history_str, current_query)
    
    if ENABLE_CACHING and _query_rewrite_cache:
        cached = _query_rewrite_cache.get(cache_key)
        if cached:
            _metrics["cache_hits"] += 1
            return cached
        _metrics["cache_misses"] += 1

    system = """You are a search query optimizer. Your task is to rewrite the user's question into a better search query.
Rules:
1. Output ONLY the improved search query, nothing else
2. Resolve pronouns using the conversation context
3. Keep it concise (under 20 words)
4. Preserve the original intent"""

    prompt = f"""Conversation Context:
{history_str}

Current User Question: {current_query}

Optimized Search Query:"""
    
    rewritten = await _call_upstream_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        model=UPSTREAM_MODEL_REWRITE,
        max_tokens=64,
        temp=0.1
    )
    
    result = rewritten.strip() if rewritten else current_query
    
    # Clean up any quotes or extra formatting
    result = result.strip('"\'')
    
    if ENABLE_CACHING and _query_rewrite_cache:
        _query_rewrite_cache.set(cache_key, result)
    
    return result


async def _hyde_expand(query: str) -> Optional[str]:
    """
    Generates a Hypothetical Document Embedding (HyDE).
    Returns None if generation fails.
    """
    cache_key = _get_cache_key("hyde", query)
    
    if ENABLE_CACHING and _hyde_cache:
        cached = _hyde_cache.get(cache_key)
        if cached:
            _metrics["cache_hits"] += 1
            return cached
        _metrics["cache_misses"] += 1

    system = """Write a short, factual 3-4 sentence excerpt from a technical document that would answer this question.
Write as if you're quoting from an authoritative source. Be specific and detailed."""

    hyde_doc = await _call_upstream_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": query}],
        model=UPSTREAM_MODEL_REWRITE,
        max_tokens=150,
        temp=0.3
    )
    
    if hyde_doc and ENABLE_CACHING and _hyde_cache:
        _hyde_cache.set(cache_key, hyde_doc)
    
    return hyde_doc


def _fast_deduplicate(docs: List[Document], similarity_threshold: int = 85) -> List[Document]:
    """
    Deduplicates documents using hash + fuzzy matching.
    Much faster than embedding-based deduplication.
    """
    if not docs:
        return []
    
    unique_docs = []
    seen_hashes: Set[str] = set()
    
    for doc in docs:
        content = doc.page_content.strip()
        if not content:
            continue
            
        # 1. Exact hash check
        doc_hash = hashlib.md5(content.encode()).hexdigest()
        if doc_hash in seen_hashes:
            continue
        
        # 2. Fuzzy similarity check (only for small result sets)
        is_duplicate = False
        if RAPIDFUZZ_AVAILABLE and len(unique_docs) <= 30:
            for kept_doc in unique_docs[-10:]:  # Only compare with recent docs
                ratio = fuzz.ratio(content[:500], kept_doc.page_content[:500])
                if ratio > similarity_threshold:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique_docs.append(doc)
            seen_hashes.add(doc_hash)
    
    return unique_docs


def _reciprocal_rank_fusion(
    results_list: List[List[Document]], 
    k: int = 60
) -> List[Tuple[Document, float]]:
    """
    Combines results from multiple retrieval methods using RRF.
    Returns documents with their fusion scores.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    for docs in results_list:
        for rank, doc in enumerate(docs):
            doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0) + (1.0 / (k + rank + 1))
    
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[id], scores[id]) for id in sorted_ids]


def _rerank_documents(query: str, docs: List[Document], top_k: int = None) -> List[Document]:
    """
    Reranks documents using CrossEncoder.
    Returns top_k documents sorted by relevance.
    """
    if not ENABLE_RERANKING or not _reranker or not docs:
        return docs[:top_k] if top_k else docs
    
    # Skip reranking for very small sets
    if len(docs) < 3:
        return docs
    
    top_k = top_k or len(docs)
    
    try:
        with timer("reranking"):
            pairs = [(query, d.page_content) for d in docs]
            scores = _reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in ranked[:top_k]]
    except Exception as e:
        logger.error(f"Reranking error: {e}")
        _metrics["errors_total"] += 1
        return docs[:top_k]


def _search_faiss_mmr(query: str, k: int, fetch_k: int) -> List[Document]:
    """Perform MMR search on FAISS index."""
    if not _vector:
        return []
    
    if ENABLE_MMR:
        return _vector.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=MMR_LAMBDA
        )
    else:
        return _vector.similarity_search(query, k=k)


def _search_faiss_with_scores(
    query: str, 
    k: int, 
    min_score: float = None
) -> List[Tuple[Document, float]]:
    """
    Perform similarity search with score filtering.
    Note: FAISS returns L2 distance (lower is better), we convert to similarity.
    """
    if not _vector:
        return []
    
    min_score = min_score if min_score is not None else RAG_MIN_SCORE
    
    results = _vector.similarity_search_with_score(query, k=k * 2)  # Fetch more to filter
    
    # FAISS returns (doc, distance) where lower distance = more similar
    # Convert to similarity score (1 / (1 + distance))
    scored_results = []
    for doc, distance in results:
        similarity = 1.0 / (1.0 + distance)
        if similarity >= min_score:
            scored_results.append((doc, similarity))
    
    # Sort by similarity (highest first) and limit
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results[:k]


def _search_bm25(query: str) -> List[Document]:
    """Perform BM25 search."""
    if not _bm25_retriever:
        return []
    return _bm25_retriever.invoke(query)


# ===============================
# Logic: Main Retrieval Pipeline
# ===============================
async def _retrieve_pipeline(messages: List[ChatMessage]) -> Dict[str, Any]:
    """
    Main retrieval pipeline with parallel query expansion and search.
    """
    global _metrics
    
    user_query = messages[-1].content.strip()
    
    # Build history string from recent messages
    history_messages = messages[-(HISTORY_WINDOW + 1):-1]
    history_str = "\n".join([f"{m.role}: {m.content}" for m in history_messages])

    # 1. Check if RAG is needed
    query_type = await _classify_query(user_query)
    if query_type == "CHAT":
        return {
            "context": "", 
            "sources": [], 
            "skip": True, 
            "query": user_query,
            "strategy": "chat"
        }

    # 2. Check retrieval cache
    cache_key = _get_cache_key("retrieval", user_query, history_str[:200])
    if ENABLE_CACHING and _retrieval_cache:
        cached = _retrieval_cache.get(cache_key)
        if cached:
            _metrics["cache_hits"] += 1
            logger.debug("Retrieval cache hit")
            return cached
        _metrics["cache_misses"] += 1

    # 3. Parallel Search Setup
    all_search_results: List[List[Document]] = []
    final_query = user_query
    
    # Start immediate search with original query
    async def search_original():
        loop = asyncio.get_event_loop()
        results = []
        
        # FAISS with MMR
        if _vector:
            faiss_results = await loop.run_in_executor(
                _executor,
                lambda: _search_faiss_mmr(user_query, k=MMR_K, fetch_k=MMR_FETCH_K)
            )
            results.extend(faiss_results)
        
        # BM25 Hybrid
        if ENABLE_HYBRID_SEARCH and _bm25_retriever:
            bm25_results = await loop.run_in_executor(
                _executor,
                lambda: _search_bm25(user_query)
            )
            results.extend(bm25_results)
        
        return results

    # Start original search immediately
    original_task = asyncio.create_task(search_original())
    
    # 4. Query Expansion (parallel with original search)
    expansion_tasks = []
    
    if RAG_QUERY_STRATEGY != "simple":
        # Query Rewriting
        if "rewrite" in RAG_QUERY_STRATEGY:
            rewrite_task = asyncio.create_task(_rewrite_query(history_str, user_query))
            expansion_tasks.append(("rewrite", rewrite_task))
        
        # HyDE
        if "hyde" in RAG_QUERY_STRATEGY:
            hyde_task = asyncio.create_task(_hyde_expand(user_query))
            expansion_tasks.append(("hyde", hyde_task))

    # 5. Collect original results
    original_results = await original_task
    all_search_results.append(original_results)

    # 6. Process expansion results and run additional searches
    secondary_search_tasks = []
    
    for name, task in expansion_tasks:
        try:
            result = await task
            if result and result != user_query:
                if name == "rewrite":
                    final_query = result  # Use rewritten query for final context
                    
                    async def search_rewritten():
                        if _vector:
                            return await asyncio.get_event_loop().run_in_executor(
                                _executor,
                                lambda: _search_faiss_mmr(result, k=MMR_K, fetch_k=MMR_FETCH_K)
                            )
                        return []
                    
                    secondary_search_tasks.append(asyncio.create_task(search_rewritten()))
                
                elif name == "hyde":
                    async def search_hyde():
                        if _vector:
                            return await asyncio.get_event_loop().run_in_executor(
                                _executor,
                                lambda: _vector.similarity_search(result, k=MMR_K // 2)
                            )
                        return []
                    
                    secondary_search_tasks.append(asyncio.create_task(search_hyde()))
        except Exception as e:
            logger.warning(f"Expansion task '{name}' failed: {e}")

    # 7. Gather secondary search results
    if secondary_search_tasks:
        secondary_results = await asyncio.gather(*secondary_search_tasks, return_exceptions=True)
        for result in secondary_results:
            if isinstance(result, list):
                all_search_results.append(result)
            elif isinstance(result, Exception):
                logger.warning(f"Secondary search failed: {result}")

    # 8. Process Results (CPU-bound, run in executor)
    def process_results():
        # Filter empty results
        valid_results = [r for r in all_search_results if r]
        if not valid_results:
            return []
        
        # RRF Fusion
        with timer("rrf_fusion"):
            fused = _reciprocal_rank_fusion(valid_results)
        
        # Extract just documents
        fused_docs = [doc for doc, score in fused]
        
        # Fast Deduplication
        with timer("deduplication"):
            deduped = _fast_deduplicate(fused_docs)
        
        # Reranking (most expensive step)
        with timer("reranking"):
            final_docs = _rerank_documents(final_query, deduped, top_k=RAG_TOP_K)
        
        return final_docs

    with timer("process_results", "info"):
        final_docs = await asyncio.get_event_loop().run_in_executor(
            _executor, 
            process_results
        )

    # 9. Format Output
    if not final_docs:
        result = {
            "context": "", 
            "sources": [], 
            "skip": False, 
            "query": final_query,
            "strategy": RAG_QUERY_STRATEGY
        }
        if ENABLE_CACHING and _retrieval_cache:
            _retrieval_cache.set(cache_key, result)
        return result

    # Format chunks with clear markers
    formatted_chunks = []
    sources_set = set()
    
    for i, doc in enumerate(final_docs):
        # Clean content: normalize whitespace
        clean_content = " ".join(doc.page_content.split())
        formatted_chunks.append(f"[Excerpt {i + 1}]: {clean_content}")
        
        source = doc.metadata.get("source", "unknown")
        sources_set.add(source)

    context_text = "\n\n".join(formatted_chunks)
    sources = list(sources_set)

    result = {
        "context": context_text,
        "sources": sources,
        "skip": False,
        "query": final_query,
        "strategy": RAG_QUERY_STRATEGY,
        "doc_count": len(final_docs)
    }
    
    if ENABLE_CACHING and _retrieval_cache:
        _retrieval_cache.set(cache_key, result)
    
    return result


# ===============================
# Endpoints
# ===============================
@app.get("/healthz")
def health():
    """Health check endpoint."""
    vector_count = 0
    if _vector and hasattr(_vector, 'index') and _vector.index:
        vector_count = _vector.index.ntotal
    
    return {
        "status": "ok",
        "vectors": vector_count,
        "documents": len(_all_docs),
        "cache_sizes": {
            "query_rewrite": len(_query_rewrite_cache) if _query_rewrite_cache else 0,
            "hyde": len(_hyde_cache) if _hyde_cache else 0,
            "retrieval": len(_retrieval_cache) if _retrieval_cache else 0,
        }
    }


@app.get("/metrics")
def metrics():
    """Metrics endpoint."""
    return {
        "metrics": _metrics,
        "config": {
            "rag_top_k": RAG_TOP_K,
            "enable_hybrid": ENABLE_HYBRID_SEARCH,
            "enable_reranking": ENABLE_RERANKING,
            "enable_mmr": ENABLE_MMR,
            "query_strategy": RAG_QUERY_STRATEGY,
        }
    }


@app.get("/v1/models")
def list_models():
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_RAG_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local"
            }
        ]
    }


@app.post("/v1/chat/completions", dependencies=[Depends(check_auth)])
async def chat_endpoint(req: ChatReq):
    """
    OpenAI-compatible chat completions endpoint with RAG.
    """
    global _metrics
    _metrics["requests_total"] += 1
    
    if req.model != MODEL_RAG_NAME:
        raise HTTPException(400, f"Unknown model '{req.model}'. Use '{MODEL_RAG_NAME}'")

    if not req.messages:
        raise HTTPException(400, "Messages cannot be empty")

    # 1. Retrieve Context
    rag_start = time.perf_counter()
    rag_result = await _retrieve_pipeline(req.messages)
    rag_time = time.perf_counter() - rag_start
    _metrics["retrieval_time_total"] += rag_time
    
    logger.info(
        f"RAG Retrieval: {rag_time:.2f}s | "
        f"Docs: {rag_result.get('doc_count', 0)} | "
        f"Strategy: {rag_result.get('strategy', 'unknown')}"
    )

    # 2. Build Final Messages
    final_messages = []
    
    if rag_result["skip"] or not rag_result["context"]:
        # Pure chat mode or no results
        final_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if not rag_result["skip"]:
            final_messages.insert(0, {
                "role": "system",
                "content": "Knowledge base search returned no relevant results. Answer based on your general knowledge, and clearly state if you're uncertain."
            })
    else:
        # Inject RAG context
        sys_prompt = f"""You are a helpful assistant with access to a knowledge base.

INSTRUCTIONS:
1. Use the provided context excerpts to answer the user's question
2. If the context contains duplicate information, mention it only once
3. Be concise and well-structured in your response
4. If the answer is not in the context, say "I don't have information about that in my knowledge base"
5. Answer in the same language as the user's question
6. Do not make up information that isn't in the context

CONTEXT:
{rag_result['context']}"""

        final_messages = [{"role": "system", "content": sys_prompt}]
        
        # Add conversation history
        history_messages = req.messages[-HISTORY_WINDOW:]
        final_messages.extend([
            {"role": m.role, "content": m.content} 
            for m in history_messages
        ])

    # 3. Stream or Sync Response
    if req.stream:
        return StreamingResponse(
            _stream_generator(final_messages, req, rag_result["sources"]),
            media_type="text/event-stream"
        )
    else:
        return await _sync_completion(final_messages, req, rag_result["sources"])


async def _sync_completion(
    messages: List[Dict[str, str]], 
    req: ChatReq, 
    sources: List[str]
) -> Dict[str, Any]:
    """Handle non-streaming completion."""
    payload = {
        "model": UPSTREAM_MODEL_RAG,
        "messages": messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.0,
        "stream": False
    }
    
    if req.max_tokens:
        payload["max_tokens"] = req.max_tokens
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    try:
        resp = await _http_client.post(
            OPENAI_CHAT_COMPLETIONS_URL, 
            json=payload, 
            headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Upstream LLM error: {e}")
        raise HTTPException(502, f"Upstream LLM error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"Upstream LLM error: {e}")
        raise HTTPException(502, f"Upstream LLM unavailable: {str(e)}")
    
    content = data["choices"][0]["message"]["content"]
    
    # Append sources
    if sources:
        source_list = ", ".join(sources[:5])  # Limit displayed sources
        if len(sources) > 5:
            source_list += f" (+{len(sources) - 5} more)"
        content += f"\n\n📚 **Sources:** {source_list}"
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_RAG_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": data.get("usage", {})
    }


async def _stream_generator(
    messages: List[Dict[str, str]], 
    req: ChatReq, 
    sources: List[str]
) -> AsyncGenerator[str, None]:
    """
    Yields SSE events for streaming responses.
    Uses the shared HTTP client.
    """
    payload = {
        "model": UPSTREAM_MODEL_RAG,
        "messages": messages,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.0,
        "stream": True
    }
    
    if req.max_tokens:
        payload["max_tokens"] = req.max_tokens
    
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    
    try:
        async with _http_client.stream(
            "POST", 
            OPENAI_CHAT_COMPLETIONS_URL, 
            json=payload, 
            headers=headers
        ) as resp:
            resp.raise_for_status()
            
            async for line in resp.aiter_lines():
                if not line:
                    continue
                    
                if line.startswith("data: "):
                    data_str = line[6:]
                    
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        # Override model name
                        data["id"] = completion_id
                        data["model"] = MODEL_RAG_NAME
                        yield f"data: {json.dumps(data)}\n\n"
                    except json.JSONDecodeError:
                        continue
    
    except httpx.HTTPStatusError as e:
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_RAG_NAME,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: Upstream LLM returned {e.response.status_code}]"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_RAG_NAME,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n[Error: {str(e)}]"},
                "finish_reason": "error"
            }]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    # Append sources at the end
    if sources:
        source_list = ", ".join(sources[:5])
        if len(sources) > 5:
            source_list += f" (+{len(sources) - 5} more)"
        
        source_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": MODEL_RAG_NAME,
            "choices": [{
                "index": 0,
                "delta": {"content": f"\n\n📚 **Sources:** {source_list}"},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(source_chunk)}\n\n"
    
    # Final done message
    yield "data: [DONE]\n\n"


@app.post("/admin/rebuild-index", dependencies=[Depends(check_auth)])
async def rebuild_index():
    """Force rebuild the vector index."""
    if not _index_lock:
        raise HTTPException(503, "System not initialized")
    
    async with _index_lock:
        await asyncio.get_event_loop().run_in_executor(_executor, _build_index)
        
        # Clear caches
        if _retrieval_cache:
            _retrieval_cache.clear()
        if _query_rewrite_cache:
            _query_rewrite_cache.clear()
        if _hyde_cache:
            _hyde_cache.clear()
    
    return {"status": "ok", "message": "Index rebuilt successfully"}


@app.post("/admin/clear-cache", dependencies=[Depends(check_auth)])
async def clear_cache():
    """Clear all caches."""
    counts = {}
    
    if _retrieval_cache:
        counts["retrieval"] = len(_retrieval_cache)
        _retrieval_cache.clear()
    if _query_rewrite_cache:
        counts["query_rewrite"] = len(_query_rewrite_cache)
        _query_rewrite_cache.clear()
    if _hyde_cache:
        counts["hyde"] = len(_hyde_cache)
        _hyde_cache.clear()
    
    return {"status": "ok", "cleared": counts}


# ===============================
# Main Entry Point
# ===============================
if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn for production
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True,
        workers=1,  # Single worker since we use ThreadPoolExecutor
    )
