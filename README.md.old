Parfait, allons-y directement.
Voici une **version “prod-ready minimaliste”** :

* un **schéma RAG**
* un **`main.py` FastAPI** prêt à servir
* un **`requirements.txt`**

L’idée : tu mets tes `.txt` dans `./data/`, tu lances le serveur, tu appelles `/ask`.

---

## 1. Schéma RAG (vue simple)

```mermaid
flowchart LR
    U[Utilisateur<br>Question] --> QEmb[Embedding de la question]
    QEmb --> VS[Vector Store<br>(Chroma)]
    VS --> Ctx[Top-k chunks pertinents]
    Ctx --> Prompt[Construction du prompt RAG]
    U --> Prompt
    Prompt --> LLM[Appel LLM<br>(API OpenAI-compatible)]
    LLM --> R[Réponse + sources]
```

---

## 2. `main.py` – FastAPI + Chroma + SentenceTransformers + LLM OpenAI-compatible

```python
from pathlib import Path
import os
from typing import List, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ==========
# CONFIG
# ==========

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "docs"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM : n'importe quel endpoint OpenAI-compatible (OpenAI, LiteLLM, vLLM, Ollama, etc.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optionnel, pour un endpoint custom
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY doit être défini dans l'environnement.")

# ==========
# INIT GLOBAL
# ==========

app = FastAPI(title="RAG minimaliste", version="0.1.0")

# Embedder chargé une fois pour toutes
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# Vector store Chroma (persistant)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# Client LLM
client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_BASE_URL:
    client_kwargs["base_url"] = OPENAI_BASE_URL
llm_client = OpenAI(**client_kwargs)


# ==========
# MODELES Pydantic
# ==========

class AskRequest(BaseModel):
    question: str
    top_k: int = 5


class Source(BaseModel):
    source: str
    chunk_index: int
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# ==========
# UTILITAIRES
# ==========

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Chunking simple par caractères, suffisant pour un exemple minimaliste.
    En prod, tu peux passer à un split par tokens / titres.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # léger recouvrement
        if start < 0:
            start = 0
    return chunks


def load_text_documents(data_dir: Path) -> List[Dict[str, Any]]:
    """
    Charge tous les fichiers .txt dans data_dir.
    Retourne une liste de dicts {doc_id, source, text}.
    """
    docs: List[Dict[str, Any]] = []
    if not data_dir.exists():
        print(f"[WARN] DATA_DIR {data_dir} n'existe pas.")
        return docs

    for path in data_dir.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        doc_id = path.stem
        docs.append({"doc_id": doc_id, "source": str(path), "text": text})
    return docs


def bootstrap_index(rebuild: bool = True) -> None:
    """
    (Re)construit l'index Chroma à partir de ./data/*.txt
    Dans un vrai setup, tu ferais l'ingestion dans un job séparé.
    """
    docs = load_text_documents(DATA_DIR)
    if not docs:
        print("[WARN] Aucun document trouvé dans ./data")
        return

    if rebuild:
        print("[INFO] Nettoyage de la collection existante...")
        try:
            collection.delete(where={})
        except Exception as e:
            print(f"[WARN] Erreur lors du delete: {e}")

    print(f"[INFO] Ingestion de {len(docs)} documents...")

    ids = []
    texts = []
    metadatas = []

    for doc in docs:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{doc['doc_id']}_{idx}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append(
                {
                    "source": doc["source"],
                    "chunk_index": idx,
                    "doc_id": doc["doc_id"],
                }
            )

    if not texts:
        print("[WARN] Aucun chunk à indexer.")
        return

    print(f"[INFO] Embedding de {len(texts)} chunks...")
    embeddings = embedder.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=True)

    print("[INFO] Ajout au vector store...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    print("[INFO] Index prêt.")


def build_context(documents: List[str], metadatas: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """
    Concatène les chunks en un contexte texte,
    en gardant une trace des sources.
    """
    pieces = []
    total_chars = 0

    for doc, meta in zip(documents, metadatas):
        header = f"[source={meta.get('source')}, chunk={meta.get('chunk_index')}]"
        block = f"{header}\n{doc}\n"
        if total_chars + len(block) > max_chars:
            break
        pieces.append(block)
        total_chars += len(block)

    return "\n---\n".join(pieces)


def call_llm(question: str, context: str) -> str:
    """
    Appel du LLM via API OpenAI-compatible.
    Prompt RAG simple, avec consignes pour rester dans le contexte.
    """
    system_msg = (
        "Tu es un assistant qui répond STRICTEMENT à partir du contexte fourni. "
        "Si l'information n'est pas dans le contexte, dis que tu ne sais pas."
    )

    user_msg = (
        f"CONTEXTE:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Réponds en français, de manière claire et concise."
    )

    resp = llm_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()


# ==========
# ÉVÈNEMENT STARTUP
# ==========

@app.on_event("startup")
def on_startup():
    print("[INFO] Démarrage de l'appli, construction de l'index...")
    bootstrap_index(rebuild=True)


# ==========
# ENDPOINTS
# ==========

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    # 1) Embedding de la question
    query_emb = embedder.encode([req.question], convert_to_numpy=True)[0]

    # 2) Recherche vectorielle
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=req.top_k,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        raise HTTPException(status_code=404, detail="Aucun passage pertinent trouvé.")

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[None]])[0]

    # 3) Construit le contexte
    context = build_context(documents, metadatas)

    # 4) Appel LLM
    answer = call_llm(req.question, context)

    # 5) Formate les sources
    # NB : Chroma retourne une distance, pas un score de similarité.
    # On peut la transformer grossièrement en pseudo-score.
    sources: List[Source] = []
    for meta, dist in zip(metadatas, distances):
        score = None
        if dist is not None:
            score = float(max(0.0, 1.0 - dist))  # pseudo-score dans [0,1] pour illustration

        sources.append(
            Source(
                source=str(meta.get("source")),
                chunk_index=int(meta.get("chunk_index", -1)),
                score=score,
            )
        )

    return AskResponse(answer=answer, sources=sources)
```

---

## 3. `requirements.txt`

```text
fastapi
uvicorn[standard]
sentence-transformers
chromadb
openai
```

---

## 4. Comment l’utiliser

1. Crée la structure :

```bash
mkdir -p rag-minimal/data
cd rag-minimal
# mets quelques fichiers .txt dans ./data
```

2. Ajoute `main.py` et `requirements.txt` ci-dessus.

3. Installe les dépendances :

```bash
pip install -r requirements.txt
```

4. Exporte tes variables d’env :

```bash
export OPENAI_API_KEY="ta_clef"
# optionnel : si tu utilises un endpoint OpenAI-compatible custom
# export OPENAI_BASE_URL="https://ton-endpoint/v1"
# export OPENAI_MODEL="gpt-4.1-mini"  # ou autre
```

5. Lance le serveur :

```bash
uvicorn main:app --reload
```

6. Test rapide (par ex. via `curl`) :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quelle est l idée principale du document X ?","top_k": 4}'
```

---

Si tu veux, on peut faire une **v3 encore plus agressive en latence** (pré-chargement, workers, cache sur les questions, ou version GPU avec vLLM / sglang).
