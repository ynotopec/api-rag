from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import SETTINGS
from .ingest import load_documents


@dataclass
class RetrievalResult:
    context: str
    sources: List[str]


class RetrievalEngine:
    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(
            model_name=SETTINGS.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vector: FAISS | None = None

    def load_or_build(self) -> None:
        index_path = os.path.join(SETTINGS.vectorstore_dir, "index.faiss")
        chunks_path = os.path.join(SETTINGS.vectorstore_dir, "chunks.pkl")

        should_rebuild = SETTINGS.force_rebuild
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            should_rebuild = True

        if should_rebuild:
            docs = load_documents()
            self._vector = FAISS.from_documents(docs, self._embeddings)
            self._vector.save_local(SETTINGS.vectorstore_dir)
            with open(chunks_path, "wb") as handle:
                pickle.dump(docs, handle)
            return

        self._vector = FAISS.load_local(
            SETTINGS.vectorstore_dir,
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        if not self._vector:
            return RetrievalResult(context="", sources=[])
        top_k = k or SETTINGS.rag_top_k
        docs: Sequence[Document] = self._vector.similarity_search(query, k=top_k)
        if not docs:
            return RetrievalResult(context="", sources=[])

        snippets = [f"[{i + 1}] {' '.join(doc.page_content.split())}" for i, doc in enumerate(docs)]
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
        return RetrievalResult(context="\n\n".join(snippets), sources=sources)
