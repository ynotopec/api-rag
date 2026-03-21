from __future__ import annotations

import os
from typing import Iterable, List

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import SETTINGS


def iter_text_paths(raw_paths: Iterable[str]) -> List[str]:
    discovered: List[str] = []
    for raw in raw_paths:
        path = raw.strip()
        if not path:
            continue
        if os.path.isfile(path):
            discovered.append(path)
            continue
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.lower().endswith(".txt"):
                        discovered.append(os.path.join(root, filename))
    return sorted(set(discovered))


def load_documents() -> List[Document]:
    paths = iter_text_paths(SETTINGS.ingestion_paths.split(","))
    docs: List[Document] = []
    for path in paths:
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        docs = [Document(page_content="Knowledge base is empty.", metadata={"source": "system"})]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
