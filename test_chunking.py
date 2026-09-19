import app
from langchain_core.documents import Document


def test_hierarchical_chunking_links_small_chunks_to_big_parent(monkeypatch):
    monkeypatch.setattr(app, "CHUNK_BIG_SIZE", 80)
    monkeypatch.setattr(app, "CHUNK_SMALL_SIZE", 30)
    monkeypatch.setattr(app, "CHUNK_BIG_OVERLAP", 10)
    monkeypatch.setattr(app, "CHUNK_SMALL_OVERLAP", 5)
    document = Document(
        page_content=" ".join(f"word-{index}" for index in range(50)),
        metadata={"source": "fixture.txt"},
    )

    chunks, parents = app._split_documents_hierarchically([document])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= app.CHUNK_SMALL_SIZE for chunk in chunks)
    assert all(chunk.metadata["parent_chunk_id"] for chunk in chunks)
    assert all(
        chunk.page_content in parents[chunk.metadata["parent_chunk_id"]].page_content
        for chunk in chunks
    )
    assert all("parent_chunk_content" not in chunk.metadata for chunk in chunks)


def test_expand_parent_chunks_collapses_siblings(monkeypatch):
    metadata = {
        "source": "fixture.txt",
        "parent_chunk_id": "parent-1",
    }
    monkeypatch.setattr(
        app,
        "_parent_docs",
        {
            "parent-1": Document(
                page_content="The complete parent context.",
                metadata={"source": "fixture.txt"},
            )
        },
    )
    chunks = [
        Document(page_content="complete parent", metadata={**metadata, "_retrieval_score": 0.1}),
        Document(page_content="parent context", metadata={**metadata, "_retrieval_score": 0.2}),
    ]

    expanded = app._expand_parent_chunks(chunks)

    assert len(expanded) == 1
    assert expanded[0].page_content == "The complete parent context."
    assert expanded[0].metadata["matched_small_chunk"] == "complete parent"
    assert expanded[0].metadata["source"] == "fixture.txt"


def test_hierarchical_chunking_rejects_small_size_larger_than_big(monkeypatch):
    monkeypatch.setattr(app, "CHUNK_BIG_SIZE", 20)
    monkeypatch.setattr(app, "CHUNK_SMALL_SIZE", 21)

    try:
        app._split_documents_hierarchically([Document(page_content="content")])
    except ValueError as exc:
        assert "CHUNK_SMALL_SIZE" in str(exc)
    else:
        raise AssertionError("Expected invalid chunk sizes to be rejected")
