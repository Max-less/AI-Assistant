import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunker import (
    Chunk,
    estimate_tokens,
    split_by_paragraphs,
    merge_to_target_size,
    chunk_document,
)
from corpus import load_corpus

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_split_by_paragraphs_double_newline():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    result = split_by_paragraphs(text)
    assert len(result) == 3
    assert result[0] == "First paragraph."


def test_split_by_paragraphs_fallback_single_newline():
    text = "Line one\nLine two\nLine three"
    result = split_by_paragraphs(text)
    assert len(result) == 3


def test_split_by_paragraphs_strips_empty():
    text = "\n\nHello\n\n\n\nWorld\n\n"
    result = split_by_paragraphs(text)
    assert len(result) == 2


def test_merge_to_target_size_basic():
    paragraphs = ["Short.", "Also short.", "Third one.", "Fourth one."]
    chunks = merge_to_target_size(paragraphs, target_tokens=10, overlap=0)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_merge_to_target_size_overlap():
    paragraphs = [f"Paragraph {i} with some content here." for i in range(6)]
    chunks = merge_to_target_size(paragraphs, target_tokens=20, overlap=1)
    # With overlap, consecutive chunks should share text
    if len(chunks) >= 2:
        last_para_of_first = chunks[0].split("\n\n")[-1]
        assert last_para_of_first in chunks[1]


def test_merge_empty():
    assert merge_to_target_size([]) == []


def test_chunk_document_on_real_file(capsys):
    """Load a real document from knowledge_base and print first 3 chunks."""
    docs = load_corpus(KB_DIR)
    assert len(docs) > 0, "No documents in knowledge_base"

    doc = docs[0]
    chunks = chunk_document(doc, target_tokens=300, overlap=1)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert len(chunk.text) > 0
        assert "::" in chunk.chunk_id
        assert chunk.metadata["chunk_index"] >= 0

    # Print first 3 chunks for manual inspection
    print(f"\n{'='*60}")
    print(f"Document: {doc.metadata.get('filename', doc.source)}")
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ({chunk.metadata['estimated_tokens']} tokens) ---")
        print(chunk.text[:300])
        if len(chunk.text) > 300:
            print("...")
    print(f"{'='*60}")
