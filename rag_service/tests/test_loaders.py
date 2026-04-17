import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loaders import load_markdown, load_pdf, load_docx, load_txt, load_document, Document
from corpus import load_corpus

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def test_load_markdown():
    text = load_markdown(os.path.join(FIXTURES_DIR, "scrum_basics.md"))
    assert len(text) > 0
    assert "Scrum" in text


def test_load_pdf():
    text = load_pdf(os.path.join(FIXTURES_DIR, "agile_manifesto.pdf"))
    assert len(text) > 0
    assert "Agile" in text


def test_load_docx():
    text = load_docx(os.path.join(FIXTURES_DIR, "devops_practices.docx"))
    assert len(text) > 0
    assert "DevOps" in text


def test_load_txt():
    text = load_txt(os.path.join(FIXTURES_DIR, "project_planning.txt"))
    assert len(text) > 0
    assert "планирование" in text.lower()


def test_load_document_returns_dataclass():
    doc = load_document(os.path.join(FIXTURES_DIR, "scrum_basics.md"))
    assert isinstance(doc, Document)
    assert len(doc.text) > 0
    assert doc.source.endswith("scrum_basics.md")
    assert doc.metadata["extension"] == ".md"
    assert doc.metadata["size_bytes"] > 0


def test_load_corpus_finds_all_fixture_formats():
    docs = load_corpus(FIXTURES_DIR)
    assert len(docs) == 4
    extensions = {doc.metadata["extension"] for doc in docs}
    assert extensions == {".md", ".pdf", ".docx", ".txt"}
    for doc in docs:
        assert len(doc.text) > 0, f"Empty text in {doc.source}"
