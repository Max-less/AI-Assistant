"""
Document loaders for various formats.
Supported: Markdown (.md), PDF (.pdf), DOCX (.docx), TXT (.txt).
"""

import os
from dataclasses import dataclass, field
from pypdf import PdfReader
from docx import Document as DocxDocument


@dataclass
class Document:
    text: str
    source: str  # file path
    metadata: dict = field(default_factory=dict)


def load_markdown(path: str) -> str:
    """Load text from a Markdown file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_docx(path: str) -> str:
    """Extract text from a DOCX file using python-docx."""
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def load_txt(path: str) -> str:
    """Load text from a TXT file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


LOADERS = {
    ".md": load_markdown,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_document(path: str) -> Document:
    """
    Dispatcher: detect format by file extension and load the document.
    Returns a Document with text, source path, and metadata.
    """
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise ValueError(f"Unsupported format: {ext} ({path})")

    text = loader(path)
    return Document(
        text=text,
        source=path,
        metadata={
            "filename": os.path.basename(path),
            "extension": ext,
            "size_bytes": os.path.getsize(path),
        },
    )