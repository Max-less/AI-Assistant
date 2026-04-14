"""
Load document corpus from a directory.
Recursively walks the folder and loads all supported files.
"""

import os
from loaders import Document, load_document, LOADERS


def load_corpus(directory: str) -> list[Document]:
    """
    Recursively walk directory, load all files
    with supported extensions (.md, .pdf, .docx, .txt).
    Returns a list of Document objects.
    """
    documents = []
    supported_extensions = set(LOADERS.keys())

    for root, _dirs, files in os.walk(directory):
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_extensions:
                continue
            filepath = os.path.join(root, filename)
            try:
                doc = load_document(filepath)
                documents.append(doc)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    return documents