"""GET /api/documents and GET /api/documents/{filename} — knowledge-base PDFs.

Intentionally unauthenticated: PDFs are opened in a new browser tab via plain
GET navigation, which can't carry the Bearer token, and the corpus is public
reference material (standards, textbooks). The endpoints proxy the RAG service,
which owns the knowledge-base files.
"""

import os

from fastapi import APIRouter, HTTPException, Request, Response

from ..rag_client import RagClient, RagError
from ..schemas import DocumentsResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
def list_documents(request: Request) -> DocumentsResponse:
    rag: RagClient = request.app.state.rag
    try:
        data = rag.list_documents()
    except RagError as e:
        raise HTTPException(status_code=502, detail=f"RAG service error: {e}")
    return DocumentsResponse(documents=data.get("documents", []))


@router.get("/documents/{filename}")
def get_document(filename: str, request: Request) -> Response:
    # Reject anything that isn't a bare filename to guard against path traversal.
    if filename != os.path.basename(filename) or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    rag: RagClient = request.app.state.rag
    try:
        doc = rag.get_document(filename)
    except RagError as e:
        raise HTTPException(status_code=502, detail=f"RAG service error: {e}")
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    content, content_type = doc
    return Response(
        content=content,
        media_type=content_type or "application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
